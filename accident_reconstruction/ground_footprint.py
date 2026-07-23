"""Orientation-invariant ground anchors from a vehicle's contact contour.

The legacy anchor is the axis-aligned mask box's bottom-centre,
``((x1 + x2) // 2, y2)`` (:func:`prompt_track_accident.mask_box_and_anchor`).
Its ``y2`` is sound -- the silhouette's lowest pixel really is on the road plane,
so the homography projects it exactly -- but ``(x1 + x2) // 2`` names a *different
physical point on the car* depending on which way the car faces:

- **Rear-on** (driving straight away from the camera): the silhouette is the car's
  back, so the box's x-centre is the rear bumper's centre. The anchor lands near
  the rear-axle ground contact. Fine.
- **Turned**: the silhouette now spans the car's SIDE, so the box's x-centre is
  the midpoint of the rear-corner-to-front-corner diagonal while ``y2`` is a
  near-side wheel. The pair is not a point on the contact patch at all -- it is
  the bottom-centre of a rectangle circumscribing an oblique object, and the
  anchor slides sideways by roughly half a car width through the turn.

That lateral slide is a pure artefact: it shows up downstream as a kink in the
trajectory and a speed spike at the corner (BMW clip, ~114 km/h false peak).

The fix here uses the fact that **every** pixel of the mask's bottom contour lies
on the road plane, so the whole contour projects exactly. Projected to metres it
traces the visible boundary of the car's ground footprint -- typically an "L" of
one full side plus one full end. Fitting a rectangle of the car's known size to
that L and taking its CENTRE gives an anchor that is orientation-invariant by
construction, because the centre is defined by the car, not by the silhouette.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Rough footprint aspect (width / length) per vehicle class, used when only a
# length is configured. A passenger car is ~4.5 x 1.8 m.
DEFAULT_WIDTH_RATIO = 0.40

# Contour columns are subsampled to at most this many points before projection.
# A 1080p car spans hundreds of columns; ~64 already pins a 2-parameter fit, and
# it keeps the sidecar small.
MAX_CONTOUR_POINTS = 64

# A contour shorter than this (in columns) is too small a slice of the footprint
# to fit reliably -- distant or badly occluded vehicles fall back to the legacy
# anchor rather than getting a confidently wrong one.
MIN_CONTOUR_POINTS = 8

# A tracking gap is bridged by straight-line interpolation only when the heading
# just before and just after it agree within this many degrees -- i.e. the
# vehicle went straight across the gap. A larger heading change means it TURNED in
# the gap, where a straight line would cut the corner, so that gap is left open
# (honest: the pipeline never saw the cornering path). See auto_reconstruct.
GAP_FILL_MAX_TURN_DEG = 25.0
# Never invent more than this many consecutive frames, even on a straight run.
GAP_FILL_MAX_FRAMES = 40

# The projected contour's major extent must land in this band, as a fraction of
# the footprint LENGTH, for the fit to be trustworthy.
#
# This is really a HOMOGRAPHY-FAITHFULNESS check. The fit seats a rigid rectangle
# of the vehicle's KNOWN metric size, so it only works where the homography maps
# that size back to itself. When the calibration under-scales -- as the BMW CCTV
# does, compressing a real 4.3 m car to ~2.4 m of projected contour (~56%) -- the
# assumed rectangle is far bigger than the data and its centre slides freely,
# adding metres of phantom jitter (worse than the legacy anchor). Requiring the
# span near 1.0x means "the projection preserves this vehicle's scale here"; a
# span far from 1.0x means it does not, so we decline and keep the legacy anchor.
MIN_SPAN_FRACTION = 0.7
MAX_SPAN_FRACTION = 1.6

# If the scale gate accepts fewer than this fraction of a vehicle's frames, the
# homography is not metrically faithful for it, so the caller discards even the
# accepted frames and keeps the whole legacy track (all-or-nothing per vehicle).
MIN_FITTED_FRACTION = 0.5


def bottom_contour(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """The mask's per-column lowest pixel -- the vehicle's ground-contact line.

    Args:
        mask: Boolean ``(H, W)`` instance mask.

    Returns:
        ``(n, 2)`` int array of ``(x, y)`` pixels, ordered by ``x``. Empty when
        the mask has no set pixels.

    Examples:
        ```python
        import numpy as np
        m = np.zeros((4, 3), dtype=bool)
        m[1:3, 0] = True     # column 0 reaches down to row 2
        m[1:2, 2] = True     # column 2 only to row 1
        bottom_contour(m).tolist()
        # [[0, 2], [2, 1]]
        ```
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    occupied = mask.any(axis=0)
    xs = np.flatnonzero(occupied)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    # argmax on the reversed rows finds the last True per column in one pass.
    ys = mask.shape[0] - 1 - np.argmax(mask[::-1, xs], axis=0)
    return np.column_stack([xs, ys]).astype(np.int32)


def subsample(points: NDArray[np.int32], limit: int = MAX_CONTOUR_POINTS):
    """Evenly thin ``points`` to at most ``limit`` rows, preserving both ends.

    Examples:
        ```python
        import numpy as np
        pts = np.arange(20).reshape(10, 2)
        subsample(pts, 3).tolist()
        # [[0, 1], [8, 9], [18, 19]]
        ```
    """
    n = len(points)
    if n <= limit:
        return points
    idx = np.linspace(0, n - 1, limit).round().astype(int)
    return points[idx]


# Half-width of the column band (as a fraction of the contour's pixel width) that
# the on-contour anchor averages the ground-contact row over, for robustness.
_ANCHOR_BAND_FRACTION = 0.12


def contour_anchor_px(contour: NDArray[np.int32]) -> tuple[float, float] | None:
    """A scale-independent ground anchor: the contact point under the body centre.

    The legacy anchor is the axis-aligned box's bottom-centre, ``((x1 + x2) // 2,
    y2)``. That pairs the horizontal box-centre with the single lowest row ``y2``
    -- which belong to DIFFERENT columns -- so the pixel it names is usually not
    on the car at all:

    - **Rear-on**, ``y2`` is a side wheel while the centre column is the higher
      bumper, so ``(mid-x, y2)`` floats on the road BELOW/behind the car (visible
      as the detached marker in the BMW rear frames).
    - **Turned**, ``mid-x`` is the midpoint of the box diagonal, which slides
      across the body as the silhouette rotates -- the lateral turn drift.

    This instead takes the contour's own median column (a robust horizontal centre
    of the visible body) and the ground-contact row THERE, so the anchor always
    lands ON the contact contour. It uses only pixel geometry -- no vehicle size,
    no homography scale -- so it is safe where the scale-dependent footprint fit
    is not (e.g. the under-scaled BMW homography). It does not recover the true
    body centre (a single silhouette can't), but it removes the gross
    box-corner artefact in both the image and, once projected, the ground plane.

    Args:
        contour: ``(n, 2)`` ground-contact pixels (from :func:`bottom_contour`).

    Returns:
        ``(x, y)`` pixel anchor, or None for an empty contour.

    Examples:
        ```python
        import numpy as np
        # A shallow "U": wheels low at the sides, bumper higher in the middle.
        c = np.array([[0, 10], [5, 6], [10, 6], [15, 6], [20, 10]])
        contour_anchor_px(c)  # median column x=10, contact row there = 6
        # (10.0, 6.0)
        ```
    """
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim != 2 or len(c) == 0:
        return None
    median_x = float(np.median(c[:, 0]))
    width = float(np.ptp(c[:, 0]))
    band = np.abs(c[:, 0] - median_x) <= max(3.0, _ANCHOR_BAND_FRACTION * width)
    contact_y = float(np.median(c[band, 1]))
    return median_x, contact_y


def rect_boundary_distance(
    points: NDArray[np.float64],
    center: NDArray[np.float64],
    heading: float,
    length_m: float,
    width_m: float,
) -> NDArray[np.float64]:
    """Distance from each point to a rotated rectangle's boundary, in metres.

    The rectangle is centred at ``center``, ``length_m`` along ``heading`` and
    ``width_m`` across it. Uses the standard box signed-distance field, taking
    the absolute value so points inside and outside are both penalised by their
    distance to the EDGE -- the contour traces the footprint's outline, not its
    interior.

    Args:
        points: ``(n, 2)`` metric points.
        center: ``(2,)`` or ``(k, 2)`` candidate centre(s) in metres.
        heading: Vehicle heading in radians (``atan2(north, east)``).
        length_m: Footprint length (along heading), > 0.
        width_m: Footprint width (across heading), > 0.

    Returns:
        ``(n,)`` distances for a single centre, else ``(k, n)``.

    Examples:
        ```python
        import numpy as np
        pts = np.array([[2.0, 0.0], [0.0, 0.0]])
        # 4 x 2 m box at the origin, pointing east: (2, 0) is ON the edge,
        # the centre is 1 m from the nearest (long) edge.
        c = np.array([0.0, 0.0])
        rect_boundary_distance(pts, c, 0.0, 4.0, 2.0).round(3).tolist()
        # [0.0, 1.0]
        ```
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    centers = np.atleast_2d(np.asarray(center, dtype=np.float64))
    # Offsets from every candidate centre to every point: (k, n, 2).
    delta = points[None, :, :] - centers[:, None, :]
    # Into the vehicle frame: s along the heading, t across it.
    s = delta[..., 0] * cos_h + delta[..., 1] * sin_h
    t = -delta[..., 0] * sin_h + delta[..., 1] * cos_h

    qx = np.abs(s) - length_m / 2.0
    qy = np.abs(t) - width_m / 2.0
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    distance = np.abs(outside + inside)
    return distance[0] if np.ndim(center) == 1 else distance


def _huber(residual: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    """Huber loss -- quadratic near zero, linear beyond ``delta``.

    Keeps a few stray contour pixels (mask bleeding onto a shadow or a kerb)
    from dragging the fit, without discarding them outright.
    """
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta,
        0.5 * absolute**2,
        delta * (absolute - 0.5 * delta),
    )


def fit_footprint_center(
    points_m: NDArray[np.float64],
    heading: float,
    length_m: float,
    width_m: float,
    *,
    search_m: float = 6.0,
    huber_delta: float = 0.35,
) -> tuple[float, float] | None:
    """Fit a known-size footprint rectangle to contact points; return its centre.

    Only the centre is free (2 parameters) -- the size comes from the vehicle and
    the heading from its track -- so a coarse-to-fine sweep is both exhaustive
    enough to avoid local minima and cheap. Four refinement levels take the grid
    step from ``search_m / 5`` down to about a centimetre.

    Args:
        points_m: ``(n, 2)`` projected contact points in metres.
        heading: Vehicle heading in radians.
        length_m: Known footprint length in metres (> 0).
        width_m: Known footprint width in metres (> 0).
        search_m: Half-width of the initial search window around the points'
            centroid. Must cover half a vehicle plus projection error.
        huber_delta: Residual (metres) beyond which outliers stop dominating.

    Returns:
        ``(east_m, north_m)`` of the footprint centre, or None when there are too
        few points to constrain the fit.

    Examples:
        ```python
        import numpy as np
        # Two full edges of a 4 x 2 m footprint centred at (10, 5), heading east:
        # the near long edge and one end. The fit should recover the centre.
        side = np.column_stack([np.linspace(8, 12, 25), np.full(25, 4.0)])
        end = np.column_stack([np.full(9, 12.0), np.linspace(4, 6, 9)])
        pts = np.vstack([side, end])
        [round(v, 1) for v in fit_footprint_center(pts, 0.0, 4.0, 2.0)]
        # [10.0, 5.0]
        ```
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or len(points) < MIN_CONTOUR_POINTS:
        return None
    if length_m <= 0 or width_m <= 0:
        return None

    # Decline when the contour is too small a slice of the footprint to pin its
    # centre: a rigid rectangle fitted to a blob far smaller than itself slides
    # freely (see MIN_SPAN_FRACTION). Major extent = larger stddev axis * range,
    # approximated by the peak-to-peak along the dominant PCA direction.
    centered = points - points.mean(axis=0)
    if len(points) >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        major_extent = np.ptp(centered @ vt[0])
        if not (
            MIN_SPAN_FRACTION * length_m <= major_extent <= MAX_SPAN_FRACTION * length_m
        ):
            return None

    center = points.mean(axis=0)
    span = search_m
    for _ in range(4):
        offsets = np.linspace(-span, span, 11)
        grid = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1)
        candidates = (center + grid.reshape(-1, 2)).astype(np.float64)
        distances = rect_boundary_distance(
            points, candidates, heading, length_m, width_m
        )
        cost = _huber(distances, huber_delta).mean(axis=1)
        center = candidates[int(np.argmin(cost))]
        span /= 5.0  # 11 samples over +/-span -> next window covers the winner's cell

    return float(center[0]), float(center[1])


def headings_from_track(
    frames: list[int],
    positions_m: NDArray[np.float64],
    *,
    window: int = 5,
    min_step_m: float = 0.15,
) -> dict[int, float]:
    """Per-frame heading (radians) from the direction of travel along a track.

    A vehicle's footprint is aligned with its motion, so the track itself supplies
    the fit's orientation. Headings are taken over a ``window``-frame span to ride
    out per-frame jitter, and a heading is only updated once the vehicle has
    actually moved ``min_step_m`` -- while it is stopped the last real heading is
    held, since a parked car keeps its orientation.

    Args:
        frames: Frame numbers, ascending.
        positions_m: ``(n, 2)`` metric positions matching ``frames``.
        window: Span in samples used for each finite difference.
        min_step_m: Movement below which the direction is treated as noise.

    Returns:
        ``{frame: heading_rad}``. Empty when the track never moves enough.

    Examples:
        ```python
        import numpy as np
        pts = np.column_stack([np.arange(6.0), np.zeros(6)])  # due east
        h = headings_from_track([0, 1, 2, 3, 4, 5], pts, window=2)
        round(h[0], 3)
        # 0.0
        ```
    """
    n = len(frames)
    positions = np.asarray(positions_m, dtype=np.float64)
    if n == 0 or len(positions) != n:
        return {}

    out: dict[int, float] = {}
    last: float | None = None
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n - 1, lo + window)
        step = positions[hi] - positions[lo]
        if float(np.hypot(*step)) >= min_step_m:
            last = float(np.arctan2(step[1], step[0]))
        if last is not None:
            out[frames[i]] = last

    # Frames before the vehicle first moved get the earliest known heading: it was
    # already pointing that way while stopped at the line.
    if out:
        first = out[min(out)]
        for frame in frames:
            out.setdefault(frame, first)
    return out


def smooth_positions(
    frames: list[int],
    positions: NDArray[np.float64],
    *,
    half_window_frames: int = 2,
) -> NDArray[np.float64]:
    """Gap-aware moving average of a centre track.

    The per-frame footprint fit is unbiased but pixel-quantised contours make it
    jitter a few centimetres frame to frame (worse on a badly-scaled homography).
    A vehicle's centre path is physically smooth, so averaging each sample with
    its neighbours within ``half_window_frames`` removes that jitter while keeping
    the mean -- and therefore the turn-bias correction -- intact.

    The window is measured in FRAMES, not samples, so it never blends across a
    tracking gap: if the nearest neighbour is 20 frames away (a dropout), the
    sample is left untouched instead of being pulled toward the far side.

    Args:
        frames: Frame numbers, ascending.
        positions: ``(n, 2)`` metric positions matching ``frames``.
        half_window_frames: Neighbours within this many frames each side are
            averaged in. ``0`` disables smoothing.

    Returns:
        ``(n, 2)`` smoothed positions.

    Examples:
        ```python
        import numpy as np
        # A single-frame spike is pulled back toward its neighbours...
        pos = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        smooth_positions([0, 1, 2], pos, half_window_frames=1)[1].round(3).tolist()
        # [0.0, 0.333]
        # ...but a sample across a 20-frame gap is left alone.
        pos2 = np.array([[0.0, 0.0], [5.0, 5.0]])
        smooth_positions([0, 20], pos2, half_window_frames=2)[0].tolist()
        # [0.0, 0.0]
        ```
    """
    n = len(frames)
    points = np.asarray(positions, dtype=np.float64)
    if half_window_frames <= 0 or n < 3:
        return points.copy()
    frame_arr = np.asarray(frames)
    out = np.empty_like(points)
    for i in range(n):
        near = np.abs(frame_arr - frame_arr[i]) <= half_window_frames
        out[i] = points[near].mean(axis=0)
    return out


# Savitzky-Golay defaults. Window 7 (half 3) with a quadratic is the sweet spot
# reported for smoothing vehicle-trajectory POSITION data (e.g. the NGSIM set):
# it removes tracking jitter while preserving the path's real shape -- crucially,
# genuine turns and the sharp motion around a collision are kept, unlike a moving
# average that rounds every corner. See docs/summary.md for the sourcing.
SAVGOL_HALF_WINDOW = 3
SAVGOL_ORDER = 2


def savgol_smooth(
    frames: list[int],
    positions: NDArray[np.float64],
    *,
    half_window_frames: int = SAVGOL_HALF_WINDOW,
    order: int = SAVGOL_ORDER,
) -> NDArray[np.float64]:
    """Savitzky-Golay position smoothing, generalised to non-uniform frames.

    For each sample, fit a low-order polynomial (default quadratic) by least
    squares to the neighbours within ``half_window_frames`` and take its value at
    the sample. Where the samples are evenly spaced this is exactly a
    Savitzky-Golay filter; fitting in the FRAME domain generalises it across
    tracking gaps and variable frame rates, and never blends across a dropout.

    Why this over the moving average (:func:`smooth_positions`): a moving average
    is an order-0 fit, so it flattens curvature and rounds real corners -- it
    would soften the very turn we are trying to represent. An order-2 fit follows
    the local curve, so it removes jitter while leaving the trajectory's shape
    (and the abrupt motion of a collision) intact. This is the documented choice
    for smoothing vehicle-trajectory datasets.

    Args:
        frames: Frame numbers, ascending.
        positions: ``(n, 2)`` metric positions matching ``frames``.
        half_window_frames: Neighbours within this many frames each side are fit.
        order: Polynomial order (2 = quadratic). Clamped so a window always has
            at least ``order + 1`` samples.

    Returns:
        ``(n, 2)`` smoothed positions. Endpoints are preserved better than a
        moving average because the local fit extrapolates rather than truncating.

    Examples:
        ```python
        import numpy as np
        # A parabola (constant curvature) is left essentially untouched -- an
        # order-2 fit reproduces it, whereas a moving average would flatten it.
        f = list(range(7))
        pos = np.array([[t, t * t] for t in range(7)], dtype=float)
        out = savgol_smooth(f, pos, half_window_frames=3)
        bool(np.allclose(out[3], [3.0, 9.0], atol=1e-9))
        # True
        ```
    """
    frame_arr = np.asarray(frames, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    n = len(frame_arr)
    if half_window_frames < 1 or n < order + 1:
        return points.copy()

    out = points.copy()
    for i in range(n):
        near = np.abs(frame_arr - frame_arr[i]) <= half_window_frames
        idx = np.flatnonzero(near)
        if len(idx) < order + 1:
            continue  # too few neighbours (e.g. isolated across gaps) -> keep raw
        dt = frame_arr[idx] - frame_arr[i]
        # Least-squares polynomial in dt; the constant term is the value at dt=0.
        design = np.vander(dt, order + 1)  # columns dt^order ... dt^0
        coef, *_ = np.linalg.lstsq(design, points[idx], rcond=None)
        out[i] = coef[-1]
    return out


def _segment_heading(a: NDArray[np.float64], b: NDArray[np.float64]) -> float | None:
    """Heading of a->b in radians, or None if the two points barely differ."""
    step = b - a
    return float(np.arctan2(step[1], step[0])) if np.hypot(*step) >= 0.05 else None


def interpolate_straight_gaps(
    frames: list[int],
    positions: NDArray[np.float64],
    *,
    max_turn_deg: float = GAP_FILL_MAX_TURN_DEG,
    max_gap_frames: int = GAP_FILL_MAX_FRAMES,
    window: int = 3,
) -> tuple[list[int], NDArray[np.float64], list[bool]]:
    """Fill a track's tracking gaps by straight-line interpolation, but ONLY where
    the vehicle went straight across the gap.

    SAM2 drops the object for stretches of frames (motion blur, the car turning
    out of a learned appearance, occlusion). A gap where the heading just before
    and just after agree (within ``max_turn_deg``) is a straight run, so the
    missing frames are linearly interpolated -- the trajectory is continuous and
    the assumption (constant heading + speed over a few frames) is safe. A gap
    where the heading swings is a TURN: a straight line would cut the corner, so
    it is left open rather than inventing a cornering path the pipeline never saw.

    Args:
        frames: Tracked frame numbers, ascending (gaps allowed).
        positions: ``(n, 2)`` metric positions matching ``frames``.
        max_turn_deg: Heading change across a gap above which it is treated as a
            turn and left open.
        max_gap_frames: Never interpolate a gap longer than this many frames.
        window: Samples each side used to estimate the local heading.

    Returns:
        ``(new_frames, new_positions, interpolated)`` -- ``interpolated[i]`` is
        True for a synthesised (gap-filled) sample, False for a tracked one.

    Examples:
        ```python
        import numpy as np
        # A straight run due east with frame 2..4 missing -> filled.
        frames = [0, 1, 5, 6]
        pos = np.array([[0.0, 0], [1, 0], [5, 0], [6, 0]])
        nf, np_, interp = interpolate_straight_gaps(frames, pos)
        nf
        # [0, 1, 2, 3, 4, 5, 6]
        [bool(x) for x in interp]
        # [False, False, True, True, True, False, False]
        ```
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(frames)
    if n < 2:
        return list(frames), positions.copy(), [False] * n

    out_frames: list[int] = []
    out_pos: list[NDArray[np.float64]] = []
    interpolated: list[bool] = []
    max_turn = np.radians(max_turn_deg)

    for i in range(n):
        out_frames.append(int(frames[i]))
        out_pos.append(positions[i])
        interpolated.append(False)
        if i + 1 >= n:
            continue
        a, b = int(frames[i]), int(frames[i + 1])
        missing = b - a - 1
        if not (1 <= missing <= max_gap_frames):
            continue
        before = _segment_heading(positions[max(0, i - window)], positions[i])
        after = _segment_heading(
            positions[i + 1], positions[min(n - 1, i + 1 + window)]
        )
        # A turn if both headings are defined and disagree beyond the threshold.
        # If either is undefined (the vehicle was ~stationary), the tiny straight
        # bridge is harmless, so treat it as straight.
        if before is not None and after is not None:
            delta = abs((after - before + np.pi) % (2 * np.pi) - np.pi)
            if delta > max_turn:
                continue  # turn -> leave the gap open
        for frame in range(a + 1, b):
            t = (frame - a) / (b - a)
            out_frames.append(frame)
            out_pos.append(positions[i] * (1 - t) + positions[i + 1] * t)
            interpolated.append(True)

    return out_frames, np.array(out_pos, dtype=np.float64), interpolated


# The rigid-rectangle footprint model only describes boxy four-wheelers. A
# motorcycle/scooter is a narrow leaning object and a pedestrian is a point, so
# fitting a rectangle to either is meaningless -- they keep the legacy anchor.
_NON_BOXY = ("motorcycle", "motorbike", "scooter", "bike", "person", "pedestrian")


def is_boxy_vehicle(label: str) -> bool:
    """True when a rigid rectangle is a reasonable footprint model for ``label``.

    Matches on the backend id/name, which is inconsistent across scenes (English
    SAM2 classes in some, Chinese display names in others).

    Examples:
        ```python
        is_boxy_vehicle("car"), is_boxy_vehicle("taxi"), is_boxy_vehicle("motorbike")
        # (True, True, False)
        ```
    """
    text = label.lower()
    if any(token in text for token in _NON_BOXY):
        return False
    if "機車" in label or "摩托" in label or "行人" in label:
        return False
    return True


def footprint_size(label: str, length_m: float | None) -> tuple[float, float] | None:
    """Footprint ``(length_m, width_m)`` for a vehicle, or None if it doesn't apply.

    Returns None for non-boxy classes (bikes, pedestrians) and for a missing
    length. Width is derived from the length via :data:`DEFAULT_WIDTH_RATIO`,
    clamped to a plausible range so a mis-configured length can't produce a
    degenerate box.

    Examples:
        ```python
        footprint_size("car", 4.5)
        # (4.5, 1.8)
        footprint_size("motorbike", 1.9) is None
        # True
        footprint_size("car", None) is None
        # True
        ```
    """
    if not length_m or length_m <= 0 or not is_boxy_vehicle(label):
        return None
    width = float(np.clip(length_m * DEFAULT_WIDTH_RATIO, 0.5, 3.0))
    return float(length_m), round(width, 3)


# --- sidecar persistence -----------------------------------------------------
# Stage 1 has the masks but no homography; Stage 2 has the homography but only a
# single anchor per frame. The contact contours bridge the two: Stage 1 writes
# them here, Stage 2 reads and projects them. Keeping this out of the tracks CSV
# leaves that schema (and every existing CSV) untouched, and lets Stage 2 rerun
# standalone to re-tune the fit without re-running SAM2.


# Each contour is stored under "<vehicle>/<frame>" so one flat npz holds every
# vehicle. "/" is not special to npz key handling; we split on the last one.
def _contour_key(vehicle: str, frame: int) -> str:
    return f"{vehicle}/{frame}"


def save_contours(
    path: Path,
    contours: dict[str, dict[int, NDArray[np.int32]]],
) -> None:
    """Persist per-vehicle per-frame contact contours to a compressed npz.

    Args:
        path: Output ``.npz`` path (parent created if missing).
        contours: ``{vehicle: {frame: (n, 2) pixels}}``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, NDArray[np.int32]] = {}
    for vehicle, by_frame in contours.items():
        for frame, pts in by_frame.items():
            flat[_contour_key(vehicle, frame)] = np.asarray(pts, dtype=np.int32)
    np.savez_compressed(path, **flat)


def load_contours(path: Path) -> dict[str, dict[int, NDArray[np.int32]]]:
    """Load contours written by :func:`save_contours`; empty if the file is absent.

    Args:
        path: The sidecar ``.npz`` path.

    Returns:
        ``{vehicle: {frame: (n, 2) pixels}}``. Empty (not an error) when the
        sidecar does not exist, so Stage 2 falls back to the legacy anchor.
    """
    if not path.exists():
        return {}
    out: dict[str, dict[int, NDArray[np.int32]]] = {}
    with np.load(path) as data:
        for key in data.files:
            vehicle, _, frame = key.rpartition("/")
            out.setdefault(vehicle, {})[int(frame)] = data[key]
    return out
