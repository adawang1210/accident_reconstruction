"""Closed loop: user-prompted SAM2 tracks -> homography -> 2D map / KML.

Takes the per-frame ground anchors produced by ``prompt_track_accident.py`` (the
user-specified vehicles, nothing else), projects them through the same homography
used by the manual pipeline, derives speed and the impact frame, and reuses the
birdseye writers to emit the aligned KML / map figure / CSV. This makes the whole
reconstruction automatic once the user has pointed at the accident vehicles.

Example:
    ```bash
    .venv/bin/python accident_reconstruction/auto_reconstruct.py
    ```
"""

from __future__ import annotations

import csv
import json
import math
import traceback
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from accident_reconstruction.birdseye_manual_annotation import (
    write_csv,
    write_kml,
    write_map_figure,
)
from accident_reconstruction.calibrate_homography import (
    VIEW_TRANSFORMER,
    load_distance_constraints,
    load_gcps,
    undistort_to_normalized,
)
from accident_reconstruction.motion import euclidean, windowed_speed
from accident_reconstruction.scene_config import SCENE, SceneConfig
from accident_reconstruction.time_axis import load_frame_times, time_axis_warning

PROMPT_TRACKS_CSV = SCENE.prompt_tracks_csv


def _load_distortion() -> dict | None:
    """The lens distortion from the scene's calibration (None if not calibrated)."""
    path = SCENE.calibration_path
    if path.exists():
        try:
            return json.loads(path.read_text()).get("distortion")
        except (json.JSONDecodeError, OSError):
            return None
    return None


DISTORTION = _load_distortion()
AUTO_KML_PATH = SCENE.out_kml
AUTO_FIGURE_PATH = SCENE.out_figure
AUTO_CSV_PATH = SCENE.out_csv

FPS = SCENE.fps


def load_anchors(csv_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Load per-vehicle per-frame ground anchors (original pixels) from a CSV.

    Args:
        csv_path: A tracks CSV with ``frame, vehicle, anchor_x, anchor_y`` columns.

    Returns:
        ``anchors[vehicle][frame] = (anchor_x, anchor_y)``.
    """
    anchors: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in csv.DictReader(csv_path.open()):
        anchors[row["vehicle"]][int(row["frame"])] = (
            float(row["anchor_x"]),
            float(row["anchor_y"]),
        )
    return anchors


def project_metric(
    anchors: dict[str, dict[int, tuple[float, float]]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Project pixel anchors onto the metric ground plane via the homography.

    Args:
        anchors: Per-vehicle pixel anchors by frame.

    Returns:
        Per-vehicle metric ``(east_m, north_m)`` by frame.
    """
    metric: dict[str, dict[int, tuple[float, float]]] = {}
    for label, by_frame in anchors.items():
        metric[label] = {}
        for frame, anchor in by_frame.items():
            point = VIEW_TRANSFORMER.transform_points(
                np.array([anchor], dtype=np.float32)
            )[0]
            metric[label][frame] = (float(point[0]), float(point[1]))
    return metric


def load_boxes(
    csv_path: Path,
) -> dict[str, dict[int, tuple[float, float, float, float]]]:
    """Load per-vehicle per-frame pixel boxes ``(x1, y1, x2, y2)`` from a CSV.

    Args:
        csv_path: A tracks CSV with ``frame, vehicle, x1, y1, x2, y2`` columns.

    Returns:
        ``boxes[vehicle][frame] = (x1, y1, x2, y2)`` (empty for a CSV lacking the
        box columns).
    """
    boxes: dict[str, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x1", "y1", "x2", "y2"} <= set(
            reader.fieldnames
        ):
            return {}
        for row in reader:
            boxes[row["vehicle"]][int(row["frame"])] = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
    return boxes


def box_long_axis_endpoints(
    box: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two endpoints of a box's LONGER axis (midpoints of its short edges).

    Projecting these through the homography measures the vehicle's longer on-ground
    dimension (roughly its length for a side-on view), which is compared to a known
    real length to reveal the local metric scale.

    Args:
        box: Pixel box ``(x1, y1, x2, y2)``.

    Returns:
        ``((ax, ay), (bx, by))`` endpoints spanning the longer box dimension.

    Examples:
        ```python
        box_long_axis_endpoints((10, 0, 30, 60))  # taller than wide
        # ((20.0, 0.0), (20.0, 60.0))
        box_long_axis_endpoints((0, 10, 60, 30))  # wider than tall
        # ((0.0, 20.0), (60.0, 20.0))
        ```
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if width >= height:
        return (float(x1), float(cy)), (float(x2), float(cy))
    return (float(cx), float(y1)), (float(cx), float(y2))


def local_scale_factors(
    boxes_by_frame: dict[int, tuple[float, float, float, float]],
    project,
    real_length_m: float,
    positions: int = 3,
) -> list[tuple[int, float]]:
    """Local homography scale (projected box length / real length) at sample frames.

    Samples ``positions`` frames evenly across the vehicle's track (front, middle,
    back by default), projects each box's long axis to metres via ``project``, and
    divides by the known real length. A factor near 1 means the homography scale is
    right there; well below 1 means it compresses distance (so speed reads low).

    IMPORTANT -- this assumes a SIDE-ON view. The box long axis is the vehicle's
    length only when it is filmed from the side; on the rear/head-on intersection
    CCTV typical of these clips the box is wider than tall, so the long axis is the
    vehicle's WIDTH (~1.8 m for a car) being divided by its length. Measured
    2026-07-20: the BMW scene's ``car`` had w/h = 1.24 / 1.39 / 2.11 at the three
    sampled frames (all horizontal long axis), as did ``taoyuan_yangmei``'s ``car``.

    So do NOT invert the factor into a speed multiplier. What IS meaningful is its
    CONSISTENCY along the track: a stable factor (BMW: 0.32 / 0.32 / 0.37) says the
    scale is uniformly off rather than blowing up by extrapolation, while a spike
    (``yilan_wujie``'s motorcycle hitting 4.66x mid-crash) just means the box
    exploded as the rider was thrown. Making this quantitative needs a view-angle
    test or 3D-model-to-box alignment -- see ``frontend/SPLAT_NOTES.md`` §11.

    Args:
        boxes_by_frame: One vehicle's ``{frame: (x1, y1, x2, y2)}`` boxes.
        project: Callable mapping an ``(n, 2)`` pixel array to metric ``(n, 2)``
            (e.g. ``VIEW_TRANSFORMER.transform_points``).
        real_length_m: The vehicle's known real length in metres (> 0).
        positions: How many frames to sample across the track.

    Returns:
        ``[(frame, scale_factor), ...]`` for the sampled frames.
    """
    frames = sorted(boxes_by_frame)
    if not frames or real_length_m <= 0:
        return []
    if positions >= len(frames):
        picks = frames
    else:
        step = (len(frames) - 1) / (positions - 1) if positions > 1 else 0
        picks = [frames[round(i * step)] for i in range(positions)]
    out: list[tuple[int, float]] = []
    for frame in picks:
        (ax, ay), (bx, by) = box_long_axis_endpoints(boxes_by_frame[frame])
        pts = np.array([[ax, ay], [bx, by]], dtype=np.float32)
        metric = np.asarray(project(pts), dtype=np.float64)
        length_m = float(np.hypot(*(metric[1] - metric[0])))
        out.append((frame, length_m / real_length_m))
    return out


def print_length_sanity(csv_path: Path = PROMPT_TRACKS_CSV) -> None:
    """Print each configured vehicle's local homography scale (diagnostic only).

    For every vehicle in ``SCENE.vehicle_length_m`` with a tracked box, print the
    local scale factor at the front/middle/back of its track (see
    :func:`local_scale_factors`). Purely informational -- it quantifies how much the
    homography under- or over-scales distance (the main cause of low speed) without
    touching any speed or position.
    """
    lengths = SCENE.vehicle_length_m or {}
    if not lengths or VIEW_TRANSFORMER is None:
        return
    boxes = load_boxes(csv_path)
    if not boxes:
        return
    print("Known-length scale check (projected box length / real length):")
    for label, real_length_m in lengths.items():
        by_frame = boxes.get(label)
        if not by_frame:
            continue
        factors = local_scale_factors(
            by_frame, VIEW_TRANSFORMER.transform_points, real_length_m
        )
        readout = ", ".join(f"f{frame}: {scale:.2f}x" for frame, scale in factors)
        print(f"  {label} (real {real_length_m:.1f} m) -> {readout}")


def _similarity_transform(source: np.ndarray, target: np.ndarray):
    """Least-squares similarity (rotate + uniform scale + translate) source->target.

    Returns a function applying the fit. Unlike a homography it is shape-preserving
    (no shear/perspective), so applying it to the recognised path keeps that path's
    curve while the fit borrows the homography's position, scale and orientation.
    """
    src_mean, tgt_mean = source.mean(axis=0), target.mean(axis=0)
    src0, tgt0 = source - src_mean, target - tgt_mean
    u, s, vt = np.linalg.svd(src0.T @ tgt0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    variance = float((src0**2).sum())
    scale = float(s.sum()) / variance if variance > 1e-12 else 1.0
    return lambda points: (scale * (points - src_mean)) @ rotation + tgt_mean


def shape_preserving_metric(
    anchors: dict[str, dict[int, tuple[float, float]]],
    metric: dict[str, dict[int, tuple[float, float]]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Place each vehicle's recognised path on the ground while KEEPING its curve.

    A single planar homography flattens the recognised trajectory (its curve is
    smaller than the calibration noise floor, so the global projective fit irons
    it out). Instead we keep the homography only for placement: per vehicle, fit a
    similarity from the (undistorted) pixel path to the homography metric path and
    apply it to the pixel path. The result sits where/at the scale the homography
    says, but its SHAPE is the recognised one -- not straightened (TPS was tried
    and amplifies GCP noise / diverges off the control points). Falls back to the
    homography path when a vehicle has too few points to fit a similarity.

    Args:
        anchors: Per-vehicle pixel anchors by frame.
        metric: The homography metric (already truncated), the placement reference.

    Returns:
        Per-vehicle shape-preserving metric ``(east_m, north_m)`` by frame.
    """
    if not DISTORTION:
        # Without a lens model the recognised pixels still carry barrel distortion
        # and frame jitter; "preserving" that shape makes the path jagged. Only
        # shape-preserve a clean (undistorted) path -- else keep the homography.
        return {label: dict(track) for label, track in metric.items()}
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for label, mtrack in metric.items():
        frames = sorted(mtrack)
        if len(frames) < 3 or label not in anchors:
            out[label] = dict(mtrack)
            continue
        pixels = np.array([anchors[label][f] for f in frames], dtype=np.float64)
        recognised = undistort_to_normalized(pixels, DISTORTION).astype(np.float64)
        homography = np.array([mtrack[f] for f in frames], dtype=np.float64)
        try:
            placed = _similarity_transform(recognised, homography)(recognised)
        except np.linalg.LinAlgError:
            out[label] = dict(mtrack)
            continue
        out[label] = {
            f: (float(placed[i, 0]), float(placed[i, 1])) for i, f in enumerate(frames)
        }
    return out


def windowed_motion(
    track: dict[int, tuple[float, float]],
    times: Mapping[int, float] | None = None,
    fps: float = FPS,
    jump_guard: bool = True,
) -> dict[int, tuple[float, float]]:
    """Compute ``(cumulative_m, speed_kmh)`` per frame for one metric track.

    Speed is the displacement across the samples inside ``SPEED_WINDOW_SECONDS``,
    divided by the elapsed time, which smooths single-frame jitter. The elapsed
    time comes from real per-frame timestamps (``times``) when supplied -- so a
    variable-frame-rate clip is not read at a wrong, uniform cadence -- and falls
    back to ``frame / fps`` otherwise.

    Args:
        track: Metric ``(east_m, north_m)`` by frame for one vehicle.
        times: Optional ``{frame: t_sec}`` real timestamps (see
            :func:`accident_reconstruction.time_axis.load_frame_times`).
        fps: Nominal frames per second for the ``times``-absent fallback.
        jump_guard: Suppress jump-dominated windows (displayed speeds); pass
            False for the settle/truncation control stream (see
            :func:`accident_reconstruction.motion.windowed_speed`).

    Returns:
        ``(cumulative_m, speed_kmh)`` by frame.
    """
    return windowed_speed(track, fps, euclidean, times, jump_guard)


# Two vehicles within this metric distance are treated as in contact. Note this
# is only reliable when the ground projection is well-scaled; under a compressed
# homography everything reads "close", which is why the closest-approach rule
# below is preferred whenever the pair cleanly separates afterwards.
CONTACT_THRESHOLD_M = 3.0
# After a real collision the two vehicles move apart again. If the distance rises
# at least this far above the closest approach, that closest approach is the
# impact; if it never does (the masks merge and the distance plateaus), we fall
# back to first-contact-under-threshold instead.
SEPARATION_MARGIN_M = 1.0
# A struck vehicle that is flung/flips moves its ground anchor faster than any car
# drives on the road; a per-frame ground step above this (metres/frame) marks the
# tumble onset, after which the bottom-centre anchor is physically meaningless.
# Urban-accident speeds are well under this (~0.5 m/frame), the flip jump is ~1.5.
FLIP_VELOCITY_M_PER_FRAME = 1.2


def pair_distances(
    metric: dict[str, dict[int, tuple[float, float]]],
    label_a: str,
    label_b: str,
) -> list[tuple[int, float]]:
    """Per-frame metric distance between two vehicles over their shared frames."""
    shared = sorted(set(metric[label_a]) & set(metric[label_b]))
    return [
        (
            frame,
            math.hypot(
                metric[label_a][frame][0] - metric[label_b][frame][0],
                metric[label_a][frame][1] - metric[label_b][frame][1],
            ),
        )
        for frame in shared
    ]


def detect_impact(metric: dict[str, dict[int, tuple[float, float]]]) -> int | None:
    """Return the impact frame, or None if there is no two-vehicle contact.

    Scene-agnostic: works for any vehicle labels and counts. With fewer than two
    tracked vehicles there is no collision (returns None). With two or more, the
    colliding pair is the one with the smallest closest approach, and the impact
    frame is chosen by:

    * **Closest approach** when the pair clearly *separates* afterwards (distance
      rises at least ``SEPARATION_MARGIN_M`` above the minimum) -- a clean
      collide-then-part, robust even when the projection is distance-compressed.
    * **First frame under ``CONTACT_THRESHOLD_M``** when they instead merge and
      the distance plateaus (e.g. one mask follows the other post-impact), where
      the closest approach would land arbitrarily late inside the plateau.

    Args:
        metric: Per-vehicle metric positions by frame (any number of vehicles).

    Returns:
        The impact frame index, or None when no pair ever shares a frame.

    Examples:
        ```python
        # clean collide-then-separate -> closest approach (frame 2)
        m = {"a": {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)},
             "b": {0: (5, 0), 1: (2, 0), 2: (1, 0), 3: (9, 0)}}
        detect_impact(m)
        # 2
        ```
    """
    labels = list(metric)
    best: tuple[int, float] | None = None  # (impact_frame, closest distance)
    for index, label_a in enumerate(labels):
        for label_b in labels[index + 1 :]:
            distances = pair_distances(metric, label_a, label_b)
            if not distances:
                continue
            min_frame, min_dist = min(distances, key=lambda item: item[1])
            after = [d for f, d in distances if f > min_frame]
            separates = bool(after) and max(after) >= min_dist + SEPARATION_MARGIN_M
            if separates:
                impact_frame = min_frame
            else:
                impact_frame = next(
                    (f for f, d in distances if d < CONTACT_THRESHOLD_M), min_frame
                )
            if best is None or min_dist < best[1]:
                best = (impact_frame, min_dist)
    return None if best is None else best[0]


def resolve_impact_frame(
    scene: SceneConfig,
    metric: dict[str, dict[int, tuple[float, float]]] | None = None,
) -> int | None:
    """The scene's impact frame: the UI override if set, else auto-detected.

    Single source of truth for ``override or detect_impact(...)``, which four
    call sites previously repeated.

    Args:
        scene: The active scene (its ``impact_frame_override`` wins when set).
        metric: Per-vehicle metric positions to detect from. Defaults to the
            scene's prompt-tracks projection; pass an already-computed metric
            (e.g. the one ``build_data`` just built, or in-memory tracking
            anchors) to avoid re-projecting.

    Returns:
        The impact frame index, or None when neither an override nor a
        two-vehicle contact is available.
    """
    if scene.impact_frame_override is not None:
        return scene.impact_frame_override
    if metric is None:
        metric = project_metric(load_anchors(scene.prompt_tracks_csv))
    return detect_impact(metric)


def settle_frame(
    motion: dict[int, tuple[float, float]],
    after_frame: int,
    min_speed_kmh: float,
    sustain: int = 3,
) -> int | None:
    """First frame after ``after_frame`` where the vehicle has come to rest.

    "At rest" = speed stays below ``min_speed_kmh`` for ``sustain`` consecutive
    frames. Used to STOP the trajectory line once a vehicle stops moving (e.g. a
    struck car that settles after the crash), so the meaningless post-stop anchor
    jitter is not drawn -- the box/marker can still mark its final position.

    Args:
        motion: ``{frame: (cumulative_m, speed_kmh)}`` for one vehicle.
        after_frame: Only consider frames after this (rest follows the collision).
        min_speed_kmh: Speed threshold; <= 0 disables (returns None).
        sustain: Consecutive sub-threshold frames required (rejects a brief dip).

    Returns:
        The first frame of the sustained low-speed run, or None.

    Examples:
        ```python
        m = {0: (0, 9.0), 1: (1, 1.0), 2: (1, 0.5), 3: (1, 0.4)}
        settle_frame(m, 0, 3.0, sustain=2)
        # 1
        ```
    """
    if min_speed_kmh <= 0:
        return None
    frames = [f for f in sorted(motion) if f > after_frame]
    run_start = None
    run = 0
    for frame in frames:
        if motion[frame][1] < min_speed_kmh:
            run_start = frame if run == 0 else run_start
            run += 1
            if run >= sustain:
                return run_start
        else:
            run = 0
            run_start = None
    return None


def flip_onset(track: dict[int, tuple[float, float]], after_frame: int) -> int | None:
    """First frame > ``after_frame`` where the path jumps non-physically (a flip).

    The ground-contact anchor (mask bottom-centre) only tracks a real position
    while the vehicle is upright on the road. When a struck vehicle is flung and
    tumbles, the anchor leaps and bounces -- a per-frame ground step no real car
    achieves. This returns the first such frame (so the caller can drop the
    tumble), or None when the vehicle stays on the ground.

    Args:
        track: ``{frame: (east_m, north_m)}`` for one vehicle.
        after_frame: Only consider frames after this (the flip follows contact).

    Returns:
        The flip-onset frame, or None.

    Examples:
        ```python
        flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0), 3: (4.0, 0)}, 0)
        # 3
        flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0)}, 0)  # all smooth
        ```
    """
    frames = sorted(track)
    previous = None
    for frame in frames:
        if previous is not None and frame > after_frame:
            gap = max(frame - previous, 1)
            step = math.hypot(
                track[frame][0] - track[previous][0],
                track[frame][1] - track[previous][1],
            )
            if step / gap > FLIP_VELOCITY_M_PER_FRAME:
                return frame
        previous = frame
    return None


def build_data(csv_path: Path = PROMPT_TRACKS_CSV):
    """Assemble ``(motion, metric, impact_frame)`` from a prompt-tracks CSV.

    If the scene names a ``stop_vehicle`` (the struck one), its physically invalid
    anchors are dropped: the path is cut at the flip onset (:func:`flip_onset`) so
    only the on-ground approach/push is kept and the post-impact tumble is removed.
    When no flip is detected (e.g. the mask just merges and plateaus) it falls back
    to truncating at the impact frame. Scenes without a stop vehicle keep every
    anchor.

    Args:
        csv_path: Prompt-tracks CSV path.

    Returns:
        ``(motion, metric, impact_frame)`` ready for the birdseye writers;
        ``impact_frame`` is None when no two-vehicle contact is found.
    """
    anchors = load_anchors(csv_path)
    metric = project_metric(anchors)
    # Real per-frame timestamps (from the tracks CSV's t_sec column) drive the
    # speed windows; without them speed falls back to frame/fps. Warn once when the
    # time axis is uneven (VFR / dropped frames) so speeds are read with suspicion.
    times = load_frame_times(csv_path)
    warning = time_axis_warning(list(times.values())) if times else None
    if warning is not None:
        print(warning)
    # The UI can pin the impact frame (overrides.json); else auto-detect.
    impact_frame = resolve_impact_frame(SCENE, metric)

    # Cut each vehicle's trajectory once it comes to rest after the collision:
    # the line stops where speed stays below ``min_traj_speed`` (post-stop anchor
    # jitter is meaningless). The struck vehicle additionally cuts at the flip
    # onset (a high-speed tumble the speed gate can't catch), or, with neither and
    # no ``struck_full``, falls back to the impact frame.
    stop_vehicle = SCENE.resolved_stop_vehicle
    min_speed = SCENE.min_traj_speed_kmh
    # CONTROL stream: no jump guard. The settle gate compares speeds against
    # ``min_speed``, and a guard zero (window flushed by a tracking gap at the
    # crash) would misread a tumbling vehicle as "at rest", moving the cut and --
    # through the alignment's anchors -- every drawn position. Spikes are harmless
    # here: they only keep a vehicle "moving", which is the historic behaviour.
    full_motion = {
        label: windowed_motion(track, times, SCENE.fps, jump_guard=False)
        for label, track in metric.items()
    }
    for label, track in list(metric.items()):
        after = impact_frame if impact_frame is not None else min(track)
        cuts = [settle_frame(full_motion[label], after, min_speed)]
        if label == stop_vehicle:
            cuts.append(flip_onset(track, after))
            if (
                not any(c is not None for c in cuts)
                and impact_frame is not None
                and not SCENE.show_struck_full
            ):
                cuts.append(impact_frame + 1)  # struck car assumed to stop here
        cut = min((c for c in cuts if c is not None), default=None)
        if cut is not None:
            metric[label] = {f: xy for f, xy in track.items() if f < cut}

    # Speed/distance come from the accurate homography metric; the DRAWN path is
    # the shape-preserving one (keeps the recognised curve the homography flattens).
    motion = {
        label: windowed_motion(track, times, SCENE.fps)
        for label, track in metric.items()
    }
    draw_metric = shape_preserving_metric(anchors, metric)
    return motion, draw_metric, impact_frame


def gcp_ground_span_m() -> float | None:
    """Real-world span (m) the scene's GCPs cover, or None if uncalibrated.

    Speed is ``metric distance / time`` and the metric is only trustworthy where
    the homography was anchored. A small GCP span means vehicles travelling beyond
    it read implausibly low speeds even though the per-GCP residual is tiny (the
    fit is good only on that patch). See ``docs/summary.md`` and the calibrate
    module's span warning.
    """
    path = SCENE.calibration_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("target_span_m")
    except (json.JSONDecodeError, OSError):
        return None


def gcp_pixel_points(store: Path | None = None) -> np.ndarray | None:
    """Pixel coords the homography was anchored on (GCPs + distance constraints).

    These pixels bound where the metric can be trusted; a vehicle straying beyond
    their convex hull is being extrapolated (see :func:`hull_coverage_ratio`). The
    pixels come from the scene's ``gcps.json`` (``gcp_store``) -- NOT the derived
    ``homography_calibration.json``, which holds only the fitted matrix and span.

    Args:
        store: GCP store path; defaults to the active scene's ``gcp_store``.

    Returns:
        ``(n, 2)`` float32 pixels, or None when uncalibrated / too few points.
    """
    path = store or SCENE.gcp_store
    if not path.exists():
        return None
    points: list[list[float]] = [
        gcp["pixel"] for gcp in load_gcps(path) if "pixel" in gcp
    ]
    for constraint in load_distance_constraints(path):
        points.append(list(constraint["pixel_a"]))
        points.append(list(constraint["pixel_b"]))
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float32)


def hull_coverage_ratio(
    trajectory_pixels: np.ndarray | None, gcp_pixels: np.ndarray | None
) -> float | None:
    """Fraction of the trajectory's pixel hull that the GCP pixel hull covers.

    Speed trust follows the calibrated region: where a vehicle's pixels leave the
    GCP convex hull the homography extrapolates and compresses distance. This is
    ``area(trajectory_hull ∩ gcp_hull) / area(trajectory_hull)`` -- ``1.0`` means
    the whole path sits inside the anchored patch; a low value flags extrapolated
    (typically under-scaled) speeds.

    Args:
        trajectory_pixels: ``(n, 2)`` vehicle anchor pixels.
        gcp_pixels: ``(m, 2)`` GCP / distance-constraint pixels.

    Returns:
        Coverage in ``[0, 1]``, or None when either hull is degenerate.

    Examples:
        ```python
        gcp = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        traj = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32)
        round(hull_coverage_ratio(traj, gcp), 2)  # 1.0 -- fully inside
        ```
    """
    if trajectory_pixels is None or gcp_pixels is None:
        return None
    traj = np.asarray(trajectory_pixels, dtype=np.float32)
    gcp = np.asarray(gcp_pixels, dtype=np.float32)
    if len(traj) < 3 or len(gcp) < 3:
        return None
    traj_hull = cv2.convexHull(traj)
    if cv2.contourArea(traj_hull) <= 0:
        return None
    gcp_hull = cv2.convexHull(gcp)
    intersection, _ = cv2.intersectConvexConvex(traj_hull, gcp_hull)
    return float(min(intersection / cv2.contourArea(traj_hull), 1.0))


def speed_reliability(csv_path: Path | None = None) -> dict:
    """GCP-scale trust facts for a scene's speeds (ground span + hull coverage).

    Combines the GCP ground span (:func:`gcp_ground_span_m`) with the fraction of
    the tracked trajectory falling inside the GCP pixel convex hull
    (:func:`hull_coverage_ratio`). Both are diagnostics, not gates.

    Args:
        csv_path: Tracks CSV to read anchors from; defaults to the scene's.

    Returns:
        ``{"gcp_ground_span_m": float | None, "hull_coverage": float}`` --
        ``hull_coverage`` is omitted when it cannot be computed.
    """
    info: dict = {"gcp_ground_span_m": gcp_ground_span_m()}
    csv_path = csv_path or PROMPT_TRACKS_CSV
    traj_pixels: np.ndarray | None = None
    if csv_path.exists():
        flat = [
            xy
            for by_frame in load_anchors(csv_path).values()
            for xy in by_frame.values()
        ]
        if len(flat) >= 3:
            traj_pixels = np.asarray(flat, dtype=np.float32)
    coverage = hull_coverage_ratio(traj_pixels, gcp_pixel_points())
    if coverage is not None:
        info["hull_coverage"] = round(coverage, 3)
    return info


def speed_reliability_caption(info: dict | None = None) -> str:
    """One-line human caption of speed reliability for figures / CSV / logs.

    Args:
        info: A :func:`speed_reliability` dict; recomputed when None.

    Returns:
        A compact zh-TW caption, e.g. ``速度可靠度:GCP 涵蓋 ~18 m、軌跡落在校正區 42%``.
    """
    info = info if info is not None else speed_reliability()
    span = info.get("gcp_ground_span_m")
    coverage = info.get("hull_coverage")
    parts: list[str] = []
    if span is not None:
        parts.append(f"GCP 涵蓋 ~{span:.0f} m")
    if coverage is not None:
        parts.append(f"軌跡落在校正區 {coverage * 100:.0f}%")
    if not parts:
        return "速度可靠度:場景未 GPS 校正"
    return "速度可靠度:" + "、".join(parts)


def print_speed_reliability(
    motion: dict[str, dict[int, tuple[float, float]]],
    csv_path: Path | None = None,
) -> None:
    """Print each vehicle's peak speed plus the GCP scale it depends on.

    Factual, not a gate: it surfaces the GCP ground span and trajectory hull
    coverage (which bound speed trust) next to the peak speeds so an under-scaled
    reading is obvious rather than silently shown.
    """
    info = speed_reliability(csv_path)
    span = info.get("gcp_ground_span_m")
    coverage = info.get("hull_coverage")
    peaks = {
        label: max((s for _, s in track.values()), default=0.0)
        for label, track in motion.items()
    }
    peak_str = ", ".join(f"{k}: {v:.0f} km/h" for k, v in peaks.items())
    print(f"Peak speed -> {peak_str}")
    if span is not None:
        print(
            f"Speed reliability: GCPs cover ~{span:.0f} m of ground. Speed is only "
            "as accurate as this scale -- if vehicles travel well beyond it, the "
            "homography compresses distance and speeds read low; re-calibrate with "
            "GCPs spread across the whole stretch the vehicles drive."
        )
    if coverage is not None:
        print(
            f"Trajectory hull coverage: {coverage * 100:.0f}% of the tracked path "
            "falls inside the GCP-anchored region (100% = fully calibrated; low "
            "values mean speeds outside it are extrapolated)."
        )


def main(csv_path: str = str(PROMPT_TRACKS_CSV)) -> None:
    """Run the closed-loop reconstruction and write KML / figure / CSV."""
    data = build_data(Path(csv_path))
    _, metric, impact_frame = data
    write_kml(data, kml_path=AUTO_KML_PATH)
    write_map_figure(data, figure_path=AUTO_FIGURE_PATH)
    write_csv(data, csv_path=AUTO_CSV_PATH)

    # Also emit the raw recognised (non-road-snapped) figure + KML -- this is what
    # the web app now displays as the primary 2D result. Deferred import because
    # recognized_route imports from this module (avoids an import cycle).
    try:
        from accident_reconstruction.recognized_route import (
            write_recognized_csv,
            write_recognized_figure,
            write_recognized_kml,
            write_reconstruction_json,
        )

        recognised_figure = write_recognized_figure()
        write_recognized_kml()
        write_recognized_csv()
        reconstruction_json = write_reconstruction_json()
        if recognised_figure is not None:
            print(f"Recognised figure: {recognised_figure.resolve()}")
        if reconstruction_json is not None:
            print(f"Reconstruction JSON: {reconstruction_json.resolve()}")
    except Exception as error:  # never let the optional view break the run
        # Print the traceback: this branch has silently swallowed import errors
        # before, so a one-line message is not enough to diagnose a real failure.
        print(f"(recognised figure skipped: {error})")
        traceback.print_exc()

    print(f"Impact frame: {impact_frame}")
    counts = ", ".join(f"{k}: {len(v)}" for k, v in metric.items())
    print(f"Frames per vehicle: {{{counts}}}")
    print_length_sanity(Path(csv_path))
    print_speed_reliability(data[0], Path(csv_path))
    print(f"KML: {AUTO_KML_PATH.resolve()}")
    print(f"Map figure: {AUTO_FIGURE_PATH.resolve()}")
    print(f"CSV: {AUTO_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
