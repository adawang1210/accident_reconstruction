"""Tests for the orientation-invariant ground-footprint anchor.

The headline test is :func:`test_footprint_anchor_is_stable_through_a_turn`,
which reproduces the defect the fit exists to remove: the legacy bbox
bottom-centre slides sideways as a car rotates, purely because the silhouette
switches from the car's rear to its side.
"""

from __future__ import annotations

import numpy as np
import pytest

from accident_reconstruction import ground_footprint as gf

# A passenger car's ground footprint.
LENGTH_M = 4.5
WIDTH_M = 1.8


def footprint_corners(
    center: tuple[float, float], heading: float, length_m: float, width_m: float
) -> np.ndarray:
    """The four ground corners of a vehicle footprint, in metres."""
    cx, cy = center
    along = np.array([np.cos(heading), np.sin(heading)])
    across = np.array([-np.sin(heading), np.cos(heading)])
    half_l, half_w = length_m / 2, width_m / 2
    return np.array(
        [
            [cx, cy] + s * half_l * along + t * half_w * across
            for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def visible_edges(
    center: tuple[float, float],
    heading: float,
    view_dir: np.ndarray,
    *,
    per_edge: int = 24,
) -> np.ndarray:
    """Sample the footprint edges a camera looking along ``view_dir`` can see.

    An edge is visible when its outward normal faces the camera. This is what the
    mask's bottom contour traces: the near side, plus the near end when oblique.
    """
    corners = footprint_corners(center, heading, LENGTH_M, WIDTH_M)
    middle = np.array(center)
    samples = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        edge_mid = (a + b) / 2
        outward = edge_mid - middle
        outward /= np.linalg.norm(outward)
        if outward @ view_dir < -1e-9:  # normal points back toward the camera
            t = np.linspace(0, 1, per_edge)[:, None]
            samples.append(a + t * (b - a))
    return np.vstack(samples) if samples else np.empty((0, 2))


def legacy_anchor(points: np.ndarray, view_dir: np.ndarray) -> np.ndarray:
    """The legacy anchor's metric equivalent: bbox x-centre at the bottom edge.

    In the image the anchor is ``((x1 + x2) // 2, y2)``. Working on the ground
    plane with the camera looking along ``view_dir``, that is the midpoint of the
    silhouette's lateral extent, at its nearest-to-camera depth.
    """
    lateral = np.array([-view_dir[1], view_dir[0]])
    across = points @ lateral
    depth = points @ view_dir
    mid_across = (across.min() + across.max()) / 2
    # y2 is the LOWEST image row = the point nearest the camera = least depth.
    return mid_across * lateral + depth.min() * view_dir


def test_contour_anchor_lands_on_the_contact_line_under_the_body_centre() -> None:
    """The scale-independent anchor sits on the contour at its median column."""
    # Wheels low at the sides, bumper higher in the middle (a shallow "U").
    contour = np.array([[0, 10], [5, 6], [10, 6], [15, 6], [20, 10]])

    anchor = gf.contour_anchor_px(contour)

    assert anchor == (10.0, 6.0)


def test_contour_anchor_avoids_the_box_corner_float() -> None:
    """Unlike the legacy box-corner anchor, it doesn't drop below the car.

    For a rear-on "U" contour the legacy ``(mid-x, max-y)`` pairs the centre
    column with a side wheel's row -- a pixel well below the central contact.
    The on-contour anchor stays on the actual contact row there.
    """
    contour = np.array([[0, 30], [4, 12], [8, 10], [12, 10], [16, 12], [20, 30]])
    legacy_y = contour[:, 1].max()  # 30, a side wheel/shadow -> floats low

    _, anchor_y = gf.contour_anchor_px(contour)

    assert anchor_y <= 12  # on the central contact, not the low corner
    assert anchor_y < legacy_y


def test_contour_anchor_of_empty_contour_is_none() -> None:
    assert gf.contour_anchor_px(np.empty((0, 2), dtype=np.int32)) is None


def test_bottom_contour_takes_the_lowest_pixel_per_column() -> None:
    """The contour is the mask's ground-contact line, not its box."""
    mask = np.zeros((10, 6), dtype=bool)
    mask[2:8, 1] = True  # tall column
    mask[2:4, 3] = True  # short column
    mask[5:6, 5] = True

    contour = gf.bottom_contour(mask)

    assert contour.tolist() == [[1, 7], [3, 3], [5, 5]]


def test_bottom_contour_of_empty_mask_is_empty() -> None:
    """An empty mask yields no contour rather than raising."""
    assert len(gf.bottom_contour(np.zeros((4, 4), dtype=bool))) == 0


def test_subsample_preserves_both_ends() -> None:
    """Thinning keeps the contour's extremes, which anchor the fit."""
    points = np.column_stack([np.arange(200), np.arange(200)])

    thinned = gf.subsample(points, 16)

    assert len(thinned) == 16
    assert thinned[0].tolist() == [0, 0]
    assert thinned[-1].tolist() == [199, 199]


def test_rect_boundary_distance_is_zero_on_the_edge() -> None:
    """Points on the footprint outline cost nothing."""
    corners = footprint_corners((3.0, -2.0), 0.7, LENGTH_M, WIDTH_M)

    distance = gf.rect_boundary_distance(
        corners, np.array([3.0, -2.0]), 0.7, LENGTH_M, WIDTH_M
    )

    assert distance == pytest.approx(np.zeros(4), abs=1e-9)


def test_rect_boundary_distance_penalises_the_interior() -> None:
    """Interior points are scored by distance to the EDGE, not to the centre."""
    centre = np.array([0.0, 0.0])

    distance = gf.rect_boundary_distance(
        np.array([[0.0, 0.0]]), centre, 0.0, LENGTH_M, WIDTH_M
    )

    assert distance[0] == pytest.approx(WIDTH_M / 2)


@pytest.mark.parametrize("heading_deg", [0.0, 25.0, 90.0, 145.0, -60.0])
def test_fit_recovers_the_centre_from_two_visible_edges(heading_deg: float) -> None:
    """Whatever the orientation, the fit lands on the true footprint centre."""
    heading = np.radians(heading_deg)
    truth = (12.0, -4.0)
    view = np.array([0.3, 0.95])
    view /= np.linalg.norm(view)

    points = visible_edges(truth, heading, view)
    found = gf.fit_footprint_center(points, heading, LENGTH_M, WIDTH_M)

    assert found == pytest.approx(truth, abs=0.05)


def test_fit_tolerates_outliers() -> None:
    """A few stray contour pixels (shadow, kerb) don't drag the fit."""
    heading = np.radians(10.0)
    truth = (0.0, 0.0)
    view = np.array([0.3, 0.95])  # oblique -> full side visible, clears the gate
    view /= np.linalg.norm(view)
    points = visible_edges(truth, heading, view)
    # A couple of stray pixels just off the body (a shadow edge, a kerb).
    points = np.vstack([points, [[2.6, 1.3], [-2.6, -1.3]]])

    found = gf.fit_footprint_center(points, heading, LENGTH_M, WIDTH_M)

    assert found == pytest.approx(truth, abs=0.15)


def test_fit_declines_when_the_contour_is_too_short() -> None:
    """Too little footprint to constrain a fit -> caller keeps the legacy anchor."""
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    assert gf.fit_footprint_center(points, 0.0, LENGTH_M, WIDTH_M) is None


def test_footprint_anchor_is_stable_through_a_turn() -> None:
    """The defect this module exists to fix.

    A car drives away from the camera and turns 80 degrees while its footprint
    centre stays put. The legacy bbox bottom-centre wanders by most of a car
    width, purely because the silhouette switches from the rear to the side; the
    footprint fit stays on the real centre.
    """
    truth = (0.0, 20.0)
    view = np.array([0.3, 0.95])  # oblique look, so the car shows a corner (an L)
    view /= np.linalg.norm(view)

    legacy_positions = []
    fitted_positions = []
    for heading_deg in np.linspace(70.0, 10.0, 9):
        heading = np.radians(heading_deg)
        points = visible_edges(truth, heading, view)
        legacy_positions.append(legacy_anchor(points, view))
        fit = gf.fit_footprint_center(points, heading, LENGTH_M, WIDTH_M)
        # The span gate declines frames showing too little footprint; the fit
        # earns its keep on the corner frames, which is what we assert on.
        if fit is not None:
            fitted_positions.append(fit)

    assert len(fitted_positions) >= 5, "the fit should engage on the corner frames"
    legacy_spread = max(np.ptp(np.array(legacy_positions), axis=0))
    fitted_spread = max(np.ptp(np.array(fitted_positions), axis=0))

    # The fitted anchor stays on the stationary centre...
    assert fitted_spread < 0.1
    # ...while the legacy anchor wanders as the silhouette rotates: that phantom
    # motion is what puts a kink in the trajectory and a spike in the speed.
    assert legacy_spread > 5 * fitted_spread


def test_headings_follow_the_direction_of_travel() -> None:
    """Heading comes from the track, since a footprint aligns with its motion."""
    frames = list(range(10))
    positions = np.column_stack([np.zeros(10), np.arange(10.0)])  # due north

    headings = gf.headings_from_track(frames, positions)

    assert len(headings) == 10
    assert all(h == pytest.approx(np.pi / 2, abs=1e-6) for h in headings.values())


def test_heading_is_held_while_stopped() -> None:
    """A stopped car keeps pointing where it was, instead of picking up noise."""
    frames = list(range(8))
    moving = np.column_stack([np.arange(4.0), np.zeros(4)])
    stopped = np.repeat([[3.0, 0.0]], 4, axis=0)
    positions = np.vstack([moving, stopped])

    headings = gf.headings_from_track(frames, positions)

    assert headings[7] == pytest.approx(0.0, abs=1e-6)


def test_smoothing_pulls_back_a_single_frame_spike() -> None:
    """A one-frame jitter spike is averaged toward its neighbours."""
    positions = np.array([[0.0, 0.0], [0.0, 0.9], [0.0, 0.0]])

    smoothed = gf.smooth_positions([0, 1, 2], positions, half_window_frames=1)

    assert smoothed[1][1] == pytest.approx(0.3)


def test_smoothing_does_not_blend_across_a_gap() -> None:
    """A sample whose neighbours are frames away (a dropout) is left untouched."""
    positions = np.array([[0.0, 0.0], [5.0, 5.0], [5.1, 5.0]])

    smoothed = gf.smooth_positions([0, 20, 21], positions, half_window_frames=2)

    assert smoothed[0].tolist() == [0.0, 0.0]


def test_smoothing_sheds_noise_without_moving_the_track() -> None:
    """Smoothing removes jitter while leaving the track's centre effectively put.

    That is what keeps the turn-bias correction intact: the fit's mean carries
    the correction, so the moving average must not shift it appreciably. (Edge
    windows shrink asymmetrically, so the mean moves by ~centimetres, not zero.)
    """
    rng = np.random.default_rng(0)
    frames = list(range(30))
    truth = np.column_stack([np.linspace(0, 10, 30), np.zeros(30)])
    noisy = truth + rng.normal(scale=0.1, size=truth.shape)

    smoothed = gf.smooth_positions(frames, noisy, half_window_frames=2)

    assert smoothed.mean(axis=0) == pytest.approx(noisy.mean(axis=0), abs=0.02)
    # and it is genuinely smoother than the input
    before = np.linalg.norm(np.diff(noisy, axis=0), axis=1).std()
    after = np.linalg.norm(np.diff(smoothed, axis=0), axis=1).std()
    assert after < before


def test_savgol_preserves_a_curved_path() -> None:
    """An order-2 fit reproduces constant curvature -- it doesn't round the turn.

    This is the whole reason to prefer Savitzky-Golay over a moving average: the
    average would flatten this parabola, distorting a real turn.
    """
    frames = list(range(9))
    curved = np.array([[t, 0.5 * t * t] for t in range(9)], dtype=float)

    smoothed = gf.savgol_smooth(frames, curved, half_window_frames=3)

    # Interior points sit on the original curve to numerical precision.
    assert np.allclose(smoothed[2:-2], curved[2:-2], atol=1e-9)


def test_savgol_removes_jitter() -> None:
    """High-frequency noise on a smooth path is reduced."""
    rng = np.random.default_rng(1)
    frames = list(range(40))
    truth = np.column_stack([np.linspace(0, 20, 40), np.sin(np.linspace(0, 3, 40))])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)

    smoothed = gf.savgol_smooth(frames, noisy, half_window_frames=3)

    # Closer to the underlying truth than the noisy input.
    assert np.abs(smoothed - truth).mean() < np.abs(noisy - truth).mean()


def test_savgol_does_not_blend_across_a_gap() -> None:
    """An isolated sample across a dropout is left as-is, not pulled over."""
    frames = [0, 1, 2, 40]
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 9.0]])

    smoothed = gf.savgol_smooth(frames, pos, half_window_frames=3)

    assert smoothed[3].tolist() == [3.0, 9.0]  # no neighbours within 3 frames


def test_interpolate_fills_a_straight_gap() -> None:
    """A gap on a straight run is linearly filled and the samples flagged."""
    frames = [0, 1, 5, 6]
    pos = np.array([[0.0, 0], [1, 0], [5, 0], [6, 0]])

    new_frames, new_pos, interp = gf.interpolate_straight_gaps(frames, pos)

    assert new_frames == [0, 1, 2, 3, 4, 5, 6]
    assert interp == [False, False, True, True, True, False, False]
    # filled points sit on the straight line
    assert np.allclose(new_pos[2:5], [[2, 0], [3, 0], [4, 0]])


def test_interpolate_leaves_a_turn_gap_open() -> None:
    """A gap where the heading swings (a turn) is NOT filled -- no corner-cutting."""
    # Heading ~east before the gap, ~north after it: a 90-degree turn across it.
    frames = [0, 1, 10, 11]
    pos = np.array([[0.0, 0], [1, 0], [5, 4], [5, 5]])

    new_frames, _, interp = gf.interpolate_straight_gaps(frames, pos)

    assert new_frames == frames  # unchanged
    assert not any(interp)


def test_interpolate_declines_an_overlong_gap() -> None:
    """Even a straight gap longer than the cap is left open (too much invented)."""
    frames = [0, 1, 100, 101]
    pos = np.array([[0.0, 0], [1, 0], [100, 0], [101, 0]])

    new_frames, _, _ = gf.interpolate_straight_gaps(frames, pos, max_gap_frames=40)

    assert new_frames == frames


def test_footprint_size_derives_width_from_length() -> None:
    """Scenes configure a length; the width follows from the class ratio."""
    assert gf.footprint_size("car", 4.5) == (4.5, 1.8)
    assert gf.footprint_size("car", None) is None
    assert gf.footprint_size("car", 0.0) is None


def test_footprint_size_declines_non_boxy_vehicles() -> None:
    """A rigid rectangle is the wrong model for bikes and pedestrians."""
    assert gf.footprint_size("motorbike", 1.9) is None
    assert gf.footprint_size("motorcycle", 1.9) is None
    assert gf.footprint_size("person", 0.5) is None
    assert gf.is_boxy_vehicle("taxi")
    assert gf.is_boxy_vehicle("police_car")
    assert not gf.is_boxy_vehicle("機車")


def test_fit_declines_when_scale_is_unfaithful() -> None:
    """A contour far smaller than the assumed footprint (under-scaled homography).

    The BMW CCTV compresses a 4.3 m car to ~2.4 m of contour; a 4.3 m rectangle
    fitted to that slides freely, so the fit must decline and keep the legacy
    anchor rather than emit a confidently wrong, jittery centre.
    """
    heading = np.radians(20.0)
    # A contour spanning only ~2.4 m for an assumed 4.3 m car (~56%, below gate).
    view = np.array([0.3, 0.95])
    view /= np.linalg.norm(view)
    points = visible_edges((0.0, 0.0), heading, view) * (2.4 / LENGTH_M)

    assert gf.fit_footprint_center(points, heading, LENGTH_M, WIDTH_M) is None
