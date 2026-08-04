"""Tests for the constant-acceleration Kalman + RTS trajectory smoother."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from accident_reconstruction import trajectory_smoothing as ts


def test_kalman_reduces_jerk_on_a_noisy_straight_run() -> None:
    """The headline property: it makes the motion kinematically plausible."""
    rng = np.random.default_rng(0)
    frames = list(range(60))
    truth = np.column_stack([np.linspace(0, 30, 60), np.zeros(60)])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, 25.0)

    jerk_noisy = ts.trajectory_jerk(frames, noisy, 25.0)
    jerk_smooth = ts.trajectory_jerk(frames, smoothed, 25.0)
    # Mean jerk collapses and no frame stays over the 15 m/s^3 plausibility line.
    assert jerk_smooth["mean_abs"] < jerk_noisy["mean_abs"] / 10
    assert jerk_smooth["frac_over_15"] == 0.0


def test_kalman_stays_close_to_the_data() -> None:
    """Smoothing sheds noise without walking off the observed path (a few cm)."""
    rng = np.random.default_rng(1)
    frames = list(range(60))
    truth = np.column_stack([np.linspace(0, 20, 60), np.sin(np.linspace(0, 2, 60))])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, 25.0)

    # Closer to the underlying truth than the noisy input, and never far from it.
    assert np.abs(smoothed - truth).mean() < np.abs(noisy - truth).mean()
    assert np.sqrt(((smoothed - noisy) ** 2).sum(1)).mean() < 0.15


def test_kalman_follows_a_turn_without_cutting_the_corner() -> None:
    """A right-angle path is tracked, not short-circuited across the corner."""
    frames = list(range(40))
    along = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    up = np.column_stack([np.full(20, 10.0), np.linspace(0.5, 10, 20)])
    path = np.vstack([along, up])

    smoothed = ts.kalman_rts_smooth(frames, path, 25.0, process_std=6.0)

    # The corner point stays near the true corner (not cut diagonally inward).
    corner_err = np.hypot(*(smoothed[19] - [10.0, 0.0]))
    assert corner_err < 1.5


def test_outlier_rejection_flags_a_spike() -> None:
    """A single implausible jump is flagged for exclusion; neighbours kept."""
    frames = list(range(10))
    pos = np.column_stack([np.arange(10.0), np.zeros(10)])
    pos[5] = [5.0, 4.0]  # a 4 m lateral jump in one frame

    keep = ts.reject_kinematic_outliers(frames, pos, 25.0)

    assert not keep[5]  # the spike itself is flagged
    # A one-frame spike also inflates the acceleration at the adjacent frame, so
    # neighbours may be flagged too; the far, clean points stay kept.
    assert keep[0]
    assert keep[2]
    assert keep[7]
    assert keep[9]


def test_spike_does_not_survive_smoothing() -> None:
    """With outlier rejection, a spike is pulled back to the smooth path."""
    frames = list(range(20))
    pos = np.column_stack([np.arange(20.0), np.zeros(20)])
    pos[10] = [10.0, 3.0]

    smoothed = ts.kalman_rts_smooth(frames, pos, 25.0)

    assert abs(smoothed[10][1]) < 1.0  # cross-track spike suppressed


def test_short_track_is_returned_unchanged() -> None:
    frames = [0, 1]
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.array_equal(ts.kalman_rts_smooth(frames, pos, 25.0), pos)


def test_jerk_metric_zero_for_constant_velocity() -> None:
    """A perfectly constant-velocity path has no jerk."""
    frames = list(range(10))
    pos = np.column_stack([np.arange(10.0), np.zeros(10)])
    assert ts.trajectory_jerk(frames, pos, 25.0)["mean_abs"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_smooth_metric_covers_every_vehicle() -> None:
    """Regression: smoothing reaches EVERY track, not only refined ones.

    Smoothing used to be a layer inside the contour refinement, which meant three
    paths silently skipped it (no sidecar, no contour for this vehicle, or the
    peak-speed guard reverting to the raw legacy anchor). It is now an
    unconditional stage over whatever the refinement returned.
    """
    from accident_reconstruction.auto_reconstruct import smooth_metric

    rng = np.random.default_rng(2)
    frames = list(range(40))
    truth = np.column_stack([np.linspace(0, 20, 40), np.zeros(40)])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)
    metric = {
        "car": {
            f: (float(noisy[i][0]), float(noisy[i][1])) for i, f in enumerate(frames)
        },
        "person": {0: (0.0, 0.0), 1: (1.0, 0.0)},  # too short to smooth
    }

    out = smooth_metric(metric)

    car_frames = sorted(out["car"])
    before = ts.trajectory_jerk(frames, noisy, 25.0)
    after = ts.trajectory_jerk(
        car_frames, np.array([out["car"][f] for f in car_frames]), 25.0
    )
    assert after["mean_abs"] < before["mean_abs"] / 10
    assert car_frames == frames  # same frames in, same frames out
    assert out["person"] == metric["person"]  # short track passes through intact


# Measurement noise of a RAW homography anchor, in metres. Every test above uses
# 0.05 m, which is the CONTOUR-REFINED anchor the smoother's defaults were tuned
# on (BMW). A raw anchor -- what a scene with no contour sidecar, or a vehicle
# whose peak-speed guard reverted to legacy, actually carries -- is an order of
# magnitude noisier. That gap is why the divergence below went unnoticed.
RAW_ANCHOR_NOISE_M = (0.3, 0.8, 1.5)


@pytest.mark.parametrize("noise", RAW_ANCHOR_NOISE_M)
def test_kalman_does_not_diverge_on_raw_anchor_noise(noise: float) -> None:
    """Regression: the smoother must FOLLOW the data, not coast away from it.

    With ``meas_std`` fixed at 0.08 m, a track this noisy puts nearly every real
    sample outside the 5-sigma innovation gate, so the filter runs predict-only
    and the constant-acceleration model extrapolates freely. On the six recorded
    scenes that walked 7 of 13 vehicle tracks off their true path -- 宜蘭五結's
    car reversed direction and ran 57 m the wrong way; 台南永康's car left the
    data by 105 m.

    Two metres is generous: the point is to catch a track that has left the
    scene, not to pin down the smoothing strength.
    """
    rng = np.random.default_rng(0)
    fps, n = 30.0, 111
    frames = list(range(n))
    seconds = np.arange(n) / fps
    # A car holding 7.5 m/s (27 km/h) in a straight line -- deliberately the
    # simplest possible motion, so any departure is the filter's doing.
    truth = np.column_stack([16.0 - 7.5 * seconds, np.full(n, 1.4)])
    noisy = truth + rng.normal(scale=noise, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, fps)

    assert np.hypot(*(smoothed - truth).T).max() < 2.0


def test_low_jerk_does_not_certify_a_smoothed_track() -> None:
    """A diverged track is SMOOTH; jerk alone cannot tell the two apart.

    This is the trap the original fix fell into: mean |jerk| dropping from 15277
    to 4.4 read as a triumph, when it partly meant the output had stopped
    following the measurements altogether. Any future tuning must assert
    closeness to the data as well as plausibility of the motion.
    """
    rng = np.random.default_rng(0)
    fps, n = 30.0, 111
    frames = list(range(n))
    truth = np.column_stack([16.0 - 7.5 * np.arange(n) / fps, np.full(n, 1.4)])
    noisy = truth + rng.normal(scale=0.8, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, fps)

    # Plausible motion AND on the observed path -- both, or the track is wrong.
    assert ts.trajectory_jerk(frames, smoothed, fps)["mean_abs"] < 15.0
    assert np.hypot(*(smoothed - truth).T).mean() < 1.0


def test_turn_survives_raw_anchor_noise() -> None:
    """Fixing the divergence must not flatten a genuine manoeuvre into a line."""
    rng = np.random.default_rng(3)
    fps, n = 30.0, 90
    frames = list(range(n))
    angle = np.linspace(0.0, np.pi / 2, n)
    truth = np.column_stack([20.0 * np.cos(angle), 20.0 * np.sin(angle)])
    noisy = truth + rng.normal(scale=0.3, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, fps)

    # Follows the arc rather than cutting across it, and beats the raw input.
    assert np.hypot(*(smoothed - truth).T).max() < 2.0
    assert np.abs(smoothed - truth).mean() < np.abs(noisy - truth).mean()


def test_estimate_measurement_std_reads_the_noise_off_the_track() -> None:
    """The per-track noise estimate, across two orders of magnitude."""
    rng = np.random.default_rng(0)
    truth = np.column_stack([np.linspace(0, 30, 200), np.zeros(200)])

    for noise in (0.05, 0.3, 1.5):
        estimate = ts.estimate_measurement_std(
            truth + rng.normal(0, noise, truth.shape)
        )
        # Within a factor of ~1.5 is ample -- meas_std sets the smoothing
        # strength, and being in the right order of magnitude is what matters.
        assert 0.67 * noise < estimate < 1.5 * noise


def test_estimate_measurement_std_ignores_real_acceleration() -> None:
    """A hard-braking car is not noise: its curvature must not inflate sigma."""
    seconds = np.arange(150) / 30.0
    braking = np.column_stack([20.0 * seconds - 3.0 * seconds**2, np.zeros(150)])

    assert ts.estimate_measurement_std(braking) == ts.MIN_MEAS_STD_M


def test_estimate_measurement_std_short_track_falls_back() -> None:
    assert ts.estimate_measurement_std(np.zeros((3, 2))) == ts.DEFAULT_MEAS_STD_M


def test_divergence_guard_returns_the_input() -> None:
    """A track the filter cannot follow comes back untouched, not relocated.

    Forced by passing a meas_std far below the real noise -- exactly the
    condition the old fixed 0.08 m created on a raw anchor.
    """
    rng = np.random.default_rng(0)
    frames = list(range(111))
    truth = np.column_stack([16.0 - 7.5 * np.arange(111) / 30.0, np.full(111, 1.4)])
    noisy = truth + rng.normal(scale=0.8, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, 30.0, meas_std=0.08)

    assert np.array_equal(smoothed, noisy)


def test_divergence_guard_does_not_fire_on_normal_smoothing() -> None:
    """The guard must not veto the ordinary case it exists to protect."""
    rng = np.random.default_rng(1)
    frames = list(range(60))
    truth = np.column_stack([np.linspace(0, 20, 60), np.zeros(60)])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, 25.0)

    assert not np.array_equal(smoothed, noisy)
    assert ts.trajectory_jerk(frames, smoothed, 25.0)["mean_abs"] < 15.0


def test_route_csv_row_keeps_enough_decimals_to_survive_differentiation() -> None:
    """The delivered CSV must not re-quantise the smoothed track back into jerk.

    7 decimals of a degree is ~1.1 cm; differentiated three times at video rate
    that alone is hundreds of m/s^3, which buried the smoother's output in the
    artefact the web map actually reads.
    """
    from accident_reconstruction.birdseye_manual_annotation import route_csv_row

    row = route_csv_row(12, "car", 25.041234567891, 121.512345678912, 33.33, 12)

    _, _, lat, lon, _, _ = row.split(",")
    assert len(lat.split(".")[1]) == 9
    assert len(lon.split(".")[1]) == 9
    assert float(lat) == pytest.approx(25.041234568, abs=1e-9)


def test_short_noisy_track_is_not_invented() -> None:
    """A track whose noise rivals its whole motion comes back untouched.

    yilan's pedestrian: 18 frames, 1.4 m of walking, ~0.39 m of anchor noise.
    With no signal to lock onto the filter follows its prior and extrapolates,
    which turned that walk into an 11 m, 71 km/h sprint. Smoothing must not be
    able to manufacture motion that is not in the data.
    """
    rng = np.random.default_rng(5)
    fps, n = 30.0, 18
    frames = list(range(n))
    truth = np.column_stack([np.linspace(0.0, 1.4, n), np.zeros(n)])
    noisy = truth + rng.normal(scale=0.39, size=truth.shape)

    smoothed = ts.kalman_rts_smooth(frames, noisy, fps)

    travelled = np.hypot(*(smoothed[-1] - smoothed[0]))
    assert travelled < 3.0  # a walk, not a sprint


# --- Shape smoothing (fit_smooth_curve) --------------------------------------
#
# The Kalman pass is judged on jerk and on position error, and passes both on the
# tracks below -- yet the drawn route came out saw-toothed anyway, which is what
# these tests exist to stop coming back. See the module docstring for why the two
# are not the same property.


def _shape(
    frames: list[int],
    positions: np.ndarray,
    fps: float,
    impact_frame: int | None = None,
) -> tuple[float, float]:
    """The two acceptance measures: p99 turn angle (deg), max cornering (m/s^2)."""
    metrics = ts.shape_metrics(frames, positions, fps, impact_frame=impact_frame)
    return metrics["turn_p99"], metrics["lateral_max"]


def _collision_track(fps: float, n: int, deflection_deg: float):
    """A straight run that changes direction sharply at the impact, then runs on.

    The shape of every clip in the recorded set: a vehicle travelling in a line,
    struck, and deflected. The corner is the finding; everything else is motion
    that should smooth.
    """
    frames = list(range(n))
    impact = n // 2
    step = 8.0 / fps
    heading = np.where(np.arange(n - 1) < impact, 0.0, np.radians(deflection_deg))
    steps = np.column_stack([step * np.cos(heading), step * np.sin(heading)])
    return frames, impact, np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def test_slow_track_is_smooth_in_shape_not_only_in_jerk() -> None:
    """Regression: a crawling vehicle drew as a zig-zag with excellent jerk.

    BMW's car covers 5.5 m in 192 frames -- under 3 cm per frame, no more than
    the anchor noise. In that regime the wobble is the same size as the step, so
    the HEADING flips frame to frame while the position error, and hence the
    jerk, stays small. Jerk read 4.5 m/s^3 and the README figure showed a saw
    blade.

    Reproduced rather than replayed: a straight crawl at the same frame rate and
    extent, with ``meas_std`` pinned to the floor, which is the production
    condition -- the pipeline's second Kalman pass estimates its noise from an
    input that has already been smoothed once, so the estimate collapses to
    :data:`MIN_MEAS_STD_M` and the pass hands its wobble straight through.
    """
    rng = np.random.default_rng(7)
    fps, n = 23.0, 192
    frames = list(range(n))
    truth = np.column_stack([np.linspace(0.0, 5.5, n), np.zeros(n)])
    noisy = truth + rng.normal(scale=0.04, size=truth.shape)

    kalman = ts.kalman_rts_smooth(frames, noisy, fps, meas_std=ts.MIN_MEAS_STD_M)
    fitted = ts.fit_smooth_curve(frames, kalman, fps, measurements=noisy)

    # The Kalman pass alone already looks kinematically fine...
    assert ts.trajectory_jerk(frames, kalman, fps)["mean_abs"] < 15.0
    # ...while its SHAPE is not, and the curve fit is what fixes that.
    assert _shape(frames, kalman, fps)[0] > ts.CURVE_TURN_TARGET_DEG
    assert _shape(frames, fitted, fps)[0] <= ts.CURVE_TURN_TARGET_DEG


def test_elbow_is_rounded_into_an_arc() -> None:
    """A corner no vehicle could take at this speed is opened out into a curve.

    Per-sample turn angle cannot see this one: 30 degrees spread over ten samples
    is 3 degrees each, well inside the zig-zag target, and still draws as two
    straight runs meeting at a point. What rules it out is the lateral
    acceleration it would take.
    """
    fps, n = 30.0, 120
    frames = list(range(n))
    # 12 m/s along +x, then the same speed rotated 30 degrees -- an elbow.
    step = 12.0 / fps
    heading = np.where(np.arange(n - 1) < n // 2, 0.0, np.radians(30.0))
    steps = np.column_stack([step * np.cos(heading), step * np.sin(heading)])
    path = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])

    fitted = ts.fit_smooth_curve(frames, path, fps, measurements=path)

    assert _shape(frames, path, fps)[1] > ts.MAX_LATERAL_ACCEL_MPS2
    assert _shape(frames, fitted, fps)[1] <= ts.MAX_LATERAL_ACCEL_MPS2
    # Rounded, not erased: the fit still gets from the same start to the same end.
    assert np.hypot(*(fitted[-1] - path[-1])) < 1.5


def test_straight_run_stays_straight() -> None:
    """The strong-penalty end of the fit is a LINE, so a straight run keeps its
    shape instead of acquiring a curve to spend the smoothing on."""
    rng = np.random.default_rng(8)
    fps, n = 25.0, 100
    frames = list(range(n))
    truth = np.column_stack([np.linspace(0.0, 25.0, n), np.zeros(n)])
    noisy = truth + rng.normal(scale=0.1, size=truth.shape)

    fitted = ts.fit_smooth_curve(frames, noisy, fps, measurements=noisy)

    assert np.abs(fitted[:, 1]).max() < 0.1
    assert np.abs(fitted - truth).mean() < np.abs(noisy - truth).mean()


def test_real_turn_is_kept_as_a_turn() -> None:
    """Smoothing the shape must not flatten a manoeuvre a vehicle really made.

    A 20 m-radius quarter circle driven in 5 s is 6.3 m/s and 2.0 m/s^2 of
    cornering -- an ordinary turn, comfortably inside
    :data:`MAX_LATERAL_ACCEL_MPS2`, so the fit has no reason to open it out.
    """
    fps, n = 30.0, 150
    frames = list(range(n))
    angle = np.linspace(0.0, np.pi / 2, n)
    truth = np.column_stack([20.0 * np.cos(angle), 20.0 * np.sin(angle)])

    fitted = ts.fit_smooth_curve(frames, truth, fps, measurements=truth)

    # Still a quarter circle of radius 20, not a chord across it.
    assert np.abs(np.hypot(fitted[:, 0], fitted[:, 1]) - 20.0).max() < 0.5


def test_curve_fit_respects_the_faithfulness_budget() -> None:
    """It may round off a zig-zag; it may not walk the route off the data.

    Driven by a genuinely sharp path the fit cannot smooth for free: the returned
    curve must still sit inside the same all-or-nothing budget
    :func:`kalman_rts_smooth` uses, rather than straightening whatever it takes to
    hit the shape targets.
    """
    fps, n = 30.0, 120
    frames = list(range(n))
    half = n // 2
    path = np.vstack(
        [
            np.column_stack([np.linspace(0, 20, half), np.zeros(half)]),
            np.column_stack([np.full(n - half, 20.0), np.linspace(0.2, 20, n - half)]),
        ]
    )

    fitted = ts.fit_smooth_curve(frames, path, fps, measurements=path)

    budget = min(
        ts.DIVERGENCE_TOLERANCE_M,
        ts.DIVERGENCE_TOLERANCE_EXTENT
        * float(np.hypot(*np.diff(path, axis=0).T).sum()),
    )
    assert float(np.median(np.hypot(*(fitted - path).T))) <= budget


def test_stationary_wobble_does_not_veto_the_fit() -> None:
    """A parked vehicle's heading is meaningless; it must not block smoothing.

    Its sub-millimetre steps flip direction at random, so a turn-angle-only rule
    would reject every candidate and hand back the unsmoothed track. Lateral
    acceleration is (yaw rate x speed), which is ~0 when the vehicle is stopped.
    """
    rng = np.random.default_rng(9)
    fps, n = 30.0, 150
    frames = list(range(n))
    moving = np.column_stack([np.linspace(0.0, 10.0, 100), np.zeros(100)])
    parked = np.column_stack([np.full(50, 10.0), np.zeros(50)])
    path = np.vstack([moving, parked]) + rng.normal(scale=0.02, size=(n, 2))

    fitted = ts.fit_smooth_curve(frames, path, fps, measurements=path)

    assert not np.array_equal(fitted, path)  # the fit ran rather than bailing out
    assert _shape(frames, fitted, fps)[1] <= ts.MAX_LATERAL_ACCEL_MPS2


def test_smooth_metric_delivers_a_smooth_shape() -> None:
    """End to end: what every writer reads is shape-smooth, not just low-jerk."""
    from accident_reconstruction.auto_reconstruct import smooth_metric

    rng = np.random.default_rng(10)
    fps, n = 23.0, 192
    frames = list(range(n))
    truth = np.column_stack([np.linspace(0.0, 5.5, n), np.zeros(n)])
    noisy = truth + rng.normal(scale=0.02, size=truth.shape)
    metric = {
        "car": {
            f: (float(noisy[i][0]), float(noisy[i][1])) for i, f in enumerate(frames)
        }
    }

    out = smooth_metric(metric)

    positions = np.array([out["car"][f] for f in frames])
    turn, cornering = _shape(frames, positions, fps)
    assert turn <= ts.CURVE_TURN_TARGET_DEG
    assert cornering <= ts.MAX_LATERAL_ACCEL_MPS2


def test_impact_deflection_is_not_smoothed_away() -> None:
    """The headline regression: the crash must survive the smoothing.

    Fitted as ONE curve, the real corner at the impact keeps the turn-angle
    criterion failing however far lambda is pushed, so the walk runs to the end
    of the grid and straightens the WHOLE track to pay for one local feature. On
    BMW's motorbike that moved the start 1.77 m, cut the path 6.04 m -> 3.54 m,
    and flattened the direction change that is the main evidence in the clip --
    while the median deviation stayed a respectable 0.22 m, so nothing in the
    numbers complained. Splitting at the impact is what keeps the corner.
    """
    fps, n = 25.0, 120
    frames, impact, path = _collision_track(fps, n, deflection_deg=55.0)

    fitted = ts.fit_smooth_curve(
        frames, path, fps, measurements=path, impact_frame=impact
    )

    def heading(positions, a, b):
        span = positions[b] - positions[a]
        return np.degrees(np.arctan2(span[1], span[0]))

    # The deflection is still there, to within a couple of degrees...
    before = heading(fitted, 2, impact - 2)
    after = heading(fitted, impact + 2, n - 3)
    assert abs((after - before) - 55.0) < 5.0
    # ...and the track was not straightened to buy it.
    assert np.hypot(*(fitted - path).T).max() < 0.5
    kept = np.hypot(*np.diff(fitted, axis=0).T).sum()
    assert kept > 0.9 * np.hypot(*np.diff(path, axis=0).T).sum()


def test_fit_spanning_the_impact_would_destroy_it() -> None:
    """Companion to the above: shows the split is doing the work, not luck."""
    fps, n = 25.0, 120
    frames, impact, path = _collision_track(fps, n, deflection_deg=55.0)

    whole = ts.fit_smooth_curve(frames, path, fps, measurements=path)
    split = ts.fit_smooth_curve(
        frames, path, fps, measurements=path, impact_frame=impact
    )

    # Fitted whole the corner is rounded off; split, it is kept.
    assert np.hypot(*(whole - path).T).max() > np.hypot(*(split - path).T).max()


def test_shape_cap_stops_a_track_being_reshaped() -> None:
    """No fit may move any sample more than a set share of the path length.

    The guard the median budget could not provide. A median stays small while a
    fit relocates one END of the track, which is exactly how BMW's motorbike lost
    1.77 m at its start without tripping anything.
    """
    fps, n = 25.0, 120
    frames, _, path = _collision_track(fps, n, deflection_deg=90.0)

    fitted = ts.fit_smooth_curve(frames, path, fps, measurements=path)

    length = float(np.hypot(*np.diff(path, axis=0).T).sum())
    assert np.hypot(*(fitted - path).T).max() <= ts.MAX_SHAPE_DEVIATION_EXTENT * length


def test_shape_metrics_excludes_the_impact_seam() -> None:
    """A kept collision corner is the finding, not a defect to report."""
    fps, n = 25.0, 120
    frames, impact, path = _collision_track(fps, n, deflection_deg=55.0)

    # Read on the cornering, not the turn angle: the seam is ONE sample wide, and
    # a p99 over 118 samples steps straight over it -- the same blind spot that
    # made ``lateral_max`` a max in the first place.
    assert _shape(frames, path, fps)[1] > ts.MAX_LATERAL_ACCEL_MPS2  # seam counted
    assert _shape(frames, path, fps, impact)[1] <= ts.MAX_LATERAL_ACCEL_MPS2
    assert ts.shape_metrics(frames, path, fps, impact_frame=impact)["runs"] == 2


def test_standstill_jitter_neither_vetoes_nor_certifies() -> None:
    """Wreckage that stops must not block smoothing -- nor fake a pass.

    Both failures happened, from the same knob. With the step floor at a fraction
    of the MEDIAN step, a track that is stationary for most of its samples sets
    that median to the standstill jitter itself, and sub-millimetre "turns" of 19
    degrees vetoed every candidate. Moving the floor up to the measurement noise
    fixed that and broke the opposite way: a track whose steps are the SIZE of
    its noise -- the whole reason this stage exists -- had every segment excluded,
    so nothing could fail and the input came back unsmoothed.
    """
    rng = np.random.default_rng(11)
    fps, n = 23.0, 160
    frames = list(range(n))
    noise = 0.03
    # Moves at ~ the noise per frame, then stops dead for the rest.
    moving = np.column_stack([np.linspace(0.0, 3.0, 100), np.zeros(100)])
    stopped = np.column_stack([np.full(60, 3.0), np.zeros(60)])
    path = np.vstack([moving, stopped]) + rng.normal(scale=noise, size=(n, 2))

    fitted = ts.fit_smooth_curve(frames, path, fps, measurements=path)

    # It ran (the standstill did not veto it)...
    assert not np.array_equal(fitted, path)
    # ...and it actually smoothed the moving part, rather than passing it through
    # because every segment had been excluded from the criteria.
    assert _shape(frames[:100], fitted[:100], fps)[0] <= ts.CURVE_TURN_TARGET_DEG
    assert _shape(frames[:100], path[:100], fps)[0] > ts.CURVE_TURN_TARGET_DEG


def test_turn_angles_of_a_straight_line_are_zero() -> None:
    straight = np.column_stack([np.arange(6.0), np.zeros(6)])
    assert ts.turn_angles_deg(straight).max() == pytest.approx(0.0, abs=1e-9)


def test_figure_markers_do_not_merge_into_a_lumpy_line() -> None:
    """Not every saw-tooth was in the data -- some of it was the drawing.

    The recognised figure drew a filled dot at EVERY frame. Wherever the vehicle
    moved less than a marker width per frame the dots overlapped into a blob
    whose ragged edge read as a jagged path, on trajectories that are clean
    curves when rendered as a bare line. Markers must stay at least a diameter
    apart.
    """
    from accident_reconstruction.recognized_route import _MARKER_RADIUS, _spaced

    crawl = [(100.0 + 0.5 * i, 200.0) for i in range(60)]  # 0.5 px per frame

    kept = _spaced(crawl, _MARKER_RADIUS)

    gaps = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(kept)]
    # Every gap except the last (which just carries the end point) clears a
    # marker diameter, and the run's extent is still drawn end to end.
    assert min(gaps[:-1]) >= 2 * _MARKER_RADIUS
    assert kept[0] == crawl[0]
    assert kept[-1] == crawl[-1]
    assert len(kept) < len(crawl)


def test_figure_markers_are_all_kept_when_already_spread() -> None:
    """A fast vehicle keeps every sample -- the thinning is density-driven."""
    from accident_reconstruction.recognized_route import _MARKER_RADIUS, _spaced

    quick = [(100.0 + 20.0 * i, 200.0) for i in range(10)]

    assert _spaced(quick, _MARKER_RADIUS) == quick


def test_short_track_skips_the_curve_fit() -> None:
    frames = [0, 1, 2]
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
    assert np.array_equal(ts.fit_smooth_curve(frames, pos, 25.0), pos)
