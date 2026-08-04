"""Tests for the constant-acceleration Kalman + RTS trajectory smoother."""

from __future__ import annotations

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
