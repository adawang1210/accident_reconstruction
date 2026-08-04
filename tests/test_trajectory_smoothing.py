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
