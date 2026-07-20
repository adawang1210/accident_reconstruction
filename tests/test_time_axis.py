"""Tests for the PTS time-axis helpers and time-aware speed windowing.

Pure-function only (no SAM2 / ffprobe subprocess): they pin that speed is timed by
real per-frame timestamps when present and falls back to ``frame / fps`` when not,
and that an uneven (variable-frame-rate) axis is flagged.
"""

from __future__ import annotations

import pytest

from accident_reconstruction.time_axis import (
    frame_seconds,
    frame_time,
    interval_cv,
    load_frame_times,
    time_axis_warning,
    window_elapsed,
)


def test_interval_cv_even_axis_is_zero() -> None:
    """A perfectly even time axis has ~zero interval variation."""
    cv = interval_cv([0.0, 0.04, 0.08, 0.12])
    assert cv is not None
    assert cv < 1e-9


def test_interval_cv_none_when_too_few_gaps() -> None:
    """Fewer than two positive gaps cannot yield a CV."""
    assert interval_cv([0.0]) is None
    assert interval_cv([1.0, 1.0, 1.0]) is None  # repeated stamps -> no gaps


def test_interval_cv_flags_variable_rate() -> None:
    """Uneven gaps (VFR / dropped frames) produce a positive CV."""
    cv = interval_cv([0.0, 0.04, 0.20, 0.24])  # one big gap
    assert cv is not None
    assert cv > 0.05


def test_frame_time_prefers_pts_then_fps() -> None:
    """Frame time reads the PTS list, falling back to frame/fps out of range."""
    assert frame_time([0.0, 0.5, 1.0], 2, 25.0) == 1.0
    assert frame_time([0.0, 0.5], 5, 25.0) == 5 / 25.0
    assert frame_time(None, 25, 25.0) == 1.0


def test_frame_seconds_prefers_map_then_fps() -> None:
    """Dict-keyed frame time reads the map, else frame/fps."""
    assert frame_seconds({10: 2.5}, 10, 25.0) == 2.5
    assert frame_seconds({10: 2.5}, 11, 25.0) == 11 / 25.0
    assert frame_seconds(None, 50, 25.0) == 2.0


def test_window_elapsed_fallback_is_single_division() -> None:
    """Without PTS, elapsed is ``(f_to - f_from) / fps`` as ONE division.

    Computing ``f_to / fps - f_from / fps`` instead drifts by a float ULP on tracks
    with frame gaps, which historically perturbed fallback speeds; this pins the
    exact-parity form. 105/25 - 90/25 == 0.6000000000000001, but the correct single
    division 15/25 == 0.6.
    """
    assert window_elapsed(None, 90, 105, 25.0) == 15 / 25.0
    assert window_elapsed({}, 90, 105, 25.0) == 15 / 25.0
    # The drift form this guards against:
    assert window_elapsed(None, 90, 105, 25.0) != 105 / 25.0 - 90 / 25.0


def test_window_elapsed_uses_real_pts_when_present() -> None:
    """With PTS for both frames, elapsed is their timestamp difference."""
    times = {90: 3.6, 105: 4.3}  # a stretched (VFR) interval
    assert window_elapsed(times, 90, 105, 25.0) == pytest.approx(0.7)
    # Missing either endpoint falls back to the frame-delta form.
    assert window_elapsed({90: 3.6}, 90, 105, 25.0) == 15 / 25.0


def test_time_axis_warning_only_for_uneven_axis() -> None:
    """No warning for a steady axis; a warning string for an uneven one."""
    assert time_axis_warning([0.0, 0.04, 0.08, 0.12]) is None
    warning = time_axis_warning([0.0, 0.04, 0.5, 0.54])
    assert warning is not None
    assert "時間軸" in warning


def test_load_frame_times_reads_column(tmp_path) -> None:
    """``t_sec`` is read per frame; a legacy CSV without it yields an empty map."""
    csv_with = tmp_path / "with.csv"
    csv_with.write_text(
        "frame,vehicle,anchor_x,anchor_y,t_sec\n"
        "10,car,1,2,0.400\n"
        "10,taxi,3,4,0.400\n"
        "11,car,5,6,0.440\n"
    )
    assert load_frame_times(csv_with) == {10: 0.4, 11: 0.44}

    legacy = tmp_path / "legacy.csv"
    legacy.write_text("frame,vehicle,anchor_x,anchor_y\n10,car,1,2\n")
    assert load_frame_times(legacy) == {}


def test_windowed_motion_uses_real_timestamps() -> None:
    """Speed reflects the real elapsed time, not the nominal frame cadence.

    A vehicle covers 10 m between frame 0 and frame 10. With a nominal 25 fps that
    is 0.4 s (90 km/h), but if the real PTS say those frames span 0.5 s the honest
    speed is 72 km/h. Both spans sit inside the 0.6 s speed window, so the only
    difference is which clock is used.
    """
    from accident_reconstruction.auto_reconstruct import windowed_motion

    # Three samples with evenly spread steps, so the speed window trusts them.
    track = {0: (0.0, 0.0), 5: (5.0, 0.0), 10: (10.0, 0.0)}
    times = {0: 0.0, 5: 0.25, 10: 0.5}
    motion = windowed_motion(track, times, fps=25.0)
    assert abs(motion[10][1] - 72.0) < 1e-6

    # Without timestamps it uses frame/fps: 10 frames / 25 fps = 0.4 s -> 90 km/h.
    motion_nominal = windowed_motion(track, None, fps=25.0)
    assert abs(motion_nominal[10][1] - 90.0) < 1e-6
