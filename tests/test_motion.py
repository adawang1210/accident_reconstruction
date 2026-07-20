"""Unit tests for the shared speed-windowing (``motion.windowed_speed``)."""

from __future__ import annotations

import pytest

from accident_reconstruction.motion import (
    SPEED_WINDOW_SECONDS,
    euclidean,
    windowed_speed,
)


def test_constant_velocity_metric() -> None:
    # 10 fps, 1 m east per frame -> 10 m/s -> 36 km/h; window = 0.6 s = 6 frames.
    track = {i: (float(i), 0.0) for i in range(10)}
    motion = windowed_speed(track, fps=10.0, distance=euclidean)
    assert motion[9][0] == pytest.approx(9.0)  # cumulative metres
    # frame 9 vs window start (frame 3): 6 m over 0.6 s = 36 km/h.
    assert motion[9][1] == pytest.approx(36.0)
    assert SPEED_WINDOW_SECONDS == 0.6


def test_first_frame_has_zero_speed() -> None:
    motion = windowed_speed({5: (0.0, 0.0)}, fps=30.0, distance=euclidean)
    assert motion[5] == (0.0, 0.0)


def test_empty_track() -> None:
    assert windowed_speed({}, fps=30.0, distance=euclidean) == {}


def test_gap_flush_jump_is_not_a_speed_spike() -> None:
    """A single-frame jump right after a tracking gap must not read as a speed.

    Reproduces the BMW clip's 114.6 km/h artifact: a >window gap flushes the
    window, so the next sample pairs with one just 1 frame back; a 1.38 m mask
    jump over 1/23 s then printed 114.6 km/h. Two-sample windows are a single
    step (indistinguishable from a jump), so no speed is reported there.
    """
    fps = 23.0
    track = {f: (0.0, 0.0) for f in range(100, 121)}  # stationary...
    track[140] = (0.0, 0.0)  # ...then a 20-frame gap (> 0.6 s window)
    track[141] = (1.38, 0.0)  # impact-frame mask jump
    motion = windowed_speed(track, fps=fps, distance=euclidean)
    # Without the guard this frame read 1.38 / (1/23) * 3.6 = 114.6 km/h.
    assert motion[141][1] == 0.0


def test_jump_dominated_window_reports_no_speed() -> None:
    """One dominant step inside a full window is a jump, not motion.

    A stationary vehicle whose mask suddenly snaps 1.38 m concentrates the whole
    window path in a single step; the old formula smeared that into a phantom
    ~8 km/h creep for the rest of the window.
    """
    track = {f: (0.0, 0.0) for f in range(0, 20)}
    track[20] = (1.38, 0.0)  # mid-track mask snap, window otherwise full
    motion = windowed_speed(track, fps=23.0, distance=euclidean)
    assert motion[20][1] == 0.0


def test_no_guard_keeps_legacy_formula_for_control_uses() -> None:
    """``jump_guard=False`` reproduces the historic two-sample speeds exactly.

    The settle/truncation logic consumes this stream; its behaviour (and thus the
    trajectory cuts and aligned geometry) must not shift under the display guard.
    """
    fps = 23.0
    track = {f: (0.0, 0.0) for f in range(100, 121)}
    track[140] = (0.0, 0.0)
    track[141] = (1.38, 0.0)  # the BMW-style impact jump
    motion = windowed_speed(track, fps=fps, distance=euclidean, jump_guard=False)
    assert motion[141][1] == pytest.approx(1.38 * fps * 3.6)  # 114.26 km/h


def test_early_track_speed_survives_with_three_samples() -> None:
    """Genuine motion right after track start keeps its speed from sample 3 on.

    Distinguishes real entry speed (displacement spread over the steps -- e.g.
    the keelung taxi entering the frame fast) from the single-step jump above:
    only the unjudgeable two-sample frame is suppressed.
    """
    track = {i: (float(i), 0.0) for i in range(10)}  # 1 m per frame at 10 fps
    motion = windowed_speed(track, fps=10.0, distance=euclidean)
    assert motion[1][1] == 0.0  # two samples: a single step, not judgeable
    assert motion[2][1] == pytest.approx(36.0)  # three samples, even steps
    assert motion[9][1] == pytest.approx(36.0)
