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
