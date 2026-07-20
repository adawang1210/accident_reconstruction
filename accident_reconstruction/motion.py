"""Shared per-frame speed windowing for the metric and aligned trajectories.

The raw-homography output (``auto_reconstruct.windowed_motion``) and the
road-aligned output (``birdseye_manual_annotation.aligned_motion``) compute speed
identically -- a displacement / elapsed-time across a fixed time window -- and
differ only in the distance metric (Euclidean on the metric plane vs haversine on
lat/lon). This module holds that one windowing implementation so the logic and the
window length live in a single place.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping

from accident_reconstruction.time_axis import window_elapsed

# Speed = displacement across the samples within this window / elapsed time.
# Smooths single-frame anchor jitter without lagging genuine acceleration.
SPEED_WINDOW_SECONDS = 0.6

# Real movement spreads across the window's steps; a tracking jump (the mask
# snapping to a new spot at impact, or right after a gap flushed the window)
# concentrates the whole displacement in ONE step. A window whose largest step
# exceeds this fraction of its path is reporting a jump, not a speed -- seen on
# the BMW clip, where a 20-frame gap flushed the window and a 1.38 m
# impact-frame jump over 1/23 s printed 114.6 km/h. Two-sample windows are a
# single step by construction (indistinguishable from a jump), so speed needs
# at least three samples.
JUMP_DOMINANCE_RATIO = 0.8

Point = tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    """Planar distance between two metric ``(east_m, north_m)`` points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def windowed_speed(
    track: dict[int, Point],
    fps: float,
    distance: Callable[[Point, Point], float],
    times: Mapping[int, float] | None = None,
    jump_guard: bool = True,
) -> dict[int, tuple[float, float]]:
    """Per-frame ``(cumulative_m, speed_kmh)`` over ``SPEED_WINDOW_SECONDS``.

    Args:
        track: ``{frame: point}`` for one vehicle, in whatever coordinate the
            ``distance`` function expects (metric plane or lat/lon).
        fps: Source frames per second; the window spans ``seconds * fps`` frames.
        distance: Metric between two points -- :func:`euclidean` for the metric
            plane, a haversine for lat/lon.
        times: Optional ``{frame: t_sec}`` real per-frame timestamps (PTS). When
            given, both the window span and the elapsed time are measured on them,
            so a variable-frame-rate clip is not timed at a wrong uniform cadence.
            Without them both fall back to the frame delta over ``fps`` (see
            :func:`accident_reconstruction.time_axis.window_elapsed`).
        jump_guard: When True (the default, for DISPLAYED speeds) a frame whose
            window is a single step or is dominated by one step reports 0 instead
            of a jump-inflated speed (see ``JUMP_DOMINANCE_RATIO``). Pass False
            for CONTROL uses that must keep the historic formula -- e.g. the
            settle/truncation logic, where a guard zero would misread a tumbling
            vehicle as "at rest" and move the trajectory cut.

    Returns:
        ``{frame: (cumulative_m, speed_kmh)}``.
    """
    motion: dict[int, tuple[float, float]] = {}
    window: deque[tuple[int, Point]] = deque()
    steps: deque[float] = deque()  # distances between consecutive window samples
    cumulative = 0.0
    previous: Point | None = None
    min_samples = 3 if jump_guard else 2
    for frame in sorted(track):
        point = track[frame]
        if previous is not None:
            cumulative += distance(previous, point)
        if window:
            steps.append(distance(window[-1][1], point))
        previous = point
        window.append((frame, point))
        while (
            len(window) >= 2
            and window_elapsed(times, window[0][0], frame, fps) > SPEED_WINDOW_SECONDS
        ):
            window.popleft()
            steps.popleft()
        speed = 0.0
        if len(window) >= min_samples:
            first_frame, first_point = window[0]
            elapsed = window_elapsed(times, first_frame, frame, fps)
            path = sum(steps)
            jump = jump_guard and path > 0 and max(steps) > JUMP_DOMINANCE_RATIO * path
            if elapsed > 0 and not jump:
                speed = distance(first_point, point) / elapsed * 3.6
        motion[frame] = (cumulative, speed)
    return motion
