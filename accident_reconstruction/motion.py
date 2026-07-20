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
from collections.abc import Callable

# Speed = displacement across the samples within this window / elapsed time.
# Smooths single-frame anchor jitter without lagging genuine acceleration.
SPEED_WINDOW_SECONDS = 0.6

Point = tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    """Planar distance between two metric ``(east_m, north_m)`` points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def windowed_speed(
    track: dict[int, Point],
    fps: float,
    distance: Callable[[Point, Point], float],
) -> dict[int, tuple[float, float]]:
    """Per-frame ``(cumulative_m, speed_kmh)`` over ``SPEED_WINDOW_SECONDS``.

    Args:
        track: ``{frame: point}`` for one vehicle, in whatever coordinate the
            ``distance`` function expects (metric plane or lat/lon).
        fps: Source frames per second; the window spans ``seconds * fps`` frames.
        distance: Metric between two points -- :func:`euclidean` for the metric
            plane, a haversine for lat/lon.

    Returns:
        ``{frame: (cumulative_m, speed_kmh)}``.
    """
    motion: dict[int, tuple[float, float]] = {}
    window: deque[tuple[int, Point]] = deque()
    cumulative = 0.0
    previous: Point | None = None
    for frame in sorted(track):
        point = track[frame]
        if previous is not None:
            cumulative += distance(previous, point)
        previous = point
        window.append((frame, point))
        while len(window) >= 2 and frame - window[0][0] > SPEED_WINDOW_SECONDS * fps:
            window.popleft()
        speed = 0.0
        if len(window) >= 2:
            first_frame, first_point = window[0]
            elapsed = (frame - first_frame) / fps
            if elapsed > 0:
                speed = distance(first_point, point) / elapsed * 3.6
        motion[frame] = (cumulative, speed)
    return motion
