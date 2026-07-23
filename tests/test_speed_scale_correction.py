"""Direction-aware (Path A) speed-scale correction in ``auto_reconstruct``.

The correction turns a known real length into a speed multiplier ONLY when the
box long axis is measured along the travel direction (side/oblique view). On a
lateral (rear/head-on) view it must abstain rather than emit a wrong number.
"""

from __future__ import annotations

import numpy as np

from accident_reconstruction.auto_reconstruct import (
    longitudinal_scale_correction,
    speed_correction_caption,
)


def _identity(points: np.ndarray) -> np.ndarray:
    """Project pixels to metres 1:1, so scale reasoning is transparent."""
    return np.asarray(points, dtype=np.float64)


def _track(width: float, height: float, steps: int = 7):
    """Build boxes + anchors for a vehicle sliding along +x by 10 units/frame.

    Each box is centred on that frame's ground anchor with the given pixel size.
    """
    boxes: dict[int, tuple[float, float, float, float]] = {}
    anchors: dict[int, tuple[float, float]] = {}
    half_w, half_h = width / 2, height / 2
    for frame in range(steps):
        cx, cy = float(frame * 10), 100.0
        boxes[frame] = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        anchors[frame] = (cx, cy)
    return boxes, anchors


def test_travel_aligned_long_axis_yields_speed_multiplier():
    # Wider-than-tall box => long axis is horizontal == the +x travel direction.
    # Projected length 2.0 vs real length 4.0 => scale 0.5 => speed x2.0.
    boxes, anchors = _track(width=2.0, height=1.0)
    factor, reason = longitudinal_scale_correction(
        boxes, anchors, _identity, real_length_m=4.0
    )
    assert factor is not None
    assert abs(factor - 2.0) < 1e-6
    assert "縱向" in reason


def test_lateral_long_axis_abstains():
    # Taller-than-wide box => long axis is vertical, perpendicular to +x travel.
    # That axis measures WIDTH, not length, so the correction must abstain.
    boxes, anchors = _track(width=1.0, height=2.0)
    factor, reason = longitudinal_scale_correction(
        boxes, anchors, _identity, real_length_m=4.0
    )
    assert factor is None
    assert "橫向" in reason


def test_inconsistent_scale_abstains():
    # Long axis stays travel-aligned, but its projected length grows every frame
    # (depth-driven foreshortening), so the scale is not uniform => abstain.
    boxes: dict[int, tuple[float, float, float, float]] = {}
    anchors: dict[int, tuple[float, float]] = {}
    for frame in range(7):
        cx, cy = float(frame * 10), 100.0
        half = 1.0 + frame  # widening long axis
        boxes[frame] = (cx - half, cy - 0.5, cx + half, cy + 0.5)
        anchors[frame] = (cx, cy)
    factor, reason = longitudinal_scale_correction(
        boxes, anchors, _identity, real_length_m=4.0
    )
    assert factor is None
    assert "不一致" in reason


def test_caption_lists_applied_factor_only():
    corrections = {"car": (2.0, "aligned"), "motorcycle": (1.0, "abstained")}
    caption = speed_correction_caption(corrections)
    assert "car" in caption
    assert "2.00" in caption
    assert "motorcycle" not in caption


def test_caption_empty_when_nothing_applied():
    assert speed_correction_caption({"car": (1.0, "abstained")}) == ""
