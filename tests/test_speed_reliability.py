"""Tests for the speed-reliability outputs (GCP span + trajectory hull coverage).

Pure-function only: they pin the convex-hull coverage ratio under controllable
pixel geometry and the human caption formatting -- no calibration file or SAM2.
"""

from __future__ import annotations

import json

import numpy as np

from accident_reconstruction.auto_reconstruct import (
    gcp_pixel_points,
    hull_coverage_ratio,
    speed_reliability_caption,
)

# A 10x10 GCP-anchored square; trajectories are compared against this hull.
GCP_SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)


def test_coverage_full_when_trajectory_inside() -> None:
    """A path wholly inside the GCP hull reports full (1.0) coverage."""
    traj = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32)
    assert hull_coverage_ratio(traj, GCP_SQUARE) == 1.0


def test_coverage_half_when_trajectory_straddles_edge() -> None:
    """A path with half its hull area outside the GCP hull reports ~0.5.

    The trajectory hull spans x in [0, 20] (area 200); only x in [0, 10]
    (area 100) overlaps the GCP square, so coverage is 100 / 200 = 0.5 -- the
    fraction of the path that is actually calibrated.
    """
    traj = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float32)
    assert hull_coverage_ratio(traj, GCP_SQUARE) == 0.5


def test_coverage_zero_when_trajectory_outside() -> None:
    """A path fully outside the GCP hull reports zero coverage (extrapolated)."""
    traj = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32)
    assert hull_coverage_ratio(traj, GCP_SQUARE) == 0.0


def test_coverage_none_for_degenerate_or_missing() -> None:
    """Too few points or a zero-area (collinear) hull yields None, not a crash."""
    assert hull_coverage_ratio(None, GCP_SQUARE) is None
    two_points = np.array([[1, 1], [2, 2]], dtype=np.float32)
    assert hull_coverage_ratio(two_points, GCP_SQUARE) is None
    collinear = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float32)
    assert hull_coverage_ratio(collinear, GCP_SQUARE) is None


def test_caption_reports_span_and_coverage() -> None:
    """The caption folds both the ground span and the hull coverage percentage."""
    caption = speed_reliability_caption(
        {"gcp_ground_span_m": 18.0, "hull_coverage": 0.42}
    )
    assert "18 m" in caption
    assert "42%" in caption


def test_caption_handles_uncalibrated_scene() -> None:
    """With neither span nor coverage the caption flags an uncalibrated scene."""
    caption = speed_reliability_caption({"gcp_ground_span_m": None})
    assert "未 GPS 校正" in caption


def test_gcp_pixel_points_reads_gcps_and_constraints(tmp_path) -> None:
    """Pixels come from ``gcps.json`` -- GCP pixels plus constraint endpoints.

    Guards the path bug where the GCP hull was read from the derived
    ``homography_calibration.json`` (no ``pixel`` field), which silently dropped
    the coverage diagnostic.
    """
    store = tmp_path / "gcps.json"
    store.write_text(
        json.dumps(
            {
                "gcps": [
                    {"pixel": [0, 0], "lat": 1.0, "lon": 1.0},
                    {"pixel": [10, 0], "lat": 1.0, "lon": 1.0},
                ],
                "distance_constraints": [
                    {"pixel_a": [10, 10], "pixel_b": [0, 10], "distance_m": 4.0}
                ],
            }
        )
    )
    points = gcp_pixel_points(store)
    assert points is not None
    # 2 GCP pixels + 2 constraint endpoints = 4 points.
    assert points.shape == (4, 2)


def test_gcp_pixel_points_none_when_store_missing(tmp_path) -> None:
    """A missing store yields None (uncalibrated), not a crash."""
    assert gcp_pixel_points(tmp_path / "absent.json") is None
