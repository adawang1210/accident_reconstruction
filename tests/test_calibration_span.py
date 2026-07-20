"""Unit tests for build_calibration's GCP-span warning (TECH_REVIEW §3.4).

A small real-world GCP spread makes a low reprojection residual misleading (the
homography is fit from a tiny patch and extrapolates); build_calibration flags
that via ``span_warning``. These build synthetic square GCP layouts of a known
ground size to exercise both sides of the MIN_GCP_SPAN_M threshold.
"""

from __future__ import annotations

import math

import pytest

from accident_reconstruction.calibrate_homography import (
    MIN_GCP_SPAN_M,
    build_calibration,
)


def _square_gcps(side_m: float, lat0: float = 25.0, lon0: float = 121.5) -> list[dict]:
    """Four GCPs forming a ``side_m`` by ``side_m`` ground square (100x100 px)."""
    dlat = side_m / 111195.0
    dlon = side_m / (111195.0 * math.cos(math.radians(lat0)))
    corners = [
        (lat0, lon0),
        (lat0, lon0 + dlon),
        (lat0 + dlat, lon0 + dlon),
        (lat0 + dlat, lon0),
    ]
    pixels = [[0, 0], [100, 0], [100, 100], [0, 100]]
    return [
        {"name": f"g{i}", "lat": la, "lon": lo, "pixel": px}
        for i, ((la, lo), px) in enumerate(zip(corners, pixels))
    ]


def test_span_warning_fires_for_small_patch() -> None:
    # 6 m square -> diagonal ~8.5 m, below the 15 m threshold.
    cal = build_calibration(_square_gcps(6.0))
    assert cal["target_span_m"] < MIN_GCP_SPAN_M
    assert cal["span_warning"] is not None


def test_no_span_warning_for_spread_points() -> None:
    # 25 m square -> diagonal ~35 m, above the threshold.
    cal = build_calibration(_square_gcps(25.0))
    assert cal["target_span_m"] >= MIN_GCP_SPAN_M
    assert cal["span_warning"] is None


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        build_calibration(_square_gcps(20.0)[:3])
