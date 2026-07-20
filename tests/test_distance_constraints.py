"""Tests for distance-constraint calibration and its backward-compatible store.

Pure-function only (no video / OpenCV window): they pin that a known-distance point
pair stretches the homography scale, that the record echoes the constraints, and
that the gcps.json store still reads old bare-list files.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from accident_reconstruction.calibrate_homography import (
    build_calibration,
    load_distance_constraints,
    load_gcps,
    save_gcps,
)

# A non-degenerate pixel square mapped to a lat/lon square (~10 m per side).
_GCPS = [
    {"name": "g1", "pixel": [0, 0], "lat": 25.0000, "lon": 121.0000},
    {"name": "g2", "pixel": [100, 0], "lat": 25.0000, "lon": 121.0001},
    {"name": "g3", "pixel": [100, 100], "lat": 25.0001, "lon": 121.0001},
    {"name": "g4", "pixel": [0, 100], "lat": 25.0001, "lon": 121.0000},
]


def _project(record: dict, pixel: tuple[float, float]) -> np.ndarray:
    """Project a raw pixel to metres through a calibration record's homography."""
    homography = np.array(record["homography_px_to_m"], dtype=np.float64)
    point = np.array([[list(pixel)]], dtype=np.float64)
    return cv2.perspectiveTransform(point, homography).reshape(2)


def _projected_length(record: dict) -> float:
    a = _project(record, (0, 0))
    b = _project(record, (100, 0))
    return float(np.hypot(*(b - a)))


def test_distance_constraint_stretches_scale() -> None:
    """A constraint saying the pair is 3x farther apart pulls the projected scale up."""
    base = build_calibration(_GCPS)
    base_len = _projected_length(base)

    constrained = build_calibration(
        _GCPS,
        distance_constraints=[
            {"pixel_a": [0, 0], "pixel_b": [100, 0], "distance_m": base_len * 3.0}
        ],
    )
    new_len = _projected_length(constrained)
    # The homography moved decisively toward the (3x) constrained length.
    assert new_len > base_len * 1.3


def test_record_echoes_constraints_and_method() -> None:
    """The record carries the constraints and flags them in ``method``."""
    constraints = [{"pixel_a": [0, 0], "pixel_b": [100, 0], "distance_m": 8.0}]
    record = build_calibration(_GCPS, distance_constraints=constraints)
    assert record["distance_constraints"] == constraints
    assert "距離約束" in record["method"]


def test_no_constraints_is_unchanged() -> None:
    """Without constraints the record has an empty list and no scale change note."""
    record = build_calibration(_GCPS)
    assert record["distance_constraints"] == []
    assert "距離約束" not in record["method"]


def test_degenerate_constraint_is_skipped() -> None:
    """A zero-length / zero-distance constraint is ignored, not crashing."""
    record = build_calibration(
        _GCPS,
        distance_constraints=[
            {"pixel_a": [5, 5], "pixel_b": [5, 5], "distance_m": 4.0},  # same pixel
            {"pixel_a": [0, 0], "pixel_b": [100, 0], "distance_m": 0.0},  # zero dist
        ],
    )
    # No usable constraint -> homography matches the GCP-only fit.
    assert "距離約束" not in record["method"]


def test_store_roundtrip_and_backward_compat(tmp_path) -> None:
    """The store keeps the legacy list shape until constraints are added."""
    store = tmp_path / "gcps.json"

    # No constraints -> legacy bare list on disk.
    save_gcps(store, _GCPS)
    assert isinstance(json.loads(store.read_text()), list)
    assert load_gcps(store) == _GCPS
    assert load_distance_constraints(store) == []

    # With constraints -> object shape, both readable.
    constraints = [{"pixel_a": [0, 0], "pixel_b": [100, 0], "distance_m": 4.0}]
    save_gcps(store, _GCPS, constraints)
    on_disk = json.loads(store.read_text())
    assert isinstance(on_disk, dict)
    assert set(on_disk) == {"gcps", "distance_constraints"}
    assert load_gcps(store) == _GCPS
    assert load_distance_constraints(store) == constraints
