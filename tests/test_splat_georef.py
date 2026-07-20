"""Tests for the splat↔scene similarity-transform solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from accident_reconstruction import splat_georef as sg


def _apply(scale: float, rot: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply the viewer's TRS convention: world = scale * R @ p + t."""
    return scale * pts @ rot.T + t


def test_recovers_known_transform() -> None:
    """A clean similarity transform is recovered to numerical precision."""
    rng = np.random.default_rng(42)
    src = rng.normal(scale=5.0, size=(8, 3))
    scale = 2.5
    rot = sg.threejs_euler_xyz_to_rotation_matrix(12.0, -47.0, 8.0)
    t = np.array([10.0, 1.5, -7.0])
    dst = _apply(scale, rot, t, src)

    result = sg.solve_splat_transform(src, dst)

    assert result.scale == pytest.approx(scale, rel=1e-9)
    assert result.pos == pytest.approx(tuple(t), abs=1e-7)
    assert result.rmse == pytest.approx(0.0, abs=1e-7)
    # The reported Euler knobs must rebuild the original rotation.
    rebuilt = sg.threejs_euler_xyz_to_rotation_matrix(*result.rot_deg)
    assert np.allclose(rebuilt, rot, atol=1e-7)


def test_euler_roundtrip_matches_threejs() -> None:
    """Matrix → Euler 'XYZ' → matrix is the identity for assorted angles."""
    for angles in [(0, 90, 0), (30, 0, 0), (0, 0, 45), (15, -60, 80), (-90, 10, 5)]:
        rot = sg.threejs_euler_xyz_to_rotation_matrix(*angles)
        decoded = sg.rotation_matrix_to_threejs_euler_xyz(rot)
        rebuilt = sg.threejs_euler_xyz_to_rotation_matrix(*decoded)
        assert np.allclose(rebuilt, rot, atol=1e-9)


def test_rotation_has_no_reflection() -> None:
    """The solved rotation is proper (det +1) even for mirror-prone point sets."""
    rng = np.random.default_rng(1)
    src = rng.normal(size=(6, 3))
    rot = sg.threejs_euler_xyz_to_rotation_matrix(0.0, 0.0, 90.0)
    dst = _apply(1.0, rot, np.zeros(3), src)
    _, solved, _ = sg.umeyama_similarity(src, dst)
    assert np.linalg.det(solved) == pytest.approx(1.0, abs=1e-9)


def test_noise_gives_small_residual_not_blowup() -> None:
    """Sub-metre measurement noise yields a sub-metre RMSE, not a degenerate fit."""
    rng = np.random.default_rng(7)
    src = rng.normal(scale=20.0, size=(12, 3))
    scale, rot, t = (
        1.3,
        sg.threejs_euler_xyz_to_rotation_matrix(5, 20, -3),
        np.array([3.0, 0.0, 4.0]),
    )
    dst = _apply(scale, rot, t, src) + rng.normal(scale=0.1, size=src.shape)
    result = sg.solve_splat_transform(src, dst)
    assert 0.0 < result.rmse < 0.5
    assert result.scale == pytest.approx(scale, rel=0.05)


def test_rejects_too_few_points() -> None:
    """Fewer than 3 correspondences cannot pin a 3D similarity."""
    with pytest.raises(ValueError, match="at least 3"):
        sg.umeyama_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


def test_rejects_mismatched_shapes() -> None:
    """Shape mismatches are caught before the maths runs."""
    with pytest.raises(ValueError, match="n, 3"):
        sg.umeyama_similarity(np.zeros((4, 3)), np.zeros((3, 3)))


def test_to_env_renders_all_knobs() -> None:
    """to_env() emits every VITE_SPLAT_* key the viewer reads."""
    env = sg.SplatTransform(
        scale=2.0, rot_deg=(0.0, 90.0, 0.0), pos=(1.0, 2.0, 3.0), rmse=0.0
    ).to_env()
    for key in (
        "VITE_SPLAT_SCALE",
        "VITE_SPLAT_ROT_X_DEG",
        "VITE_SPLAT_ROT_Y_DEG",
        "VITE_SPLAT_ROT_Z_DEG",
        "VITE_SPLAT_X",
        "VITE_SPLAT_Y",
        "VITE_SPLAT_Z",
    ):
        assert key in env


def test_load_correspondences_roundtrip(tmp_path: Path) -> None:
    """A pairs JSON file is parsed into matching (n, 3) arrays."""
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"splat": [0, 0, 0], "scene": [1, 2, 3]},
                    {"splat": [1, 0, 0], "scene": [2, 2, 3]},
                    {"splat": [0, 1, 0], "scene": [1, 3, 3]},
                ]
            }
        ),
        encoding="utf-8",
    )
    source, target = sg.load_correspondences(path)
    assert source.shape == (3, 3)
    assert target.shape == (3, 3)
    assert target[0].tolist() == [1.0, 2.0, 3.0]


def test_load_correspondences_rejects_empty(tmp_path: Path) -> None:
    """A file with no pairs is a clear error, not a silent empty solve."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"pairs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="pairs"):
        sg.load_correspondences(path)
