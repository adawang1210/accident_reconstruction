"""Tests for the single-frame depth backdrop (unprojection + .splat writer).

Pure-function only: geometry, coordinate conventions and the binary splat
layout are pinned without running the depth model.
"""

from __future__ import annotations

import numpy as np
import pytest

from accident_reconstruction.depth_backdrop import (
    SH_C0,
    camera_to_viewer,
    focal_from_known_width,
    median_plate,
    splat_scales,
    unproject_depth,
    write_ply,
    write_splat,
)


def test_focal_from_known_width_matches_pinhole() -> None:
    """A 1.8 m car spanning 381 px at 9.2 m implies f = 381*9.2/1.8 px."""
    assert focal_from_known_width(381.0, 9.2, 1.8) == pytest.approx(1947.333, abs=1e-3)
    with pytest.raises(ValueError, match="real_width_m"):
        focal_from_known_width(100.0, 5.0, 0.0)


def test_unproject_centre_pixel_lands_on_axis() -> None:
    """The centre pixel unprojects to (0, 0, z); offsets scale with z/f."""
    depth = np.full((10, 10), 20.0, dtype=np.float32)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    points, _ = unproject_depth(depth, image, focal_px=100.0, stride=5)
    centre = points[np.argmin(np.abs(points[:, 0]) + np.abs(points[:, 1]))]
    assert centre[2] == 20.0
    # pixel (0,0) sits 5 px left/up of centre -> -5 * 20 / 100 = -1 m on each axis
    corner = points[0]
    assert corner[0] == pytest.approx(-1.0)
    assert corner[1] == pytest.approx(-1.0)


def test_unproject_filters_depth_range_and_maps_color() -> None:
    """Out-of-range depths are dropped; colors come back RGB, not BGR."""
    depth = np.array([[1.0, 10.0], [10.0, 99.0]], dtype=np.float32)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 1] = (255, 0, 0)  # BGR blue at the only kept pixel of row 0
    points, colors = unproject_depth(depth, image, focal_px=10.0, stride=1)
    assert len(points) == 2  # 1.0 and 99.0 fall outside DEPTH_RANGE_M
    assert colors[0].tolist() == [0, 0, 255]  # BGR (255,0,0) -> RGB (0,0,255)


def test_camera_to_viewer_flips_y_and_z() -> None:
    """Camera (x right, y down, z forward) -> viewer (y up, scene at -z)."""
    out = camera_to_viewer(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert out.tolist() == [[1.0, -2.0, -3.0]]


def test_splat_scales_cover_pixel_footprint() -> None:
    """Radius = z * stride / focal, so far points get proportionally larger."""
    scales = splat_scales(np.array([10.0, 40.0]), focal_px=1000.0, stride=3)
    assert scales.tolist() == pytest.approx([0.03, 0.12])


def test_write_splat_binary_layout(tmp_path) -> None:
    """32 bytes per splat: pos f32x3, scale f32x3, RGBA u8, identity rotation."""
    points = np.array([[1.0, -2.0, -10.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30]], dtype=np.uint8)
    path = write_splat(tmp_path / "b.splat", points, colors, np.array([0.05]))
    raw = path.read_bytes()
    assert len(raw) == 32
    record = np.frombuffer(
        raw,
        dtype=[
            ("pos", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rot", "u1", 4),
        ],
    )[0]
    assert record["pos"].tolist() == [1.0, -2.0, -10.0]
    assert record["scale"].tolist() == pytest.approx([0.05, 0.05, 0.05])
    assert record["rgba"].tolist() == [10, 20, 30, 255]  # opaque
    assert record["rot"].tolist() == [255, 128, 128, 128]  # identity quaternion


def test_write_ply_header_and_color_roundtrip(tmp_path) -> None:
    """Binary PLY has the 3DGS fields and encodes color so SH-DC decode inverts.

    The viewer decodes color as ``0.5 + SH_C0 * f_dc``; a mid-grey (128) point
    must round-trip back through that formula.
    """
    points = np.array([[1.0, -2.0, -10.0]], dtype=np.float32)
    colors = np.array([[128, 128, 128]], dtype=np.uint8)
    path = write_ply(tmp_path / "b.ply", points, colors, np.array([0.05]))
    raw = path.read_bytes()
    header, _, body = raw.partition(b"end_header\n")
    text = header.decode("ascii")
    assert "format binary_little_endian 1.0" in text
    assert "element vertex 1" in text
    for field in ("f_dc_0", "opacity", "scale_0", "rot_0"):
        assert f"property float {field}" in text
    record = np.frombuffer(
        body,
        dtype=[
            (n, "<f4")
            for n in (
                "x",
                "y",
                "z",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            )
        ],
    )[0]
    assert [record["x"], record["y"], record["z"]] == [1.0, -2.0, -10.0]
    decoded = 0.5 + SH_C0 * record["f_dc_0"]
    assert decoded == pytest.approx(128 / 255, abs=1e-4)
    assert record["scale_0"] == pytest.approx(np.log(0.05), abs=1e-5)
    assert record["rot_0"] == 1.0  # identity quaternion, w first


def test_median_plate_removes_moving_object() -> None:
    """A bright block that moves each frame vanishes under the temporal median."""
    frames = np.full((3, 4, 6, 3), 100, np.uint8)
    frames[0, :, 0] = 255  # block at col 0 in frame 0
    frames[1, :, 2] = 255  # col 2 in frame 1
    frames[2, :, 4] = 255  # col 4 in frame 2
    plate = median_plate(frames)
    assert plate.shape == (4, 6, 3)
    assert np.all(plate == 100)  # every pixel is background in >= 2/3 frames


def test_median_plate_inpaints_residual_box() -> None:
    """A box that survives the median (e.g. a parked car) is painted over."""
    frames = np.full((3, 20, 20, 3), 120, np.uint8)
    frames[:, 4:8, 4:8] = 0  # a black patch present in ALL frames (parked)
    plate = median_plate(frames, inpaint_boxes=[(4, 4, 8, 8)])
    # the patch is gone -- filled from the surrounding grey, not left black
    assert plate[5, 5].tolist() != [0, 0, 0]
    assert plate[5, 5, 0] > 100
