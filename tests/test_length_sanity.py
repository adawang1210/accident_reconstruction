"""Tests for the known-vehicle-length homography scale diagnostic.

Pure-function only: they pin the box long-axis extraction, the local scale factor
(projected length / real length) under a controllable projection, and the box
loader -- no SAM2 / calibration file needed.
"""

from __future__ import annotations

import numpy as np

from accident_reconstruction.auto_reconstruct import (
    box_long_axis_endpoints,
    load_boxes,
    local_scale_factors,
)


def test_long_axis_picks_taller_dimension() -> None:
    """A taller-than-wide box spans its vertical (long) axis."""
    assert box_long_axis_endpoints((10, 0, 30, 60)) == ((20.0, 0.0), (20.0, 60.0))


def test_long_axis_picks_wider_dimension() -> None:
    """A wider-than-tall box spans its horizontal (long) axis."""
    assert box_long_axis_endpoints((0, 10, 60, 30)) == ((0.0, 20.0), (60.0, 20.0))


def test_local_scale_factor_detects_compression() -> None:
    """A projection that halves pixel distances yields a 0.5x scale factor.

    The box long axis is 40 px; a projection scaling pixels by 0.05 m/px maps it to
    2.0 m. Against a real length of 4.0 m that is a 0.5x (compressed) local scale --
    exactly the under-scaling that makes speeds read low.
    """

    def project(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * 0.05  # 0.05 m per pixel

    boxes = {5: (0.0, 0.0, 10.0, 40.0)}  # 40 px tall
    factors = local_scale_factors(boxes, project, real_length_m=4.0)
    assert factors == [(5, 0.5)]


def test_local_scale_factor_samples_front_mid_back() -> None:
    """Three sample positions map to the first, middle and last tracked frames."""

    def project(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)

    boxes = {f: (0.0, 0.0, 10.0, 100.0) for f in (0, 3, 6, 9, 12)}
    frames = [f for f, _ in local_scale_factors(boxes, project, real_length_m=100.0)]
    assert frames == [0, 6, 12]


def test_load_boxes_reads_columns(tmp_path) -> None:
    """Boxes load per vehicle per frame; a CSV without box columns yields empty."""
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "frame,vehicle,x1,y1,x2,y2,anchor_x,anchor_y,t_sec\n"
        "10,car,1,2,3,4,2,4,0.4\n"
        "11,car,5,6,7,8,6,8,0.44\n"
    )
    boxes = load_boxes(csv_path)
    assert boxes["car"][10] == (1.0, 2.0, 3.0, 4.0)
    assert boxes["car"][11] == (5.0, 6.0, 7.0, 8.0)

    no_box = tmp_path / "nobox.csv"
    no_box.write_text("frame,vehicle,anchor_x,anchor_y\n10,car,1,2\n")
    assert load_boxes(no_box) == {}
