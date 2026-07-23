"""Round-trip tests for ViewTransformer's pixel<->metric inverse.

The overlay video back-projects the metric trajectory to pixels via
``inverse_transform_points``; it must invert ``transform_points`` closely, with
and without a lens-distortion model.
"""

from __future__ import annotations

import numpy as np

from accident_reconstruction.calibrate_homography import (
    ViewTransformer,
    _redistort_from_normalized,
    undistort_to_normalized,
)

# A simple non-degenerate homography: a 20 m square seen in perspective.
_SOURCE = np.array([[100, 600], [1180, 600], [1000, 200], [280, 200]], dtype=np.float32)
_TARGET = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)


def test_inverse_round_trips_without_distortion() -> None:
    """Pixels -> metric -> pixels returns the original points."""
    vt = ViewTransformer(source=_SOURCE, target=_TARGET)
    pixels = np.array([[640, 400], [300, 550], [900, 300]], dtype=np.float32)

    metric = vt.transform_points(pixels)
    back = vt.inverse_transform_points(metric)

    assert np.allclose(back, pixels, atol=1e-2)


def test_inverse_round_trips_with_distortion() -> None:
    """The radial re-distortion inverts the undistortion for a mild k1."""
    distortion = {"k1": -0.18, "cx": 640, "cy": 360, "f": 720}
    vt = ViewTransformer(source=_SOURCE, target=_TARGET, distortion=distortion)
    pixels = np.array([[500, 500], [820, 300], [700, 450]], dtype=np.float32)

    metric = vt.transform_points(pixels)
    back = vt.inverse_transform_points(metric)

    assert np.allclose(back, pixels, atol=0.5)


def test_redistort_inverts_undistort() -> None:
    """`_redistort_from_normalized` is the inverse of `undistort_to_normalized`."""
    distortion = {"k1": -0.25, "cx": 640, "cy": 360, "f": 700}
    pixels = np.array([[820.0, 300.0], [400.0, 520.0]])

    normalized = undistort_to_normalized(pixels, distortion)
    recovered = _redistort_from_normalized(normalized, distortion)

    assert np.allclose(recovered, pixels, atol=1e-2)


def test_inverse_of_empty_is_empty() -> None:
    vt = ViewTransformer(source=_SOURCE, target=_TARGET)
    assert vt.inverse_transform_points(np.empty((0, 2), dtype=np.float32)).size == 0
