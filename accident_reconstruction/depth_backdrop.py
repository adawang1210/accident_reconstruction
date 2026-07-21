"""Single-frame CCTV -> monocular metric depth -> 3D backdrop ``.splat``.

All six scenes are FIXED cameras (measured global shift 0.01-0.96 px), so
multi-view reconstruction from the accident footage itself is physically
impossible. What IS possible without going on site: lift one frame to a colored
point cloud with a monocular METRIC depth model, and write it as a Gaussian
``.splat`` file (antimatter15 layout) that the existing frontend viewer loads
directly via ``VITE_SPLAT_URL`` -- a schematic 3D backdrop, valid for small view
changes (roughly +/-15 deg; occlusion holes appear beyond that).

This is NOT measurement-grade geometry: speeds/positions stay with the 2D
homography pipeline. See ``frontend/SPLAT_NOTES.md`` (10.1, 11.3).

Example:
    ```bash
    ACCIDENT_SCENE=車禍影片_BMW神之鬼切_2026_05_16_臺北市大安區基隆路四段 \
      .venv/bin/python -m accident_reconstruction.depth_backdrop --stride 3
    ```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from accident_reconstruction.scene_config import SCENE

#: HuggingFace id of the metric (absolute-metres) outdoor depth model.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"

#: Depth range kept in the backdrop (metres): below ~2 m is the camera housing /
#: mounting pole, beyond ~60 m the model's guesses (and the sky) dominate.
DEPTH_RANGE_M = (2.0, 60.0)


def focal_from_known_width(
    width_px: float, depth_m: float, real_width_m: float
) -> float:
    """Focal length (px) from an object of known real width at known depth.

    The pinhole relation ``width_px = focal * real_width / depth`` solved for
    ``focal``. Handy when a tracked vehicle's box width and its depth are known
    (a car is ~1.8 m wide).

    Args:
        width_px: The object's apparent width in pixels.
        depth_m: Its distance from the camera in metres.
        real_width_m: Its real-world width in metres (> 0).

    Returns:
        The focal length in pixels.

    Examples:
        ```python
        focal_from_known_width(381.0, 9.2, 1.8)
        # 1947.3333333333333
        ```
    """
    if real_width_m <= 0:
        raise ValueError("real_width_m must be > 0")
    return width_px * depth_m / real_width_m


def unproject_depth(
    depth_m: np.ndarray,
    image_bgr: np.ndarray,
    focal_px: float,
    stride: int = 3,
    z_range: tuple[float, float] = DEPTH_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a depth map + image to camera-frame 3D points with RGB colors.

    Pinhole back-projection about the image centre: ``x = (u - cx) * z / f``,
    ``y = (v - cy) * z / f`` -- camera frame is x right, y DOWN, z forward
    (convert with :func:`camera_to_viewer` before writing a splat).

    Args:
        depth_m: ``(H, W)`` metric depth.
        image_bgr: ``(H, W, 3)`` BGR frame (OpenCV order).
        focal_px: Focal length in pixels.
        stride: Sample every ``stride``-th pixel in both axes.
        z_range: Keep points with ``z_min <= z <= z_max`` (metres).

    Returns:
        ``(points, colors)`` -- ``(N, 3)`` float32 metres and ``(N, 3)`` uint8
        RGB.

    Examples:
        ```python
        depth = np.full((4, 4), 10.0, dtype=np.float32)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        points, colors = unproject_depth(depth, image, focal_px=100.0, stride=2)
        points.shape, float(points[0, 2])
        # ((4, 3), 10.0)
        ```
    """
    height, width = depth_m.shape
    cx, cy = width / 2.0, height / 2.0
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[vs, us].astype(np.float32).ravel()
    keep = (z >= z_range[0]) & (z <= z_range[1])
    u = us.ravel()[keep].astype(np.float32)
    v = vs.ravel()[keep].astype(np.float32)
    z = z[keep]
    points = np.stack([(u - cx) * z / focal_px, (v - cy) * z / focal_px, z], axis=1)
    colors = image_bgr[vs.ravel()[keep], us.ravel()[keep]][:, ::-1]  # BGR -> RGB
    return points.astype(np.float32), colors.astype(np.uint8)


def camera_to_viewer(points: np.ndarray) -> np.ndarray:
    """Map camera-frame points (x right, y down, z forward) to viewer axes.

    The Three.js viewer is right-handed with y UP and the camera looking down
    -z, so the scene must sit at negative z: flip y and z.

    Args:
        points: ``(N, 3)`` camera-frame points.

    Returns:
        ``(N, 3)`` viewer-frame points (same dtype).

    Examples:
        ```python
        camera_to_viewer(np.array([[1.0, 2.0, 3.0]])).tolist()
        # [[1.0, -2.0, -3.0]]
        ```
    """
    out = points.copy()
    out[:, 1] *= -1
    out[:, 2] *= -1
    return out


def splat_scales(z_m: np.ndarray, focal_px: float, stride: int) -> np.ndarray:
    """Isotropic per-splat radius covering one sampled pixel's 3D footprint.

    A pixel at depth ``z`` spans ``z / focal`` metres; multiplying by ``stride``
    makes neighbouring splats just touch, hiding the sampling grid without
    blurring near geometry (far points get proportionally larger).

    Args:
        z_m: ``(N,)`` depths in metres.
        focal_px: Focal length in pixels.
        stride: The sampling stride used in :func:`unproject_depth`.

    Returns:
        ``(N,)`` float32 radii in metres.
    """
    return (np.asarray(z_m, dtype=np.float32) * stride / focal_px).astype(np.float32)


def write_splat(
    path: Path,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    scales_m: np.ndarray,
) -> Path:
    """Write isotropic Gaussians in the antimatter15 ``.splat`` binary layout.

    32 bytes per splat: position ``3xf32``, scale ``3xf32``, color ``RGBA u8``
    (alpha 255), rotation ``4xu8`` (identity quaternion -- irrelevant for
    isotropic splats). This is the format the frontend's mkkellogg viewer loads
    natively.

    Args:
        path: Output ``.splat`` path (parent dirs created).
        points: ``(N, 3)`` positions, viewer frame.
        colors_rgb: ``(N, 3)`` uint8 RGB.
        scales_m: ``(N,)`` isotropic radii in metres.

    Returns:
        The path written.
    """
    n = len(points)
    record = np.zeros(
        n,
        dtype=[
            ("pos", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rot", "u1", 4),
        ],
    )
    record["pos"] = np.asarray(points, dtype=np.float32)
    record["scale"] = np.repeat(
        np.asarray(scales_m, dtype=np.float32)[:, None], 3, axis=1
    )
    record["rgba"][:, :3] = colors_rgb
    record["rgba"][:, 3] = 255
    record["rot"] = (255, 128, 128, 128)  # identity quaternion, byte-encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(record.tobytes())
    return path


#: Zeroth-order spherical-harmonic normalisation (``1 / (2*sqrt(pi))``): the INRIA
#: PLY stores DC color as an SH coefficient, decoded as ``0.5 + SH_C0 * f_dc``.
SH_C0 = 0.28209479177387814


def write_ply(
    path: Path,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    scales_m: np.ndarray,
) -> Path:
    """Write isotropic Gaussians as an INRIA-format binary ``.ply``.

    The standard 3DGS PLY the mkkellogg viewer's well-tested PLY loader expects:
    ``x y z``, ``f_dc_0..2`` (SH DC color), ``opacity`` (logit), ``scale_0..2``
    (log-metres), ``rot_0..3`` (quaternion, w first). Prefer this over
    :func:`write_splat` -- the hand-written ``.splat`` path renders invisibly in
    the current viewer (see ``frontend/SPLAT_NOTES.md`` §3), while the PLY path
    does not.

    Args:
        path: Output ``.ply`` path (parent dirs created).
        points: ``(N, 3)`` positions, viewer frame.
        colors_rgb: ``(N, 3)`` uint8 RGB.
        scales_m: ``(N,)`` isotropic radii in metres (> 0).

    Returns:
        The path written.
    """
    n = len(points)
    rgb = np.clip(np.asarray(colors_rgb, dtype=np.float32) / 255.0, 1e-4, 1 - 1e-4)
    f_dc = (rgb - 0.5) / SH_C0  # invert the SH DC decode
    log_scale = np.log(np.maximum(np.asarray(scales_m, dtype=np.float32), 1e-6))
    fields = [
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
    ]
    record = np.zeros(n, dtype=[(name, "<f4") for name in fields])
    record["x"], record["y"], record["z"] = np.asarray(points, dtype=np.float32).T
    record["f_dc_0"], record["f_dc_1"], record["f_dc_2"] = f_dc.T
    record["opacity"] = 6.0  # sigmoid(6) ~= 0.9975, effectively opaque
    for axis in ("scale_0", "scale_1", "scale_2"):
        record[axis] = log_scale
    record["rot_0"] = 1.0  # identity quaternion (w, x, y, z)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in fields)
        + "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(record.tobytes())
    return path


def estimate_depth(image_bgr: np.ndarray) -> np.ndarray:
    """Run the metric depth model on a BGR frame (downloads weights on first use).

    Args:
        image_bgr: ``(H, W, 3)`` BGR frame.

    Returns:
        ``(H, W)`` float32 metric depth, resized to the frame size.
    """
    import torch  # deferred: heavy, only needed for the CLI path
    from PIL import Image
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=DEPTH_MODEL_ID, device=device)
    rgb = Image.fromarray(image_bgr[:, :, ::-1])
    predicted = np.asarray(pipe(rgb)["predicted_depth"], dtype=np.float32)
    if predicted.shape != image_bgr.shape[:2]:
        predicted = cv2.resize(predicted, (image_bgr.shape[1], image_bgr.shape[0]))
    return predicted


def main() -> None:
    """Build the active scene's 3D backdrop splat from one source-video frame."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", type=int, default=0, help="source frame index")
    parser.add_argument("--stride", type=int, default=3, help="pixel sampling stride")
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="focal length in px (default: frame width, a typical CCTV FOV; "
        "derive a better one with focal_from_known_width)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output .splat path")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(SCENE.source_video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} of {SCENE.source_video}")

    focal = args.focal if args.focal is not None else float(frame.shape[1])
    depth = estimate_depth(frame)
    points, colors = unproject_depth(depth, frame, focal, stride=args.stride)
    scales = splat_scales(points[:, 2], focal, args.stride)
    out = args.out or SCENE.out_csv.with_name(f"{SCENE.name}_backdrop.splat")
    write_splat(out, camera_to_viewer(points), colors, scales)
    size_mb = out.stat().st_size / 1e6
    print(
        f"Backdrop splat: {out.resolve()}  ({len(points):,} splats, {size_mb:.1f} MB)"
    )
    print("View it: copy into frontend/public/ and set VITE_SPLAT_URL=/<name>.splat")
    print("(schematic backdrop only -- speeds/positions stay with the 2D pipeline)")


if __name__ == "__main__":
    main()
