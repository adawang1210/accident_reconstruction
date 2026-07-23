"""Interactive ground-control-point (GCP) calibration for the active scene.

The bird's-eye projection needs a homography mapping camera pixels onto a real,
metric ground plane. We build it from *ground control points*: features visible
BOTH in the video (click the pixel) AND on a satellite map (read the lat/lon).
Good GCPs are flat on the road with sharp edges -- zebra-crossing corners, lane
mark ends, manhole covers -- spread across where the vehicles travel.

Workflow (no source editing -- everything is interactive and per-scene):
    1. Pick the scene with the ``ACCIDENT_SCENE`` env var (see ``scene_config``).
    2. Run this script. The frame opens; for each new point: LEFT-CLICK its pixel,
       then type its ``lat, lon`` (read from Google Maps satellite) in the
       terminal. Already-saved points are reused (green) and not re-clicked.
       Keys: ``u`` undo last new point, ``s``/Enter save, ``q`` quit without saving.
    3. Points persist in the scene's ``gcps.json``; the homography (>4 points ->
       robust MAGSAC++) is written to the scene's ``homography_calibration.json``.

Examples:
    ```python
    import numpy as np
    from accident_reconstruction.calibrate_homography import latlon_to_local_meters
    origin = (23.026861, 120.249608)
    latlon_to_local_meters(np.array([origin]), origin).round(2).tolist()
    # [[0.0, 0.0]]
    ```
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accident_reconstruction.scene_config import SCENE, SceneConfig

# Static ground features are visible on any frame; frame 0 is usually clearest.
CALIBRATION_FRAME_INDEX = 0

# MAGSAC++ inlier threshold, in target METERS. A control point whose reprojection
# is off by more than this is treated as a mismatch and down-weighted (robust
# camera-to-satellite recipe). Used only when there are more than 4 points.
RANSAC_THRESHOLD_M = 2.0

# Minimum real-world spread of the control points, in METERS (diagonal of their
# ground bounding box). Below this the homography is fit from a tiny patch and
# extrapolates/compresses badly across the frame -- a low reprojection residual is
# then misleading. Points must span the actual scene (near AND far, left AND
# right), not cluster near one crosswalk. See the keelung clip: 11 GCPs spanned the
# whole frame in pixels but only ~6 m of ground, compressing a ~30 m crossing to
# ~10 m and collapsing impact detection.
MIN_GCP_SPAN_M = 15.0

# Mean Earth radius (meters); good enough for the ~10 m span of these scenes.
EARTH_RADIUS_M = 6_371_000.0


def latlon_to_local_meters(
    latlon: np.ndarray, origin: tuple[float, float]
) -> np.ndarray:
    """Project lat/lon degrees to a local east/north meter plane (equirectangular).

    Accurate to millimeters over the small (~10 m) extent of these scenes. East
    (``+x``) increases with longitude; north (``+y``) increases with latitude.

    Args:
        latlon: Array of shape ``(n, 2)`` holding ``(lat, lon)`` in degrees.
        origin: ``(lat, lon)`` degrees mapped to ``(0, 0)`` meters.

    Returns:
        Array of shape ``(n, 2)`` holding ``(east_m, north_m)``.

    Examples:
        ```python
        import numpy as np
        pts = np.array([[23.026861, 120.249608], [23.026861, 120.249618]])
        latlon_to_local_meters(pts, (23.026861, 120.249608)).round(2).tolist()
        # [[0.0, 0.0], [1.02, 0.0]]
        ```
    """
    origin_lat, origin_lon = origin
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    east = (lon - origin_lon_rad) * math.cos(origin_lat_rad) * EARTH_RADIUS_M
    north = (lat - origin_lat_rad) * EARTH_RADIUS_M
    return np.column_stack([east, north]).astype(np.float32)


def load_gcps(store: Path) -> list[dict]:
    """Load saved ground control points (``[{name, lat, lon, pixel:[x,y]}]``).

    Reads both the legacy bare-list format and the extended
    ``{"gcps": [...], "distance_constraints": [...]}`` object (see
    :func:`save_gcps`), returning just the GCP list in either case.
    """
    if not store.exists():
        return []
    data = json.loads(store.read_text())
    if isinstance(data, dict):
        return data.get("gcps", [])
    return data


def load_distance_constraints(store: Path) -> list[dict]:
    """Load saved distance constraints, or ``[]`` for a legacy/absent store.

    A distance constraint is ``{"pixel_a": [x, y], "pixel_b": [x, y],
    "distance_m": float}`` -- two image points a KNOWN real distance apart (e.g. the
    ends of a lane dash, 4 m by Taiwan spec). Stored alongside the GCPs so scale can
    be pinned even where the GCPs cover only a small patch.
    """
    if not store.exists():
        return []
    data = json.loads(store.read_text())
    if isinstance(data, dict):
        return data.get("distance_constraints", [])
    return []


def save_gcps(
    store: Path, gcps: list[dict], distance_constraints: list[dict] | None = None
) -> None:
    """Write ground control points (and optional distance constraints) to the store.

    Backward compatible: with no ``distance_constraints`` the file stays a bare GCP
    list (the legacy shape old readers expect). When constraints are supplied it is
    written as ``{"gcps": [...], "distance_constraints": [...]}``, which
    :func:`load_gcps` / :func:`load_distance_constraints` both read.
    """
    store.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict] | dict[str, list[dict]]
    if distance_constraints:
        payload = {"gcps": gcps, "distance_constraints": distance_constraints}
    else:
        payload = gcps
    store.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_calibration_frame(scene: SceneConfig) -> np.ndarray:
    """Read the frame used for clicking GCPs from the scene's source video.

    Args:
        scene: The active scene.

    Returns:
        The decoded BGR frame.

    Raises:
        FileNotFoundError: If the source video is missing.
        RuntimeError: If the frame cannot be decoded.
    """
    if not scene.source_video.exists():
        raise FileNotFoundError(f"Missing video: {scene.source_video}")
    capture = cv2.VideoCapture(str(scene.source_video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME_INDEX)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not read the calibration frame.")
    return frame


def _draw_overlay(
    frame: np.ndarray, existing: list[dict], new: list[dict]
) -> np.ndarray:
    """Render the banner, reused points (green) and newly added points (red)."""
    canvas = frame.copy()
    prompt = (
        f"Saved {len(existing)} | new {len(new)}.  "
        "LEFT-CLICK a point then type lat,lon in terminal.  u=undo s=save q=quit"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    for point in existing:
        x, y = point["pixel"]
        cv2.circle(canvas, (int(x), int(y)), 6, (0, 200, 0), -1)
        cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), 1)
    for index, point in enumerate(new):
        x, y = point["pixel"]
        cv2.circle(canvas, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            f"n{index + 1}",
            (int(x) + 9, int(y) - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return canvas


def _prompt_latlon() -> tuple[float, float] | None:
    """Read one ``lat, lon`` from the terminal; None if blank/invalid."""
    raw = input("    lat, lon (blank to cancel this point): ").strip()
    if not raw:
        return None
    parts = [p for p in raw.replace(",", " ").split() if p]
    try:
        return (float(parts[0]), float(parts[1]))
    except (IndexError, ValueError):
        print("    !! could not parse; point cancelled")
        return None


def collect_new_points(frame: np.ndarray, existing: list[dict]) -> list[dict]:
    """Open a window; for each LEFT-CLICK, prompt the terminal for its lat/lon.

    Args:
        frame: The BGR frame to click on.
        existing: Already-saved points (shown green, not re-clicked).

    Returns:
        Newly added points as ``[{name, lat, lon, pixel:[x,y]}]``.
    """
    window = "GCP calibration"
    new: list[dict] = []
    pending: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, _draw_overlay(frame, existing, new))
        key = cv2.waitKey(20) & 0xFF
        if pending:
            x, y = pending.pop()
            print(f"  clicked pixel ({x}, {y})")
            latlon = _prompt_latlon()
            if latlon is not None:
                new.append(
                    {
                        "name": f"gcp{len(existing) + len(new) + 1}",
                        "lat": latlon[0],
                        "lon": latlon[1],
                        "pixel": [int(x), int(y)],
                    }
                )
            continue
        if key == ord("q"):
            new = []
            break
        if key == ord("u") and new:
            new.pop()
        if key in (13, 10, ord("s")):
            break
    cv2.destroyWindow(window)
    return new


def undistort_to_normalized(pixels, distortion: dict | None):
    """Map pixels to undistorted, normalised image coords using a radial term.

    The homography is fit on, and applied to, the OUTPUT of this function, so a
    single radial coefficient ``k1`` removes the barrel distortion a wide-angle
    CCTV adds before the planar map. ``distortion`` is ``{"k1","cx","cy","f"}``
    (centre + focal in px); None returns the pixels unchanged (no lens model).

    Examples:
        ```python
        import numpy as np
        undistort_to_normalized(np.array([[640.0, 360.0]]),
                                {"k1": -0.5, "cx": 640, "cy": 360, "f": 640}).tolist()
        # [[0.0, 0.0]]
        ```
    """
    pts = np.asarray(pixels, dtype=np.float64)
    if not distortion:
        return pts.astype(np.float32)
    k1, cx, cy, f = (
        distortion["k1"],
        distortion["cx"],
        distortion["cy"],
        distortion["f"],
    )
    xn = (pts[:, 0] - cx) / f
    yn = (pts[:, 1] - cy) / f
    scale = 1.0 + k1 * (xn * xn + yn * yn)
    return np.stack([xn * scale, yn * scale], axis=1).astype(np.float32)


def _loo_residual_m(source, target, distortion) -> float:
    """Mean leave-one-out reprojection error (m) of the undistorted homography."""
    n = len(source)
    errors = []
    for i in range(n):
        keep = np.arange(n) != i
        homography, _ = cv2.findHomography(
            undistort_to_normalized(source[keep], distortion), target[keep], method=0
        )
        if homography is None:
            return float("inf")
        point = cv2.perspectiveTransform(
            undistort_to_normalized(source[i : i + 1], distortion).reshape(-1, 1, 2),
            homography,
        ).reshape(2)
        errors.append(float(np.linalg.norm(point - target[i])))
    return float(np.mean(errors))


def estimate_radial_k1(source, target, image_size) -> dict | None:
    """Pick the radial ``k1`` (barrel distortion) minimising leave-one-out error.

    A wide-angle lens bends straight ground lines, so a single planar homography
    fits the control points yet EXTRAPOLATES badly (large LOO error, the keelung
    clip's plain homography is ~26 m). Undistorting with one radial term first
    makes the points consistent with a plane. Returns ``{"k1","cx","cy","f"}`` or
    None when k1=0 is already best, or there are too few points to cross-validate.
    """
    width, height = image_size
    base = {"k1": 0.0, "cx": width / 2.0, "cy": height / 2.0, "f": width / 2.0}
    if len(source) < 6:
        return None
    best_k1, best_loo = 0.0, _loo_residual_m(source, target, base)
    for k1 in np.arange(-1.2, 0.31, 0.05):
        loo = _loo_residual_m(source, target, {**base, "k1": float(k1)})
        if loo < best_loo:
            best_loo, best_k1 = loo, float(k1)
    return None if abs(best_k1) < 1e-6 else {**base, "k1": round(best_k1, 4)}


def _synthesize_distance_targets(
    fit: np.ndarray,
    target: np.ndarray,
    distortion: dict | None,
    distance_constraints: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Turn each distance constraint into two synthesized (pixel, metric) points.

    Simplified, scipy-free scale constraint: an initial GCP-only homography gives
    each constraint pair's DIRECTION on the ground; we keep endpoint ``a`` where the
    homography places it and move ``b`` along that direction to the KNOWN real
    distance. Fed back in as extra correspondences, these pull the homography's
    scale toward the true metric even where the GCPs themselves are clustered.

    Args:
        fit: GCP pixels, already undistorted (``(n, 2)``).
        target: GCP metric targets (``(n, 2)``).
        distortion: The lens model (applied to the constraint pixels too), or None.
        distance_constraints: ``[{pixel_a, pixel_b, distance_m}, ...]``.

    Returns:
        ``(extra_fit, extra_target)`` undistorted pixels and synthesized metric
        points (each of shape ``(2k, 2)`` for ``k`` usable constraints; empty when
        none are usable).
    """
    base_h, _ = cv2.findHomography(fit, target, method=0)
    extra_fit: list[np.ndarray] = []
    extra_target: list[np.ndarray] = []
    if base_h is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    for constraint in distance_constraints:
        try:
            distance_m = float(constraint["distance_m"])
            pixel_a = np.array([constraint["pixel_a"]], dtype=np.float32)
            pixel_b = np.array([constraint["pixel_b"]], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if distance_m <= 0:
            continue
        a_ud = undistort_to_normalized(pixel_a, distortion)
        b_ud = undistort_to_normalized(pixel_b, distortion)
        ma = cv2.perspectiveTransform(a_ud.reshape(-1, 1, 2), base_h).reshape(2)
        mb = cv2.perspectiveTransform(b_ud.reshape(-1, 1, 2), base_h).reshape(2)
        vector = mb - ma
        norm = float(np.hypot(vector[0], vector[1]))
        if norm < 1e-9:
            continue
        synth_b = ma + (vector / norm) * distance_m
        extra_fit.extend([a_ud[0], b_ud[0]])
        extra_target.extend([ma, synth_b])
    if not extra_fit:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return (
        np.array(extra_fit, dtype=np.float32),
        np.array(extra_target, dtype=np.float32),
    )


def build_calibration(
    gcps: list[dict],
    image_size: tuple[int, int] | None = None,
    distance_constraints: list[dict] | None = None,
) -> dict:
    """Compute the pixel->meter homography and assemble the calibration record.

    When ``image_size`` is given (>= 6 points), a single radial distortion term is
    estimated and the homography is fit on UNDISTORTED image coords -- essential
    for wide-angle CCTV where the plain homography overfits and extrapolates badly.

    ``distance_constraints`` add KNOWN real distances between image point pairs (a
    lane dash is 4 m, its gap 6 m by Taiwan spec §182). Absolute positioning still
    comes from the satellite GCPs; the constraints only pin SCALE, so a homography
    fit from a small GCP patch stops compressing distance beyond it (the main cause
    of low speed). See :func:`_synthesize_distance_targets` for the simplified,
    scipy-free fitting.

    Args:
        gcps: Ground control points ``[{name, lat, lon, pixel:[x,y]}]`` (>= 4).
        image_size: ``(width, height)`` of the video, enabling lens undistortion.
        distance_constraints: Optional ``[{pixel_a, pixel_b, distance_m}, ...]``.

    Returns:
        A JSON-serializable calibration dict (``distortion`` is None when unused;
        ``distance_constraints`` echoes the input list).

    Raises:
        ValueError: If fewer than 4 control points are supplied.
    """
    if len(gcps) < 4:
        raise ValueError(f"Need at least 4 control points, got {len(gcps)}.")
    distance_constraints = distance_constraints or []
    latlon = np.array([[g["lat"], g["lon"]] for g in gcps], dtype=np.float64)
    origin = (float(latlon[:, 0].mean()), float(latlon[:, 1].mean()))
    source = np.array([g["pixel"] for g in gcps], dtype=np.float32)
    target = latlon_to_local_meters(latlon, origin)

    distortion = estimate_radial_k1(source, target, image_size) if image_size else None
    fit = undistort_to_normalized(source, distortion)

    # The homography is fit on the GCP points PLUS any synthesized distance-
    # constraint points; residuals/inliers are still reported per GCP only.
    extra_fit, extra_target = _synthesize_distance_targets(
        fit, target, distortion, distance_constraints
    )
    fit_h = np.vstack([fit, extra_fit]) if len(extra_fit) else fit
    target_h = np.vstack([target, extra_target]) if len(extra_target) else target

    if len(fit_h) == 4:
        homography = cv2.getPerspectiveTransform(
            fit_h.astype(np.float32), target_h.astype(np.float32)
        )
        method = "getPerspectiveTransform (4 points, exact)"
        inlier_mask = [True] * len(fit)
    else:
        least_squares, _ = cv2.findHomography(fit_h, target_h, method=0)
        robust, mask = cv2.findHomography(
            fit_h, target_h, cv2.USAC_MAGSAC, RANSAC_THRESHOLD_M
        )
        inliers = mask.ravel().astype(bool) if mask is not None else None
        enough = inliers is not None and int(inliers.sum()) >= len(fit_h) - 2
        if robust is not None and inliers is not None and enough:
            homography = robust
            method = f"MAGSAC++ ({int(inliers.sum())}/{len(fit_h)} inliers)"
            inlier_mask = inliers[: len(fit)].tolist()
        else:
            homography = least_squares
            kept = 0 if inliers is None else int(inliers.sum())
            method = f"least-squares (MAGSAC kept only {kept}/{len(fit_h)})"
            inlier_mask = [True] * len(fit)
    if distortion is not None:
        method += f" + 去畸變 k1={distortion['k1']}"
    if len(extra_target):
        method += f" + {len(extra_target) // 2} 距離約束"

    projected = cv2.perspectiveTransform(fit.reshape(-1, 1, 2), homography).reshape(
        -1, 2
    )
    residuals = np.linalg.norm(projected - target, axis=1)

    # Real-world spread of the control points (diagonal of their ground bounding
    # box). A small span means the homography was fit from a tiny patch and a low
    # residual is misleading -- warn so the user spreads the points out. Distance
    # constraints extend the trustworthy extent, so their synthesized targets count
    # toward the span.
    span_target = np.vstack([target, extra_target]) if len(extra_target) else target
    extent = span_target.max(axis=0) - span_target.min(axis=0)
    target_span_m = float(np.hypot(extent[0], extent[1]))
    span_warning = (
        None
        if target_span_m >= MIN_GCP_SPAN_M
        else (
            f"控制點真實範圍僅 ~{target_span_m:.0f} m"
            f"（建議 ≥ {MIN_GCP_SPAN_M:.0f} m）。點全擠在一小塊地面，"
            "投影會壓縮/外插失真，重投影誤差再小也不可靠。"
            "請補上彼此拉開的控制點：近端與遠端、最左與最右，"
            "或提供沿路標線的距離約束點對。"
        )
    )
    return {
        "source_points_px": source.tolist(),
        "target_points_m": target.tolist(),
        "distortion": distortion,
        "origin_latlon": list(origin),
        "gcp_names": [g["name"] for g in gcps],
        "gcp_latlon": latlon.tolist(),
        "homography_px_to_m": homography.tolist(),
        "method": method,
        "inlier_mask": inlier_mask,
        "residuals_m": residuals.tolist(),
        "max_residual_m": float(residuals.max()),
        "mean_residual_m": float(residuals.mean()),
        "target_span_m": target_span_m,
        "span_warning": span_warning,
        "distance_constraints": distance_constraints,
    }


# ---------------------------------------------------------------------------
# ViewTransformer – pixel → metric ground-plane projection
# ---------------------------------------------------------------------------

# Metres per degree of latitude (spherical Earth). Shared with the geo writers
# so the 111_195 magic number is defined exactly once.
METERS_PER_DEG_LAT = math.radians(1.0) * EARTH_RADIUS_M


class ViewTransformer:
    """Map points from the camera image onto the metric bird's-eye plane.

    Wraps a perspective homography (optionally preceded by radial undistortion)
    so all pixel → metric projections go through a single, well-tested path.

    Args:
        source: Road-plane points in camera pixel coordinates (≥ 4 points).
        target: Matching points in real-world meters.
        distortion: Optional ``{"k1","cx","cy","f"}`` radial-undistortion model.

    Examples:
        ```python
        import numpy as np
        source = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        target = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        transformer = ViewTransformer(source=source, target=target)
        transformer.transform_points(np.array([[5, 5]])).round(2).tolist()
        # [[0.5, 0.5]]
        ```
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        distortion: dict | None = None,
    ) -> None:
        self.distortion = distortion
        source_ud = undistort_to_normalized(source, distortion)
        target = target.astype(np.float32)
        if len(source_ud) == 4:
            self.m = cv2.getPerspectiveTransform(source_ud, target)
        else:
            self.m, _ = cv2.findHomography(source_ud, target, method=0)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an ``(n, 2)`` pixel array to metric ground coordinates.

        Args:
            points: Array of shape ``(n, 2)`` in camera pixels.

        Returns:
            Array of shape ``(n, 2)`` in metric ``(east_m, north_m)``.
        """
        if points.size == 0:
            return points.astype(np.float32)
        reshaped = undistort_to_normalized(points, self.distortion).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(reshaped, self.m).reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map ``(n, 2)`` metric ground points back to camera pixels.

        The inverse of :meth:`transform_points`: apply the inverse homography to
        reach undistorted normalised coords, then re-apply the lens distortion to
        land on real pixels. Used to draw a metric trajectory back onto the video.

        Args:
            points: Array of shape ``(n, 2)`` in metric ``(east_m, north_m)``.

        Returns:
            Array of shape ``(n, 2)`` in camera pixels.
        """
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points
        inverse = np.linalg.inv(self.m)
        normalized = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), inverse
        ).reshape(-1, 2)
        return _redistort_from_normalized(normalized, self.distortion)


def _redistort_from_normalized(
    normalized: np.ndarray, distortion: dict | None
) -> np.ndarray:
    """Inverse of :func:`undistort_to_normalized`: normalised coords -> pixels.

    With no lens model the normalised coords are already the pixels. Otherwise the
    forward map is ``xu = xn * (1 + k1 * r^2)``; recover ``xn`` from ``xu`` by a few
    fixed-point iterations (converges for the mild ``k1`` a single-coefficient CCTV
    model uses), then scale back to pixels by ``f`` and the centre.

    Examples:
        ```python
        import numpy as np
        # Round-trips undistort_to_normalized for a mild k1.
        d = {"k1": -0.2, "cx": 640, "cy": 360, "f": 700}
        px = np.array([[820.0, 300.0]])
        norm = undistort_to_normalized(px, d)
        _redistort_from_normalized(norm, d).round(2).tolist()
        # [[820.0, 300.0]]
        ```
    """
    pts = np.asarray(normalized, dtype=np.float64)
    if not distortion:
        return pts.astype(np.float32)
    k1, cx, cy, f = (
        distortion["k1"],
        distortion["cx"],
        distortion["cy"],
        distortion["f"],
    )
    xu, yu = pts[:, 0], pts[:, 1]
    xn, yn = xu.copy(), yu.copy()
    for _ in range(10):
        scale = 1.0 + k1 * (xn * xn + yn * yn)
        xn, yn = xu / scale, yu / scale
    return np.stack([xn * f + cx, yn * f + cy], axis=1).astype(np.float32)


# Active-scene transformer + calibration state (populated by _load_calibration).
VIEW_TRANSFORMER: ViewTransformer | None = None
ORIGIN_LATLON: tuple[float, float] | None = None
USING_GPS_CALIBRATION: bool = False


def _load_calibration() -> bool:
    """Load the active scene's homography calibration and populate module globals.

    Sets :data:`VIEW_TRANSFORMER`, :data:`ORIGIN_LATLON`, and
    :data:`USING_GPS_CALIBRATION`. Called once at import time; call again after
    writing a new calibration file to refresh.

    Returns:
        True when the calibration file was found and loaded.
    """
    global VIEW_TRANSFORMER, ORIGIN_LATLON, USING_GPS_CALIBRATION
    path = SCENE.calibration_path
    if not path.exists():
        USING_GPS_CALIBRATION = False
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        USING_GPS_CALIBRATION = False
        return False
    source = np.array(data["source_points_px"], dtype=np.float32)
    target = np.array(data["target_points_m"], dtype=np.float32)
    VIEW_TRANSFORMER = ViewTransformer(
        source=source, target=target, distortion=data.get("distortion")
    )
    ORIGIN_LATLON = (float(data["origin_latlon"][0]), float(data["origin_latlon"][1]))
    USING_GPS_CALIBRATION = True
    return True


_load_calibration()


def metric_to_latlon(point: tuple[float, float]) -> tuple[float, float] | None:
    """Convert a bird's-eye ``(east_m, north_m)`` point to ``(lat, lon)``.

    Inverse of the equirectangular projection used during GCP calibration.

    Args:
        point: Metric position ``(east_m, north_m)`` on the ground plane.

    Returns:
        ``(lat, lon)`` in decimal degrees, or ``None`` when no GPS calibration
        is loaded.

    Examples:
        ```python
        metric_to_latlon((0.0, 0.0)) is None  # without calibration loaded
        # True
        ```
    """
    if ORIGIN_LATLON is None:
        return None
    east_m, north_m = point
    origin_lat, origin_lon = ORIGIN_LATLON
    lat = origin_lat + north_m / METERS_PER_DEG_LAT
    lon = origin_lon + east_m / (
        METERS_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    )
    return (lat, lon)


def main(scene: SceneConfig = SCENE) -> None:
    """Run the interactive calibration for ``scene`` and write the result."""
    print(f"Scene: {scene.name}  video: {scene.source_video}")
    existing = load_gcps(scene.gcp_store)
    print(f"Loaded {len(existing)} saved control points from {scene.gcp_store}")
    frame = read_calibration_frame(scene)
    new = collect_new_points(frame, existing)
    if not new and not existing:
        print("No control points; nothing saved.")
        return

    gcps = existing + new
    # Distance constraints (if any) are entered out of band (edited into gcps.json
    # or via the web app); the interactive click flow only adds satellite GCPs.
    constraints = load_distance_constraints(scene.gcp_store)
    save_gcps(scene.gcp_store, gcps, constraints)
    calibration = build_calibration(
        gcps,
        image_size=(frame.shape[1], frame.shape[0]),
        distance_constraints=constraints,
    )
    scene.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    scene.calibration_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False)
    )
    print(f"Saved {len(new)} new point(s); {len(gcps)} total -> {scene.gcp_store}")
    print(f"Calibration -> {scene.calibration_path}")
    print(f"Fit: {calibration['method']}")
    print(
        f"Reprojection residual: mean {calibration['mean_residual_m']:.2f} m, "
        f"max {calibration['max_residual_m']:.2f} m"
    )
    print(f"Control-point ground span: {calibration['target_span_m']:.1f} m")
    if calibration["span_warning"]:
        print(f"⚠️  {calibration['span_warning']}")
    print("Per-point error (look for outliers to re-read or remove):")
    for name, (x, y), residual, inlier in zip(
        calibration["gcp_names"],
        calibration["source_points_px"],
        calibration["residuals_m"],
        calibration["inlier_mask"],
    ):
        flag = "" if inlier else "  <-- rejected (likely mis-read)"
        print(f"  {name}: pixel ({x:.0f}, {y:.0f})  err {residual:.2f} m{flag}")


if __name__ == "__main__":
    main()
