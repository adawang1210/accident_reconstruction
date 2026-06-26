from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accident_reconstruction.calibrate_homography import (
    ORIGIN_LATLON,
    USING_GPS_CALIBRATION,
    latlon_to_local_meters,
    metric_to_latlon,
)
from accident_reconstruction.manual_pre_impact_motorcycle_annotation import (
    BIRDSEYE_PX_PER_M,
    END_FRAME_INDEX,
    MANUAL_TRACKS,
    ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH,
    SOURCE_VIDEO_PATH,
    STATIC_FRAME_DIFF_THRESHOLD,
    TrackState,
    create_metric_birdseye_base,
    frame_difference,
    metric_to_panel,
    update_track_speed,
)
from accident_reconstruction.scene_config import SCENE

_PREFIX = SCENE.artifact_dir.parent / SCENE.name
BIRDSEYE_TARGET_VIDEO_PATH = SCENE.video_path("birdseye_split_slow_2x")
BIRDSEYE_CHECK_SHEET_PATH = SCENE.artifact_dir / "birdseye_result_sheet.jpg"
BIRDSEYE_SUMMARY_IMAGE_PATH = Path(f"{_PREFIX}_birdseye_summary.png")
BIRDSEYE_KML_PATH = Path(f"{_PREFIX}_route.kml")
MAP_FIGURE_PATH = Path(f"{_PREFIX}_map_figure.png")
ROUTE_CSV_PATH = Path(f"{_PREFIX}_route.csv")

# The KML/projection carries a small bias (offset + slight rotation/scale) from
# fisheye + control-point noise -- NOT GCJ-02 (~500 m). We remove it with a
# two-point SIMILARITY alignment (translation + rotation + uniform scale, which
# does NOT distort the curve shape): the recognised impact maps to TRUE_IMPACT and
# the recognised car start maps to TRUE_CAR_START, both read off the basemap by
# the user. Both anchor points land exactly, so the impact is preserved.
# Override-aware (the step-2 UI can edit these per scene via overrides.json).
TRUE_IMPACT_LATLON = SCENE.resolved_true_impact_latlon
TRUE_CAR_START_LATLON = SCENE.true_car_start_latlon
# Optional per-vehicle real start positions ``{label: (lat, lon)}``. A vehicle
# listed here is aligned by a two-point similarity (impact + start, with SCALE), so
# its compressed path is stretched to the real distance -- counters the fisheye
# homography distance compression. Vehicles absent here keep rotation-only.
TRUE_VEHICLE_STARTS = SCENE.resolved_true_vehicle_starts


# Label of the moving vehicle whose start anchors to TRUE_CAR_START (the second
# alignment point). Resolved per call from the scene + UI overrides; "car" keeps
# the original 永康 behaviour.
def _moving_vehicle() -> str:
    """The vehicle whose start anchors the alignment (scene/UI, else 'car')."""
    return SCENE.resolved_moving_vehicle or "car"


# A CJK-capable font for the figures. The macOS system font is tried first, then
# common Linux paths, so rendering does not crash off-Mac.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
CJK_FONT_PATH = next(
    (path for path in _FONT_CANDIDATES if Path(path).exists()), _FONT_CANDIDATES[0]
)

# --- 2D figure styling -------------------------------------------------------
# Per-scene vehicle labels + RGB trail colours (from scene_config).
VEHICLE_DISPLAY = SCENE.vehicle_display

_BG = (18, 18, 18)
_INK = (220, 220, 220)
_RED = (235, 70, 70)


def _font(size: int):
    """A CJK font at ``size``, falling back to PIL's default if none is found."""
    try:
        return ImageFont.truetype(CJK_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy, text, size, fill) -> None:
    """Draw text with a dark halo so it stays readable over the panel."""
    draw.text(xy, text, font=_font(size), fill=fill, stroke_width=3, stroke_fill=_BG)


def _latlon_text(metric_xy: tuple[float, float]) -> str | None:
    """Format the lat/lon of a metric point, or None without GPS calibration.

    Args:
        metric_xy: `(east_m, north_m)` point on the metric plane.

    Returns:
        A ``"lat, lon"`` string to 6 decimals, or None.
    """
    latlon = metric_to_latlon(metric_xy)
    if latlon is None:
        return None
    return f"{latlon[0]:.6f}, {latlon[1]:.6f}"


# --- Geo-anchored schematic --------------------------------------------------
# A single planar homography on this fisheye dashcam cannot put the recognised
# trajectory accurately on the road (leave-one-out error ~4.5 m, capped by lens
# distortion + control-point noise -- proven, not fixable by more points). So the
# final figure is a GEO-ANCHORED SCHEMATIC: the two roads are drawn from real
# OpenStreetMap centrelines, and each vehicle is placed ON its road by its real
# travelled distance (from the recognition), anchored so both reach the shared
# intersection node at the impact frame. Positions are reconstructed (geometry
# from the map, distance/speed from the video); the lat/lon are real.
INTERSECTION_LATLON = SCENE.intersection_latlon
ROAD_CENTERLINES = SCENE.road_centerlines or {}
ROAD_NAMES = SCENE.road_names
# Geo-anchored map figure / KML need the scene's road + anchor data.
GEO_READY = USING_GPS_CALIBRATION and SCENE.is_geo_ready


def _road_metric(label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a road centreline in metres, its arc lengths, and the intersection s.

    Args:
        label: Track label whose road centreline to build.

    Returns:
        A `(points_m, cumulative_arc, s_intersection)` tuple.
    """
    points = latlon_to_local_meters(
        np.array(ROAD_CENTERLINES[label], dtype=np.float64), ORIGIN_LATLON
    ).astype(np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    node = latlon_to_local_meters(
        np.array([INTERSECTION_LATLON], dtype=np.float64), ORIGIN_LATLON
    )[0]
    index = int(np.argmin(np.linalg.norm(points - node, axis=1)))
    return points, cumulative, float(cumulative[index])


def _point_at_arc(
    points: np.ndarray, cumulative: np.ndarray, target: float
) -> tuple[float, float]:
    """Interpolate the metric point at a given arc length along a polyline."""
    target = float(np.clip(target, cumulative[0], cumulative[-1]))
    index = max(0, min(int(np.searchsorted(cumulative, target)) - 1, len(points) - 2))
    span = cumulative[index + 1] - cumulative[index]
    ratio = 0.0 if span == 0 else (target - cumulative[index]) / span
    point = points[index] + (points[index + 1] - points[index]) * ratio
    return (float(point[0]), float(point[1]))


def _project_arc(points: np.ndarray, cumulative: np.ndarray, pt: np.ndarray) -> float:
    """Arc length of ``pt``'s nearest-point projection onto a metric polyline."""
    best_distance, best_arc = float("inf"), 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        length2 = float(ab @ ab) + 1e-12
        t = float(np.clip((pt - a) @ ab / length2, 0.0, 1.0))
        projection = a + t * ab
        distance = float(np.linalg.norm(pt - projection))
        if distance < best_distance:
            best_distance = distance
            best_arc = float(cumulative[i]) + t * float(np.linalg.norm(ab))
    return best_arc


def geo_anchored_metric(
    motion: dict[str, dict[int, tuple[float, float]]],
    metric: dict[str, dict[int, tuple[float, float]]],
    impact_frame: int,
) -> dict[str, dict[int, tuple[float, float]]]:
    """Place each vehicle on its real road centreline by travelled distance.

    The along-road position is ``cumulative_distance - cumulative_at_impact``
    measured from the intersection node, so both vehicles meet at the impact.
    Falls back to the raw projection when no GPS calibration is loaded.

    Args:
        motion: Per-vehicle `(cumulative_m, speed_kmh)` by frame.
        metric: Per-vehicle raw projection `(east_m, north_m)` by frame (fallback).
        impact_frame: Frame of closest approach.

    Returns:
        Per-vehicle geo-anchored `(east_m, north_m)` by frame.
    """
    if not USING_GPS_CALIBRATION or ORIGIN_LATLON is None:
        return metric
    anchored: dict[str, dict[int, tuple[float, float]]] = {}
    for label in metric:
        if label not in ROAD_CENTERLINES:
            anchored[label] = metric[label]
            continue
        points, cumulative, s_node = _road_metric(label)
        impact_distance = motion[label][impact_frame][0]
        anchored[label] = {
            frame: _point_at_arc(
                points, cumulative, s_node + distance - impact_distance
            )
            for frame, (distance, _) in motion[label].items()
            if frame in metric[label]
        }
    return anchored


def _draw_roads(image: Image.Image) -> None:
    """Draw the two real road centrelines as grey bands on the panel."""
    if not USING_GPS_CALIBRATION or ORIGIN_LATLON is None:
        return
    draw = ImageDraw.Draw(image)
    width_px = max(8, round(SCENE.road_width_m * BIRDSEYE_PX_PER_M))
    for label in ROAD_CENTERLINES:
        points, _, _ = _road_metric(label)
        panel = [metric_to_panel((float(x), float(y))) for x, y in points]
        draw.line(panel, fill=(70, 74, 82), width=width_px, joint="curve")
        for px, py in panel:  # round the band ends/joints
            draw.ellipse(
                [
                    px - width_px // 2,
                    py - width_px // 2,
                    px + width_px // 2,
                    py + width_px // 2,
                ],
                fill=(70, 74, 82),
            )


def build_real_base() -> Image.Image:
    """Build the static bird's-eye background from the real homography panel.

    Wraps the annotation module's metric panel (camera-aligned grid, surveyed
    GCP patch, and underpass marker) as an RGB PIL image so vehicle paths and
    CJK labels can be drawn on top.

    Returns:
        An RGB PIL image of the empty, camera-oriented bird's-eye plane.

    Examples:
        ```python
        build_real_base().mode
        # 'RGB'
        ```
    """
    base_bgr = create_metric_birdseye_base()
    image = Image.fromarray(base_bgr[:, :, ::-1])
    _draw_roads(image)
    title = (
        "事故 2D 俯視重建(地理錨定:路形依地圖,距離/速度由影片)"
        if USING_GPS_CALIBRATION
        else "事故 2D 俯視重建(未校正)"
    )
    _label(ImageDraw.Draw(image), (10, 30), title, 14, _INK)
    return image


def collect_vehicle_motion() -> tuple[
    dict[str, dict[int, tuple[float, float]]],
    dict[str, dict[int, tuple[float, float]]],
    int,
]:
    """Replay the clip and return each vehicle's raw recognition projection.

    The trajectory is the manual tracking boxes' ground anchors projected through
    the homography -- exactly as recognised, with NO snapping, straightening, or
    smoothing of the line. Speed comes from the windowed metric displacement.

    Returns:
        A `(motion, metric, impact_frame)` tuple. ``motion[label][frame]`` is
        `(cumulative_distance_m, speed_kmh)`, ``metric[label][frame]`` is the
        projected `(east_m, north_m)` position, and ``impact_frame`` is the frame
        of closest approach between the vehicles.
    """
    source_cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not source_cap.isOpened():
        raise FileNotFoundError(f"Could not open {SOURCE_VIDEO_PATH}")

    states = {track.label: TrackState() for track in MANUAL_TRACKS}
    motion: dict[str, dict[int, tuple[float, float]]] = {
        track.label: {} for track in MANUAL_TRACKS
    }
    metric: dict[str, dict[int, tuple[float, float]]] = {
        track.label: {} for track in MANUAL_TRACKS
    }
    cumulative = {track.label: 0.0 for track in MANUAL_TRACKS}
    last_xy: dict[str, tuple[float, float] | None] = {
        track.label: None for track in MANUAL_TRACKS
    }
    previous_frame = None
    frame_index = 0
    while True:
        ok, source_frame = source_cap.read()
        if not ok or frame_index > END_FRAME_INDEX:
            break
        is_static = (
            frame_difference(previous_frame=previous_frame, frame=source_frame)
            < STATIC_FRAME_DIFF_THRESHOLD
        )
        for track in MANUAL_TRACKS:
            state = states[track.label]
            metric_xy = update_track_speed(state, track, frame_index, 25.0)
            if metric_xy is not None:
                previous_xy = last_xy[track.label]
                if previous_xy is not None and not is_static:
                    cumulative[track.label] += math.hypot(
                        metric_xy[0] - previous_xy[0], metric_xy[1] - previous_xy[1]
                    )
                last_xy[track.label] = metric_xy
                if not is_static:
                    metric[track.label][frame_index] = metric_xy
            motion[track.label][frame_index] = (
                cumulative[track.label],
                state.speed_kmh,
            )
        previous_frame = source_frame
        frame_index += 1
    source_cap.release()

    labels = [track.label for track in MANUAL_TRACKS]
    shared = [f for f in metric[labels[0]] if f in metric[labels[1]]]
    impact_frame = (
        min(
            shared,
            key=lambda f: math.hypot(
                metric[labels[0]][f][0] - metric[labels[1]][f][0],
                metric[labels[0]][f][1] - metric[labels[1]][f][1],
            ),
        )
        if shared
        else END_FRAME_INDEX
    )
    return motion, metric, impact_frame


def _panel_path(
    label: str,
    metric: dict[str, dict[int, tuple[float, float]]],
    upto_frame: int | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Project a vehicle's metric trace onto the panel, optionally up to a frame.

    Args:
        label: Track label.
        metric: Per-vehicle `(east_m, north_m)` positions by frame.
        upto_frame: Keep only frames ``<= upto_frame``; None keeps the full path.

    Returns:
        A `(panel_points, frames)` tuple of pixel points and their frame indices.
    """
    track = metric.get(label, {})
    frames = sorted(f for f in track if upto_frame is None or f <= upto_frame)
    return [metric_to_panel(track[f]) for f in frames], frames


def _impact_point(
    metric: dict[str, dict[int, tuple[float, float]]], impact_frame: int
) -> tuple[tuple[int, int], tuple[float, float]] | None:
    """Return the panel pixel and metric midpoint of the two vehicles at impact.

    Args:
        metric: Per-vehicle `(east_m, north_m)` positions by frame.
        impact_frame: Frame of closest approach.

    Returns:
        A `(panel_point, metric_xy)` tuple, or None if no shared frame exists.
    """
    positions = [
        metric[label][impact_frame] for label in metric if impact_frame in metric[label]
    ]
    if not positions:
        return None
    mid = (
        sum(p[0] for p in positions) / len(positions),
        sum(p[1] for p in positions) / len(positions),
    )
    return metric_to_panel(mid), mid


def _draw_impact(draw: ImageDraw.ImageDraw, panel_xy: tuple[int, int]) -> None:
    """Draw the red collision starburst at a panel point.

    Args:
        draw: Target PIL draw context.
        panel_xy: Panel pixel position of the impact.
    """
    cx, cy = panel_xy
    for degrees in range(0, 360, 45):
        radians = math.radians(degrees)
        tip = (cx + 16 * math.cos(radians), cy + 16 * math.sin(radians))
        draw.line([(cx, cy), tip], fill=_RED, width=3)
    draw.ellipse([cx - 19, cy - 19, cx + 19, cy + 19], outline=_RED, width=3)


def write_summary_image() -> None:
    """Render the static top-down accident figure: real paths, impact, lat/lon."""
    motion, metric, impact_frame = collect_vehicle_motion()
    image = build_real_base()
    draw = ImageDraw.Draw(image)

    for label, display in VEHICLE_DISPLAY.items():
        points, frames = _panel_path(label, metric)
        if len(points) < 2:
            continue
        color = display["rgb"]
        draw.line(points, fill=color, width=4)
        for corner, frame in ((points[0], frames[0]), (points[-1], frames[-1])):
            cx, cy = corner
            draw.ellipse(
                [cx - 6, cy - 6, cx + 6, cy + 6], fill=color, outline=(0, 0, 0)
            )
        # 90th percentile, not max: the raw homography produces single-frame
        # speed spikes (e.g. 123 km/h) that the percentile rejects.
        peak = float(np.percentile([s for _, s in motion[label].values()], 90))
        sx, sy = points[0]
        _label(
            draw, (sx + 9, sy - 9), f"{display['name']} 起 約{peak:.0f}km/h", 16, color
        )
        start_ll = _latlon_text(metric[label][frames[0]])
        end_ll = _latlon_text(metric[label][frames[-1]])
        if start_ll:
            _label(draw, (sx + 9, sy + 8), start_ll, 12, color)
        if end_ll:
            ex, ey = points[-1]
            _label(draw, (ex + 9, ey + 4), end_ll, 12, color)

    impact = _impact_point(metric, impact_frame)
    if impact is not None:
        panel_xy, metric_xy = impact
        _draw_impact(draw, panel_xy)
        _label(draw, (panel_xy[0] + 14, panel_xy[1] - 24), "撞擊點", 16, _RED)
        impact_ll = _latlon_text(metric_xy)
        if impact_ll:
            _label(draw, (panel_xy[0] + 14, panel_xy[1] - 6), impact_ll, 12, _RED)

    BIRDSEYE_SUMMARY_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(BIRDSEYE_SUMMARY_IMAGE_PATH))


def render_real_panel(
    base: Image.Image,
    frame_index: int,
    motion: dict[str, dict[int, tuple[float, float]]],
    metric: dict[str, dict[int, tuple[float, float]]],
    impact_frame: int,
) -> np.ndarray:
    """Draw the projected paths/dots up to one frame onto a copy of the base.

    Args:
        base: The static bird's-eye background from `build_real_base`.
        frame_index: Current frame index.
        motion: Per-vehicle distance/speed, from `collect_vehicle_motion`.
        metric: Per-vehicle `(east_m, north_m)` positions by frame.
        impact_frame: Frame of closest approach.

    Returns:
        A BGR ``np.ndarray`` panel ready to composite with the video frame.
    """
    image = base.copy()
    draw = ImageDraw.Draw(image)

    if frame_index >= impact_frame:
        impact = _impact_point(metric, impact_frame)
        if impact is not None:
            _draw_impact(draw, impact[0])

    for label, display in VEHICLE_DISPLAY.items():
        points, _ = _panel_path(label, metric, upto_frame=frame_index)
        if not points:
            continue
        color = display["rgb"]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        px, py = points[-1]
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=color, outline=(0, 0, 0))
        speed = motion[label][frame_index][1] if frame_index in motion[label] else 0.0
        # Clamp single-frame homography speed spikes for a stable readout.
        cap = float(np.percentile([s for _, s in motion[label].values()], 95))
        speed = min(speed, cap)
        _label(
            draw, (px + 12, py - 24), f"{display['name']} {speed:.0f} km/h", 16, color
        )

    return np.ascontiguousarray(np.array(image)[:, :, ::-1])


def write_birdseye_split_video() -> None:
    """Create a split-screen video: camera view beside the real 2D bird's-eye."""
    annotated_cap = cv2.VideoCapture(str(ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH))
    if not annotated_cap.isOpened():
        raise FileNotFoundError(
            f"Could not open {ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH}"
        )

    fps = annotated_cap.get(cv2.CAP_PROP_FPS) or 12.5
    frame_height = int(annotated_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(annotated_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    motion, metric, impact_frame = collect_vehicle_motion()
    base = build_real_base()
    panel_width = round(base.width * frame_height / base.height)
    output_size = (frame_width + panel_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    BIRDSEYE_TARGET_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(BIRDSEYE_TARGET_VIDEO_PATH), fourcc, fps, output_size)

    frame_index = 0
    while True:
        annotated_ok, annotated_frame = annotated_cap.read()
        if not annotated_ok or frame_index > END_FRAME_INDEX:
            break
        panel = render_real_panel(base, frame_index, motion, metric, impact_frame)
        panel = cv2.resize(
            panel, (panel_width, frame_height), interpolation=cv2.INTER_AREA
        )
        writer.write(np.hstack([annotated_frame, panel]))
        frame_index += 1

    annotated_cap.release()
    writer.release()


def _kml_linestring(
    name: str, color_abgr: str, coords: list[tuple[float, float]]
) -> str:
    """Build a KML Placemark LineString from lat/lon coordinates.

    Args:
        name: Placemark name.
        color_abgr: KML colour as ``aabbggrr`` hex.
        coords: ``(lat, lon)`` vertices in order.

    Returns:
        The Placemark XML string (empty when fewer than two points).
    """
    if len(coords) < 2:
        return ""
    points = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in coords)
    return (
        f"  <Placemark><name>{name}</name>"
        f"<Style><LineStyle><color>{color_abgr}</color><width>4</width></LineStyle></Style>"
        f"<LineString><tessellate>1</tessellate>"
        f"<coordinates>{points}</coordinates></LineString></Placemark>\n"
    )


def _travel_bearing(points_m: list[np.ndarray]) -> float:
    """Principal travel direction (radians) of a metric path, start->end oriented.

    Uses the path's principal axis (robust to wiggle), then flips it to point from
    the first sample toward the last (the direction of travel).
    """
    pts = np.asarray(points_m, dtype=np.float64)
    centred = pts - pts.mean(axis=0)
    if len(pts) < 2 or np.allclose(centred, 0):
        return 0.0
    axis = np.linalg.svd(centred)[2][0]
    if np.dot(pts[-1] - pts[0], axis) < 0:
        axis = -axis
    return math.atan2(axis[1], axis[0])


def _road_bearing(
    centreline_m: list[np.ndarray], intersection_m: np.ndarray, reference: float
) -> float:
    """Road tangent (radians) near the intersection, oriented to ``reference``.

    Takes the centreline tangent at the vertex nearest the intersection and flips
    it to within 90 degrees of ``reference`` (the recognised travel direction), so
    the vehicle is oriented along its road the way it actually drove.
    """
    pts = np.asarray(centreline_m, dtype=np.float64)
    i = int(np.argmin(np.linalg.norm(pts - intersection_m, axis=1)))
    before, after = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
    bearing = math.atan2(after[1] - before[1], after[0] - before[0])
    if math.cos(bearing - reference) < 0:
        bearing += math.pi
    return bearing


def build_alignment(
    metric: dict[str, dict[int, tuple[float, float]]], impact_frame: int | None
):
    """Build a PER-VEHICLE alignment of recognised lat/lon onto the real roads.

    Each vehicle's recognised path is rotated about its own position at the impact
    frame so its travel direction matches its OWN road's bearing (from
    ``ROAD_CENTERLINES`` near ``INTERSECTION_LATLON``), then that pivot is placed on
    ``TRUE_IMPACT_LATLON``. This generalises the old single-road two-point
    alignment: every vehicle follows its real street (e.g. a struck car on the
    cross road aligns to the cross road), not just the moving one -- with no
    per-scene anchor tuning, so it works for any clip that has road centrelines.
    The recognised curve SHAPE and length are preserved (rotation only, no scale).

    Args:
        metric: Per-vehicle recognised `(east_m, north_m)` by frame.
        impact_frame: Frame of closest approach (None disables alignment).

    Returns:
        ``align(latlon, label)`` mapping a vehicle's `(lat, lon)` to the map; an
        unknown label is returned unchanged.
    """
    if not GEO_READY or impact_frame is None:
        return lambda latlon, label=None: latlon

    clat, clon = TRUE_IMPACT_LATLON
    m_lat = 111195.0
    m_lon = 111195.0 * math.cos(math.radians(clat))

    def to_m(latlon: tuple[float, float]) -> np.ndarray:
        return np.array([(latlon[1] - clon) * m_lon, (latlon[0] - clat) * m_lat])

    def to_ll(point: np.ndarray) -> tuple[float, float]:
        return (clat + point[1] / m_lat, clon + point[0] / m_lon)

    intersection_m = to_m(INTERSECTION_LATLON) if INTERSECTION_LATLON else None
    centrelines = ROAD_CENTERLINES or {}
    transforms: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for label, track in metric.items():
        frames = sorted(track)
        path_m = [to_m(ll) for f in frames if (ll := metric_to_latlon(track[f]))]
        pivot_f = min(frames, key=lambda f: abs(f - impact_frame))
        pivot_ll = metric_to_latlon(track[pivot_f])
        if len(path_m) < 2 or pivot_ll is None:
            continue
        pivot_m = to_m(pivot_ll)
        scale = 1.0
        true_start = TRUE_VEHICLE_STARTS.get(label)
        recognised_start = path_m[0]  # earliest tracked frame
        rec_vec = recognised_start - pivot_m
        if true_start is not None and float(np.hypot(*rec_vec)) > 1e-6:
            # Two-point similarity (impact + real start, WITH scale): the real
            # start defines both bearing and length, stretching the compressed
            # path to the true distance. This overrides the road-bearing rotation.
            # The impact maps to the origin (TRUE_IMPACT), so the target vector
            # impact->start is just the real start in impact-centred metres.
            tgt_vec = to_m(tuple(true_start))
            angle = math.atan2(tgt_vec[1], tgt_vec[0]) - math.atan2(
                rec_vec[1], rec_vec[0]
            )
            scale = float(np.hypot(*tgt_vec) / np.hypot(*rec_vec))
        else:
            # Rotation-only road alignment (shape/length preserved; may be short).
            recognised = _travel_bearing(path_m)
            angle = 0.0
            if intersection_m is not None and label in centrelines:
                road_m = [to_m(p) for p in centrelines[label]]
                angle = _road_bearing(road_m, intersection_m, recognised) - recognised
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        transforms[label] = (pivot_m, rotation, scale)

    def align(latlon: tuple[float, float], label: str | None = None):
        entry = transforms.get(label)
        if entry is None:
            return latlon
        pivot_m, rotation, scale = entry
        return to_ll(scale * (rotation @ (to_m(latlon) - pivot_m)))

    return align


def write_kml(data=None, kml_path: Path = BIRDSEYE_KML_PATH) -> None:
    """Write the recognised (un-straightened) trajectories as an aligned KML.

    Uses the trajectory exactly as recognised/projected (its real curve, NO
    snapping or straightening), then applies the per-vehicle road alignment (see
    ``build_alignment``) so each vehicle lands on the impact and follows its own
    road. Opens directly in Google Earth / My Maps.

    Args:
        data: Optional ``(motion, metric, impact_frame)``; None replays manual.
        kml_path: Output KML path.
    """
    if not GEO_READY:
        print("Scene geo data / calibration incomplete; skipping KML export.")
        return
    _, metric, impact_frame = data if data is not None else collect_vehicle_motion()
    aligned = _aligned_latlon(metric, impact_frame)

    placemarks = ""
    for label, display in VEHICLE_DISPLAY.items():
        if label not in aligned:
            continue
        r, g, b = display["rgb"]
        color = f"ff{b:02x}{g:02x}{r:02x}"  # KML aabbggrr
        road = ROAD_NAMES.get(label, label)
        coords = [
            aligned[label][f] for f in sorted(aligned[label]) if aligned[label][f]
        ]
        placemarks += _kml_linestring(f"{display['name']} ({road})", color, coords)

    # Every vehicle's impact-frame position maps to the true impact, so the marker
    # is the true impact point itself.
    if _impact_point(metric, impact_frame) is not None:
        clat, clon = TRUE_IMPACT_LATLON
        placemarks += (
            f"  <Placemark><name>撞擊點</name>"
            f"<Point><coordinates>{clon:.7f},{clat:.7f},0</coordinates>"
            f"</Point></Placemark>\n"
        )

    title = " / ".join(
        f"{d['name']} {ROAD_NAMES.get(label, label)}"
        for label, d in VEHICLE_DISPLAY.items()
    )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
        f"  <name>車禍 2D 路線 ({title})</name>\n"
        f"{placemarks}</Document>\n</kml>\n"
    )
    kml_path.parent.mkdir(parents=True, exist_ok=True)
    kml_path.write_text(kml, encoding="utf-8")


def _aligned_latlon(
    metric: dict[str, dict[int, tuple[float, float]]], impact_frame: int
) -> dict[str, dict[int, tuple[float, float]]]:
    """Per-vehicle lat/lon, CONSTRAINED to the real road centreline when possible.

    For a vehicle with both an OSM road centreline and a true start point, the
    drawn position comes from the road (never off-road): it is placed on the
    centreline at an arc length interpolated from the true start to the true
    impact by the recognised travelled distance (a robust 1-D progress, since the
    noisy 2-D projection drifts metres off the road). Vehicles lacking that data
    fall back to the rotation alignment (:func:`build_alignment`).
    """
    align = build_alignment(metric, impact_frame)
    road_ok = bool(
        USING_GPS_CALIBRATION and ORIGIN_LATLON is not None and ROAD_CENTERLINES
    )
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for label in metric:
        frames = sorted(metric[label])
        true_start = TRUE_VEHICLE_STARTS.get(label)
        if not (
            road_ok
            and impact_frame is not None
            and label in ROAD_CENTERLINES
            and true_start is not None
            and len(frames) >= 2
        ):
            out[label] = {
                f: align(ll, label)
                for f in frames
                if (ll := metric_to_latlon(metric[label][f])) is not None
            }
            continue
        points, cumulative, _ = _road_metric(label)
        s_start = _project_arc(
            points,
            cumulative,
            latlon_to_local_meters(np.array([true_start]), ORIGIN_LATLON)[0],
        )
        s_impact = _project_arc(
            points,
            cumulative,
            latlon_to_local_meters(np.array([TRUE_IMPACT_LATLON]), ORIGIN_LATLON)[0],
        )
        travelled, accumulated, previous = {}, 0.0, None
        for f in frames:
            current = np.array(metric[label][f])
            if previous is not None:
                accumulated += float(np.linalg.norm(current - previous))
            travelled[f] = accumulated
            previous = current
        impact_f = min(frames, key=lambda f: abs(f - impact_frame))
        denominator = (travelled[impact_f] - travelled[frames[0]]) or 1.0
        out[label] = {}
        for f in frames:
            progress = (travelled[f] - travelled[frames[0]]) / denominator
            arc = s_start + progress * (s_impact - s_start)
            out[label][f] = metric_to_latlon(_point_at_arc(points, cumulative, arc))
    return out


def write_map_figure(data=None, figure_path: Path = MAP_FIGURE_PATH) -> None:
    """Render a north-up map figure: OSM roads + aligned recognised tracks.

    Args:
        data: Optional ``(motion, metric, impact_frame)`` to plot (e.g. from an
            automatic tracker). When None, the manual tracks are replayed.
        figure_path: Output PNG path.
    """
    if not GEO_READY:
        print("Scene geo data / calibration incomplete; skipping map figure.")
        return
    motion, metric, impact_frame = (
        data if data is not None else collect_vehicle_motion()
    )
    aligned = _aligned_latlon(metric, impact_frame)

    clat, clon = TRUE_IMPACT_LATLON
    scale, size, half = 13.0, 880, 880 / 2
    m_lon = 111195.0 * math.cos(math.radians(clat))

    def to_px(latlon: tuple[float, float]) -> tuple[float, float]:
        x = (latlon[1] - clon) * m_lon
        y = (latlon[0] - clat) * 111195.0
        return (half + x * scale, half - y * scale)

    image = Image.new("RGB", (size, size), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    road_w = int(SCENE.road_width_m * scale)
    for label in ROAD_CENTERLINES:
        pts = [to_px(ll) for ll in ROAD_CENTERLINES[label]]
        draw.line(pts, fill=(176, 182, 193), width=road_w, joint="curve")
        for px, py in pts:
            draw.ellipse(
                [px - road_w / 2, py - road_w / 2, px + road_w / 2, py + road_w / 2],
                fill=(176, 182, 193),
            )
    for label, centerline in ROAD_CENTERLINES.items():
        mid = centerline[len(centerline) // 2]
        _label(draw, to_px(mid), ROAD_NAMES.get(label, label), 20, (90, 96, 108))

    for label, display in VEHICLE_DISPLAY.items():
        if label not in aligned:
            continue
        frames = sorted(aligned[label])
        pts = [to_px(aligned[label][f]) for f in frames]
        if len(pts) < 2:
            continue
        rgb = display["rgb"]  # already RGB for PIL
        draw.line(pts, fill=rgb, width=4, joint="curve")
        sx, sy = pts[0]
        draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=rgb, outline=(0, 0, 0))
        peak = float(np.percentile([s for _, s in motion[label].values()], 90))
        _label(
            draw, (sx + 9, sy - 22), f"{display['name']} 起 約{peak:.0f}km/h", 17, rgb
        )

    ix, iy = to_px(TRUE_IMPACT_LATLON)
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        draw.line(
            [(ix, iy), (ix + 16 * math.cos(rad), iy + 16 * math.sin(rad))],
            fill=(214, 51, 51),
            width=3,
        )
    draw.ellipse([ix - 19, iy - 19, ix + 19, iy + 19], outline=(214, 51, 51), width=3)
    _label(
        draw, (ix + 16, iy + 10), f"撞擊點 {clat:.6f}, {clon:.6f}", 14, (214, 51, 51)
    )
    _label(
        draw,
        (12, 12),
        "事故 2D 重建 (北上;辨識軌跡;各車對齊到自己道路)",
        16,
        (60, 60, 60),
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(figure_path))


def write_csv(data=None, csv_path: Path = ROUTE_CSV_PATH) -> None:
    """Write the aligned per-frame trajectory (lat/lon + speed) as CSV.

    Args:
        data: Optional ``(motion, metric, impact_frame)``; None replays manual.
        csv_path: Output CSV path.
    """
    if not GEO_READY:
        print("Scene geo data / calibration incomplete; skipping CSV.")
        return
    motion, metric, impact_frame = (
        data if data is not None else collect_vehicle_motion()
    )
    aligned = _aligned_latlon(metric, impact_frame)
    lines = ["frame,vehicle,lat,lon,speed_kmh,is_impact"]
    for label in aligned:
        for frame in sorted(aligned[label]):
            lat, lon = aligned[label][frame]
            speed = motion[label][frame][1] if frame in motion[label] else 0.0
            lines.append(
                f"{frame},{label},{lat:.7f},{lon:.7f},{speed:.1f},"
                f"{int(frame == impact_frame)}"
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_check_sheet() -> None:
    """Create a contact sheet for the split-screen bird's-eye output."""
    cap = cv2.VideoCapture(str(BIRDSEYE_TARGET_VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {BIRDSEYE_TARGET_VIDEO_PATH}")

    thumbnails = []
    for frame_index in [80, 100, 120, 140, 160, 170]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        thumbnails.append(
            cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        )
    cap.release()

    rows = []
    for index in range(0, len(thumbnails), 2):
        row = thumbnails[index : index + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 24))
        rows.append(np.hstack(row))

    BIRDSEYE_CHECK_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(BIRDSEYE_CHECK_SHEET_PATH), np.vstack(rows))


if __name__ == "__main__":
    write_map_figure()
    write_kml()
    write_csv()
    write_birdseye_split_video()
    write_summary_image()
    write_check_sheet()
    print(f"Map figure (north-up, aligned): {MAP_FIGURE_PATH.resolve()}")
    print(f"Route KML: {BIRDSEYE_KML_PATH.resolve()}")
    print(f"Route CSV: {ROUTE_CSV_PATH.resolve()}")
    print(f"Bird's-eye split video: {BIRDSEYE_TARGET_VIDEO_PATH.resolve()}")
    print(f"Camera-panel figure: {BIRDSEYE_SUMMARY_IMAGE_PATH.resolve()}")
    print(f"Check sheet: {BIRDSEYE_CHECK_SHEET_PATH.resolve()}")
