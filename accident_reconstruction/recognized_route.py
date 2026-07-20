"""Draw the trajectory the model ACTUALLY recognised -- no road snapping.

Every other writer in this pipeline imposes the OSM road geometry on the path
(``_aligned_latlon`` pins it to the centreline, ``optimize_route`` keeps it in a
road/Frenet frame). That is why they all produce a long line that follows the
street rather than what SAM2 saw. This module does the opposite: it takes the
per-frame SAM2 ground anchors, projects them through the homography, and draws
the result *exactly as recognised* -- raw shape, raw length, raw position -- on a
view auto-zoomed to the path so the real (often small / compressed) trajectory is
actually visible. The roads are drawn underneath only as faint context.

For a well-calibrated clip this is the honest trajectory. For a clip whose
homography is broken (e.g. keelung, where the GCPs are clustered so distances are
compressed 3-4x) the recognised path comes out tiny -- which is the point: it
shows what the camera + calibration really produced, instead of hiding it behind a
road-length schematic.

Example:
    ```bash
    .venv/bin/python -m accident_reconstruction.recognized_route
    ACCIDENT_SCENE=keelung_xinwu_yier \
        .venv/bin/python -m accident_reconstruction.recognized_route
    ```
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from accident_reconstruction.auto_reconstruct import (
    gcp_ground_span_m,
    load_anchors,
    project_metric,
    resolve_impact_frame,
    windowed_motion,
)
from accident_reconstruction.birdseye_manual_annotation import (
    ORIGIN_LATLON,
    ROAD_CENTERLINES,
    ROAD_NAMES,
    ROUTE_CSV_HEADER,
    TRUE_IMPACT_LATLON,
    USING_GPS_CALIBRATION,
    VEHICLE_DISPLAY,
    MapProjection,
    _kml_linestring,
    _label,
    aligned_motion,
    route_csv_row,
)
from accident_reconstruction.calibrate_homography import (
    METERS_PER_DEG_LAT,
    latlon_to_local_meters,
    metric_to_latlon,
)
from accident_reconstruction.scene_config import SCENE


def _vehicle_start_metric() -> dict[str, tuple[float, float]]:
    """Per-vehicle user-set START lat/lon (step 3), as metric-plane points.

    The mark-vehicles UI lets the user type each vehicle's start coordinates
    (``start_latlon`` = ``[lat, lon]`` per object in ``vehicle_boxes.json``). It is
    where the recognised route should depart from; :func:`_recognized_metric`
    translates each path so its first sample lands there (the curve/shape is
    preserved, only the placement is anchored). Converting to local metres keeps the
    translation Euclidean, consistent with the projected path.

    Returns:
        ``{vehicle_name: (east_m, north_m)}`` for vehicles that have a start set.
    """
    path = SCENE.vehicle_boxes
    if not path.exists() or ORIGIN_LATLON is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    objects = data.get("objects") if isinstance(data, dict) else None
    starts: dict[str, tuple[float, float]] = {}
    if isinstance(objects, list):
        for obj in objects:
            start = obj.get("start_latlon")
            name = obj.get("name")
            if name and start and len(start) == 2:
                point = latlon_to_local_meters(
                    np.array([[start[0], start[1]]], dtype=np.float64), ORIGIN_LATLON
                )[0]
                starts[name] = (float(point[0]), float(point[1]))
    return starts


def _recognized_metric() -> dict[str, dict[int, tuple[float, float]]]:
    """Raw recognised metric path per vehicle, translated to its user-set start.

    Projects the SAM2 anchors through the homography (no road snapping), then, for
    any vehicle with a step-3 start point, shifts the whole path so its first sample
    sits on that start -- preserving the recognised shape while anchoring placement.
    Vehicles without a start are returned exactly as projected.
    """
    metric = project_metric(load_anchors(SCENE.prompt_tracks_csv))
    starts = _vehicle_start_metric()
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for label, track in metric.items():
        frames = sorted(track)
        if not frames:
            continue
        start = starts.get(label)
        if start is not None:
            first_x, first_y = track[frames[0]]
            dx, dy = start[0] - first_x, start[1] - first_y
        else:
            dx = dy = 0.0
        out[label] = {f: (track[f][0] + dx, track[f][1] + dy) for f in frames}
    return out


#: Fallback colours for clips whose scene has no ``vehicle_display`` (e.g. a freshly
#: downloaded clip): the recognised figure/KML still draw every tracked label.
_FALLBACK_RGB = [
    (255, 193, 7),
    (33, 150, 243),
    (76, 175, 80),
    (233, 30, 99),
    (156, 39, 176),
]


@functools.lru_cache(maxsize=1)
def _vehicle_colors() -> dict[str, tuple[int, int, int]]:
    """Per-vehicle RGB taken from the user-drawn boxes (``vehicle_boxes.json``).

    The web workbench stores each object's colour as ``bgr``. Reusing it here keeps
    a vehicle's recognised trajectory the SAME colour as its tracking box, and -- as
    the workbench assigns distinct colours per object -- keeps same-class vehicles
    (``car`` / ``car2``) visually apart.

    Cached: ``_display_for`` calls this once per vehicle in the writer loops, but
    ``SCENE`` (hence ``vehicle_boxes``) is fixed for a process, so the file is read
    and parsed once instead of per vehicle.

    Returns:
        ``{vehicle_name: (r, g, b)}`` for objects that carry a colour.
    """
    path = SCENE.vehicle_boxes
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    objects = data.get("objects") if isinstance(data, dict) else None
    colors: dict[str, tuple[int, int, int]] = {}
    if isinstance(objects, list):
        for obj in objects:
            name, bgr = obj.get("name"), obj.get("bgr")
            if name and isinstance(bgr, (list, tuple)) and len(bgr) == 3:
                blue, green, red = bgr
                colors[name] = (int(red), int(green), int(blue))
    return colors


def _display_for(label: str, index: int) -> dict:
    """Display name + RGB for a label.

    Prefers the scene's authored styling, then the user-drawn box colour (so the
    trajectory matches the tracking box and same-class vehicles stay distinct), and
    finally a per-index fallback palette.
    """
    info = VEHICLE_DISPLAY.get(label)
    if info:
        return info
    color = _vehicle_colors().get(label)
    if color is not None:
        return {"name": label, "rgb": color}
    return {"name": label, "rgb": _FALLBACK_RGB[index % len(_FALLBACK_RGB)]}


def _calibration_ready() -> bool:
    """True when the homography is georeferenced -- all the recognised view needs.

    Unlike the road-aligned outputs (which also need road centrelines, the
    intersection and the true impact point), the raw recognised projection only
    needs a GPS-anchored homography. So a freshly calibrated clip can show its
    recognised trajectory even before any road/impact geo data is added.
    """
    return bool(USING_GPS_CALIBRATION and ORIGIN_LATLON is not None)


def recognized_latlon() -> dict[str, dict[int, tuple[float, float]]]:
    """Per-vehicle ``{frame: (lat, lon)}`` of the recognised path.

    Every tracked SAM2 anchor is projected straight through the homography to
    lat/lon -- NO road snapping, and (unlike ``build_data``) NO impact-frame
    truncation or flip removal. The only adjustment is an optional rigid
    translation so the path departs from the user's step-3 start point
    (:func:`_recognized_metric`); the recognised shape is untouched.

    Returns:
        ``paths[label][frame] = (lat, lon)``.
    """
    paths: dict[str, dict[int, tuple[float, float]]] = {}
    for label, track in _recognized_metric().items():
        latlon = {}
        for frame in sorted(track):
            ll = metric_to_latlon(track[frame])
            if ll is not None:
                latlon[frame] = ll
        if latlon:
            paths[label] = latlon
    return paths


def build_reconstruction() -> dict:
    """Assemble the whole 2D reconstruction as one JSON-ready dict for a frontend.

    A single self-contained document so a 3D/web viewer (e.g. Three.js) needs no
    CSV parsing or re-projection: per-vehicle tracks (local metres ``x_m``/``z_m``
    AND lat/lon, time, speed, impact flag), the road centrelines, the impact point,
    fps and a speed-reliability hint. All positions share ONE local east/north
    metre plane (``x_m`` = east, ``z_m`` = north, origin = the homography's
    ``origin_latlon``), so vehicles, roads and the impact line up directly as
    ground-plane coordinates.

    Returns:
        ``{"scene", "ready", "fps", "impact_frame", "axes", "origin_latlon",
        "impact", "vehicles", "roads", "speed_reliability"}``. ``ready`` is False
        (with a ``reason``) when the homography is not georeferenced yet.

    Examples:
        ```python
        data = build_reconstruction()
        data["ready"] and data["vehicles"]["car"]["track"][0].keys()
        # dict_keys(['frame', 't_sec', 'x_m', 'z_m', 'lat', 'lon', 'speed_kmh',
        #            'is_impact'])
        ```
    """
    if not _calibration_ready():
        return {
            "scene": SCENE.name,
            "ready": False,
            "reason": "homography not GPS-calibrated",
        }
    paths = recognized_latlon()
    fps = float(SCENE.fps)
    impact_frame = resolve_impact_frame(SCENE)
    motion = aligned_motion(paths, fps)  # speed consistent with these positions
    origin = ORIGIN_LATLON

    def to_xz(lat: float, lon: float) -> tuple[float, float]:
        east, north = latlon_to_local_meters(
            np.array([[lat, lon]], dtype=np.float64), origin
        )[0]
        return float(east), float(north)

    vehicles: dict[str, dict] = {}
    for index, (label, track) in enumerate(paths.items()):
        display = _display_for(label, index)
        speeds = motion.get(label, {})
        samples = []
        for frame in sorted(track):
            lat, lon = track[frame]
            east, north = to_xz(lat, lon)
            samples.append(
                {
                    "frame": frame,
                    "t_sec": round(frame / fps, 4),
                    "x_m": round(east, 3),
                    "z_m": round(north, 3),
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "speed_kmh": round(speeds.get(frame, (0.0, 0.0))[1], 1),
                    "is_impact": frame == impact_frame,
                }
            )
        vehicles[label] = {
            "name": display.get("name", label),
            "color_rgb": [int(c) for c in display["rgb"]],
            "road": ROAD_NAMES.get(label, label),
            "track": samples,
        }

    roads: dict[str, list] = {}
    for label, centreline in (ROAD_CENTERLINES or {}).items():
        points = []
        for lat, lon in centreline:
            east, north = to_xz(lat, lon)
            points.append(
                {"x_m": round(east, 3), "z_m": round(north, 3), "lat": lat, "lon": lon}
            )
        roads[label] = points

    impact = None
    if TRUE_IMPACT_LATLON is not None:
        east, north = to_xz(*TRUE_IMPACT_LATLON)
        impact = {
            "frame": impact_frame,
            "lat": TRUE_IMPACT_LATLON[0],
            "lon": TRUE_IMPACT_LATLON[1],
            "x_m": round(east, 3),
            "z_m": round(north, 3),
        }

    return {
        "scene": SCENE.name,
        "ready": True,
        "fps": fps,
        "impact_frame": impact_frame,
        "axes": "x_m=east, z_m=north (metres, north-up ground plane)",
        "origin_latlon": [origin[0], origin[1]] if origin else None,
        "impact": impact,
        "vehicles": vehicles,
        "roads": roads,
        "speed_reliability": {"gcp_ground_span_m": gcp_ground_span_m()},
    }


def write_reconstruction_json(path: Path | None = None) -> Path | None:
    """Write :func:`build_reconstruction` to ``<scene>_reconstruction.json``.

    Args:
        path: Output path; defaults to the scene's ``<name>_reconstruction.json``
            next to the other route outputs.

    Returns:
        The path written, or None when the scene is not calibrated yet.
    """
    data = build_reconstruction()
    if not data.get("ready"):
        return None
    path = path or SCENE.out_csv.with_name(f"{SCENE.name}_reconstruction.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_recognized_figure(figure_path: Path | None = None) -> Path | None:
    """Render the raw recognised trajectories, auto-zoomed to their real extent."""
    if not _calibration_ready():
        print("No GPS-anchored calibration; skipping recognised figure.")
        return None
    figure_path = figure_path or SCENE.out_figure.with_name(
        f"{SCENE.name}_route_recognized.png"
    )
    paths = recognized_latlon()
    if not paths:
        print("No recognised path to draw.")
        return None

    # Auto-fit the view to the path (+ impact) so a small compressed path is still
    # visible, instead of the fixed road-scale view the other figures use.
    lats = [lat for track in paths.values() for lat, _ in track.values()]
    lons = [lon for track in paths.values() for _, lon in track.values()]
    # Centre on the true impact when known; otherwise on the path centroid (a clip
    # that is only calibrated, with no impact point set yet, still draws fine).
    if TRUE_IMPACT_LATLON is not None:
        clat, clon = TRUE_IMPACT_LATLON
        lats.append(clat)
        lons.append(clon)
    else:
        clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    mid_lat = sum(lats) / len(lats)
    m_lat = METERS_PER_DEG_LAT
    m_lon = METERS_PER_DEG_LAT * math.cos(math.radians(mid_lat))

    xs = [(lon - clon) * m_lon for lon in lons]
    ys = [(lat - clat) * m_lat for lat in lats]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0)
    size, pad = 880, 70
    scale = (size - 2 * pad) / (span * 1.25)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    to_px = MapProjection(
        clat=clat, clon=clon, m_lon=m_lon, scale=scale, size=size, cx=cx, cy=cy
    ).to_px

    image = Image.new("RGB", (size, size), (245, 246, 248))
    draw = ImageDraw.Draw(image)

    # Faint road context underneath.
    road_w = max(2, int(SCENE.road_width_m * scale))
    for label in ROAD_CENTERLINES:
        pts = [to_px(ll) for ll in ROAD_CENTERLINES[label]]
        if len(pts) >= 2:
            draw.line(pts, fill=(210, 214, 221), width=road_w, joint="curve")
    for label, centerline in ROAD_CENTERLINES.items():
        mid = centerline[len(centerline) // 2]
        _label(draw, to_px(mid), ROAD_NAMES.get(label, label), 18, (170, 176, 186))

    # The recognised paths, exactly as projected (markers at every frame). Iterate
    # the actual tracked labels (not just VEHICLE_DISPLAY) so clips without scene
    # styling still draw, using the fallback palette.
    for index, label in enumerate(sorted(paths)):
        display = _display_for(label, index)
        track = paths.get(label)
        if not track or len(track) < 2:
            continue
        rgb = display["rgb"]
        frames = sorted(track)
        pts = [to_px(track[f]) for f in frames]
        draw.line(pts, fill=rgb, width=3, joint="curve")
        for px, py in pts:  # every recognised sample, so the real shape is visible
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=rgb)
        sx, sy = pts[0]
        draw.ellipse(
            [sx - 6, sy - 6, sx + 6, sy + 6], fill=(255, 255, 255), outline=rgb, width=2
        )
        ll0, ll1 = track[frames[0]], track[frames[-1]]
        travelled = math.hypot((ll1[1] - ll0[1]) * m_lon, (ll1[0] - ll0[0]) * m_lat)
        _label(
            draw,
            (sx + 9, sy - 22),
            f"{display['name']} 辨識軌跡 {len(frames)}幀 ~{travelled:.1f}m",
            16,
            rgb,
        )

    if TRUE_IMPACT_LATLON is not None:  # only when an impact point has been set
        ix, iy = to_px(TRUE_IMPACT_LATLON)
        draw.line([(ix - 12, iy), (ix + 12, iy)], fill=(214, 51, 51), width=3)
        draw.line([(ix, iy - 12), (ix, iy + 12)], fill=(214, 51, 51), width=3)
        draw.ellipse(
            [ix - 15, iy - 15, ix + 15, iy + 15], outline=(214, 51, 51), width=2
        )
        _label(draw, (ix + 14, iy + 8), "撞擊點", 14, (214, 51, 51))

    bar_m = 5.0
    bx, by = pad, size - pad
    draw.line([(bx, by), (bx + bar_m * scale, by)], fill=(60, 60, 60), width=3)
    _label(draw, (bx, by - 20), f"{bar_m:.0f} m", 14, (60, 60, 60))
    _label(
        draw,
        (12, 12),
        "模型辨識軌跡(原始投影,未貼路;比例尺見左下)",
        16,
        (60, 60, 60),
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(figure_path))
    return figure_path


def write_recognized_kml(kml_path: Path | None = None) -> Path | None:
    """Write the fully-raw recognised trajectories as KML (no road snapping)."""
    if not _calibration_ready():
        print("No GPS-anchored calibration; skipping recognised KML.")
        return None
    kml_path = kml_path or SCENE.out_kml.with_name(f"{SCENE.name}_route_recognized.kml")
    paths = recognized_latlon()
    placemarks = ""
    for index, label in enumerate(sorted(paths)):
        display = _display_for(label, index)
        track = paths.get(label)
        if not track:
            continue
        r, g, b = display["rgb"]
        color = f"ff{b:02x}{g:02x}{r:02x}"  # KML aabbggrr
        road = ROAD_NAMES.get(label, label)
        coords = [track[f] for f in sorted(track)]
        placemarks += _kml_linestring(
            f"{display['name']} ({road}) 辨識軌跡", color, coords
        )
    if TRUE_IMPACT_LATLON is not None:  # only when an impact point has been set
        clat, clon = TRUE_IMPACT_LATLON
        placemarks += (
            f"  <Placemark><name>撞擊點(地圖判讀)</name>"
            f"<Point><coordinates>{clon:.7f},{clat:.7f},0</coordinates></Point></Placemark>\n"
        )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
        "  <name>模型辨識軌跡(原始投影,未貼路)</name>\n"
        f"{placemarks}</Document>\n</kml>\n"
    )
    kml_path.parent.mkdir(parents=True, exist_ok=True)
    kml_path.write_text(kml, encoding="utf-8")
    return kml_path


def write_recognized_csv(csv_path: Path | None = None) -> Path | None:
    """Write the fully-raw recognised path as CSV (the web map plots this).

    Columns ``frame,vehicle,lat,lon,speed_kmh,is_impact`` -- the same schema the
    aligned ``write_csv`` uses, so the interactive map and metric cards read it
    unchanged, but the positions are the raw recognised projection (no snapping).
    Speed is the windowed metric speed; ``is_impact`` marks the contact frame.
    """
    if not _calibration_ready():
        print("No GPS-anchored calibration; skipping recognised CSV.")
        return None
    csv_path = csv_path or SCENE.out_csv.with_name(f"{SCENE.name}_route_recognized.csv")
    # Positions follow the (start-anchored) recognised path; speed and the impact
    # frame come from the ORIGINAL projection -- a per-vehicle translation does not
    # change speeds, and impact detection needs the untranslated inter-vehicle
    # geometry (each vehicle is shifted independently to its own start).
    original = project_metric(load_anchors(SCENE.prompt_tracks_csv))
    motion = {label: windowed_motion(track) for label, track in original.items()}
    impact_frame = resolve_impact_frame(SCENE, original)
    lines = [ROUTE_CSV_HEADER]
    for label, track in _recognized_metric().items():
        for frame in sorted(track):
            ll = metric_to_latlon(track[frame])
            if ll is None:
                continue
            speed = motion[label][frame][1] if frame in motion.get(label, {}) else 0.0
            lines.append(route_csv_row(frame, label, ll[0], ll[1], speed, impact_frame))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def main() -> None:
    """Build and write the recognised-trajectory figure + KML + CSV; report extents."""
    figure = write_recognized_figure()
    kml_path = write_recognized_kml()
    csv_path = write_recognized_csv()
    paths = recognized_latlon()
    for label, track in paths.items():
        frames = sorted(track)
        ll0, ll1 = track[frames[0]], track[frames[-1]]
        m_lon = METERS_PER_DEG_LAT * math.cos(math.radians(ll0[0]))
        travelled = math.hypot(
            (ll1[1] - ll0[1]) * m_lon, (ll1[0] - ll0[0]) * METERS_PER_DEG_LAT
        )
        print(
            f"  {label}: {len(frames)} frames, "
            f"recognised straight extent ~{travelled:.1f} m"
        )
    if figure:
        print(f"Recognised figure: {figure.resolve()}")
    if kml_path:
        print(f"Recognised KML: {kml_path.resolve()}")
    if csv_path:
        print(f"Recognised CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
