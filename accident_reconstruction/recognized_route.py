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

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from accident_reconstruction.auto_reconstruct import (
    detect_impact,
    load_anchors,
    project_metric,
    windowed_motion,
)
from accident_reconstruction.birdseye_manual_annotation import (
    ORIGIN_LATLON,
    ROAD_CENTERLINES,
    ROAD_NAMES,
    TRUE_IMPACT_LATLON,
    USING_GPS_CALIBRATION,
    VEHICLE_DISPLAY,
    _kml_linestring,
    _label,
)
from accident_reconstruction.calibrate_homography import (
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


def _display_for(label: str, index: int) -> dict:
    """Display name + RGB for a label, from the scene or a fallback palette."""
    info = VEHICLE_DISPLAY.get(label)
    if info:
        return info
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
    m_lat = 111195.0
    m_lon = 111195.0 * math.cos(math.radians(mid_lat))

    xs = [(lon - clon) * m_lon for lon in lons]
    ys = [(lat - clat) * m_lat for lat in lats]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0)
    size, pad = 880, 70
    scale = (size - 2 * pad) / (span * 1.25)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    def to_px(latlon: tuple[float, float]) -> tuple[float, float]:
        x = (latlon[1] - clon) * m_lon
        y = (latlon[0] - clat) * m_lat
        return (size / 2 + (x - cx) * scale, size / 2 - (y - cy) * scale)

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
    impact_frame = SCENE.impact_frame_override
    if impact_frame is None:
        impact_frame = detect_impact(original)
    lines = ["frame,vehicle,lat,lon,speed_kmh,is_impact"]
    for label, track in _recognized_metric().items():
        for frame in sorted(track):
            ll = metric_to_latlon(track[frame])
            if ll is None:
                continue
            speed = motion[label][frame][1] if frame in motion.get(label, {}) else 0.0
            lines.append(
                f"{frame},{label},{ll[0]:.7f},{ll[1]:.7f},{speed:.1f},"
                f"{int(frame == impact_frame)}"
            )
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
        m_lon = 111195.0 * math.cos(math.radians(ll0[0]))
        travelled = math.hypot((ll1[1] - ll0[1]) * m_lon, (ll1[0] - ll0[0]) * 111195.0)
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
