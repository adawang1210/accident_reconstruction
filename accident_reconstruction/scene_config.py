"""Per-scene configuration for the accident-reconstruction pipeline.

Everything scene-specific lives here so the same code runs on any clip. Select
the active scene with the ``ACCIDENT_SCENE`` environment variable (defaults to the
original 永康 motorcycle clip), e.g.::

    ACCIDENT_SCENE=keelung_xinwu_yier \
        .venv/bin/python -m accident_reconstruction.run_pipeline

The stage modules read :data:`SCENE` for paths, and ``run_pipeline`` injects the
geo fields (road centrelines, true anchors, vehicle styling) into the writers, so
adding a clip is "add a SceneConfig here" -- no edits in the stage modules.

Examples:
    ```python
    from accident_reconstruction.scene_config import SCENES
    SCENES["pre_impact_motorcycle"].calibration_path.name
    # 'homography_calibration.json'
    ```
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

#: All clips and rendered videos live together here (sources + pipeline outputs),
#: instead of being scattered across each scene's data folder.
VIDEO_DIR = Path("data/videos")


@dataclass(frozen=True)
class SceneConfig:
    """All inputs and output locations for one accident clip.

    Attributes:
        name: Scene id; also the artifact subfolder and output filename stem.
        source_video: Input clip path.
        artifact_dir: Folder for per-scene artifacts (calibration, GCPs, tracks).
        start_frame: First frame of the analysis window.
        end_frame: Last frame (inclusive).
        fps: Source frames per second.
        weights: SAM2 weights file for the tracker.
        youtube_url: Source YouTube URL of the clip, if downloaded. Used to look up
            a committed calibration record (``scene_records/``) so a re-downloaded
            clip is auto-calibrated. See :mod:`accident_reconstruction.scene_records`.
        init_vehicles: ``{label: {"box": [x1,y1,x2,y2], "bgr": (b,g,r), "frame"?}}``
            default vehicle prompts (overridden by ``select_vehicles.py`` output).
        vehicle_display: ``{label: {"name": str, "rgb": (r,g,b)}}`` figure styling.
        road_names: ``{label: road name}`` for KML / labels.
        road_centerlines: ``{label: [(lat, lon), ...]}`` OSM centreline per road.
        intersection_latlon: ``(lat, lon)`` of the shared intersection node.
        true_impact_latlon: ``(lat, lon)`` of the impact, read off the basemap.
        true_car_start_latlon: ``(lat, lon)`` of the second vehicle's start.
        stop_vehicle: Label of the vehicle that stops at impact (its SAM2 mask
            merges with the other afterwards), truncated at the impact frame.
        moving_vehicle: Label of the vehicle whose start anchors to
            ``true_car_start_latlon`` in the two-point alignment (the one that
            keeps moving through the collision). Defaults to ``"car"`` downstream.
        true_vehicle_starts: ``{label: (lat, lon)}`` real start position of each
            vehicle (read off the basemap at its first tracked frame). When set,
            the alignment scales that vehicle's recognised path so its start lands
            on this point (translation + rotation + uniform scale), countering the
            fisheye homography's distance compression. Without it a vehicle keeps
            the rotation-only road alignment (shape/length preserved, may be short).
    """

    name: str
    source_video: Path
    artifact_dir: Path
    start_frame: int
    end_frame: int
    fps: float = 25.0
    weights: str = "sam2.1_t.pt"
    road_width_m: float = 5.6
    youtube_url: str | None = None
    stop_vehicle: str | None = None
    moving_vehicle: str | None = None
    init_vehicles: dict = field(default_factory=dict)
    vehicle_display: dict = field(default_factory=dict)
    road_names: dict = field(default_factory=dict)
    road_centerlines: dict | None = None
    intersection_latlon: tuple[float, float] | None = None
    true_impact_latlon: tuple[float, float] | None = None
    true_car_start_latlon: tuple[float, float] | None = None
    true_vehicle_starts: dict | None = None

    # --- shared video folder paths ------------------------------------------
    def video_path(self, suffix: str) -> Path:
        """A rendered output video for this scene, under the shared video folder.

        Args:
            suffix: The output kind, e.g. ``"prompt_tracked"`` ->
                ``data/videos/<name>_prompt_tracked.mp4``.

        Examples:
            ```python
            SCENES["pre_impact_motorcycle"].video_path("prompt_tracked").name
            # 'pre_impact_motorcycle_prompt_tracked.mp4'
            ```
        """
        return VIDEO_DIR / f"{self.name}_{suffix}.mp4"

    @property
    def prompt_tracked_video(self) -> Path:
        """SAM2-prompt-tracked overlay video (written by ``prompt_track_accident``)."""
        return self.video_path("prompt_tracked")

    # --- derived per-scene paths --------------------------------------------
    @property
    def calibration_path(self) -> Path:
        """Homography calibration JSON written by ``calibrate_homography``."""
        return self.artifact_dir / "homography_calibration.json"

    @property
    def gcp_store(self) -> Path:
        """Per-scene ground-control-point store (pixel + lat/lon pairs)."""
        return self.artifact_dir / "gcps.json"

    @property
    def vehicle_boxes(self) -> Path:
        """Vehicle prompt boxes written by ``select_vehicles``."""
        return self.artifact_dir / "vehicle_boxes.json"

    @property
    def prompt_tracks_csv(self) -> Path:
        """Per-frame SAM2 anchors written by ``prompt_track_accident``."""
        return self.artifact_dir / "prompt_tracks.csv"

    @property
    def out_kml(self) -> Path:
        return self.artifact_dir.parent / f"{self.name}_route_auto.kml"

    @property
    def out_figure(self) -> Path:
        return self.artifact_dir.parent / f"{self.name}_map_figure_auto.png"

    @property
    def out_csv(self) -> Path:
        return self.artifact_dir.parent / f"{self.name}_route_auto.csv"

    @property
    def is_geo_ready(self) -> bool:
        """True when the scene has the geo data needed for the map figure/KML.

        The per-vehicle road alignment needs road centrelines, the intersection,
        and the true impact point. ``true_car_start_latlon`` is no longer required
        (each vehicle now orients to its own road, not a single car-start anchor).
        """
        return None not in (
            self.road_centerlines,
            self.intersection_latlon,
            self.true_impact_latlon,
        )

    # --- per-scene user overrides (written by the web workbench) -------------
    @property
    def overrides_path(self) -> Path:
        """Per-scene user settings the UI can edit (impact frame, roles, gates)."""
        return self.artifact_dir / "overrides.json"

    @property
    def overrides(self) -> dict:
        """Load ``overrides.json`` (empty when absent/unreadable).

        Keys (all optional): ``impact_frame`` (int), ``stop_vehicle`` (str),
        ``moving_vehicle`` (str), ``gates`` ("strict"|"loose"|"off").
        """
        path = self.overrides_path
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @property
    def impact_frame_override(self) -> int | None:
        """Manual impact frame from the UI, or None to auto-detect."""
        value = self.overrides.get("impact_frame")
        return int(value) if value not in (None, "") else None

    @property
    def resolved_stop_vehicle(self) -> str | None:
        """UI ``stop_vehicle`` if set, else the scene default."""
        return self.overrides.get("stop_vehicle") or self.stop_vehicle

    @property
    def resolved_moving_vehicle(self) -> str | None:
        """UI ``moving_vehicle`` if set, else the scene default."""
        return self.overrides.get("moving_vehicle") or self.moving_vehicle

    @property
    def gate_mode(self) -> str:
        """Tracking gate strictness: 'strict' (default), 'loose', or 'off'."""
        return self.overrides.get("gates") or "strict"

    @property
    def show_struck_full(self) -> bool:
        """If true, keep the struck vehicle's FULL on-ground path (don't truncate
        it at the impact frame) so its post-impact push -- e.g. shoved into a shop
        -- is shown. The genuine tumble is still dropped by the flip detector."""
        return bool(self.overrides.get("struck_full"))

    @property
    def resolved_start_frame(self) -> int:
        """UI ``start_frame`` (analysis-window first frame) if set, else scene."""
        value = self.overrides.get("start_frame")
        return int(value) if value not in (None, "") else self.start_frame

    @property
    def resolved_end_frame(self) -> int:
        """UI ``end_frame`` (analysis-window last frame) if set, else scene."""
        value = self.overrides.get("end_frame")
        return int(value) if value not in (None, "") else self.end_frame

    @property
    def min_traj_speed_kmh(self) -> float:
        """Speed (km/h) below which a vehicle is considered stopped, so its
        trajectory line stops being drawn (the box/marker can stay). The UI can
        override ``min_traj_speed``; 0 disables the speed cutoff."""
        value = self.overrides.get("min_traj_speed")
        return float(value) if value not in (None, "") else 3.0

    @property
    def resolved_true_impact_latlon(self) -> tuple[float, float] | None:
        """UI ``true_impact_latlon`` override (collision point) if set, else scene."""
        value = self.overrides.get("true_impact_latlon")
        if value and len(value) == 2:
            return (float(value[0]), float(value[1]))
        return self.true_impact_latlon

    @property
    def resolved_true_vehicle_starts(self) -> dict:
        """Per-vehicle real start points: scene defaults merged with UI overrides.

        ``{label: (lat, lon)}`` where each vehicle's first-frame position was read
        off the basemap. The override entries win, so the UI can add/replace a
        single vehicle without restating the others.
        """
        merged: dict = dict(self.true_vehicle_starts or {})
        for label, value in (self.overrides.get("true_vehicle_starts") or {}).items():
            if value and len(value) == 2:
                merged[label] = (float(value[0]), float(value[1]))
        return merged

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (for persisting dynamic scenes)."""

        def latlon(value):
            return list(value) if value else None

        return {
            "name": self.name,
            "source_video": str(self.source_video),
            "artifact_dir": str(self.artifact_dir),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": self.fps,
            "weights": self.weights,
            "youtube_url": self.youtube_url,
            "stop_vehicle": self.stop_vehicle,
            "moving_vehicle": self.moving_vehicle,
            "init_vehicles": self.init_vehicles,
            "vehicle_display": self.vehicle_display,
            "road_names": self.road_names,
            "road_centerlines": self.road_centerlines,
            "intersection_latlon": latlon(self.intersection_latlon),
            "true_impact_latlon": latlon(self.true_impact_latlon),
            "true_car_start_latlon": latlon(self.true_car_start_latlon),
            "true_vehicle_starts": self.true_vehicle_starts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneConfig:
        """Rebuild a SceneConfig from :meth:`to_dict` output."""

        def latlon(value):
            return tuple(value) if value else None

        return cls(
            name=data["name"],
            source_video=Path(data["source_video"]),
            artifact_dir=Path(data["artifact_dir"]),
            start_frame=int(data.get("start_frame", 0)),
            end_frame=int(data.get("end_frame", 0)),
            fps=float(data.get("fps", 25.0)),
            weights=data.get("weights", "sam2.1_t.pt"),
            youtube_url=data.get("youtube_url"),
            stop_vehicle=data.get("stop_vehicle"),
            moving_vehicle=data.get("moving_vehicle"),
            init_vehicles=data.get("init_vehicles") or {},
            vehicle_display=data.get("vehicle_display") or {},
            road_names=data.get("road_names") or {},
            road_centerlines=data.get("road_centerlines"),
            intersection_latlon=latlon(data.get("intersection_latlon")),
            true_impact_latlon=latlon(data.get("true_impact_latlon")),
            true_car_start_latlon=latlon(data.get("true_car_start_latlon")),
            true_vehicle_starts=data.get("true_vehicle_starts"),
        )


PRE_IMPACT_MOTORCYCLE = SceneConfig(
    name="pre_impact_motorcycle",
    source_video=Path("data/videos/pre_impact_motorcycle_source.mp4"),
    artifact_dir=Path("data/scenes/pre_impact_motorcycle/scene"),
    start_frame=80,
    end_frame=180,
    fps=25.0,
    init_vehicles={
        "motorcycle": {"box": [225, 335, 270, 398], "bgr": (0, 196, 255)},
        "car": {"box": [699, 649, 947, 694], "bgr": (240, 140, 40)},
    },
    vehicle_display={
        "motorcycle": {"name": "機車", "rgb": (255, 196, 0)},
        "car": {"name": "汽車", "rgb": (40, 140, 240)},
    },
    road_names={"motorcycle": "高速一街二段", "car": "自強路"},
    road_centerlines={
        "motorcycle": [
            (23.0247826, 120.2497741),
            (23.0256827, 120.2496670),
            (23.0268405, 120.2496047),
            (23.0277663, 120.2495920),
            (23.0287256, 120.2496005),
        ],
        "car": [
            (23.0278193, 120.2509967),
            (23.0276197, 120.2507594),
            (23.0272222, 120.2501418),
            (23.0271996, 120.2501042),
            (23.0270726, 120.2499276),
            (23.0268405, 120.2496047),
            (23.0267880, 120.2495188),
            (23.0266758, 120.2493529),
            (23.0265350, 120.2491486),
        ],
    },
    intersection_latlon=(23.0268405, 120.2496047),
    true_impact_latlon=(23.026871, 120.249608),
    true_car_start_latlon=(23.026900, 120.249650),
    stop_vehicle="motorcycle",
    moving_vehicle="car",
)


# 基隆市中正區 信五路 與 義二路口: a yellow taxi (left->right) T-bones a police car
# that flips over. Two tracked vehicles -- the moving taxi (striker) and the struck
# police car (stops/flips at impact). Vehicle identities confirmed from the clip;
# the two true anchors below are homography-projected estimates (keelung calibration
# residual ~0.7 m), refine off the basemap if the aligned map looks rotated.
KEELUNG_XINWU_YIER = SceneConfig(
    name="keelung_xinwu_yier",
    source_video=Path("data/videos/keelung_xinwu_yier_source.mp4"),
    artifact_dir=Path("data/scenes/keelung_xinwu_yier/scene"),
    start_frame=120,
    end_frame=245,
    fps=29.0,
    # Defaults; vehicle_boxes.json (web workbench / drawn here) overrides these.
    init_vehicles={
        "taxi": {"frame": 130, "box": [12, 205, 238, 298], "bgr": (0, 193, 255)},
        "police_car": {"frame": 148, "box": [268, 315, 562, 478], "bgr": (243, 90, 33)},
    },
    vehicle_display={
        "taxi": {"name": "計程車", "rgb": (255, 193, 7)},
        "police_car": {"name": "警車", "rgb": (33, 90, 243)},
    },
    road_names={"taxi": "義二路", "police_car": "信五路"},
    # OSM centrelines through the intersection (overpass, 2026-06). Taxi travels
    # roughly east on 義二路; the police car crosses north on 信五路 -- swap if the
    # aligned map shows them on the wrong roads.
    road_centerlines={
        "police_car": [  # 信五路
            (25.1336137, 121.7483291),
            (25.1338772, 121.7478537),
            (25.1341019, 121.7474411),
            (25.1343264, 121.7470368),
            (25.1346359, 121.7464735),
        ],
        "taxi": [  # 義二路
            (25.1331847, 121.7468295),
            (25.1336386, 121.7471322),
            (25.1341019, 121.7474411),
            (25.1344215, 121.7476538),
            (25.1350084, 121.74803),
        ],
    },
    intersection_latlon=(25.1341019, 121.7474411),
    # Collision point read off the basemap by the user.
    true_impact_latlon=(25.1341166, 121.7474306),
    true_car_start_latlon=(25.1341163, 121.7474077),
    # Real start positions read off the basemap (taxi@f120 entering from the west,
    # police@f140 from the south) -> per-vehicle scale alignment stretches the
    # fisheye-compressed paths back to true length along each road.
    true_vehicle_starts={
        "taxi": (25.1340989, 121.7474192),
        "police_car": (25.1340959, 121.7474648),
    },
    stop_vehicle="police_car",
    moving_vehicle="taxi",
)


SCENES: dict[str, SceneConfig] = {
    PRE_IMPACT_MOTORCYCLE.name: PRE_IMPACT_MOTORCYCLE,
    KEELUNG_XINWU_YIER.name: KEELUNG_XINWU_YIER,
}

DATA_ROOT = Path("data")


def discover_scenes() -> None:
    """Register dynamic scenes persisted on disk as ``data/**/scene.json``.

    The web workbench writes a ``scene.json`` when the user picks/downloads a clip
    that has no built-in scene, so downloaded clips can be calibrated and run. The
    curated scenes above take precedence (a disk scene never overrides them).
    """
    if not DATA_ROOT.exists():
        return
    for path in sorted(DATA_ROOT.rglob("scene.json")):
        try:
            scene = SceneConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
        SCENES.setdefault(scene.name, scene)


discover_scenes()

#: The active scene, selected by the ``ACCIDENT_SCENE`` env var.
SCENE: SceneConfig = SCENES.get(
    os.environ.get("ACCIDENT_SCENE", PRE_IMPACT_MOTORCYCLE.name),
    PRE_IMPACT_MOTORCYCLE,
)
