"""End-to-end accident-reconstruction pipeline (one command, one config).

Ties together the working stages into a single runnable pipeline:

    (one-time)  calibrate_homography.py  ->  homography_calibration.json
    stage 1     prompt-track the user-specified vehicles (SAM2 video memory)
    stage 2     project to the ground plane, reconstruct, two-point map alignment
                ->  aligned KML / north-up map figure / per-frame CSV

Everything scene-specific lives in one ``SceneConfig`` below: the input clip, the
user's vehicle init boxes, the frame window, and the two true map anchor points
(impact + car start) used to remove the residual offset/rotation. To run another
clip, copy the config and re-point it -- no code changes in the stage modules.

Prerequisite: run ``calibrate_homography.py`` once for the scene (it writes
``homography_calibration.json``, loaded by the projection module on import).

Example:
    ```bash
    .venv/bin/python -m accident_reconstruction.run_pipeline
    ```
"""

from __future__ import annotations

import json
from pathlib import Path

import accident_reconstruction.auto_reconstruct as reconstruct
import accident_reconstruction.calibrate_homography as calibrate
import accident_reconstruction.prompt_track_accident as track
from accident_reconstruction.scene_config import SCENE, SceneConfig


def load_init_vehicles(vehicle_boxes_path: Path) -> dict:
    """Load saved boxes into the tracker's ``{name: {frame, box, bgr}}`` form.

    Two on-disk shapes are accepted:

    * **Workbench format** ``{"objects": [{"name", "bgr", "boxes": [{"frame",
      "box"}, ...]}, ...]}`` -- an object may be boxed on several frames; the
      whole frame-sorted ``boxes`` list is kept so the tracker can re-seed at
      each one, and ``frame``/``box`` mirror the earliest box (first appearance).
    * **Legacy flat format** ``{name: {"frame"?, "box", "bgr"}}`` -- one box each.

    Args:
        vehicle_boxes_path: Path to the scene's ``vehicle_boxes.json``.

    Returns:
        ``{name: {"frame"?, "box", "bgr", "boxes"?}}`` for the tracker.

    Examples:
        ```python
        import json, tempfile
        from pathlib import Path
        data = {"objects": [{"name": "motorcycle", "bgr": [0, 196, 255],
                             "boxes": [{"frame": 50, "box": [2, 2, 8, 8]},
                                       {"frame": 10, "box": [1, 1, 5, 5]}]}]}
        p = Path(tempfile.mkdtemp()) / "vehicle_boxes.json"
        _ = p.write_text(json.dumps(data))
        sorted(load_init_vehicles(p)["motorcycle"]["boxes"][0].items())
        # [('box', [1, 1, 5, 5]), ('frame', 10)]
        ```
    """
    loaded = json.loads(vehicle_boxes_path.read_text())
    objects = loaded.get("objects") if isinstance(loaded, dict) else None
    if isinstance(objects, list):
        init: dict = {}
        for obj in objects:
            raw = obj.get("boxes") or []
            if not raw:
                continue
            boxes = sorted(
                ({"frame": int(b.get("frame", 0)), "box": b["box"]} for b in raw),
                key=lambda b: b["frame"],
            )
            init[obj["name"]] = {
                "frame": boxes[0]["frame"],
                "box": boxes[0]["box"],
                "bgr": tuple(obj["bgr"]),
                "boxes": boxes,
            }
        return init
    return {
        name: {
            **({"frame": spec["frame"]} if "frame" in spec else {}),
            "box": spec["box"],
            "bgr": tuple(spec["bgr"]),
        }
        for name, spec in loaded.items()
    }


def run(config: SceneConfig = SCENE) -> None:
    """Run the full pipeline for one scene config.

    Args:
        config: The scene to reconstruct (defaults to the active scene from the
            ``ACCIDENT_SCENE`` env var).

    Raises:
        SystemExit: If no homography calibration has been produced for the scene.
    """
    if not calibrate.USING_GPS_CALIBRATION:
        raise SystemExit(
            f"No calibration at {config.calibration_path} -- run "
            f"ACCIDENT_SCENE={config.name} python accident_reconstruction/"
            "calibrate_homography.py first."
        )

    # Vehicle boxes drawn in the web workbench / select_vehicles.py override the
    # config default.
    init_vehicles = dict(config.init_vehicles)
    if config.vehicle_boxes.exists():
        init_vehicles = load_init_vehicles(config.vehicle_boxes)
        print(f"Using user-selected vehicle boxes from {config.vehicle_boxes}")
    track.INIT_VEHICLES = init_vehicles

    print("[1/2] Prompt-tracking the specified vehicles with SAM2 video memory ...")
    track.main(
        source_video_path=str(config.source_video),
        start_frame=config.resolved_start_frame,
        end_frame=config.resolved_end_frame,
        weights=config.weights,
    )

    print("[2/2] Projecting, reconstructing, and aligning to the real roads ...")
    reconstruct.main()

    print("\nPipeline complete. Overlay the KML on Google My Maps to review.")


if __name__ == "__main__":
    run()
