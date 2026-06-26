"""Web workbench: pick / download a clip (left) beside a Google map (right).

FastAPI is used (not the stdlib server of ``calibrate_web``) because video
playback needs HTTP Range support for seeking, which ``FileResponse`` provides
for free. The left pane lists the clips already under ``data/`` and can download a
new one from a YouTube URL into a named folder; the right pane is a Google map
centred on the active scene.

Run::

    .venv/bin/python -m accident_reconstruction.web_app   # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import yt_dlp
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accident_reconstruction.calibrate_homography import (
    build_calibration,
    load_gcps,
    save_gcps,
)
from accident_reconstruction.scene_config import SCENE, SCENES, VIDEO_DIR, SceneConfig


def _register_dynamic_scene(relpath: str):
    """Build + persist a scene for a clip that has no built-in SceneConfig.

    Lets a freshly downloaded/selected clip be calibrated and run: a minimal scene
    (name from its folder, frame window from the video) is added to ``SCENES`` and
    written to ``data/<folder>/scene.json`` so the run_pipeline subprocess (which
    re-imports scene_config) can load it via ``ACCIDENT_SCENE``. Geo outputs (map
    figure / KML) still need road centrelines + true anchors added later.

    Args:
        relpath: Clip path under ``data/``.

    Returns:
        The new :class:`SceneConfig`, or None if the video is missing.
    """
    video = DATA_ROOT / relpath
    if not video.is_file():
        return None
    parent = video.parent
    # All clips now live together in data/videos/, so the scene name comes from the
    # video FILE (not its folder, which would be the shared "videos") and artifacts
    # get their own per-clip folder data/<name>/ -- keeping videos consolidated while
    # each clip still has a private place for calibration / tracks / scene.json.
    if parent.resolve() == VIDEO_DIR.resolve():
        name = re.sub(r"[^\w]+", "_", video.stem).strip("_") or "clip"
        artifact_parent = DATA_ROOT / name
    else:  # legacy: a clip kept in its own data/<folder>/
        rel = parent.relative_to(DATA_ROOT) if parent != DATA_ROOT else Path(video.stem)
        name = re.sub(r"[^\w]+", "_", str(rel)).strip("_") or "clip"
        artifact_parent = parent
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    scene = SceneConfig(
        name=name,
        source_video=video,
        artifact_dir=artifact_parent / "scene",
        start_frame=0,
        end_frame=max(0, frames - 1),
        fps=float(fps),
    )
    SCENES[scene.name] = scene
    scene.artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_parent / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return scene


def _scene_for_video(relpath: str):
    """Return the scene for clip ``relpath`` (under data/), creating one if needed.

    A built-in SceneConfig whose ``source_video`` matches wins; otherwise a dynamic
    scene is registered so any selected/downloaded clip is usable.
    """
    target = (DATA_ROOT / relpath).resolve()
    for scene in SCENES.values():
        if scene.source_video.resolve() == target:
            return scene
    return _register_dynamic_scene(relpath)


DATA_ROOT = Path("data")
PAGE = Path(__file__).with_name("web_app.html")
# Clips listed for selection exclude pipeline OUTPUT videos (these name fragments).
OUTPUT_MARKERS = (
    "route",
    "figure",
    "tracked",
    "annotation",
    "birdseye",
    "summary",
    "preview",
    "montage",
    "trajectory",
    "_cmp",
    "scene_replay",
)

app = FastAPI()


class DownloadRequest(BaseModel):
    """A YouTube download request from the front-end.

    ``folder`` defaults to the shared ``videos`` folder so all clips land together
    (the UI no longer asks for a folder name); each clip still gets its own
    artifact folder via :func:`_register_dynamic_scene`.
    """

    url: str
    folder: str = "videos"
    start: float | None = None
    end: float | None = None


def _ffmpeg_location() -> str | None:
    """Find an ffmpeg dir for yt-dlp (PATH first, then common installs)."""
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)
    for candidate in (
        Path.home() / "miniconda3/bin/ffmpeg",
        Path.home() / "anaconda3/bin/ffmpeg",
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ):
        if candidate.exists():
            return str(candidate.parent)
    return None


def download_youtube(request: DownloadRequest) -> Path:
    """Download a (optionally trimmed) YouTube clip into ``data/<folder>``.

    Args:
        request: The validated download request.

    Returns:
        Path to the downloaded mp4 (relative to ``data/``).

    Raises:
        RuntimeError: If no mp4 results from the download.
    """
    folder = re.sub(r"[^\w./-]", "_", request.folder).strip("/") or "videos"
    dest_dir = DATA_ROOT / folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    options: dict = {
        "format": (
            "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        ),
        "outtmpl": str(dest_dir / "%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    ffmpeg = _ffmpeg_location()
    if ffmpeg:
        options["ffmpeg_location"] = ffmpeg
    if request.start is not None and request.end is not None:
        options["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(request.start, request.end)]
        )
        options["force_keyframes_at_cuts"] = True

    before = {p for p in dest_dir.glob("*.mp4")}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.extract_info(request.url, download=True)
    new = sorted(set(dest_dir.glob("*.mp4")) - before, key=lambda p: p.stat().st_mtime)
    if not new:
        new = sorted(dest_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError("download produced no mp4 file")
    return new[-1].relative_to(DATA_ROOT)


@app.get("/")
def index() -> HTMLResponse:
    """Serve the page with the scene name, map centre, and source clip injected."""
    center = list(SCENE.intersection_latlon or (23.5, 121.0))
    try:
        source_rel = str(SCENE.source_video.resolve().relative_to(DATA_ROOT.resolve()))
    except ValueError:
        source_rel = ""
    html = (
        PAGE.read_text(encoding="utf-8")
        .replace("__SCENE__", SCENE.name)
        .replace("__CENTER__", json.dumps(center))
        .replace("__SOURCE__", source_rel)
    )
    return HTMLResponse(html)


@app.get("/api/videos")
def list_videos() -> dict:
    """List selectable source/downloaded clips under ``data/`` (not outputs)."""
    videos = []
    for path in sorted(DATA_ROOT.rglob("*.mp4")):
        if any(marker in path.name for marker in OUTPUT_MARKERS):
            continue
        videos.append(str(path.relative_to(DATA_ROOT)))
    return {"videos": videos}


@app.get("/media/{relpath:path}")
def media(relpath: str):
    """Serve a video file with Range support (so the player can seek)."""
    target = (DATA_ROOT / relpath).resolve()
    if not str(target).startswith(str(DATA_ROOT.resolve())) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


@app.post("/api/download")
def download(request: DownloadRequest) -> dict:
    """Download a YouTube clip; return the new file path or an error."""
    try:
        return {"ok": True, "file": str(download_youtube(request))}
    except Exception as error:  # surface the reason to the UI
        return {"ok": False, "error": str(error)}


@app.get("/api/frame")
def frame(video: str, index: int = 0):
    """Return one frame of a clip as JPEG (the clickable calibration image)."""
    target = (DATA_ROOT / video).resolve()
    if not str(target).startswith(str(DATA_ROOT.resolve())) or not target.is_file():
        return JSONResponse({"error": "video not found"}, status_code=404)
    capture = cv2.VideoCapture(str(target))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    _, buffer = cv2.imencode(".jpg", image)
    return Response(buffer.tobytes(), media_type="image/jpeg")


@app.get("/api/crop")
def crop(video: str, index: int = 0, box: str = ""):
    """Return a JPEG crop of ``box`` on frame ``index`` (a marked-object thumbnail).

    Args:
        video: Clip path under ``data/``.
        index: Zero-based frame index the box was drawn on.
        box: ``"x1,y1,x2,y2"`` in original-frame pixels.
    """
    target = (DATA_ROOT / video).resolve()
    if not str(target).startswith(str(DATA_ROOT.resolve())) or not target.is_file():
        return JSONResponse({"error": "video not found"}, status_code=404)
    try:
        x1, y1, x2, y2 = (round(float(v)) for v in box.split(","))
    except ValueError:
        return JSONResponse({"error": "bad box"}, status_code=400)
    capture = cv2.VideoCapture(str(target))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    height, width = image.shape[:2]
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return JSONResponse({"error": "empty crop"}, status_code=400)
    _, buffer = cv2.imencode(".jpg", image[y1:y2, x1:x2])
    return Response(buffer.tobytes(), media_type="image/jpeg")


@app.get("/api/gcps")
def get_gcps(video: str | None = None) -> dict:
    """Return GCPs + map centre for the scene matching ``video`` (else active).

    Args:
        video: A clip path under ``data/``. The scene whose source video matches
            decides which GCP store and map centre to return; an unrecognised clip
            has no scene (empty GCPs) so switching clips clears stale points.
    """
    scene = _scene_for_video(video) if video else SCENE
    if scene is None:
        return {"scene": None, "gcps": [], "center": None}
    center = list(scene.intersection_latlon) if scene.intersection_latlon else None
    return {
        "scene": scene.name,
        "gcps": load_gcps(scene.gcp_store),
        "center": center,
    }


@app.get("/api/scene")
def scene_info(video: str | None = None) -> dict:
    """Return everything the UI needs for the clip's scene (GCPs, vehicles, …)."""
    scene = _scene_for_video(video) if video else SCENE
    if scene is None:
        return {
            "scene": None,
            "center": None,
            "gcps": [],
            "vehicles": [],
            "vehicle_boxes": {},
        }
    vehicles = []
    for key, display in (scene.vehicle_display or {}).items():
        bgr = (scene.init_vehicles.get(key) or {}).get("bgr")
        if bgr is None:
            r, g, b = display.get("rgb", (255, 255, 255))
            bgr = [b, g, r]
        vehicles.append(
            {"name": key, "display": display.get("name", key), "bgr": list(bgr)}
        )
    boxes = {}
    if scene.vehicle_boxes.exists():
        boxes = json.loads(scene.vehicle_boxes.read_text())
    return {
        "scene": scene.name,
        "center": (
            list(scene.intersection_latlon) if scene.intersection_latlon else None
        ),
        "gcps": load_gcps(scene.gcp_store),
        "vehicles": vehicles,
        "vehicle_boxes": boxes,
        "start_frame": scene.start_frame,
        "end_frame": scene.end_frame,
        "true_impact": (
            list(scene.resolved_true_impact_latlon)
            if scene.resolved_true_impact_latlon
            else None
        ),
        "true_vehicle_starts": {
            k: list(v) for k, v in scene.resolved_true_vehicle_starts.items()
        },
    }


class VehiclesRequest(BaseModel):
    """Per-vehicle prompt boxes to save for a clip's scene."""

    vehicles: dict
    video: str | None = None


@app.post("/api/vehicles")
def save_vehicles(request: VehiclesRequest) -> dict:
    """Write the user-drawn vehicle boxes to the scene's ``vehicle_boxes.json``."""
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    scene.vehicle_boxes.parent.mkdir(parents=True, exist_ok=True)
    scene.vehicle_boxes.write_text(
        json.dumps(request.vehicles, indent=2, ensure_ascii=False)
    )
    objects = request.vehicles.get("objects")
    count = len(objects) if isinstance(objects, list) else len(request.vehicles)
    return {"ok": True, "file": str(scene.vehicle_boxes), "count": count}


class CalibrateRequest(BaseModel):
    """Ground control points to save and calibrate from, for a clip's scene."""

    gcps: list[dict]
    video: str | None = None


@app.post("/api/calibrate")
def calibrate(request: CalibrateRequest) -> dict:
    """Save the GCPs and rebuild the homography; return the per-point error."""
    try:
        scene = _scene_for_video(request.video) if request.video else SCENE
        if scene is None:
            return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
        save_gcps(scene.gcp_store, request.gcps)
        if len(request.gcps) < 4:
            msg = f"需要 >= 4 點，目前 {len(request.gcps)}（已存）"
            return {"ok": False, "error": msg}
        image_size = None
        capture = (
            cv2.VideoCapture(str((DATA_ROOT / request.video).resolve()))
            if request.video
            else None
        )
        if capture is not None and capture.isOpened():
            image_size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            capture.release()
        calibration = build_calibration(request.gcps, image_size=image_size)
        scene.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        scene.calibration_path.write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False)
        )
        per_point = [
            {"name": n, "err": round(e, 2), "inlier": bool(i)}
            for n, e, i in zip(
                calibration["gcp_names"],
                calibration["residuals_m"],
                calibration["inlier_mask"],
            )
        ]
        return {
            "ok": True,
            "method": calibration["method"],
            "mean": round(calibration["mean_residual_m"], 2),
            "max": round(calibration["max_residual_m"], 2),
            "span_m": round(calibration["target_span_m"], 1),
            "span_warning": calibration["span_warning"],
            "per_point": per_point,
        }
    except Exception as error:
        return {"ok": False, "error": str(error)}


REPO_ROOT = Path(__file__).resolve().parents[1]
# One running pipeline job per scene -> {"log": deque, "done", "returncode"}.
_JOBS: dict[str, dict] = {}


def _start_job(scene, module: str) -> dict:
    """Spawn ``module`` for ``scene`` in a subprocess, streaming its log.

    A subprocess (not an in-process call) is used because the stage modules bind
    their paths from the active scene at import time; running the scene the user
    picked is therefore done by re-launching with ``ACCIDENT_SCENE`` set, exactly
    as on the CLI. Its combined stdout/stderr is collected by a reader thread.

    Args:
        scene: The :class:`SceneConfig` to reconstruct.
        module: The module to run, e.g. ``"accident_reconstruction.run_pipeline"`` (full
            pipeline) or ``"accident_reconstruction.auto_reconstruct"`` (stage 2 only).

    Returns:
        The job record (``proc``, ``log`` deque, ``done``, ``returncode``).
    """
    env = {**os.environ, "ACCIDENT_SCENE": scene.name, "PYTHONUNBUFFERED": "1"}
    # `module` is a fixed internal value (run_pipeline / auto_reconstruct), not
    # user input; argv is a list (no shell), so this is not a shell-injection risk.
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", module],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job = {
        "proc": proc,
        "log": collections.deque(maxlen=500),
        "done": False,
        "returncode": None,
    }

    def _reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            job["log"].append(line.rstrip())
        proc.wait()
        job["returncode"] = proc.returncode
        job["done"] = True

    threading.Thread(target=_reader, daemon=True).start()
    return job


def _result_files(scene) -> dict:
    """Map result kinds to their on-disk paths for ``scene`` (existing only).

    The displayed figure/KML are the RAW recognised (non-road-snapped) outputs the
    user asked for; they fall back to the road-aligned ``auto`` outputs only when a
    recognised file has not been produced yet (e.g. an older run).
    """
    recognised_figure = scene.out_figure.with_name(f"{scene.name}_route_recognized.png")
    recognised_kml = scene.out_kml.with_name(f"{scene.name}_route_recognized.kml")
    recognised_csv = scene.out_csv.with_name(f"{scene.name}_route_recognized.csv")
    candidates = {
        "figure": recognised_figure if recognised_figure.exists() else scene.out_figure,
        "kml": recognised_kml if recognised_kml.exists() else scene.out_kml,
        "csv": recognised_csv if recognised_csv.exists() else scene.out_csv,
        "tracked": scene.prompt_tracked_video,
    }
    return {kind: path for kind, path in candidates.items() if path.exists()}


class RunRequest(BaseModel):
    """A request to run the pipeline for a clip's scene."""

    video: str | None = None


@app.post("/api/run")
def run_pipeline_endpoint(request: RunRequest) -> dict:
    """Start (or report) the full reconstruction pipeline for the clip's scene."""
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    job = _JOBS.get(scene.name)
    if job and not job["done"]:
        return {"ok": True, "scene": scene.name, "running": True}
    _JOBS[scene.name] = _start_job(scene, "accident_reconstruction.run_pipeline")
    return {"ok": True, "scene": scene.name, "running": True}


@app.post("/api/reconstruct")
def reconstruct_endpoint(request: RunRequest) -> dict:
    """Re-run only stage 2 (projection/impact/align) -- fast, reuses the tracks.

    Used after editing settings (impact frame, vehicle roles, …) so the user does
    not pay for SAM2 tracking again just to re-project with new settings.
    """
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    if not scene.prompt_tracks_csv.exists():
        return {"ok": False, "error": "尚無追蹤結果，請先執行完整 pipeline（④ 執行）"}
    job = _JOBS.get(scene.name)
    if job and not job["done"]:
        return {"ok": True, "scene": scene.name, "running": True}
    _JOBS[scene.name] = _start_job(scene, "accident_reconstruction.auto_reconstruct")
    return {"ok": True, "scene": scene.name, "running": True}


@app.post("/api/run/stop")
def stop_run(request: RunRequest) -> dict:
    """Cancel the scene's running job (terminate the subprocess; SIGKILL if slow)."""
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    job = _JOBS.get(scene.name)
    if not job or job["done"]:
        return {"ok": True, "running": False}
    job["cancelled"] = True
    proc = job["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"ok": True, "running": False}


class OverridesRequest(BaseModel):
    """Per-scene UI settings (impact frame, vehicle roles, gate strictness)."""

    overrides: dict
    video: str | None = None


@app.get("/api/overrides")
def get_overrides(video: str | None = None) -> dict:
    """Return saved overrides + the choices the UI needs to populate the form."""
    scene = _scene_for_video(video) if video else SCENE
    if scene is None:
        return {"scene": None, "overrides": {}, "vehicles": [], "frame_window": None}
    vehicles: list[str] = []
    if scene.vehicle_boxes.exists():
        try:
            data = json.loads(scene.vehicle_boxes.read_text())
            objects = data.get("objects") if isinstance(data, dict) else None
            vehicles = (
                [o["name"] for o in objects]
                if isinstance(objects, list)
                else list(data)
            )
        except (json.JSONDecodeError, OSError, KeyError):
            vehicles = []
    return {
        "scene": scene.name,
        "overrides": scene.overrides,
        "vehicles": vehicles,
        "defaults": {
            "stop_vehicle": scene.stop_vehicle,
            "moving_vehicle": scene.moving_vehicle,
            "start_frame": scene.start_frame,
            "end_frame": scene.end_frame,
        },
        "frame_window": [scene.resolved_start_frame, scene.resolved_end_frame],
    }


@app.post("/api/overrides")
def save_overrides(request: OverridesRequest) -> dict:
    """Persist the scene's ``overrides.json`` (drops empty/None entries).

    Geo-anchor keys (``true_impact_latlon``, ``true_vehicle_starts``) written by
    the step-2 anchor form are preserved so this step-4 save does not wipe them.
    """
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    cleaned = {k: v for k, v in request.overrides.items() if v not in (None, "")}
    for key in ("true_impact_latlon", "true_vehicle_starts"):
        if key not in cleaned and scene.overrides.get(key):
            cleaned[key] = scene.overrides[key]
    scene.overrides_path.parent.mkdir(parents=True, exist_ok=True)
    scene.overrides_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    return {"ok": True, "file": str(scene.overrides_path), "overrides": cleaned}


class AnchorsRequest(BaseModel):
    """Geo anchors read off the basemap in step 2 (2D-alignment correspondences)."""

    video: str | None = None
    true_impact: list[float] | None = None
    true_vehicle_starts: dict | None = None


@app.post("/api/anchors")
def save_anchors(request: AnchorsRequest) -> dict:
    """Merge the collision point + per-vehicle real starts into ``overrides.json``.

    Only the geo-anchor keys are touched; existing step-4 settings (impact frame,
    roles, gates) are left intact. Empty entries clear that anchor.
    """
    scene = _scene_for_video(request.video) if request.video else SCENE
    if scene is None:
        return {"ok": False, "error": "此影片沒有對應的場景設定（scene_config）"}
    overrides = dict(scene.overrides)
    if request.true_impact and len(request.true_impact) == 2:
        overrides["true_impact_latlon"] = list(request.true_impact)
    else:
        overrides.pop("true_impact_latlon", None)
    starts = {
        label: list(value)
        for label, value in (request.true_vehicle_starts or {}).items()
        if value and len(value) == 2
    }
    if starts:
        overrides["true_vehicle_starts"] = starts
    else:
        overrides.pop("true_vehicle_starts", None)
    scene.overrides_path.parent.mkdir(parents=True, exist_ok=True)
    scene.overrides_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False))
    return {"ok": True, "file": str(scene.overrides_path), "overrides": overrides}


@app.get("/api/run/status")
def run_status(video: str | None = None) -> dict:
    """Return the running pipeline's log + (when finished) the result kinds."""
    scene = _scene_for_video(video) if video else SCENE
    if scene is None:
        return {"scene": None, "running": False, "log": []}
    job = _JOBS.get(scene.name)
    if job is None:
        return {"scene": scene.name, "running": False, "log": [], "started": False}
    results = list(_result_files(scene)) if job["done"] else []
    return {
        "scene": scene.name,
        "started": True,
        "running": not job["done"],
        "returncode": job["returncode"],
        "cancelled": job.get("cancelled", False),
        "log": list(job["log"]),
        "results": results,
    }


@app.get("/api/result")
def result_file(video: str, kind: str):
    """Serve one pipeline output (``kind`` = figure | kml | csv | tracked)."""
    scene = _scene_for_video(video)
    if scene is None:
        return JSONResponse({"error": "no scene"}, status_code=404)
    path = _result_files(scene).get(kind)
    if path is None:
        return JSONResponse({"error": "not ready"}, status_code=404)
    media = {
        "figure": "image/png",
        "kml": "application/vnd.google-earth.kml+xml",
        "csv": "text/csv",
        "tracked": "video/mp4",
    }.get(kind)
    return FileResponse(path, media_type=media, filename=path.name)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Accident-reconstruction web workbench."
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("WEB_PORT", "8000"))
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes (dev)"
    )
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port}  (scene: {SCENE.name})")
    if args.reload:
        uvicorn.run(
            "accident_reconstruction.web_app:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)
