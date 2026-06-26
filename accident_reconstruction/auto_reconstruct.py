"""Closed loop: user-prompted SAM2 tracks -> homography -> 2D map / KML.

Takes the per-frame ground anchors produced by ``prompt_track_accident.py`` (the
user-specified vehicles, nothing else), projects them through the same homography
used by the manual pipeline, derives speed and the impact frame, and reuses the
birdseye writers to emit the aligned KML / map figure / CSV. This makes the whole
reconstruction automatic once the user has pointed at the accident vehicles.

Example:
    ```bash
    .venv/bin/python accident_reconstruction/auto_reconstruct.py
    ```
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from accident_reconstruction.birdseye_manual_annotation import (
    write_csv,
    write_kml,
    write_map_figure,
)
from accident_reconstruction.calibrate_homography import (
    VIEW_TRANSFORMER,
    undistort_to_normalized,
)
from accident_reconstruction.scene_config import SCENE

PROMPT_TRACKS_CSV = SCENE.prompt_tracks_csv


def _load_distortion() -> dict | None:
    """The lens distortion from the scene's calibration (None if not calibrated)."""
    path = SCENE.calibration_path
    if path.exists():
        try:
            return json.loads(path.read_text()).get("distortion")
        except (json.JSONDecodeError, OSError):
            return None
    return None


DISTORTION = _load_distortion()
AUTO_KML_PATH = SCENE.out_kml
AUTO_FIGURE_PATH = SCENE.out_figure
AUTO_CSV_PATH = SCENE.out_csv

FPS = SCENE.fps
SPEED_WINDOW_SECONDS = 0.6


def load_anchors(csv_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Load per-vehicle per-frame ground anchors (original pixels) from a CSV.

    Args:
        csv_path: A tracks CSV with ``frame, vehicle, anchor_x, anchor_y`` columns.

    Returns:
        ``anchors[vehicle][frame] = (anchor_x, anchor_y)``.
    """
    anchors: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in csv.DictReader(csv_path.open()):
        anchors[row["vehicle"]][int(row["frame"])] = (
            float(row["anchor_x"]),
            float(row["anchor_y"]),
        )
    return anchors


def project_metric(
    anchors: dict[str, dict[int, tuple[float, float]]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Project pixel anchors onto the metric ground plane via the homography.

    Args:
        anchors: Per-vehicle pixel anchors by frame.

    Returns:
        Per-vehicle metric ``(east_m, north_m)`` by frame.
    """
    metric: dict[str, dict[int, tuple[float, float]]] = {}
    for label, by_frame in anchors.items():
        metric[label] = {}
        for frame, anchor in by_frame.items():
            point = VIEW_TRANSFORMER.transform_points(
                np.array([anchor], dtype=np.float32)
            )[0]
            metric[label][frame] = (float(point[0]), float(point[1]))
    return metric


def _similarity_transform(source: np.ndarray, target: np.ndarray):
    """Least-squares similarity (rotate + uniform scale + translate) source->target.

    Returns a function applying the fit. Unlike a homography it is shape-preserving
    (no shear/perspective), so applying it to the recognised path keeps that path's
    curve while the fit borrows the homography's position, scale and orientation.
    """
    src_mean, tgt_mean = source.mean(axis=0), target.mean(axis=0)
    src0, tgt0 = source - src_mean, target - tgt_mean
    u, s, vt = np.linalg.svd(src0.T @ tgt0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    variance = float((src0**2).sum())
    scale = float(s.sum()) / variance if variance > 1e-12 else 1.0
    return lambda points: (scale * (points - src_mean)) @ rotation + tgt_mean


def shape_preserving_metric(
    anchors: dict[str, dict[int, tuple[float, float]]],
    metric: dict[str, dict[int, tuple[float, float]]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Place each vehicle's recognised path on the ground while KEEPING its curve.

    A single planar homography flattens the recognised trajectory (its curve is
    smaller than the calibration noise floor, so the global projective fit irons
    it out). Instead we keep the homography only for placement: per vehicle, fit a
    similarity from the (undistorted) pixel path to the homography metric path and
    apply it to the pixel path. The result sits where/at the scale the homography
    says, but its SHAPE is the recognised one -- not straightened (TPS was tried
    and amplifies GCP noise / diverges off the control points). Falls back to the
    homography path when a vehicle has too few points to fit a similarity.

    Args:
        anchors: Per-vehicle pixel anchors by frame.
        metric: The homography metric (already truncated), the placement reference.

    Returns:
        Per-vehicle shape-preserving metric ``(east_m, north_m)`` by frame.
    """
    if not DISTORTION:
        # Without a lens model the recognised pixels still carry barrel distortion
        # and frame jitter; "preserving" that shape makes the path jagged. Only
        # shape-preserve a clean (undistorted) path -- else keep the homography.
        return {label: dict(track) for label, track in metric.items()}
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for label, mtrack in metric.items():
        frames = sorted(mtrack)
        if len(frames) < 3 or label not in anchors:
            out[label] = dict(mtrack)
            continue
        pixels = np.array([anchors[label][f] for f in frames], dtype=np.float64)
        recognised = undistort_to_normalized(pixels, DISTORTION).astype(np.float64)
        homography = np.array([mtrack[f] for f in frames], dtype=np.float64)
        try:
            placed = _similarity_transform(recognised, homography)(recognised)
        except np.linalg.LinAlgError:
            out[label] = dict(mtrack)
            continue
        out[label] = {
            f: (float(placed[i, 0]), float(placed[i, 1])) for i, f in enumerate(frames)
        }
    return out


def windowed_motion(
    track: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Compute ``(cumulative_m, speed_kmh)`` per frame for one metric track.

    Speed is the displacement across the samples inside ``SPEED_WINDOW_SECONDS``,
    divided by the elapsed time, which smooths single-frame jitter.

    Args:
        track: Metric ``(east_m, north_m)`` by frame for one vehicle.

    Returns:
        ``(cumulative_m, speed_kmh)`` by frame.
    """
    motion: dict[int, tuple[float, float]] = {}
    samples: deque[tuple[int, tuple[float, float]]] = deque()
    cumulative = 0.0
    previous: tuple[float, float] | None = None
    for frame in sorted(track):
        point = track[frame]
        if previous is not None:
            cumulative += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
        samples.append((frame, point))
        while len(samples) >= 2 and frame - samples[0][0] > SPEED_WINDOW_SECONDS * FPS:
            samples.popleft()
        speed = 0.0
        if len(samples) >= 2:
            first_frame, first_point = samples[0]
            elapsed = (frame - first_frame) / FPS
            if elapsed > 0:
                distance = math.hypot(
                    point[0] - first_point[0], point[1] - first_point[1]
                )
                speed = distance / elapsed * 3.6
        motion[frame] = (cumulative, speed)
    return motion


# Two vehicles within this metric distance are treated as in contact. Note this
# is only reliable when the ground projection is well-scaled; under a compressed
# homography everything reads "close", which is why the closest-approach rule
# below is preferred whenever the pair cleanly separates afterwards.
CONTACT_THRESHOLD_M = 3.0
# After a real collision the two vehicles move apart again. If the distance rises
# at least this far above the closest approach, that closest approach is the
# impact; if it never does (the masks merge and the distance plateaus), we fall
# back to first-contact-under-threshold instead.
SEPARATION_MARGIN_M = 1.0
# A struck vehicle that is flung/flips moves its ground anchor faster than any car
# drives on the road; a per-frame ground step above this (metres/frame) marks the
# tumble onset, after which the bottom-centre anchor is physically meaningless.
# Urban-accident speeds are well under this (~0.5 m/frame), the flip jump is ~1.5.
FLIP_VELOCITY_M_PER_FRAME = 1.2


def pair_distances(
    metric: dict[str, dict[int, tuple[float, float]]],
    label_a: str,
    label_b: str,
) -> list[tuple[int, float]]:
    """Per-frame metric distance between two vehicles over their shared frames."""
    shared = sorted(set(metric[label_a]) & set(metric[label_b]))
    return [
        (
            frame,
            math.hypot(
                metric[label_a][frame][0] - metric[label_b][frame][0],
                metric[label_a][frame][1] - metric[label_b][frame][1],
            ),
        )
        for frame in shared
    ]


def detect_impact(metric: dict[str, dict[int, tuple[float, float]]]) -> int | None:
    """Return the impact frame, or None if there is no two-vehicle contact.

    Scene-agnostic: works for any vehicle labels and counts. With fewer than two
    tracked vehicles there is no collision (returns None). With two or more, the
    colliding pair is the one with the smallest closest approach, and the impact
    frame is chosen by:

    * **Closest approach** when the pair clearly *separates* afterwards (distance
      rises at least ``SEPARATION_MARGIN_M`` above the minimum) -- a clean
      collide-then-part, robust even when the projection is distance-compressed.
    * **First frame under ``CONTACT_THRESHOLD_M``** when they instead merge and
      the distance plateaus (e.g. one mask follows the other post-impact), where
      the closest approach would land arbitrarily late inside the plateau.

    Args:
        metric: Per-vehicle metric positions by frame (any number of vehicles).

    Returns:
        The impact frame index, or None when no pair ever shares a frame.

    Examples:
        ```python
        # clean collide-then-separate -> closest approach (frame 2)
        m = {"a": {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)},
             "b": {0: (5, 0), 1: (2, 0), 2: (1, 0), 3: (9, 0)}}
        detect_impact(m)
        # 2
        ```
    """
    labels = list(metric)
    best: tuple[int, float] | None = None  # (impact_frame, closest distance)
    for index, label_a in enumerate(labels):
        for label_b in labels[index + 1 :]:
            distances = pair_distances(metric, label_a, label_b)
            if not distances:
                continue
            min_frame, min_dist = min(distances, key=lambda item: item[1])
            after = [d for f, d in distances if f > min_frame]
            separates = bool(after) and max(after) >= min_dist + SEPARATION_MARGIN_M
            if separates:
                impact_frame = min_frame
            else:
                impact_frame = next(
                    (f for f, d in distances if d < CONTACT_THRESHOLD_M), min_frame
                )
            if best is None or min_dist < best[1]:
                best = (impact_frame, min_dist)
    return None if best is None else best[0]


def settle_frame(
    motion: dict[int, tuple[float, float]],
    after_frame: int,
    min_speed_kmh: float,
    sustain: int = 3,
) -> int | None:
    """First frame after ``after_frame`` where the vehicle has come to rest.

    "At rest" = speed stays below ``min_speed_kmh`` for ``sustain`` consecutive
    frames. Used to STOP the trajectory line once a vehicle stops moving (e.g. a
    struck car that settles after the crash), so the meaningless post-stop anchor
    jitter is not drawn -- the box/marker can still mark its final position.

    Args:
        motion: ``{frame: (cumulative_m, speed_kmh)}`` for one vehicle.
        after_frame: Only consider frames after this (rest follows the collision).
        min_speed_kmh: Speed threshold; <= 0 disables (returns None).
        sustain: Consecutive sub-threshold frames required (rejects a brief dip).

    Returns:
        The first frame of the sustained low-speed run, or None.

    Examples:
        ```python
        m = {0: (0, 9.0), 1: (1, 1.0), 2: (1, 0.5), 3: (1, 0.4)}
        settle_frame(m, 0, 3.0, sustain=2)
        # 1
        ```
    """
    if min_speed_kmh <= 0:
        return None
    frames = [f for f in sorted(motion) if f > after_frame]
    run_start = None
    run = 0
    for frame in frames:
        if motion[frame][1] < min_speed_kmh:
            run_start = frame if run == 0 else run_start
            run += 1
            if run >= sustain:
                return run_start
        else:
            run = 0
            run_start = None
    return None


def flip_onset(track: dict[int, tuple[float, float]], after_frame: int) -> int | None:
    """First frame > ``after_frame`` where the path jumps non-physically (a flip).

    The ground-contact anchor (mask bottom-centre) only tracks a real position
    while the vehicle is upright on the road. When a struck vehicle is flung and
    tumbles, the anchor leaps and bounces -- a per-frame ground step no real car
    achieves. This returns the first such frame (so the caller can drop the
    tumble), or None when the vehicle stays on the ground.

    Args:
        track: ``{frame: (east_m, north_m)}`` for one vehicle.
        after_frame: Only consider frames after this (the flip follows contact).

    Returns:
        The flip-onset frame, or None.

    Examples:
        ```python
        flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0), 3: (4.0, 0)}, 0)
        # 3
        flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0)}, 0)  # all smooth
        ```
    """
    frames = sorted(track)
    previous = None
    for frame in frames:
        if previous is not None and frame > after_frame:
            gap = max(frame - previous, 1)
            step = math.hypot(
                track[frame][0] - track[previous][0],
                track[frame][1] - track[previous][1],
            )
            if step / gap > FLIP_VELOCITY_M_PER_FRAME:
                return frame
        previous = frame
    return None


def build_data(csv_path: Path = PROMPT_TRACKS_CSV):
    """Assemble ``(motion, metric, impact_frame)`` from a prompt-tracks CSV.

    If the scene names a ``stop_vehicle`` (the struck one), its physically invalid
    anchors are dropped: the path is cut at the flip onset (:func:`flip_onset`) so
    only the on-ground approach/push is kept and the post-impact tumble is removed.
    When no flip is detected (e.g. the mask just merges and plateaus) it falls back
    to truncating at the impact frame. Scenes without a stop vehicle keep every
    anchor.

    Args:
        csv_path: Prompt-tracks CSV path.

    Returns:
        ``(motion, metric, impact_frame)`` ready for the birdseye writers;
        ``impact_frame`` is None when no two-vehicle contact is found.
    """
    anchors = load_anchors(csv_path)
    metric = project_metric(anchors)
    # The UI can pin the impact frame (overrides.json); else auto-detect.
    impact_frame = SCENE.impact_frame_override
    if impact_frame is None:
        impact_frame = detect_impact(metric)

    # Cut each vehicle's trajectory once it comes to rest after the collision:
    # the line stops where speed stays below ``min_traj_speed`` (post-stop anchor
    # jitter is meaningless). The struck vehicle additionally cuts at the flip
    # onset (a high-speed tumble the speed gate can't catch), or, with neither and
    # no ``struck_full``, falls back to the impact frame.
    stop_vehicle = SCENE.resolved_stop_vehicle
    min_speed = SCENE.min_traj_speed_kmh
    full_motion = {label: windowed_motion(track) for label, track in metric.items()}
    for label, track in list(metric.items()):
        after = impact_frame if impact_frame is not None else min(track)
        cuts = [settle_frame(full_motion[label], after, min_speed)]
        if label == stop_vehicle:
            cuts.append(flip_onset(track, after))
            if (
                not any(c is not None for c in cuts)
                and impact_frame is not None
                and not SCENE.show_struck_full
            ):
                cuts.append(impact_frame + 1)  # struck car assumed to stop here
        cut = min((c for c in cuts if c is not None), default=None)
        if cut is not None:
            metric[label] = {f: xy for f, xy in track.items() if f < cut}

    # Speed/distance come from the accurate homography metric; the DRAWN path is
    # the shape-preserving one (keeps the recognised curve the homography flattens).
    motion = {label: windowed_motion(track) for label, track in metric.items()}
    draw_metric = shape_preserving_metric(anchors, metric)
    return motion, draw_metric, impact_frame


def main(csv_path: str = str(PROMPT_TRACKS_CSV)) -> None:
    """Run the closed-loop reconstruction and write KML / figure / CSV."""
    data = build_data(Path(csv_path))
    _, metric, impact_frame = data
    write_kml(data, kml_path=AUTO_KML_PATH)
    write_map_figure(data, figure_path=AUTO_FIGURE_PATH)
    write_csv(data, csv_path=AUTO_CSV_PATH)

    # Also emit the raw recognised (non-road-snapped) figure + KML -- this is what
    # the web app now displays as the primary 2D result. Deferred import because
    # recognized_route imports from this module (avoids an import cycle).
    try:
        from accident_reconstruction.recognized_route import (
            write_recognized_csv,
            write_recognized_figure,
            write_recognized_kml,
        )

        recognised_figure = write_recognized_figure()
        write_recognized_kml()
        write_recognized_csv()
        if recognised_figure is not None:
            print(f"Recognised figure: {recognised_figure.resolve()}")
    except Exception as error:  # never let the optional view break the run
        print(f"(recognised figure skipped: {error})")

    print(f"Impact frame: {impact_frame}")
    counts = ", ".join(f"{k}: {len(v)}" for k, v in metric.items())
    print(f"Frames per vehicle: {{{counts}}}")
    print(f"KML: {AUTO_KML_PATH.resolve()}")
    print(f"Map figure: {AUTO_FIGURE_PATH.resolve()}")
    print(f"CSV: {AUTO_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
