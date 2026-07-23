"""User-prompted vehicle tracking with SAM2 video memory (no detector tuning).

Pipeline design: the user SPECIFIES the accident vehicles as input (one box per
vehicle, each drawn on the frame where that vehicle first appears -- not
necessarily the window start). SAM2's video predictor then segments and
propagates each one through the clip using its cross-frame MEMORY -- which holds
small / occluded objects (e.g. the motorcycle here) far better than a detector or
naive per-frame re-prompting (that leaks). No class filtering, confidence tuning,
or track-id matching to babysit.

The ground-contact anchor (mask bottom-centre) feeds the same homography / map
alignment as the rest of this folder.

Each vehicle is tracked independently from its own init frame (``spec["frame"]``)
and the per-vehicle tracks are merged, so a vehicle that enters after the window
start is handled.

Example:
    ```bash
    .venv/bin/python accident_reconstruction/prompt_track_accident.py
    ```
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.sam import SAM2VideoPredictor

from accident_reconstruction import ground_footprint as gf
from accident_reconstruction.ffmpeg_util import ensure_readable_mp4
from accident_reconstruction.scene_config import SCENE
from accident_reconstruction.time_axis import (
    frame_time,
    probe_pts_times,
    time_axis_warning,
)

# The user's input: one box per accident vehicle, in ORIGINAL (1280x720) pixels,
# on ``start_frame``. Insertion order defines the prompt/mask order. (Seeded here
# from known positions; in the product the user draws these.)
INIT_VEHICLES = dict(SCENE.init_vehicles)

SOURCE_VIDEO = str(SCENE.source_video)
TARGET_VIDEO = str(SCENE.prompt_tracked_video)
TRACKS_CSV = str(SCENE.prompt_tracks_csv)

# A real vehicle's box changes only gradually frame-to-frame, so a box whose
# width or height jumps outside this ratio of the last accepted one is treated as
# a SAM2 mask leak/merge and dropped (keeps box sizes stable, trajectory clean).
SIZE_RATIO_MIN = 0.6
SIZE_RATIO_MAX = 1.7

# The vehicles move away from their start toward/through the impact and then stop;
# they never drive back to the start. If the anchor's distance-from-start drops
# more than this many pixels below the max reached, it is a SAM2 mask jumping back
# to the initial spot (post-collision) -- drop it so the trajectory can't reverse.
BACKTRACK_TOLERANCE_PX = 40.0

# Stop propagating a re-seeded segment after SAM2 loses the object for this many
# consecutive frames -- avoids running a lost track all the way to the window end
# (every box now propagates to end_frame, so this keeps that affordable).
LOST_PATIENCE_FRAMES = 20

# Two vehicles' boxes are treated as MERGED when the smaller box is largely inside
# the larger one -- a SAM2 mask leak where one track's mask has grown onto the
# overlapping other vehicle (e.g. the struck motorcycle's mask swallowed by the car
# it hit, so its box collapses onto the car's). Past this overlap fraction (of the
# smaller box's area) the smaller box no longer is that vehicle, so it is dropped
# from the merge frame on -- the two boxes can no longer become one.
MERGE_CONTAINMENT_RATIO = 0.6
# Require the overlap to persist this many consecutive shared frames before cutting,
# so a single-frame brush-past (the striker driving close) is not mistaken for a
# mask merge.
MERGE_SUSTAIN_FRAMES = 3


def _select_device() -> str:
    """Pick the fastest available torch device for SAM2 (MPS > CUDA > CPU)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


# When a box's height collapses below this fraction of the last full-body height,
# the visible mask is only the object's TOP (e.g. a rider's helmet once the impact
# occludes the rest). Its bottom-centre then sits far above the real wheel/ground
# contact, so the ground anchor is reconstructed from the box top instead.
OCCLUSION_HEIGHT_RATIO = 0.55


def trim_clip(source: str, start: int, end: int, out_path: str) -> float:
    """Write frames ``[start, end]`` of ``source`` to ``out_path``.

    SAM2's video memory propagates from the prompt on the FIRST frame, so the clip
    must begin on the frame where the vehicles are prompted.

    Args:
        source: Input video path.
        start: First frame to copy (inclusive).
        end: Last frame to copy (inclusive).
        out_path: Output clip path.

    Returns:
        The clip's frames-per-second.
    """
    capture = cv2.VideoCapture(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start + 1):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    capture.release()
    writer.release()
    return fps


def mask_box_and_anchor(mask: np.ndarray):
    """Return a mask's tight box and ground-contact anchor, or None if empty.

    Args:
        mask: Boolean ``(H, W)`` mask.

    Returns:
        ``((x1, y1, x2, y2), (anchor_x, anchor_y))`` or None.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (x1, y1, x2, y2), ((x1 + x2) // 2, y2)


def occlusion_corrected_anchor(
    box: tuple[int, int, int, int],
    raw_anchor: tuple[int, int],
    ref_height: float | None,
    frame_height: int,
) -> tuple[tuple[int, int], bool]:
    """Reconstruct the ground anchor when the box is an occluded top fragment.

    When ``box`` is far shorter than the last full-body height ``ref_height`` (see
    :data:`OCCLUSION_HEIGHT_RATIO`), only the object's top is visible, so the
    mask's bottom-centre (``raw_anchor``) sits above the true ground contact. The
    anchor is then taken at the box's horizontal centre, ``ref_height`` pixels
    below the box TOP (clamped to the frame), approximating the wheel contact.

    Args:
        box: Visible box ``(x1, y1, x2, y2)``.
        raw_anchor: Mask bottom-centre ``(x, y)``.
        ref_height: Last accepted full-body box height, or None if unknown.
        frame_height: Frame height in pixels (anchor is clamped to it).

    Returns:
        ``(anchor, occluded)`` -- the (possibly corrected) anchor and whether the
        correction fired.

    Examples:
        ```python
        # full box (height 60) vs ref 60 -> unchanged
        occlusion_corrected_anchor((10, 0, 30, 60), (20, 60), 60.0, 720)
        # ((20, 60), False)
        # collapsed box (height 12) -> extend from top (y=100) by ref 60
        occlusion_corrected_anchor((10, 100, 30, 112), (20, 112), 60.0, 720)
        # ((20, 160), True)
        ```
    """
    if ref_height is None:
        return raw_anchor, False
    height = box[3] - box[1]
    if height >= OCCLUSION_HEIGHT_RATIO * ref_height:
        return raw_anchor, False
    center_x = (box[0] + box[2]) // 2
    ground_y = min(int(box[1] + ref_height), frame_height - 1)
    return (center_x, ground_y), True


def _box_near(existing: tuple[int, int, int, int], drawn: list[int]) -> bool:
    """True if ``existing``'s centre is within half the ``drawn`` box of its centre.

    Used to skip re-seeding a user box that an earlier segment already tracks well
    there (a redundant box), versus a real correction where the track diverged.

    Examples:
        ```python
        _box_near((10, 10, 30, 30), [12, 12, 32, 32])
        # True
        _box_near((10, 10, 30, 30), [200, 200, 220, 220])
        # False
        ```
    """
    ex = ((existing[0] + existing[2]) / 2, (existing[1] + existing[3]) / 2)
    dx = ((drawn[0] + drawn[2]) / 2, (drawn[1] + drawn[3]) / 2)
    tol = 0.5 * max(drawn[2] - drawn[0], drawn[3] - drawn[1], 1)
    return abs(ex[0] - dx[0]) <= tol and abs(ex[1] - dx[1]) <= tol


def anchor_boxes(spec: dict, start_frame: int) -> list[tuple[int, list[int]]]:
    """User prompt boxes for one vehicle as ``[(frame, [x1,y1,x2,y2]), ...]``.

    Two input shapes are accepted, sorted by frame:

    * **Multi-box** ``spec["boxes"] = [{"frame", "box"}, ...]`` -- one re-seed
      point per drawn frame (e.g. a correction at the impact frame where the
      motorcycle is occluded down to its helmet).
    * **Single-box** ``spec["box"]`` (+ optional ``spec["frame"]``).

    Args:
        spec: The vehicle spec.
        start_frame: Fallback prompt frame when none is given.

    Returns:
        Frame-sorted ``(frame, box)`` anchors (at least one).

    Examples:
        ```python
        anchor_boxes({"box": [1, 2, 3, 4]}, 80)
        # [(80, [1, 2, 3, 4])]
        anchor_boxes({"boxes": [{"frame": 50, "box": [5, 6, 7, 8]},
                                {"frame": 10, "box": [1, 1, 2, 2]}]}, 0)
        # [(10, [1, 1, 2, 2]), (50, [5, 6, 7, 8])]
        ```
    """
    raw = spec.get("boxes")
    if raw:
        anchors = [(int(b.get("frame", start_frame)), list(b["box"])) for b in raw]
    else:
        anchors = [(int(spec.get("frame", start_frame)), list(spec["box"]))]
    anchors.sort(key=lambda frame_box: frame_box[0])
    return anchors


def _build_predictor(weights: str) -> SAM2VideoPredictor:
    """A fresh SAM2 video predictor on the best available device."""
    return SAM2VideoPredictor(
        overrides=dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model=weights,
            save=False,
            verbose=False,
            device=_select_device(),
        )
    )


def _segment_masks(
    predictor: SAM2VideoPredictor,
    source_video_path: str,
    seg_start: int,
    seg_end: int,
    box: list[int],
) -> dict[int, tuple]:
    """Track one re-seeded segment ``[seg_start, seg_end]`` from ``box``.

    Each segment is an independent SAM2 video-memory run prompted by the user's
    box on its first frame (a clean re-acquire after occlusion). The predictor is
    REUSED across a vehicle's segments (the model loads once) but its per-video
    state is reset here first: in this ultralytics build ``on_predict_start`` does
    not fully clear the memory bank / stored prompts between calls, so without this
    a later re-seed -- e.g. a manual post-impact box -- is dragged back to an
    earlier segment's mask instead of honouring its own prompt box.

    Args:
        predictor: A reusable SAM2 video predictor (reset per call here).
        source_video_path: Input video.
        seg_start: First frame of the segment (the re-seed/prompt frame).
        seg_end: Last frame of the segment (inclusive).
        box: Prompt box ``[x1, y1, x2, y2]`` in original-frame pixels.

    Returns:
        ``{frame_index: (box, anchor, mask)}`` raw (no gates applied yet).
    """
    predictor.inference_state = {}  # drop the previous segment's video memory
    predictor.prompts = {}  # and its stored prompts (else they override this box)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        clip_path = handle.name
    trim_clip(source_video_path, seg_start, seg_end, clip_path)
    out: dict[int, tuple] = {}
    lost = 0
    try:
        results = predictor(source=clip_path, bboxes=[box], stream=True)
        for offset, result in enumerate(results):
            masks = (
                result.masks.data.cpu().numpy()
                if result.masks is not None
                else np.empty((0,))
            )
            found = mask_box_and_anchor(masks[0] > 0.5) if len(masks) else None
            if found is None:
                lost += 1
                if lost > LOST_PATIENCE_FRAMES:
                    break  # object lost; stop wasting frames to the window end
                continue
            lost = 0
            mbox, anchor = found
            out[seg_start + offset] = (mbox, anchor, masks[0] > 0.5)
    finally:
        Path(clip_path).unlink(missing_ok=True)
    return out


def box_containment(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    """Intersection area as a fraction of the SMALLER box's area (0..1).

    Containment (rather than IoU) detects a merge where a small box sits wholly
    inside a much larger one: their IoU stays low (the union is dominated by the
    big box) but the small box is fully swallowed, which is exactly the SAM2 mask
    leak we want to catch.

    Args:
        box_a: ``(x1, y1, x2, y2)``.
        box_b: ``(x1, y1, x2, y2)``.

    Returns:
        Overlap fraction of whichever box is smaller; 0 if either is degenerate.

    Examples:
        ```python
        # small box fully inside the large one -> 1.0
        box_containment((10, 10, 20, 20), (0, 0, 100, 100))
        # 1.0
        # disjoint boxes -> 0.0
        box_containment((0, 0, 10, 10), (50, 50, 60, 60))
        # 0.0
        ```
    """
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def merge_suppression_cuts(
    per_vehicle: dict[str, dict[int, tuple]],
    anchor_frames: dict[str, set[int]] | None = None,
) -> dict[str, int]:
    """Frame from which each vehicle's box has merged into another's (drop after).

    Each vehicle is tracked by an independent SAM2 run, so when two of them overlap
    at impact one track's mask can leak onto the other -- the smaller (struck)
    vehicle's box grows to cover the larger one, and the two drawn boxes collapse
    into one. This finds, per pair, the first run of :data:`MERGE_SUSTAIN_FRAMES`
    consecutive shared frames where the smaller box is at least
    :data:`MERGE_CONTAINMENT_RATIO` inside the larger one, and marks the SMALLER
    vehicle to be cut from the run's first frame (its box is no longer that
    vehicle). The trajectory's own flip/impact truncation is separate; this guards
    the drawn boxes and the tracks CSV.

    A vehicle the user has manually re-seeded inside the merge window is NOT cut:
    a user box at frame >= the run start asserts the vehicle really is there (e.g.
    a motorcycle re-marked post-impact while crushed against the car it hit, which
    legitimately overlaps the car's box). Auto leaks have no such anchor, so they
    are still dropped.

    Args:
        per_vehicle: ``{vehicle: {frame: (box, anchor, mask)}}``.
        anchor_frames: ``{vehicle: {frame, ...}}`` user-drawn (re-seed) frames per
            vehicle; a cut is skipped when the smaller vehicle is anchored at or
            after the merge run's start. Defaults to none (no manual boxes).

    Returns:
        ``{vehicle: cut_frame}`` -- only for vehicles that merge; absent otherwise.

    Examples:
        ```python
        # small box swallowed by the big one from frame 2 on (sustain=3) -> cut@2
        small = {f: ((10, 10, 20, 20), (15, 20), None) for f in range(2, 6)}
        big = {f: ((0, 0, 100, 100), (50, 100), None) for f in range(0, 6)}
        merge_suppression_cuts({"motorcycle": small, "car": big})
        # {'motorcycle': 2}
        # a user re-seed at frame 3 keeps the motorcycle (no cut)
        merge_suppression_cuts({"motorcycle": small, "car": big},
                               {"motorcycle": {3}})
        # {}
        ```
    """
    anchor_frames = anchor_frames or {}
    cuts: dict[str, int] = {}
    names = list(per_vehicle)
    for index, label_a in enumerate(names):
        for label_b in names[index + 1 :]:
            shared = sorted(set(per_vehicle[label_a]) & set(per_vehicle[label_b]))
            run = 0
            run_start: int | None = None
            smaller: str | None = None
            for frame in shared:
                box_a = per_vehicle[label_a][frame][0]
                box_b = per_vehicle[label_b][frame][0]
                if box_containment(box_a, box_b) >= MERGE_CONTAINMENT_RATIO:
                    if run == 0:
                        run_start = frame
                        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                        smaller = label_a if area_a <= area_b else label_b
                    run += 1
                    if run >= MERGE_SUSTAIN_FRAMES and smaller is not None:
                        # A manual re-seed inside/after the run means the user keeps
                        # this vehicle here on purpose -- never cut it.
                        if any(f >= run_start for f in anchor_frames.get(smaller, ())):
                            break
                        prior = cuts.get(smaller)
                        cuts[smaller] = (
                            run_start if prior is None else min(prior, run_start)
                        )
                        break
                else:
                    run = 0
                    run_start = None
                    smaller = None
    return cuts


def split_overlapping_masks(
    per_vehicle: dict[str, dict[int, tuple]],
) -> dict[str, dict[int, tuple]]:
    """Split a fused blob: clip the larger vehicle's mask out of the smaller's box.

    After a fusing collision the two vehicles share one connected blob, so the
    larger vehicle's track segments the whole thing and its box/mask swallow the
    smaller one (even when the smaller is separately tracked). Per frame, for each
    overlapping pair, the smaller vehicle's box region is erased from the LARGER
    vehicle's mask and its box/anchor recomputed -- so the larger box/mask stops at
    the smaller vehicle instead of covering it. The smaller vehicle (typically a
    manual post-impact re-seed) keeps its own box. A no-op where boxes don't
    overlap, so it is safe to run over every frame.

    Args:
        per_vehicle: ``{vehicle: {frame: (box, anchor, mask)}}`` (masks required).

    Returns:
        The same dict, with larger boxes/masks trimmed where they swallowed a
        smaller vehicle (mutated in place and returned).
    """
    names = list(per_vehicle)
    all_frames: set[int] = set()
    for records in per_vehicle.values():
        all_frames |= set(records)
    for frame in sorted(all_frames):
        present = [name for name in names if frame in per_vehicle[name]]
        if len(present) < 2:
            continue
        # Smallest first, so each larger vehicle is clipped by every smaller one.
        present.sort(
            key=lambda name: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(
                per_vehicle[name][frame][0]
            )
        )
        for rank, small in enumerate(present):
            sbox = per_vehicle[small][frame][0]
            for large in present[rank + 1 :]:
                lbox, _, lmask = per_vehicle[large][frame]
                if lmask is None or box_containment(sbox, lbox) <= 0.0:
                    continue
                clipped = lmask.copy()
                h, w = clipped.shape
                # Cut a full strip off the larger mask on the side the smaller
                # vehicle sits (the axis where their centres are most separated),
                # so the larger body keeps only its own side of the seam. A strip
                # (not just the smaller box) is needed because the fused blob stays
                # connected around the box otherwise.
                s_cx, s_cy = (sbox[0] + sbox[2]) / 2, (sbox[1] + sbox[3]) / 2
                l_cx, l_cy = (lbox[0] + lbox[2]) / 2, (lbox[1] + lbox[3]) / 2
                if abs(l_cx - s_cx) >= abs(l_cy - s_cy):
                    if s_cx < l_cx:  # smaller on the left -> drop columns up to it
                        clipped[:, : min(sbox[2] + 1, w)] = False
                    else:  # smaller on the right
                        clipped[:, max(sbox[0], 0) :] = False
                elif s_cy < l_cy:  # smaller above
                    clipped[: min(sbox[3] + 1, h), :] = False
                else:  # smaller below
                    clipped[max(sbox[1], 0) :, :] = False
                found = mask_box_and_anchor(clipped)
                if found is None:  # strip emptied it -- keep the original
                    continue
                nbox, nanchor = found
                per_vehicle[large][frame] = (nbox, nanchor, clipped)
    return per_vehicle


def track_vehicle(
    name: str,
    spec: dict,
    source_video_path: str,
    start_frame: int,
    end_frame: int,
    weights: str,
):
    """Track one vehicle with SAM2 video memory, re-seeding at each user box.

    Each user box re-seeds a SAM2 segment propagated to ``end_frame``; per frame
    the most recent re-seed that still holds the object wins, falling back to an
    earlier box's longer continuous track where a later (fresh, memory-less)
    re-seed was lost. So a vehicle lost to occlusion (e.g. the motorcycle reduced
    to a helmet at impact) is re-acquired wherever the user re-marks it, and extra
    boxes never shorten coverage. A single box is the original single-prompt track.

    The size-stability and no-backtrack gates run over the merged frames to drop
    SAM2 mask leaks. A user-anchored frame bypasses the size gate (its occluded box
    is legitimately small) but NOT the no-backtrack gate -- a re-seed that snaps
    back to the original prompt location after the collision is still dropped. For a
    struck/flung vehicle that is legitimately shoved back past its start, set the
    scene's gates to "loose" (drops the no-backtrack gate) -- see SCENE.gate_mode.

    Note:
        The ground-contact anchor is the mask's bottom-centre. Under heavy
        occlusion the box covers only the visible part, so the anchor sits higher
        than the true wheel contact -- a small bias at the impact frame. TODO:
        carry the last full-body height to estimate ground contact when occluded.

    Args:
        name: Vehicle label (for messages).
        spec: ``{"box"|"boxes", "frame"?, "bgr"}`` (see :func:`anchor_boxes`).
        source_video_path: Input video.
        start_frame: Pipeline window start (fallback prompt frame).
        end_frame: Pipeline window end (inclusive).
        weights: SAM2 weights.

    Returns:
        ``{frame_index: (box, anchor, mask)}`` for accepted frames.
    """
    anchors = anchor_boxes(spec, start_frame)
    anchor_frames = {frame for frame, _ in anchors}
    usable = [(frame, box) for frame, box in anchors if frame <= end_frame]

    # One predictor for the whole vehicle (the model loads once); _segment_masks
    # resets its per-video memory per call. Re-seed one segment per user box. Each
    # only needs to cover until the NEXT re-seed takes over -- the most recent
    # re-seed wins per frame -- so a segment runs to the next anchor plus a short
    # ``LOST_PATIENCE_FRAMES`` margin (the last runs to end_frame). The margin
    # overlaps the next re-seed so that if it is immediately lost, this segment
    # still fills the gap; ``raw.update`` in ascending order lets the later re-seed
    # overwrite the overlap where it holds. This caps the SAM2 propagation -- else
    # every segment runs to end_frame, re-inferring the same long tail many times.
    predictor = _build_predictor(weights)
    raw: dict[int, tuple] = {}
    for index, (anchor_frame, box) in enumerate(usable):
        # Skip a redundant box: if an earlier segment already tracks this frame with
        # a box near the user's, re-seeding here just repeats work. Re-seed only when
        # the frame is uncovered or the track diverged (a real correction).
        if index > 0 and anchor_frame in raw and _box_near(raw[anchor_frame][0], box):
            continue
        if index + 1 < len(usable):
            seg_end = min(usable[index + 1][0] + LOST_PATIENCE_FRAMES, end_frame)
        else:
            seg_end = end_frame
        seg_end = max(seg_end, anchor_frame)
        raw.update(
            _segment_masks(predictor, source_video_path, anchor_frame, seg_end, box)
        )

    # Gate pass over the merged frames. The strictness is scene/UI-configurable
    # (overrides.json "gates"): "strict" keeps both motion gates (default, best for
    # a vehicle driving normally through frame -- also right for a struck vehicle
    # that flips, whose tumble is dropped later by the flip-onset truncation),
    # "loose" widens the size gate and drops the no-backtrack gate (only for a
    # vehicle that legitimately reverses / backs up smoothly), and "off" disables
    # both. User-anchored frames always
    # bypass the size gate (an occluded box is legitimately small).
    mode = SCENE.gate_mode
    size_min, size_max = (
        (0.3, 3.0) if mode == "loose" else (SIZE_RATIO_MIN, SIZE_RATIO_MAX)
    )
    size_gate_on = mode != "off"
    backtrack_gate_on = mode == "strict"
    records: dict[int, tuple] = {}
    prev_size: tuple[int, int] | None = None
    start_anchor: tuple[int, int] | None = None
    ref_height: float | None = None
    max_dist = 0.0
    for frame_index in sorted(raw):
        box, anchor, mask = raw[frame_index]
        width, height = box[2] - box[0], box[3] - box[1]
        is_anchor = frame_index in anchor_frames
        # Reconstruct the ground anchor when the box is an occluded top fragment.
        anchor, occluded = occlusion_corrected_anchor(
            box, anchor, ref_height, mask.shape[0]
        )
        # Size-stability gate: drop sudden box growth/shrink (SAM2 mask leak).
        # User-anchored frames bypass it (an occluded box is legitimately small).
        if size_gate_on and not is_anchor and prev_size is not None:
            w_ratio, h_ratio = (
                width / max(prev_size[0], 1),
                height / max(prev_size[1], 1),
            )
            if not (
                size_min <= w_ratio <= size_max and size_min <= h_ratio <= size_max
            ):
                continue
        # No-backtrack gate -- a real vehicle never teleports back toward its start,
        # so this drops a SAM2 re-seed that snapped to the original prompt location
        # after the collision. Disabled in loose/off for struck/reversing vehicles
        # (a struck/flung motorcycle the truck shoves back past its start needs
        # "loose" gates -- see SCENE.gate_mode). A user-anchored frame bypasses it
        # (the user vouches for this position) AND redefines the furthest-distance
        # reference, so the propagated frames of a manual post-impact re-seed --
        # legitimately closer to the start than the run-up's furthest point, e.g. a
        # motorcycle crushed back against the car it hit -- survive instead of being
        # dropped as a "teleport back to start".
        if start_anchor is None:
            start_anchor = anchor
        else:
            distance = float(
                np.hypot(anchor[0] - start_anchor[0], anchor[1] - start_anchor[1])
            )
            if (
                backtrack_gate_on
                and not is_anchor
                and distance < max_dist - BACKTRACK_TOLERANCE_PX
            ):
                continue
            max_dist = distance if is_anchor else max(max_dist, distance)
        prev_size = (width, height)
        if not occluded:  # only un-occluded frames update the full-body height
            ref_height = float(height)
        records[frame_index] = (box, anchor, mask)

    return records


def main(
    source_video_path: str = SOURCE_VIDEO,
    target_video_path: str = TARGET_VIDEO,
    tracks_csv_path: str = TRACKS_CSV,
    weights: str = SCENE.weights,
    start_frame: int = SCENE.start_frame,
    end_frame: int = SCENE.end_frame,
) -> None:
    """Track each user-specified vehicle (each from its own frame); write outputs.

    Args:
        source_video_path: Input video.
        target_video_path: Output annotated video.
        tracks_csv_path: Output per-frame trajectory CSV.
        weights: SAM2 weights (auto-downloaded by ultralytics).
        start_frame: First frame of the output window.
        end_frame: Last frame (inclusive).
    """
    names = list(INIT_VEHICLES)
    Path(tracks_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(target_video_path).parent.mkdir(parents=True, exist_ok=True)

    # Track each vehicle independently from its own appearance frame.
    per_vehicle = {
        name: track_vehicle(
            name,
            INIT_VEHICLES[name],
            source_video_path,
            start_frame,
            end_frame,
            weights,
        )
        for name in names
    }

    # Suppress SAM2 mask merges: once a vehicle's box has leaked onto / collapsed
    # into another's (e.g. the struck motorcycle swallowed by the car it hit), drop
    # it from that frame on so the two boxes never render as one. This is purely a
    # box/CSV guard -- the trajectory's flip/impact truncation is handled in
    # ``auto_reconstruct`` and is independent of this. A vehicle the user manually
    # re-seeds inside the merge window is kept (a deliberate post-impact box).
    anchor_frames = {
        name: {frame for frame, _ in anchor_boxes(INIT_VEHICLES[name], start_frame)}
        for name in names
    }
    for cut_name, cut_frame in merge_suppression_cuts(
        per_vehicle, anchor_frames
    ).items():
        per_vehicle[cut_name] = {
            frame: record
            for frame, record in per_vehicle[cut_name].items()
            if frame < cut_frame
        }

    # Split any fused blob: where two vehicles overlap (a struck vehicle crushed
    # against the striker, each separately tracked), clip the larger vehicle's
    # mask/box out of the smaller's box so the larger one stops at, rather than
    # swallows, the smaller. Lets a manual post-impact re-seed render as its own box
    # beside the striker instead of inside it.
    split_overlapping_masks(per_vehicle)

    # Truncate every vehicle's box at the impact frame when the scene opts in
    # (``truncate_boxes_at_impact``). After a fusing collision -- the struck
    # vehicle crushed against the striker -- SAM2 cannot separate the two, so the
    # surviving track's box grows to swallow the other. Dropping every box after
    # impact stops that (only the on-approach boxes and the impact marker remain).
    # The impact frame is detected the same way as ``auto_reconstruct`` (deferred
    # import: it pulls the geo writers, unneeded for a plain track run otherwise).
    if SCENE.truncate_boxes_at_impact:
        from accident_reconstruction.auto_reconstruct import (
            project_metric,
            resolve_impact_frame,
        )

        anchors = {
            name: {frame: record[1] for frame, record in records.items()}
            for name, records in per_vehicle.items()
        }
        impact_frame = resolve_impact_frame(SCENE, project_metric(anchors))
        if impact_frame is not None:
            for name, records in per_vehicle.items():
                per_vehicle[name] = {
                    frame: record
                    for frame, record in records.items()
                    if frame <= impact_frame
                }

    # The struck vehicle's trace line is stopped at its flip onset so the drawn
    # route does not loop through the post-impact tumble (where the ground anchor
    # is meaningless). The flip onset is the frame whose anchor is FURTHEST from
    # the start -- forward motion ends there; the tumble only bounces back.
    stop_vehicle = SCENE.resolved_stop_vehicle
    trace_cut: int | None = None
    if per_vehicle.get(stop_vehicle):
        recs = per_vehicle[stop_vehicle]
        start_xy = recs[min(recs)][1]
        trace_cut = max(
            recs,
            key=lambda f: (
                (recs[f][1][0] - start_xy[0]) ** 2 + (recs[f][1][1] - start_xy[1]) ** 2
            ),
        )

    # Real per-frame timestamps (PTS) so speed uses actual elapsed time, not the
    # nominal frame/fps (defends against VFR / dropped-frame YouTube & CCTV clips).
    # Falls back to frame/fps when ffprobe is unavailable; warn if the axis is
    # uneven (variable frame rate) so downstream speeds are read with suspicion.
    pts_times = probe_pts_times(source_video_path)
    if pts_times is None:
        print("(ffprobe unavailable; using nominal frame/fps for t_sec)")
    else:
        warning = time_axis_warning(pts_times[start_frame : end_frame + 1])
        if warning is not None:
            print(warning)

    # Composite render over the full window, overlaying whichever vehicles appear.
    capture = cv2.VideoCapture(source_video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer: cv2.VideoWriter | None = None
    trace: dict[str, list[tuple[int, int]]] = {name: [] for name in names}
    rows: list[list[object]] = []
    # Per-vehicle ground-contact contours -> sidecar npz for Stage 2's
    # orientation-invariant footprint anchor (see ground_footprint).
    contours: dict[str, dict[int, np.ndarray]] = {name: {} for name in names}
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                target_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        t_sec = frame_time(pts_times, frame_index, fps)
        for name in names:
            record = per_vehicle[name].get(frame_index)
            if record is None:
                continue
            box, anchor, mask = record
            contour = gf.subsample(gf.bottom_contour(mask))
            if len(contour):
                contours[name][frame_index] = contour
            color = INIT_VEHICLES[name]["bgr"]
            frame[mask] = (0.45 * np.array(color) + 0.55 * frame[mask]).astype(np.uint8)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.circle(frame, anchor, 5, color, -1)
            cv2.circle(frame, anchor, 6, (255, 255, 255), 1)
            cv2.putText(
                frame,
                name,
                (box[0], box[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            if not (
                name == stop_vehicle
                and trace_cut is not None
                and frame_index > trace_cut
            ):
                trace[name].append(anchor)  # stop the struck vehicle's line at the flip
            rows.append([frame_index, name, *box, *anchor, round(t_sec, 6)])
        for name in names:
            if len(trace[name]) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(trace[name], dtype=np.int32)],
                    False,
                    INIT_VEHICLES[name]["bgr"],
                    2,
                    cv2.LINE_AA,
                )
        writer.write(frame)
    capture.release()

    # Write the tracks CSV -- the most valuable product -- BEFORE the optional
    # video transcode, so a re-encode hiccup can never cost the tracking result.
    with open(tracks_csv_path, "w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(
            [
                "frame",
                "vehicle",
                "x1",
                "y1",
                "x2",
                "y2",
                "anchor_x",
                "anchor_y",
                "t_sec",
            ]
        )
        csv_writer.writerows(rows)

    # Sidecar contours next to the CSV, so Stage 2 can fit footprint anchors.
    contour_path = Path(tracks_csv_path).with_name("contact_contours.npz")
    gf.save_contours(contour_path, contours)

    if writer is not None:
        writer.release()
        ensure_readable_mp4(target_video_path)

    counts = {name: len(per_vehicle[name]) for name in names}
    print(f"Annotated video: {Path(target_video_path).resolve()}")
    print(f"Tracks CSV: {Path(tracks_csv_path).resolve()} ({len(rows)} rows)")
    n_contours = sum(len(by_frame) for by_frame in contours.values())
    print(f"Contact contours: {contour_path.resolve()} ({n_contours} frames)")
    print(f"Frames tracked per vehicle: {counts}")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
