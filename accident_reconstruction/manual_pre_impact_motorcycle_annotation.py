from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accident_reconstruction.calibrate_homography import (
    USING_GPS_CALIBRATION,  # re-export
    VIEW_TRANSFORMER,  # re-export
    ViewTransformer,
)
from accident_reconstruction.scene_config import SCENE

# NOTE: the cropped/zoomed clip (source_accident_crop_zoom.mp4) was removed; this
# legacy annotator now reads the regular source. All videos live under data/videos/.
SOURCE_VIDEO_PATH = SCENE.source_video
ORIGINAL_SOURCE_VIDEO_PATH = SCENE.source_video
TARGET_VIDEO_PATH = SCENE.video_path("manual_annotation")
SLOW_TARGET_VIDEO_PATH = SCENE.video_path("manual_annotation_slow_2x")
ORIGINAL_SIZE_TARGET_VIDEO_PATH = SCENE.video_path("manual_annotation_original_size")
ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH = SCENE.video_path(
    "manual_annotation_original_size_slow_2x"
)
CHECK_SHEET_PATH = Path(
    "data/scenes/pre_impact_motorcycle/scene/manual_annotation_result_sheet.jpg"
)
END_FRAME_INDEX = 170
STATIC_FRAME_DIFF_THRESHOLD = 0.05
ZOOM_TO_ORIGINAL_SCALE = 0.59375
ZOOM_TO_ORIGINAL_OFFSET = np.array([80.0, 80.0], dtype=np.float32)

# --- Real-world (bird's-eye) calibration -------------------------------------
# Speed cannot be measured in image pixels: under perspective, a fixed real
# speed produces a different pixel speed depending on the distance from the
# camera, so px/s is meaningless and the two vehicles are not comparable. We
# instead project each vehicle's ground-contact point (bottom-center of its box)
# onto a flat, metric "bird's-eye" plane and measure displacement there.
#
# SOURCE_POINTS are four points on the road plane in ORIGINAL (1280x720) camera
# coordinates. They must form a wide quad (NOT a near-straight line, which makes
# the homography degenerate) roughly bounding the region the vehicles travel
# through. Order: near-left, near-right, far-right, far-left.
SOURCE_POINTS = np.array(
    [[300.0, 700.0], [1000.0, 560.0], [760.0, 440.0], [380.0, 470.0]],
    dtype=np.float32,
)
# Road width across the carriageway, from the on-map measurement (~5.6 m). This
# sets the SHAPE (aspect) of the bird's-eye plane; the ABSOLUTE scale is fixed by
# the along-road distance below.
ROAD_WIDTH_M = 5.6
# Base length of the calibration rectangle along the road (target units). The
# plane is uniformly rescaled so the projected impact->underpass distance equals
# IMPACT_TO_UNDERPASS_M, so this base value only affects the aspect ratio.
BASE_ROAD_LENGTH_M = 15.0

# Absolute-scale calibration. Two reliable points on the car's labeled path, in
# ORIGINAL (1280x720) camera coordinates: where the car is at the moment of the
# collision, and where it ends up inside the underpass. The real distance between
# them (measured along the road on the map) sets the speed scale -- every km/h
# value is linear in IMPACT_TO_UNDERPASS_M.
CAR_IMPACT_CAM = (669.0, 491.0)
CAR_FINAL_CAM = (675.0, 407.0)
# Approximate calibration: the exact value is hard to pin down from this camera,
# so it is set to give plausible urban speeds (motorcycle peak ~55 km/h near the
# posted 35 km/h underpass). Change this one number to rescale every speed.
IMPACT_TO_UNDERPASS_M = 10.0

# Sliding window used to smooth the per-frame speed estimate (seconds).
SPEED_WINDOW_SECONDS = 0.6

# --- Bird's-eye panel rendering ----------------------------------------------
# The panel auto-frames the vehicle paths (see view_bounds); these control only
# its resolution and styling.
BIRDSEYE_PX_PER_M = 26
BIRDSEYE_MARGIN_M = 4.0
BIRDSEYE_GRID_M = 5.0
BIRDSEYE_BG_COLOR = (24, 24, 24)
BIRDSEYE_ROAD_COLOR = (54, 54, 54)
BIRDSEYE_GRID_COLOR = (70, 70, 70)


def _calibrated_target() -> np.ndarray:
    """Build the metric target rectangle, scaled to the real road distance.

    A base rectangle (``ROAD_WIDTH_M`` x ``BASE_ROAD_LENGTH_M``) fixes the plane
    shape; it is then uniformly rescaled so the projected distance between
    ``CAR_IMPACT_CAM`` and ``CAR_FINAL_CAM`` equals ``IMPACT_TO_UNDERPASS_M``.

    Returns:
        The four target points in meters.
    """
    base = np.array(
        [
            [0.0, 0.0],
            [ROAD_WIDTH_M, 0.0],
            [ROAD_WIDTH_M, BASE_ROAD_LENGTH_M],
            [0.0, BASE_ROAD_LENGTH_M],
        ],
        dtype=np.float32,
    )
    base_transformer = ViewTransformer(source=SOURCE_POINTS, target=base)
    reference = base_transformer.transform_points(
        np.array([CAR_IMPACT_CAM, CAR_FINAL_CAM], dtype=np.float32)
    )
    base_distance = float(np.linalg.norm(reference[1] - reference[0]))
    scale = IMPACT_TO_UNDERPASS_M / base_distance
    return base * scale


TARGET_POINTS = _calibrated_target()
# VIEW_TRANSFORMER / USING_GPS_CALIBRATION / ORIGIN_LATLON / metric_to_latlon
# are now canonical in calibrate_homography and re-imported above; the GPS
# calibration overrides TARGET_POINTS/VIEW_TRANSFORMER there at import time.


@dataclass(frozen=True)
class KeyFrameBox:
    """A manually labeled object box on one video frame.

    Attributes:
        frame_index: Zero-based frame index.
        xyxy: Bounding box in `(x_min, y_min, x_max, y_max)` format.

    Examples:
        ```python
        KeyFrameBox(frame_index=95, xyxy=(480, 480, 555, 595))
        ```
    """

    frame_index: int
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class ManualTrack:
    """A manually tracked object with display styling.

    Attributes:
        label: Object label shown in the video.
        key_frame_boxes: Manually labeled boxes for interpolation.
        box_color: BGR color used for the box and label background.
        trace_color: BGR color used for the path line.
        current_point_color: BGR color used for the current center point.

    Examples:
        ```python
        ManualTrack(
            label="motorcycle",
            key_frame_boxes=[KeyFrameBox(0, (0, 0, 10, 10))],
            box_color=(86, 255, 255),
            trace_color=(0, 255, 255),
            current_point_color=(0, 128, 255),
        )
        ```
    """

    label: str
    key_frame_boxes: list[KeyFrameBox]
    box_color: tuple[int, int, int]
    trace_color: tuple[int, int, int]
    current_point_color: tuple[int, int, int]


@dataclass
class TrackState:
    """Runtime state for a manually tracked object.

    Attributes:
        trace_points: Accumulated image-space center points used to draw the path
            on the camera view.
        metric_trace: Accumulated bird's-eye plane points (meters) used to draw
            the path on the 2D panel.
        metric_samples: Recent `(frame_index, (x_m, y_m))` samples inside the
            speed window, used to compute a smoothed speed.
        speed_kmh: Most recent speed estimate in kilometers per hour.

    Examples:
        ```python
        TrackState().speed_kmh
        # 0.0
        ```
    """

    trace_points: list[tuple[int, int]] = field(default_factory=list)
    metric_trace: list[tuple[float, float]] = field(default_factory=list)
    metric_samples: deque[tuple[int, tuple[float, float]]] = field(
        default_factory=deque
    )
    speed_kmh: float = 0.0


MOTORCYCLE_KEY_FRAME_BOXES = [
    KeyFrameBox(frame_index=80, xyxy=(245, 430, 320, 535)),
    KeyFrameBox(frame_index=85, xyxy=(320, 440, 395, 550)),
    KeyFrameBox(frame_index=90, xyxy=(390, 455, 465, 570)),
    KeyFrameBox(frame_index=95, xyxy=(480, 480, 555, 595)),
    KeyFrameBox(frame_index=100, xyxy=(635, 470, 738, 616)),
    KeyFrameBox(frame_index=105, xyxy=(852, 490, 982, 655)),
    KeyFrameBox(frame_index=110, xyxy=(902, 500, 1012, 648)),
    KeyFrameBox(frame_index=115, xyxy=(902, 500, 1012, 648)),
    KeyFrameBox(frame_index=120, xyxy=(900, 505, 1010, 650)),
    KeyFrameBox(frame_index=130, xyxy=(900, 505, 1015, 650)),
]

CAR_KEY_FRAME_BOXES = [
    KeyFrameBox(frame_index=80, xyxy=(1044, 1002, 1465, 1078)),
    KeyFrameBox(frame_index=85, xyxy=(1090, 805, 1280, 876)),
    KeyFrameBox(frame_index=90, xyxy=(965, 735, 1280, 876)),
    KeyFrameBox(frame_index=95, xyxy=(945, 650, 1225, 876)),
    KeyFrameBox(frame_index=100, xyxy=(955, 590, 1180, 810)),
    KeyFrameBox(frame_index=105, xyxy=(930, 555, 1125, 735)),
    KeyFrameBox(frame_index=110, xyxy=(895, 525, 1070, 690)),
    KeyFrameBox(frame_index=115, xyxy=(885, 515, 1065, 680)),
    KeyFrameBox(frame_index=120, xyxy=(900, 525, 1105, 700)),
    KeyFrameBox(frame_index=130, xyxy=(900, 520, 1105, 690)),
    KeyFrameBox(frame_index=140, xyxy=(940, 500, 1135, 650)),
    KeyFrameBox(frame_index=150, xyxy=(1000, 480, 1190, 625)),
    KeyFrameBox(frame_index=160, xyxy=(970, 450, 1165, 590)),
    KeyFrameBox(frame_index=170, xyxy=(920, 425, 1085, 550)),
]

MANUAL_TRACKS = [
    ManualTrack(
        label="motorcycle",
        key_frame_boxes=MOTORCYCLE_KEY_FRAME_BOXES,
        box_color=(86, 255, 255),
        trace_color=(0, 255, 255),
        current_point_color=(0, 128, 255),
    ),
    ManualTrack(
        label="car",
        key_frame_boxes=CAR_KEY_FRAME_BOXES,
        box_color=(255, 170, 50),
        trace_color=(255, 120, 0),
        current_point_color=(255, 255, 255),
    ),
]


def interpolate_box(
    frame_index: int, key_frames: list[KeyFrameBox]
) -> np.ndarray | None:
    """Interpolate a manual box for a frame between two labeled keyframes.

    Args:
        frame_index: Zero-based frame index.
        key_frames: Manually labeled keyframe boxes sorted by frame index.

    Returns:
        Interpolated `(x_min, y_min, x_max, y_max)` box, or `None` when the frame is
        outside the labeled range.

    Examples:
        ```python
        boxes = [
            KeyFrameBox(0, (0, 0, 10, 10)),
            KeyFrameBox(10, (10, 10, 20, 20)),
        ]
        interpolate_box(5, boxes).tolist()
        # [5, 5, 15, 15]
        ```
    """
    if (
        frame_index < key_frames[0].frame_index
        or frame_index > key_frames[-1].frame_index
    ):
        return None

    for start, end in pairwise(key_frames):
        if start.frame_index <= frame_index <= end.frame_index:
            span = end.frame_index - start.frame_index
            if span == 0:
                return np.array(start.xyxy, dtype=np.int32)

            ratio = (frame_index - start.frame_index) / span
            start_box = np.array(start.xyxy, dtype=np.float32)
            end_box = np.array(end.xyxy, dtype=np.float32)
            return np.rint(start_box + (end_box - start_box) * ratio).astype(np.int32)

    return np.array(key_frames[-1].xyxy, dtype=np.int32)


def get_box_center(box: np.ndarray) -> tuple[int, int]:
    """Calculate the center point of a box.

    Args:
        box: `(x_min, y_min, x_max, y_max)` box.

    Returns:
        Center point as `(x, y)`.

    Examples:
        ```python
        get_box_center(np.array([10, 20, 30, 60]))
        # (20, 40)
        ```
    """
    x_min, y_min, x_max, y_max = box.tolist()
    return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))


def get_ground_anchor(box: np.ndarray) -> tuple[int, int]:
    """Return the ground-contact anchor (bottom-center) of a box.

    The bottom-center point sits on the road plane, so it is the correct anchor
    for a perspective projection. The box center, by contrast, drifts upward as
    the box grows when the vehicle approaches the camera.

    Args:
        box: `(x_min, y_min, x_max, y_max)` box.

    Returns:
        Bottom-center point as `(x, y)`.

    Examples:
        ```python
        get_ground_anchor(np.array([10, 20, 30, 60]))
        # (20, 60)
        ```
    """
    x_min, _, x_max, y_max = box.tolist()
    return (int((x_min + x_max) / 2), int(y_max))


def update_metric_speed(
    state: TrackState,
    anchor: tuple[int, int],
    frame_index: int,
    fps: float,
    transformer: ViewTransformer = VIEW_TRANSFORMER,
    window_seconds: float = SPEED_WINDOW_SECONDS,
) -> tuple[float, float]:
    """Update the speed estimate from a ground anchor in the metric plane.

    The anchor is projected to the bird's-eye plane (meters) and stored. Speed is
    the displacement across the samples inside ``window_seconds`` divided by the
    elapsed time, which smooths out single-frame jitter.

    Args:
        state: Mutable tracking state.
        anchor: Ground-contact point in ORIGINAL camera coordinates.
        frame_index: Current frame index.
        fps: Source video frames per second.
        transformer: Camera-to-metric transformer.
        window_seconds: Length of the smoothing window in seconds.

    Returns:
        The projected `(x_m, y_m)` metric position of ``anchor``.

    Examples:
        ```python
        state = TrackState()
        _ = update_metric_speed(state, (640, 700), 0, 25)
        metric = update_metric_speed(state, (640, 470), 25, 25)
        len(metric)
        # 2
        ```
    """
    metric = transformer.transform_points(np.array([anchor], dtype=np.float32))[0]
    metric_xy = (float(metric[0]), float(metric[1]))
    state.metric_samples.append((frame_index, metric_xy))

    window_frames = window_seconds * fps
    while (
        len(state.metric_samples) >= 2
        and frame_index - state.metric_samples[0][0] > window_frames
    ):
        state.metric_samples.popleft()

    if len(state.metric_samples) >= 2:
        first_frame, first_xy = state.metric_samples[0]
        last_frame, last_xy = state.metric_samples[-1]
        elapsed_seconds = (last_frame - first_frame) / fps
        if elapsed_seconds > 0:
            distance_m = float(np.linalg.norm(np.array(last_xy) - np.array(first_xy)))
            state.speed_kmh = distance_m / elapsed_seconds * 3.6

    return metric_xy


def update_track_speed(
    state: TrackState, track: ManualTrack, frame_index: int, fps: float
) -> tuple[float, float] | None:
    """Advance a track's metric speed from its keyframe position.

    Speed is driven by the manual keyframe motion model, which already encodes
    any source-video freeze as a repeated (static) box. Computing it every frame
    -- independent of the video-freeze detection -- keeps it correct on frozen
    frames: a vehicle stopped at impact reads ~0 instead of holding its last
    pre-impact speed.

    Args:
        state: Mutable tracking state.
        track: Manual track whose keyframes define the motion.
        frame_index: Current frame index.
        fps: Source video frames per second.

    Returns:
        The projected metric `(x_m, y_m)` position, or `None` when the track has
        no keyframe at this frame.

    Examples:
        ```python
        state = TrackState()
        _ = update_track_speed(state, MANUAL_TRACKS[1], 80, 25)
        isinstance(update_track_speed(state, MANUAL_TRACKS[1], 90, 25), tuple)
        # True
        ```
    """
    box = interpolate_box(frame_index=frame_index, key_frames=track.key_frame_boxes)
    if box is None:
        return None
    anchor = get_ground_anchor(project_box_to_original_size(box))
    return update_metric_speed(
        state=state, anchor=anchor, frame_index=frame_index, fps=fps
    )


def format_speed(speed_kmh: float) -> str:
    """Format a speed estimate for display.

    Args:
        speed_kmh: Speed in kilometers per hour.

    Returns:
        Speed label in km/h.

    Examples:
        ```python
        format_speed(43.7)
        # '44 km/h'
        ```
    """
    return f"{round(speed_kmh)} km/h"


def project_box_to_original_size(box: np.ndarray) -> np.ndarray:
    """Project a crop-zoom box back to the original video coordinates.

    Args:
        box: Crop-zoom `(x_min, y_min, x_max, y_max)` box.

    Returns:
        Original-video `(x_min, y_min, x_max, y_max)` box.

    Examples:
        ```python
        project_box_to_original_size(np.array([0, 0, 100, 100])).tolist()
        # [80, 80, 139, 139]
        ```
    """
    points = np.array(
        [[box[0], box[1]], [box[2], box[3]]],
        dtype=np.float32,
    )
    projected = points * ZOOM_TO_ORIGINAL_SCALE + ZOOM_TO_ORIGINAL_OFFSET
    return np.rint(projected.reshape(-1)).astype(np.int32)


def frame_difference(previous_frame: np.ndarray | None, frame: np.ndarray) -> float:
    """Measure visual change between two consecutive frames.

    Args:
        previous_frame: Previous video frame, or `None` for the first frame.
        frame: Current video frame.

    Returns:
        Mean absolute grayscale pixel difference. Lower values indicate that the
        source video may be frozen.

    Examples:
        ```python
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame_difference(frame, frame)
        # 0.0
        ```
    """
    if previous_frame is None:
        return float("inf")

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.resize(previous_gray, (320, 219))
    frame_gray = cv2.resize(frame_gray, (320, 219))
    return float(np.mean(cv2.absdiff(previous_gray, frame_gray)))


def draw_trace(
    frame: np.ndarray,
    trace_points: list[tuple[int, int]],
    trace_color: tuple[int, int, int],
    current_point_color: tuple[int, int, int],
) -> np.ndarray:
    """Draw an accumulated object path.

    Args:
        frame: Video frame.
        trace_points: Accumulated center points.
        trace_color: BGR line color.
        current_point_color: BGR current point color.

    Returns:
        Annotated frame.

    Examples:
        ```python
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_trace(frame, [(10, 10), (20, 20)], (0, 255, 255), (0, 128, 255)).shape
        # (100, 100, 3)
        ```
    """
    if not trace_points:
        return frame

    if len(trace_points) >= 2:
        for start, end in pairwise(trace_points):
            cv2.line(frame, start, end, trace_color, 4, cv2.LINE_AA)

    for point in trace_points:
        cv2.circle(frame, point, 4, trace_color, -1, cv2.LINE_AA)

    cv2.circle(frame, trace_points[-1], 8, current_point_color, -1, cv2.LINE_AA)
    return frame


def draw_persisted_trace(
    frame: np.ndarray, track: ManualTrack, trace_points: list[tuple[int, int]]
) -> np.ndarray:
    """Draw an object's path even when its current box is unavailable.

    Args:
        frame: Video frame.
        track: Manual track display configuration.
        trace_points: Accumulated center points.

    Returns:
        Annotated frame with the persisted path.

    Examples:
        ```python
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_persisted_trace(frame, MANUAL_TRACKS[0], [(10, 10), (20, 20)]).shape
        # (100, 100, 3)
        ```
    """
    return draw_trace(
        frame=frame,
        trace_points=trace_points,
        trace_color=track.trace_color,
        current_point_color=track.current_point_color,
    )


def draw_manual_track(
    frame: np.ndarray,
    box: np.ndarray,
    trace_points: list[tuple[int, int]],
    track: ManualTrack,
    speed_label: str | None = None,
) -> np.ndarray:
    """Draw a manually tracked object box and path.

    Args:
        frame: Video frame.
        box: `(x_min, y_min, x_max, y_max)` box.
        trace_points: Accumulated center points.
        track: Manual track display configuration.
        speed_label: Optional speed label appended after the object label.

    Returns:
        Annotated frame.

    Examples:
        ```python
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.array([10, 10, 20, 20])
        draw_manual_track(frame, box, [(15, 15)], MANUAL_TRACKS[0]).shape
        # (100, 100, 3)
        ```
    """
    x_min, y_min, x_max, y_max = box.tolist()
    display_label = (
        track.label if speed_label is None else f"{track.label} {speed_label}"
    )
    label_width = 260 if speed_label is not None else 150
    if track.label == "car" and speed_label is None:
        label_width = 80
    label_offset = 62 if track.label == "motorcycle" else 36
    label_top = max(0, y_min - label_offset)
    label_bottom = max(label_top + 28, y_min - label_offset + 32)
    frame = draw_trace(
        frame=frame,
        trace_points=trace_points,
        trace_color=track.trace_color,
        current_point_color=track.current_point_color,
    )
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), track.box_color, 3)
    cv2.rectangle(
        frame,
        (x_min, label_top),
        (min(frame.shape[1], x_min + label_width), label_bottom),
        track.box_color,
        -1,
    )
    cv2.putText(
        frame,
        display_label,
        (x_min + 8, label_bottom - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def _camera_up_rotation() -> np.ndarray:
    """Build the rotation that aligns the metric plane with the camera heading.

    The user wants the bird's-eye panel oriented like the dashcam, not north-up.
    We read the camera's "up" direction (decreasing pixel y) as a vector in the
    metric plane via the homography, then rotate so it points to panel-up. The
    rotation is an isometry, so speeds (measured in raw meters) are unaffected.
    Falls back to identity when no GPS calibration is loaded.

    Returns:
        A ``(2, 2)`` rotation matrix from metric meters to display meters.
    """
    if not USING_GPS_CALIBRATION:
        return np.eye(2, dtype=np.float32)
    low = VIEW_TRANSFORMER.transform_points(np.array([[640.0, 680.0]], np.float32))[0]
    high = VIEW_TRANSFORMER.transform_points(np.array([[640.0, 430.0]], np.float32))[0]
    up = high - low
    theta = math.pi / 2 - math.atan2(float(up[1]), float(up[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)


_DISPLAY_ROTATION = _camera_up_rotation()


def to_display(point_m: tuple[float, float]) -> tuple[float, float]:
    """Rotate a raw metric point into the camera-aligned display frame.

    Args:
        point_m: `(east_m, north_m)` point on the metric plane.

    Returns:
        `(x_m, y_m)` in the rotated display frame.

    Examples:
        ```python
        len(to_display((1.0, 2.0)))
        # 2
        ```
    """
    vector = _DISPLAY_ROTATION @ np.array(point_m, dtype=np.float32)
    return (float(vector[0]), float(vector[1]))


_VIEW_BOUNDS: tuple[float, float, float, float] | None = None


def view_bounds() -> tuple[float, float, float, float]:
    """Return the display extent shown in the bird's-eye panel.

    Computed once from the projected vehicle paths (plus a margin), in the
    camera-aligned display frame, so the panel auto-frames the data.

    Returns:
        `(x_min, x_max, y_min, y_max)` in display meters.

    Examples:
        ```python
        len(view_bounds())
        # 4
        ```
    """
    global _VIEW_BOUNDS
    if _VIEW_BOUNDS is not None:
        return _VIEW_BOUNDS

    points = []
    for track in MANUAL_TRACKS:
        for key_frame in track.key_frame_boxes:
            box = project_box_to_original_size(np.array(key_frame.xyxy))
            metric = VIEW_TRANSFORMER.transform_points(
                np.array([get_ground_anchor(box)], dtype=np.float32)
            )[0]
            points.append(to_display((float(metric[0]), float(metric[1]))))
    for target in TARGET_POINTS:
        points.append(to_display((float(target[0]), float(target[1]))))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = BIRDSEYE_MARGIN_M
    _VIEW_BOUNDS = (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )
    return _VIEW_BOUNDS


def display_to_panel(point_m: tuple[float, float]) -> tuple[int, int]:
    """Convert a display-frame point to panel pixel coordinates.

    The panel shows `view_bounds` left-to-right and bottom-to-top (larger ``y``
    sits higher). Use this for axis-aligned panel furniture (grid, bounds).

    Args:
        point_m: `(x_m, y_m)` point already in the display frame.

    Returns:
        `(x_px, y_px)` point on the panel.

    Examples:
        ```python
        len(display_to_panel((0.0, 0.0)))
        # 2
        ```
    """
    x_min, _, y_min, y_max = view_bounds()
    height_px = round((y_max - y_min) * BIRDSEYE_PX_PER_M)
    x_px = round((point_m[0] - x_min) * BIRDSEYE_PX_PER_M)
    y_px = height_px - round((point_m[1] - y_min) * BIRDSEYE_PX_PER_M)
    return (x_px, y_px)


def metric_to_panel(point_m: tuple[float, float]) -> tuple[int, int]:
    """Convert a raw metric bird's-eye point to panel pixel coordinates.

    Applies the camera-alignment rotation, then maps to the panel. Use this for
    projected data (vehicle paths, markers) given in raw metric meters.

    Args:
        point_m: `(east_m, north_m)` point on the metric plane.

    Returns:
        `(x_px, y_px)` point on the panel.

    Examples:
        ```python
        len(metric_to_panel((0.0, 0.0)))
        # 2
        ```
    """
    return display_to_panel(to_display(point_m))


def create_metric_birdseye_base() -> np.ndarray:
    """Create the empty bird's-eye panel with a meter grid and road overlay.

    Returns:
        Base panel image.

    Examples:
        ```python
        create_metric_birdseye_base().shape[2]
        # 3
        ```
    """
    x_min, x_max, y_min, y_max = view_bounds()
    width_px = round((x_max - x_min) * BIRDSEYE_PX_PER_M)
    height_px = round((y_max - y_min) * BIRDSEYE_PX_PER_M)
    panel = np.full((height_px, width_px, 3), BIRDSEYE_BG_COLOR, dtype=np.uint8)

    # Shade the surveyed ground patch: the CONVEX HULL of the GCP calibration
    # points (real meters), which marks where the homography is interpolating
    # rather than extrapolating. Hull keeps it a clean region for any point count.
    quad = np.array(
        [metric_to_panel((float(x), float(y))) for x, y in TARGET_POINTS],
        dtype=np.int32,
    )
    hull = cv2.convexHull(quad)
    cv2.fillPoly(panel, [hull], BIRDSEYE_ROAD_COLOR, cv2.LINE_AA)

    # Meter grid (axis-aligned in the display frame).
    grid_x = np.arange(
        np.ceil(x_min / BIRDSEYE_GRID_M) * BIRDSEYE_GRID_M, x_max, BIRDSEYE_GRID_M
    )
    for value in grid_x:
        top = display_to_panel((value, y_max))
        bottom = display_to_panel((value, y_min))
        cv2.line(panel, top, bottom, BIRDSEYE_GRID_COLOR, 1, cv2.LINE_AA)
    grid_y = np.arange(
        np.ceil(y_min / BIRDSEYE_GRID_M) * BIRDSEYE_GRID_M, y_max, BIRDSEYE_GRID_M
    )
    for value in grid_y:
        left = display_to_panel((x_min, value))
        right = display_to_panel((x_max, value))
        cv2.line(panel, left, right, BIRDSEYE_GRID_COLOR, 1, cv2.LINE_AA)

    # Mark the underpass mouth, where the car ends up after the impact.
    tunnel = VIEW_TRANSFORMER.transform_points(
        np.array([CAR_FINAL_CAM], dtype=np.float32)
    )[0]
    tunnel_px = metric_to_panel((float(tunnel[0]), float(tunnel[1])))
    cv2.drawMarker(
        panel, tunnel_px, (180, 180, 180), cv2.MARKER_TRIANGLE_UP, 18, 2, cv2.LINE_AA
    )
    cv2.putText(
        panel,
        "underpass",
        (tunnel_px[0] - 36, tunnel_px[1] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"2D bird's-eye (grid = {BIRDSEYE_GRID_M:.0f} m)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_metric_birdseye(
    panel: np.ndarray,
    track: ManualTrack,
    metric_trace: list[tuple[float, float]],
    speed_label: str,
    is_current: bool,
) -> np.ndarray:
    """Draw a track's path and current position on the bird's-eye panel.

    Args:
        panel: Bird's-eye panel to draw on.
        track: Manual track display configuration.
        metric_trace: Accumulated metric `(x_m, y_m)` points.
        speed_label: Current speed label.
        is_current: Whether the object is visible on the current frame.

    Returns:
        Annotated panel.

    Examples:
        ```python
        panel = create_metric_birdseye_base()
        trace = [(0.0, 5.0)]
        draw_metric_birdseye(panel, MANUAL_TRACKS[0], trace, "44 km/h", True).shape[2]
        # 3
        ```
    """
    points = [metric_to_panel(point) for point in metric_trace]
    if len(points) >= 2:
        for start, end in pairwise(points):
            cv2.line(panel, start, end, track.trace_color, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(panel, point, 3, track.trace_color, -1, cv2.LINE_AA)

    if not points:
        return panel

    current = points[-1]
    radius = 9 if is_current else 6
    cv2.circle(panel, current, radius, track.current_point_color, -1, cv2.LINE_AA)
    cv2.circle(panel, current, radius, track.box_color, 2, cv2.LINE_AA)

    label = f"{track.label} {speed_label}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    text_x = min(panel.shape[1] - text_width - 6, current[0] + 12)
    text_y = max(text_height + 6, current[1] - 8)
    cv2.rectangle(
        panel,
        (text_x - 4, text_y - text_height - 4),
        (text_x + text_width + 4, text_y + 4),
        track.box_color,
        -1,
    )
    cv2.putText(
        panel,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return panel


def write_manual_annotation_video() -> None:
    """Create regular-speed and slow manual annotation videos."""
    cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {SOURCE_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    TARGET_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    regular_writer = cv2.VideoWriter(
        str(TARGET_VIDEO_PATH), fourcc, fps, (width, height)
    )
    slow_writer = cv2.VideoWriter(
        str(SLOW_TARGET_VIDEO_PATH), fourcc, fps / 2, (width, height)
    )

    frame_index = 0
    previous_frame: np.ndarray | None = None
    last_boxes_by_label: dict[str, np.ndarray] = {}
    states_by_label: dict[str, TrackState] = {
        track.label: TrackState() for track in MANUAL_TRACKS
    }
    while True:
        ok, frame = cap.read()
        if not ok or frame_index > END_FRAME_INDEX:
            break

        raw_frame = frame.copy()
        is_static_frame = (
            frame_difference(previous_frame=previous_frame, frame=frame)
            < STATIC_FRAME_DIFF_THRESHOLD
        )

        for track in MANUAL_TRACKS:
            state = states_by_label[track.label]
            metric_xy = update_track_speed(state, track, frame_index, fps)

            if is_static_frame:
                box = last_boxes_by_label.get(track.label)
            else:
                box = interpolate_box(
                    frame_index=frame_index, key_frames=track.key_frame_boxes
                )
            if box is None:
                if state.trace_points:
                    frame = draw_persisted_trace(
                        frame=frame,
                        track=track,
                        trace_points=state.trace_points,
                    )
                continue

            last_boxes_by_label[track.label] = box
            if not is_static_frame:
                state.trace_points.append(get_box_center(box=box))
                if metric_xy is not None:
                    state.metric_trace.append(metric_xy)
            frame = draw_manual_track(
                frame=frame,
                box=box,
                trace_points=state.trace_points,
                track=track,
                speed_label=format_speed(state.speed_kmh),
            )

        regular_writer.write(frame)
        slow_writer.write(frame)
        previous_frame = raw_frame
        frame_index += 1

    cap.release()
    regular_writer.release()
    slow_writer.release()


def write_original_size_annotation_video() -> None:
    """Create regular-speed and slow annotation videos in the original dimensions."""
    source_cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    original_cap = cv2.VideoCapture(str(ORIGINAL_SOURCE_VIDEO_PATH))
    if not source_cap.isOpened():
        raise FileNotFoundError(f"Could not open {SOURCE_VIDEO_PATH}")
    if not original_cap.isOpened():
        raise FileNotFoundError(f"Could not open {ORIGINAL_SOURCE_VIDEO_PATH}")

    fps = original_cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ORIGINAL_SIZE_TARGET_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    regular_writer = cv2.VideoWriter(
        str(ORIGINAL_SIZE_TARGET_VIDEO_PATH), fourcc, fps, (width, height)
    )
    slow_writer = cv2.VideoWriter(
        str(ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH), fourcc, fps / 2, (width, height)
    )

    frame_index = 0
    previous_source_frame: np.ndarray | None = None
    last_boxes_by_label: dict[str, np.ndarray] = {}
    states_by_label: dict[str, TrackState] = {
        track.label: TrackState() for track in MANUAL_TRACKS
    }
    while True:
        source_ok, source_frame = source_cap.read()
        original_ok, original_frame = original_cap.read()
        if not source_ok or not original_ok or frame_index > END_FRAME_INDEX:
            break

        is_static_frame = (
            frame_difference(previous_frame=previous_source_frame, frame=source_frame)
            < STATIC_FRAME_DIFF_THRESHOLD
        )

        for track in MANUAL_TRACKS:
            state = states_by_label[track.label]
            metric_xy = update_track_speed(state, track, frame_index, fps)

            if is_static_frame:
                box = last_boxes_by_label.get(track.label)
            else:
                box = interpolate_box(
                    frame_index=frame_index, key_frames=track.key_frame_boxes
                )
                if box is not None:
                    box = project_box_to_original_size(box=box)
            if box is None:
                if state.trace_points:
                    original_frame = draw_persisted_trace(
                        frame=original_frame,
                        track=track,
                        trace_points=state.trace_points,
                    )
                continue

            last_boxes_by_label[track.label] = box
            if not is_static_frame:
                state.trace_points.append(get_box_center(box=box))
                if metric_xy is not None:
                    state.metric_trace.append(metric_xy)
            original_frame = draw_manual_track(
                frame=original_frame,
                box=box,
                trace_points=state.trace_points,
                track=track,
                speed_label=format_speed(state.speed_kmh),
            )

        regular_writer.write(original_frame)
        slow_writer.write(original_frame)
        previous_source_frame = source_frame
        frame_index += 1

    source_cap.release()
    original_cap.release()
    regular_writer.release()
    slow_writer.release()


def write_check_sheet() -> None:
    """Create a contact sheet around the manually labeled impact frames."""
    cap = cv2.VideoCapture(str(TARGET_VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {TARGET_VIDEO_PATH}")

    thumbnails = []
    for frame_index in [80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        cv2.rectangle(frame, (16, 16), (210, 58), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"frame {frame_index}",
            (28, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(
            cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        )

    cap.release()

    rows = []
    for index in range(0, len(thumbnails), 2):
        row = thumbnails[index : index + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 245))
        rows.append(np.hstack(row))

    CHECK_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CHECK_SHEET_PATH), np.vstack(rows))


if __name__ == "__main__":
    write_manual_annotation_video()
    write_original_size_annotation_video()
    write_check_sheet()
    print(f"Manual annotation video: {TARGET_VIDEO_PATH.resolve()}")
    print(f"Slow manual annotation video: {SLOW_TARGET_VIDEO_PATH.resolve()}")
    print(f"Original-size video: {ORIGINAL_SIZE_TARGET_VIDEO_PATH.resolve()}")
    print(f"Original-size slow video: {ORIGINAL_SIZE_SLOW_TARGET_VIDEO_PATH.resolve()}")
    print(f"Check sheet: {CHECK_SHEET_PATH.resolve()}")
