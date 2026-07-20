"""Video time-axis helpers: real per-frame timestamps (PTS) and their stability.

Speed is ``distance / time``, so a wrong time axis biases every speed reading.
YouTube downloads and CCTV recorders routinely emit variable-frame-rate (VFR)
clips -- dropped, duplicated or unevenly spaced frames -- which makes the naive
``t = frame / fps`` assumption drift and skews speed systematically.

These helpers pull the real presentation timestamps (PTS) with ``ffprobe`` and
quantify how uneven they are, so the pipeline can record a real ``t_sec`` per
frame (falling back to ``frame / fps`` when ffprobe is unavailable) and warn when
the time axis is not trustworthy. Pure/stdlib only -- no OpenCV or ML imports --
so it stays importable on a bare CI runner.

Examples:
    ```python
    from accident_reconstruction.time_axis import interval_cv
    round(interval_cv([0.0, 0.04, 0.08, 0.12]), 6)
    # 0.0
    ```
"""

from __future__ import annotations

import csv
import itertools
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Above this coefficient of variation of frame intervals the time axis is
#: considered unreliable (VFR / dropped frames), so speeds derived from it should
#: be treated with suspicion. 5% per the task spec.
TIME_AXIS_CV_WARN = 0.05


def _find_ffprobe() -> str | None:
    """Locate an ffprobe binary (PATH first, then common install locations)."""
    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path
    candidates = [
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        str(Path.home() / "miniconda3/bin/ffprobe"),
        str(Path.home() / "anaconda3/bin/ffprobe"),
    ]
    return next((path for path in candidates if Path(path).exists()), None)


def probe_pts_times(video_path: str) -> list[float] | None:
    """Return every video frame's presentation timestamp (seconds), or None.

    Runs ``ffprobe -select_streams v:0 -show_entries frame=pts_time -of csv`` and
    parses the ``pts_time`` column. The result is sorted ascending, so index ``i``
    is the ``i``-th frame in presentation order (matching how OpenCV reads
    frames). Returns None when ffprobe is missing, fails, or yields no timestamps,
    so callers can fall back to ``frame / fps``.

    Args:
        video_path: Path to the source clip.

    Returns:
        Per-frame ``pts_time`` in seconds (presentation order), or None.
    """
    ffprobe = _find_ffprobe()
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 -- fixed args, path is our own clip
            [
                ffprobe,
                "-loglevel",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pts_time",
                "-of",
                "csv",
                "-i",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    times: list[float] = []
    for line in result.stdout.splitlines():
        # Each line is "frame,<pts_time>"; some frames may lack a pts_time.
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            times.append(float(parts[1]))
        except ValueError:
            continue
    if not times:
        return None
    times.sort()
    return times


def frame_time(pts_times: Sequence[float] | None, frame: int, fps: float) -> float:
    """Time (seconds) of ``frame``: real PTS when available, else ``frame / fps``.

    Args:
        pts_times: Per-frame PTS from :func:`probe_pts_times`, or None.
        frame: Absolute source-frame index.
        fps: Nominal frames per second, used for the fallback.

    Returns:
        The frame's timestamp in seconds.

    Examples:
        ```python
        frame_time([0.0, 0.5, 1.0], 2, 25.0)
        # 1.0
        frame_time(None, 25, 25.0)  # fallback
        # 1.0
        ```
    """
    if pts_times and 0 <= frame < len(pts_times):
        return pts_times[frame]
    return frame / fps if fps else 0.0


def interval_cv(times: Sequence[float]) -> float | None:
    """Coefficient of variation (std / mean) of consecutive time gaps.

    A steady time axis has CV ~0; VFR / dropped frames push it up. Only strictly
    positive gaps are counted (repeated timestamps are ignored). Returns None when
    there are too few gaps to judge, or the mean gap is non-positive.

    Args:
        times: Timestamps in seconds (any order; sorted internally).

    Returns:
        The CV of the frame intervals, or None.

    Examples:
        ```python
        round(interval_cv([0.0, 0.04, 0.08, 0.12]), 6)  # perfectly even
        # 0.0
        interval_cv([0.0])  # too few gaps
        ```
    """
    ordered = sorted(times)
    gaps = [b - a for a, b in itertools.pairwise(ordered) if b - a > 0]
    if len(gaps) < 2:
        return None
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return None
    variance = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
    return (variance**0.5) / mean


def time_axis_warning(
    times: Sequence[float], threshold: float = TIME_AXIS_CV_WARN
) -> str | None:
    """A warning string when the frame intervals vary more than ``threshold``.

    Args:
        times: Frame timestamps in seconds.
        threshold: CV above which the axis is flagged (default 5%).

    Returns:
        A human-readable warning, or None when the axis looks steady / unknown.

    Examples:
        ```python
        time_axis_warning([0.0, 0.04, 0.08, 0.12]) is None
        # True
        ```
    """
    cv = interval_cv(times)
    if cv is None or cv <= threshold:
        return None
    return (
        f"⚠️  此影片時間軸不可靠：幀間隔變異係數 {cv * 100:.1f}%"
        f"（> {threshold * 100:.0f}%），可能為變動幀率／掉幀／重複幀，"
        "由此推得的車速會有系統性誤差。"
    )


def load_frame_times(csv_path: Path) -> dict[int, float]:
    """Load ``{frame: t_sec}`` from a tracks CSV that carries a ``t_sec`` column.

    Frame timestamps depend only on the frame (not the vehicle), so later rows for
    the same frame just overwrite with the same value. Returns an empty mapping for
    a legacy CSV without the column, letting callers fall back to ``frame / fps``.

    Args:
        csv_path: A tracks CSV (``frame, vehicle, ..., t_sec``).

    Returns:
        ``{frame: t_sec}`` (empty when the column is absent/blank).
    """
    times: dict[int, float] = {}
    if not csv_path.exists():
        return times
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "t_sec" not in reader.fieldnames:
            return times
        for row in reader:
            value = row.get("t_sec")
            frame = row.get("frame")
            if not value or not frame:
                continue
            try:
                times[int(frame)] = float(value)
            except ValueError:
                continue
    return times


def frame_seconds(times: Mapping[int, float] | None, frame: int, fps: float) -> float:
    """Time of ``frame`` from a ``{frame: t_sec}`` map, else ``frame / fps``.

    The dict-keyed companion to :func:`frame_time` (which indexes a PTS list). Used
    by the speed windows, which key their samples by frame index.

    Args:
        times: ``{frame: t_sec}`` (e.g. from :func:`load_frame_times`), or None.
        frame: Frame index.
        fps: Nominal frames per second, used for the fallback.

    Returns:
        The frame's timestamp in seconds.
    """
    if times and frame in times:
        return times[frame]
    return frame / fps if fps else 0.0


def window_elapsed(
    times: Mapping[int, float] | None,
    frame_from: int,
    frame_to: int,
    fps: float,
) -> float:
    """Seconds elapsed between two frames, for the speed-window computation.

    With real per-frame timestamps (``times``) it is the PTS difference, so a
    variable-frame-rate clip is timed correctly. Without them it is the plain
    frame-count delta over ``fps`` -- computed as a single ``(frame_to -
    frame_from) / fps`` division so the fallback is bit-identical to the historic
    ``(frame - first) / fps`` speed formula (individually dividing each frame by
    ``fps`` and subtracting would introduce floating-point drift on tracks with
    frame gaps, perturbing speeds even when no PTS is present).

    Args:
        times: ``{frame: t_sec}`` real timestamps, or None/empty for the fallback.
        frame_from: Earlier frame index.
        frame_to: Later frame index.
        fps: Nominal frames per second, used for the fallback.

    Returns:
        Elapsed seconds from ``frame_from`` to ``frame_to``.

    Examples:
        ```python
        window_elapsed(None, 90, 105, 25.0)  # 0.6 -- single division
        window_elapsed({90: 3.6, 105: 4.3}, 90, 105, 25.0)  # 0.7 -- real PTS
        ```
    """
    if times and frame_from in times and frame_to in times:
        return times[frame_to] - times[frame_from]
    return (frame_to - frame_from) / fps if fps else 0.0
