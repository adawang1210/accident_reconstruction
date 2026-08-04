"""Per-vehicle track transforms: the dict<->array round-trip, done once.

The pipeline's central data structure is a *track set* --
``{label: {frame: (x, y)}}`` -- but every numerical stage (smoothing, contour
refinement, back-projection, shape preservation) wants a dense, frame-ordered
``(n, 2)`` array instead. So each stage re-implemented the same five steps by
hand: sort the frames, bail out when the track is too short, stack the values,
call the array routine, and rebuild the dict.

That is not a cosmetic repetition. The bail-outs are *policy* -- "what happens to
a vehicle this stage cannot process" -- and hand-writing them at every call site
is precisely how :func:`auto_reconstruct.smooth_metric` came to be skipped on
three separate paths, leaving 7 of 13 recorded vehicle tracks unsmoothed (see
``docs/TRAJECTORY_SMOOTHING.md``). Centralising the round-trip makes the
pass-through the *default* a stage gets for free, rather than something each
caller has to remember to write.

:func:`map_tracks` is the whole interface: give it a track set and a function
over arrays, get a track set back.

Examples:
    ```python
    import numpy as np

    shifted = map_tracks(
        {"car": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}},
        lambda t, label: t.positions + np.array([10.0, 0.0]),
    )
    shifted["car"][1]
    # (11.0, 0.0)
    ```
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# One vehicle's positions by frame, and a set of them by label. Coordinates are
# whatever the stage is working in -- metric (east_m, north_m), pixels, or
# lat/lon -- so these aliases stay unit-agnostic on purpose.
Track = dict[int, tuple[float, float]]
TrackSet = dict[str, Track]


@dataclass(frozen=True)
class TrackArray:
    """One vehicle's track as frame-ordered parallel arrays.

    Attributes:
        frames: Ascending frame numbers. Gaps are preserved, not filled -- the
            numerical stages need the real spacing to compute dt.
        positions: ``(n, 2)`` float64 coordinates aligned to ``frames``.

    Examples:
        ```python
        arr = TrackArray.from_track({2: (1.0, 2.0), 0: (0.0, 0.0)})
        arr.frames
        # [0, 2]
        arr.positions.tolist()
        # [[0.0, 0.0], [1.0, 2.0]]
        ```
    """

    frames: list[int]
    positions: NDArray[np.float64]

    @classmethod
    def from_track(cls, track: Track) -> TrackArray:
        """Stack a ``{frame: (x, y)}`` mapping into frame-ordered arrays."""
        frames = sorted(track)
        positions = np.array([track[f] for f in frames], dtype=np.float64)
        # An empty track yields shape (0,), which breaks the (n, 2) contract that
        # every consumer indexes on; normalise it here rather than at each site.
        if not frames:
            positions = positions.reshape(0, 2)
        return cls(frames, positions)

    def with_positions(self, positions: NDArray[np.float64]) -> Track:
        """Rebuild a ``{frame: (x, y)}`` mapping from new positions.

        Args:
            positions: ``(n, 2)`` coordinates, one row per frame in ``frames``.

        Returns:
            The track mapping, with plain floats (not NumPy scalars, which
            serialise as ``np.float64(...)`` and break the JSON/CSV writers).

        Raises:
            ValueError: If ``positions`` has a different length than ``frames``.
                A stage that drops or inserts samples silently would misalign
                every frame after the change, so fail loudly instead.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) != len(self.frames):
            raise ValueError(
                f"transform returned {len(positions)} points for "
                f"{len(self.frames)} frames; it must preserve the sampling"
            )
        return {
            frame: (float(positions[i][0]), float(positions[i][1]))
            for i, frame in enumerate(self.frames)
        }

    def __len__(self) -> int:
        return len(self.frames)


# A stage: given one vehicle's arrays and its label, return its new positions.
# The label is passed because several stages are label-dependent (per-vehicle
# lengths, the pixel anchors of that same vehicle, per-vehicle colours).
TrackTransform = Callable[[TrackArray, str], NDArray[np.float64] | None]


def map_tracks(
    tracks: TrackSet,
    transform: TrackTransform,
    *,
    min_samples: int = 3,
    on_error: type[Exception] | tuple[type[Exception], ...] = (),
) -> TrackSet:
    """Apply an array transform to every vehicle, with a uniform pass-through.

    A vehicle is returned UNCHANGED (a copy, never the caller's dict) when it is
    shorter than ``min_samples``, when ``transform`` returns None, or when
    ``transform`` raises one of ``on_error``. Every vehicle present in ``tracks``
    is present in the result -- a stage can decline to process a track, but it
    cannot make one disappear.

    Args:
        tracks: ``{label: {frame: (x, y)}}`` to transform.
        transform: Called as ``transform(track_array, label)``; returns the new
            ``(n, 2)`` positions, or None to keep the vehicle as-is.
        min_samples: Skip (pass through) tracks shorter than this. Default 3,
            the shortest track from which an acceleration can be estimated.
        on_error: Exception types to treat as "declined" rather than propagate.
            Pass explicitly -- the default catches nothing, so a genuine bug in a
            stage surfaces instead of silently degrading every vehicle to raw.

    Returns:
        A new track set with the same labels.

    Examples:
        ```python
        import numpy as np

        tracks = {"car": {0: (0.0, 0.0), 1: (1.0, 1.0)}}
        map_tracks(tracks, lambda t, label: t.positions * 2, min_samples=3)
        # {'car': {0: (0.0, 0.0), 1: (1.0, 1.0)}}
        ```
    """
    out: TrackSet = {}
    for label, track in tracks.items():
        array = TrackArray.from_track(track)
        if len(array) < min_samples:
            out[label] = dict(track)
            continue
        try:
            positions = transform(array, label)
        except on_error:
            out[label] = dict(track)
            continue
        out[label] = (
            dict(track) if positions is None else array.with_positions(positions)
        )
    return out


def track_extent_m(track: Track) -> float:
    """Straight-line distance from a track's first sample to its last.

    Reported next to each vehicle in the CLI summaries as a sanity check: an
    extent far larger than the calibrated GCP span means the track leaves the
    region the homography was fitted on, where positions (and so speeds) are
    extrapolated and unreliable.

    Examples:
        ```python
        track_extent_m({0: (0.0, 0.0), 5: (3.0, 4.0)})
        # 5.0
        ```
    """
    array = TrackArray.from_track(track)
    if len(array) < 2:
        return 0.0
    return float(np.hypot(*(array.positions[-1] - array.positions[0])))


def common_frames(*tracks: Track) -> list[int]:
    """Ascending frames present in every given track.

    Used where two per-frame mappings must be read in lockstep (boxes with
    anchors, metric with pixels); taking the intersection avoids the KeyError
    that a plain ``sorted(a)`` walk hits wherever one side has a tracking gap.

    Examples:
        ```python
        common_frames({0: (0.0, 0.0), 1: (1.0, 1.0)}, {1: (2.0, 2.0), 2: (3.0, 3.0)})
        # [1]
        ```
    """
    if not tracks:
        return []
    shared: set[int] = set(tracks[0])
    for track in tracks[1:]:
        shared &= set(track)
    return sorted(shared)


def iter_ordered(track: Track) -> Iterable[tuple[int, tuple[float, float]]]:
    """Yield ``(frame, (x, y))`` in ascending frame order.

    Examples:
        ```python
        list(iter_ordered({1: (1.0, 1.0), 0: (0.0, 0.0)}))
        # [(0, (0.0, 0.0)), (1, (1.0, 1.0))]
        ```
    """
    for frame in sorted(track):
        yield frame, track[frame]
