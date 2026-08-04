"""Tests for the per-vehicle track transform helper."""

from __future__ import annotations

import numpy as np
import pytest

from accident_reconstruction.track_ops import (
    TrackArray,
    common_frames,
    iter_ordered,
    map_tracks,
    track_extent_m,
)


def test_from_track_orders_by_frame():
    array = TrackArray.from_track({5: (2.0, 2.0), 0: (0.0, 0.0), 2: (1.0, 1.0)})
    assert array.frames == [0, 2, 5]
    assert array.positions.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_from_track_empty_keeps_two_columns():
    # A (0,) array would break any consumer that indexes [:, 0].
    assert TrackArray.from_track({}).positions.shape == (0, 2)


def test_with_positions_returns_plain_floats():
    # NumPy scalars serialise as "np.float64(...)" and corrupt the CSV/JSON writers.
    array = TrackArray.from_track({0: (0.0, 0.0), 1: (1.0, 1.0)})
    track = array.with_positions(np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert track == {0: (3.0, 4.0), 1: (5.0, 6.0)}
    assert all(type(v) is float for xy in track.values() for v in xy)


def test_with_positions_rejects_length_change():
    # Silently accepting this would misalign every frame after the dropped sample.
    array = TrackArray.from_track({0: (0.0, 0.0), 1: (1.0, 1.0)})
    with pytest.raises(ValueError, match="preserve the sampling"):
        array.with_positions(np.array([[0.0, 0.0]]))


def test_map_tracks_applies_transform():
    out = map_tracks(
        {"car": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}},
        lambda t, _: t.positions + np.array([10.0, 0.0]),
    )
    assert out["car"] == {0: (10.0, 0.0), 1: (11.0, 0.0), 2: (12.0, 0.0)}


def test_map_tracks_passes_label_to_transform():
    seen: list[str] = []
    map_tracks(
        {"car": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}},
        lambda t, label: (seen.append(label), t.positions)[1],
    )
    assert seen == ["car"]


def test_map_tracks_short_track_passes_through():
    tracks = {"bike": {0: (0.0, 0.0), 1: (1.0, 1.0)}}
    out = map_tracks(tracks, lambda t, _: t.positions * 99.0, min_samples=3)
    assert out["bike"] == tracks["bike"]


def test_map_tracks_none_passes_through():
    tracks = {"car": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}}
    out = map_tracks(tracks, lambda t, _: None)
    assert out["car"] == tracks["car"]


def test_map_tracks_declined_error_passes_through():
    def failing(track_array, label):
        raise np.linalg.LinAlgError("singular")

    tracks = {"car": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}}
    out = map_tracks(tracks, failing, on_error=np.linalg.LinAlgError)
    assert out["car"] == tracks["car"]


def test_map_tracks_does_not_swallow_unlisted_errors():
    # A genuine bug in a stage must surface, not degrade every vehicle to raw.
    def buggy(track_array, label):
        raise KeyError("typo")

    with pytest.raises(KeyError):
        map_tracks(
            {"car": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}},
            buggy,
            on_error=np.linalg.LinAlgError,
        )


def test_map_tracks_never_drops_a_vehicle():
    # The regression this module exists to prevent: a stage may decline a track,
    # but every input label must still appear in the output.
    tracks = {
        "long": {f: (float(f), 0.0) for f in range(10)},
        "short": {0: (0.0, 0.0)},
        "declines": {f: (float(f), 1.0) for f in range(10)},
    }
    out = map_tracks(
        tracks,
        lambda t, label: None if label == "declines" else t.positions * 2.0,
    )
    assert set(out) == set(tracks)
    assert out["short"] == tracks["short"]
    assert out["declines"] == tracks["declines"]
    assert out["long"][5] == (10.0, 0.0)


def test_map_tracks_copies_input():
    tracks = {"car": {0: (0.0, 0.0)}}
    out = map_tracks(tracks, lambda t, _: t.positions)
    out["car"][99] = (1.0, 1.0)
    assert 99 not in tracks["car"]


def test_track_extent_m():
    assert track_extent_m({0: (0.0, 0.0), 5: (3.0, 4.0)}) == pytest.approx(5.0)
    assert track_extent_m({0: (0.0, 0.0)}) == 0.0
    assert track_extent_m({}) == 0.0


def test_common_frames_intersects():
    a = {0: (0.0, 0.0), 1: (1.0, 1.0), 3: (3.0, 3.0)}
    b = {1: (1.0, 1.0), 2: (2.0, 2.0), 3: (3.0, 3.0)}
    assert common_frames(a, b) == [1, 3]
    assert common_frames() == []
    assert common_frames(a) == [0, 1, 3]


def test_iter_ordered():
    assert list(iter_ordered({2: (2.0, 2.0), 0: (0.0, 0.0)})) == [
        (0, (0.0, 0.0)),
        (2, (2.0, 2.0)),
    ]
