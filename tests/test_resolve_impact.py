"""Unit tests for ``auto_reconstruct.resolve_impact_frame``.

The four call sites (build_data, recognised build/CSV, prompt-track truncate)
share this one resolver: UI override wins, else detect from the metric track.
"""

from __future__ import annotations

from types import SimpleNamespace

from accident_reconstruction.auto_reconstruct import resolve_impact_frame


def test_override_wins_without_touching_metric() -> None:
    scene = SimpleNamespace(impact_frame_override=42)
    # metric deliberately empty: an override must short-circuit detection.
    assert resolve_impact_frame(scene, metric={}) == 42


def test_detects_from_metric_when_no_override() -> None:
    scene = SimpleNamespace(impact_frame_override=None)
    # From detect_impact's docstring example: clean collide-then-separate -> f2.
    metric = {
        "a": {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)},
        "b": {0: (5, 0), 1: (2, 0), 2: (1, 0), 3: (9, 0)},
    }
    assert resolve_impact_frame(scene, metric) == 2


def test_none_when_single_vehicle() -> None:
    scene = SimpleNamespace(impact_frame_override=None)
    assert resolve_impact_frame(scene, {"a": {0: (0, 0), 1: (1, 0)}}) is None
