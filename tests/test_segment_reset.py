"""Regression test: a re-seed segment resets the predictor's per-video state.

Reusing one SAM2VideoPredictor across a vehicle's re-seed segments leaks video
memory in this ultralytics build, so a later re-seed (e.g. a manual post-impact
box) gets dragged back to an earlier segment's mask. ``_segment_masks`` guards
against this by clearing ``inference_state`` and ``prompts`` before each call.
This test pins that contract with a fake predictor -- no SAM2/weights needed --
so a regression (dropping the reset, or reusing without clearing) fails fast.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

import cv2
import numpy as np

from accident_reconstruction.prompt_track_accident import _segment_masks


class _FakePredictor:
    """Stand-in that records its state at call time and yields no masks."""

    def __init__(self) -> None:
        # Pretend a previous segment left memory and stored prompts behind.
        self.inference_state = {"leftover_memory": object()}
        self.prompts = {"bboxes": [[1, 2, 3, 4]]}
        self.state_at_call: tuple[dict, dict] | None = None

    def __call__(self, source, bboxes, stream):
        self.state_at_call = (dict(self.inference_state), dict(self.prompts))
        return iter(())  # no results -- we only care about the reset


def _tiny_clip(path: str) -> None:
    """Write a 3-frame black clip so ``trim_clip`` has something to read."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()


def test_segment_masks_resets_state_before_inference(tmp_path) -> None:
    """The predictor's memory and stored prompts are cleared before the call."""
    clip = tmp_path / "clip.mp4"
    _tiny_clip(str(clip))
    predictor = _FakePredictor()

    _segment_masks(predictor, str(clip), 0, 2, [5, 6, 7, 8])

    assert predictor.state_at_call is not None, "predictor was never invoked"
    state_at_call, prompts_at_call = predictor.state_at_call
    # Stale memory and prompts must be gone by the time SAM2 runs, else the box
    # prompt is overridden by the previous segment's (the contamination bug).
    assert state_at_call == {}
    assert prompts_at_call == {}
