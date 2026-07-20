"""Unit tests for the pure geometry/detection functions in auto_reconstruct.

Covers the two-rule impact detector and the two trajectory-truncation helpers
(settle-to-rest and flip onset), which directly decide the drawn output and have
several branches (TECH_REVIEW §3.4). All are pure and import torch-free.
"""

from __future__ import annotations

from accident_reconstruction.auto_reconstruct import (
    detect_impact,
    flip_onset,
    settle_frame,
)


class TestDetectImpact:
    def test_clean_collide_then_separate_uses_closest_approach(self) -> None:
        # From the docstring: b approaches a to its closest at f2, then separates.
        metric = {
            "a": {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)},
            "b": {0: (5, 0), 1: (2, 0), 2: (1, 0), 3: (9, 0)},
        }
        assert detect_impact(metric) == 2

    def test_merge_plateau_uses_first_under_threshold(self) -> None:
        # They get close and STAY close (no separation) -> first frame under 3 m.
        metric = {
            "a": {0: (0, 0), 1: (0, 0), 2: (0, 0)},
            "b": {0: (5, 0), 1: (2, 0), 2: (2, 0)},
        }
        assert detect_impact(metric) == 1

    def test_single_vehicle_has_no_impact(self) -> None:
        assert detect_impact({"a": {0: (0, 0), 1: (1, 0)}}) is None

    def test_empty_returns_none(self) -> None:
        assert detect_impact({}) is None


class TestSettleFrame:
    def test_docstring_example(self) -> None:
        motion = {0: (0, 9.0), 1: (1, 1.0), 2: (1, 0.5), 3: (1, 0.4)}
        assert settle_frame(motion, 0, 3.0, sustain=2) == 1

    def test_brief_dip_does_not_count(self) -> None:
        # A single sub-threshold frame (f1) resets; the sustained run starts at f3.
        motion = {0: (0, 9.0), 1: (1, 1.0), 2: (1, 5.0), 3: (1, 0.5), 4: (1, 0.4)}
        assert settle_frame(motion, 0, 3.0, sustain=2) == 3

    def test_disabled_when_threshold_non_positive(self) -> None:
        motion = {0: (0, 0.0), 1: (1, 0.0)}
        assert settle_frame(motion, 0, 0.0, sustain=2) is None

    def test_never_settles_returns_none(self) -> None:
        motion = {0: (0, 9.0), 1: (1, 9.0), 2: (2, 9.0)}
        assert settle_frame(motion, 0, 3.0, sustain=2) is None


class TestFlipOnset:
    def test_detects_non_physical_jump(self) -> None:
        # Steps of 0.5 m then a 3.0 m leap at f3 (> FLIP_VELOCITY_M_PER_FRAME=1.2).
        assert flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0), 3: (4.0, 0)}, 0) == 3

    def test_all_smooth_returns_none(self) -> None:
        assert flip_onset({0: (0, 0), 1: (0.5, 0), 2: (1.0, 0)}, 0) is None

    def test_only_considers_frames_after(self) -> None:
        # The big jump is at f1; with after_frame=1 it is not considered.
        track = {0: (0, 0), 1: (5, 0), 2: (5.2, 0), 3: (5.4, 0)}
        assert flip_onset(track, after_frame=1) is None
