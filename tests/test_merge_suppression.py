"""Regression tests for the cross-vehicle mask-merge suppression.

Each accident vehicle is tracked by an independent SAM2 run, so at impact one
track's mask can leak onto the overlapping other vehicle: the smaller (struck)
box grows to cover the larger one and the two drawn boxes collapse into a single
box. ``merge_suppression_cuts`` detects that and cuts the smaller vehicle from the
merge frame on. These tests pin that behaviour without running SAM2.
"""

from __future__ import annotations

import pytest

# The tracker module imports ultralytics at import time; skip where it is absent
# (CI without ML deps), consistent with the smoke tests.
pytest.importorskip("ultralytics")

from accident_reconstruction.prompt_track_accident import (
    MERGE_SUSTAIN_FRAMES,
    box_containment,
    merge_suppression_cuts,
)


def _record(box):
    """A per-frame ``(box, anchor, mask)`` record (anchor/mask unused here)."""
    return (box, ((box[0] + box[2]) // 2, box[3]), None)


def test_containment_fully_inside() -> None:
    """A small box wholly inside a large one is fully contained (1.0)."""
    assert box_containment((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0


def test_containment_disjoint() -> None:
    """Disjoint boxes have zero containment."""
    assert box_containment((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_containment_degenerate_box_is_zero() -> None:
    """A zero-area box never reports containment (no divide-by-zero)."""
    assert box_containment((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_merge_cuts_smaller_vehicle_at_run_start() -> None:
    """The struck motorcycle's box swallowed by the car is cut from the merge."""
    # Motorcycle box collapses onto (sits inside) the car box from frame 5 on.
    car = {f: _record((0, 0, 100, 100)) for f in range(0, 10)}
    motorcycle = {f: _record((220, 330, 270, 398)) for f in range(0, 5)}
    motorcycle.update({f: _record((10, 10, 30, 30)) for f in range(5, 10)})

    cuts = merge_suppression_cuts({"motorcycle": motorcycle, "car": car})

    assert cuts == {"motorcycle": 5}


def test_no_merge_when_boxes_stay_apart() -> None:
    """Vehicles that never overlap are never cut."""
    car = {f: _record((0, 0, 100, 100)) for f in range(0, 10)}
    motorcycle = {f: _record((200, 200, 250, 260)) for f in range(0, 10)}

    assert merge_suppression_cuts({"motorcycle": motorcycle, "car": car}) == {}


def test_brief_overlap_below_sustain_is_ignored() -> None:
    """A short brush-past (fewer than the sustain run) is not treated as a merge."""
    car = {f: _record((0, 0, 100, 100)) for f in range(0, 10)}
    motorcycle = {f: _record((200, 200, 250, 260)) for f in range(0, 10)}
    # Overlap for one frame fewer than required to trigger a cut.
    for f in range(3, 3 + MERGE_SUSTAIN_FRAMES - 1):
        motorcycle[f] = _record((10, 10, 30, 30))

    assert merge_suppression_cuts({"motorcycle": motorcycle, "car": car}) == {}
