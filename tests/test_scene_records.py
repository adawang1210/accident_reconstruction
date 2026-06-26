"""Tests for the URL-keyed calibration record registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from accident_reconstruction import scene_records as sr


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abc", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        (None, None),
        ("", None),
        ("https://example.com/no-id-here", None),
    ],
)
def test_youtube_id(url: str | None, expected: str | None) -> None:
    """Video ids are extracted from every common URL form (and tracking params)."""
    assert sr.youtube_id(url) == expected


def test_save_and_find_roundtrip(tmp_path: Path) -> None:
    """A saved record is found again by an equivalent (params-changed) URL."""
    sr.save_record(
        "demo",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        gcps=[{"name": "g1", "pixel": [1, 2], "lat": 25.0, "lon": 121.0}],
        source_video="data/videos/demo.mp4",
        records_dir=tmp_path,
    )
    # Same clip, different surface form -> still matches by id.
    found = sr.find_record_by_url("https://youtu.be/dQw4w9WgXcQ?si=x", tmp_path)
    assert found is not None
    assert found["name"] == "demo"
    assert len(found["gcps"]) == 1


def test_find_returns_none_without_match(tmp_path: Path) -> None:
    """No record and no id both resolve to None rather than raising."""
    assert sr.find_record_by_url("https://youtu.be/UNKNOWN12345", tmp_path) is None
    assert sr.find_record_by_url(None, tmp_path) is None


def test_committed_records_are_loadable() -> None:
    """The four seeded records on disk parse and carry marked GCP points."""
    records = {r["name"]: r for r in sr.load_records()}
    assert "keelung_xinwu_yier" in records
    assert records["keelung_xinwu_yier"]["gcps"], "expected marked GCPs"
