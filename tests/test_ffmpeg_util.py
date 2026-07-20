"""Unit tests for the shared ffmpeg locator (``ffmpeg_util.find_ffmpeg``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from accident_reconstruction import ffmpeg_util


def test_find_ffmpeg_prefers_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_util.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert ffmpeg_util.find_ffmpeg() == Path("/usr/bin/ffmpeg")


def test_find_ffmpeg_falls_back_to_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "ffmpeg"
    binary.write_text("")
    monkeypatch.setattr(ffmpeg_util.shutil, "which", lambda _: None)
    monkeypatch.setattr(ffmpeg_util, "_FFMPEG_CANDIDATES", (binary,))
    assert ffmpeg_util.find_ffmpeg() == binary


def test_find_ffmpeg_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg_util.shutil, "which", lambda _: None)
    monkeypatch.setattr(ffmpeg_util, "_FFMPEG_CANDIDATES", ())
    assert ffmpeg_util.find_ffmpeg() is None
