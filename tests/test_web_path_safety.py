"""Regression tests for web_app's data-path traversal guard.

Covers ``_safe_data_path`` (backing ``/media``, ``/api/frame``, ``/api/crop``).
The previous string ``startswith`` check let a sibling directory sharing the
``data`` prefix (``../data_backup/x``) through; ``is_relative_to`` must reject it.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_safe_data_path_guards_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from accident_reconstruction import web_app

    data_root = tmp_path / "data"
    data_root.mkdir()
    inside = data_root / "clip.mp4"
    inside.write_bytes(b"x")

    # A sibling dir sharing the "data" prefix: the old startswith check served
    # this; the resolved-component check must reject it.
    sibling = tmp_path / "data_backup"
    sibling.mkdir()
    (sibling / "secret.mp4").write_bytes(b"x")

    monkeypatch.setattr(web_app, "DATA_ROOT", data_root)

    assert web_app._safe_data_path("clip.mp4") == inside.resolve()
    assert web_app._safe_data_path("../data_backup/secret.mp4") is None
    assert web_app._safe_data_path("missing.mp4") is None
    assert web_app._safe_data_path(".") is None  # a directory, not a regular file
