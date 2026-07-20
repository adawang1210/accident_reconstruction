"""Locate an ffmpeg binary, shared by the tracker and the web app.

Kept dependency-free (only ``shutil``/``pathlib``) so ``web_app`` can import it
without pulling in the heavy ML stack that ``prompt_track_accident`` carries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Common non-PATH install locations, in preference order (Homebrew first on mac).
_FFMPEG_CANDIDATES = (
    Path("/opt/homebrew/bin/ffmpeg"),
    Path("/usr/local/bin/ffmpeg"),
    Path.home() / "miniconda3/bin/ffmpeg",
    Path.home() / "anaconda3/bin/ffmpeg",
)


def find_ffmpeg() -> Path | None:
    """Return the ffmpeg binary path (PATH first, then common installs).

    Returns:
        The resolved ffmpeg binary as a ``Path``, or ``None`` if not found.
        Callers wanting the directory (e.g. yt-dlp's ``ffmpeg_location``) take
        ``.parent``.
    """
    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path)
    return next((path for path in _FFMPEG_CANDIDATES if path.exists()), None)
