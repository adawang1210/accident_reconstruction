"""Locate an ffmpeg binary, shared by the tracker and the web app.

Kept dependency-free (only ``shutil``/``pathlib``) so ``web_app`` can import it
without pulling in the heavy ML stack that ``prompt_track_accident`` carries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


def ensure_readable_mp4(path: str) -> None:
    """Re-encode a video to H.264 so headless OpenCV (and browsers) can read it.

    The OpenCV ``mp4v`` writer in the venv produces files this headless build
    cannot read back. When ffmpeg is available we transcode to H.264 yuv420p in
    place; otherwise the original (still player-playable) file is left as is.

    A transcode failure (disk full, missing encoder) is downgraded to a warning
    rather than raised, so a cosmetic re-encode never discards a good result.

    Args:
        path: Path to the just-written video.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return
    transcoded = f"{path}.h264.mp4"
    try:
        subprocess.run(  # noqa: S603 -- fixed ffmpeg args, no untrusted input
            [
                str(ffmpeg),
                "-y",
                "-loglevel",
                "error",
                "-i",
                path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                transcoded,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"[warn] ffmpeg re-encode failed ({error}); keeping original {path}.")
        Path(transcoded).unlink(missing_ok=True)
        return
    os.replace(transcoded, path)
