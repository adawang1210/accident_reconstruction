"""URL-keyed calibration records: the marked GCP lat/lon points per scene.

Each record (``scene_records/<name>.json``) ties a **YouTube URL** to the
ground-control points the user marked during calibration. Because ``data/`` is
git-ignored (videos are large and copyright-bound), these small records are the
**committed, shareable** copy of the calibration work.

Mechanism:
    - When a clip is (re)downloaded from a YouTube URL, the web workbench calls
      :func:`find_record_by_url`. A match means the marked points already exist,
      so the clip is auto-calibrated (no re-marking).
    - After a fresh calibration of a clip whose URL is known,
      :func:`save_record` persists the URL -> GCPs mapping for next time.

Matching is by **video id** (the 11-char YouTube id), so tracking/query params
(``&t=``, ``?si=``...) and the ``youtu.be`` / ``watch?v=`` / ``embed`` forms all
resolve to the same record.

Record schema (``scene_records/<name>.json``)::

    {
      "name": "keelung_xinwu_yier",
      "youtube_url": "https://www.youtube.com/watch?v=XXXXXXXXXXX",
      "source_video": "data/videos/keelung_xinwu_yier_source.mp4",
      "origin_latlon": [25.1341258, 121.7473854],
      "distortion": null,
      "gcps": [{"name": "g1", "pixel": [650, 460], "lat": 25.13, "lon": 121.74}, ...]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: Committed directory of per-scene calibration records (sibling of ``data/``).
RECORDS_DIR = Path("scene_records")

_YOUTUBE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def youtube_id(url: str | None) -> str | None:
    """Extract the 11-char YouTube video id from any common URL form.

    Args:
        url: A YouTube URL (``watch?v=``, ``youtu.be/``, ``embed/`` ...) or None.

    Returns:
        The video id, or None if ``url`` is empty or holds no id.

    Examples:
        ```python
        youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s")
        # 'dQw4w9WgXcQ'
        youtube_id("https://youtu.be/dQw4w9WgXcQ?si=abc")
        # 'dQw4w9WgXcQ'
        youtube_id(None) is None
        # True
        ```
    """
    if not url:
        return None
    patterns = (
        r"(?:v=|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",  # bare id
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_records(records_dir: Path = RECORDS_DIR) -> list[dict]:
    """Load every committed scene record.

    Args:
        records_dir: Directory holding ``<name>.json`` records.

    Returns:
        The parsed records (unreadable / malformed files are skipped).
    """
    if not records_dir.is_dir():
        return []
    records = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def find_record_by_url(
    url: str | None, records_dir: Path = RECORDS_DIR
) -> dict | None:
    """Return the record whose ``youtube_url`` matches ``url`` by video id.

    Args:
        url: The YouTube URL to look up.
        records_dir: Directory holding the records.

    Returns:
        The matching record, or None when ``url`` has no id or nothing matches.
    """
    wanted = youtube_id(url)
    if not wanted:
        return None
    for record in load_records(records_dir):
        if youtube_id(record.get("youtube_url")) == wanted:
            return record
    return None


def save_record(
    name: str,
    youtube_url: str | None,
    gcps: list[dict],
    *,
    source_video: str | None = None,
    origin_latlon: list[float] | None = None,
    distortion: object | None = None,
    records_dir: Path = RECORDS_DIR,
) -> Path:
    """Write (or overwrite) the record for scene ``name``.

    Args:
        name: Scene id (becomes the file stem ``<name>.json``).
        youtube_url: Source URL the record is keyed by.
        gcps: Marked ground-control points (``{name, pixel, lat, lon}``).
        source_video: Clip path under the repo, for reference.
        origin_latlon: Local-frame origin, copied from the calibration.
        distortion: Lens-distortion block, copied from the calibration.
        records_dir: Output directory.

    Returns:
        The path written.
    """
    records_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "youtube_url": youtube_url,
        "source_video": source_video,
        "origin_latlon": origin_latlon,
        "distortion": distortion,
        "gcps": gcps,
    }
    path = records_dir / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
