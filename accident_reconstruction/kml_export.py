"""KML document assembly, shared by every trajectory exporter.

Two writers -- :func:`birdseye_manual_annotation.write_kml` (road-aligned) and
:func:`recognized_route.write_recognized_kml` (raw projection) -- produced the
same document by hand: the same XML envelope, the same BGR-to-``aabbggrr``
colour conversion, the same impact-point Placemark, the same write. Only the
document title and the source of the coordinates actually differed.

Keeping two copies of an XML serialiser is how they drift: the escaping fix that
:mod:`tests.test_kml_escape` guards reached one writer only because they happened
to share the one private helper the other module reached across for. This module
gives that shared logic a home, so a fix lands once.

Everything here is pure string assembly -- no scene state, no I/O beyond
:func:`write_kml_document` -- so it is testable without a calibrated scene.

Examples:
    ```python
    line = linestring_placemark("car", "ff0000ff", [(25.0, 121.5), (25.1, 121.6)])
    doc = kml_document("路線", [line])
    doc.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    # True
    ```
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from xml.sax.saxutils import escape

# Coordinate decimals in KML output. SEVEN, deliberately -- ~1.1 cm, the
# convention for map data, and all a viewer needs to draw a line in the right
# place. This is NOT the 9 decimals the route CSV and reconstruction JSON carry:
# those are DIFFERENTIATED downstream (speed, and jerk at the third derivative),
# where rounding is amplified by fps per derivative. Nothing differentiates a
# KML, so the extra digits would only bloat the file. See
# ``birdseye_manual_annotation.route_csv_row`` for the other side of this.
COORD_DECIMALS = 7


def rgb_to_kml_color(rgb: tuple[int, int, int], alpha: int = 0xFF) -> str:
    """Convert a display ``(r, g, b)`` to KML's ``aabbggrr`` hex string.

    KML orders its colour channels backwards from RGB and puts alpha first,
    which is easy to get subtly wrong by hand (blue and red swap, and the
    mistake is invisible on a monochrome track).

    Args:
        rgb: Display colour, each channel 0-255.
        alpha: Opacity 0-255; defaults to fully opaque.

    Returns:
        The eight-character ``aabbggrr`` string.

    Examples:
        ```python
        rgb_to_kml_color((255, 0, 0))
        # 'ff0000ff'
        rgb_to_kml_color((0, 128, 255))
        # 'ffff8000'
        ```
    """
    r, g, b = rgb
    return f"{alpha:02x}{b:02x}{g:02x}{r:02x}"


def _coords_text(coords: Iterable[tuple[float, float]]) -> str:
    """Serialise ``(lat, lon)`` pairs as KML's ``lon,lat,alt`` triples."""
    return " ".join(
        f"{lon:.{COORD_DECIMALS}f},{lat:.{COORD_DECIMALS}f},0" for lat, lon in coords
    )


def linestring_placemark(
    name: str, color_abgr: str, coords: list[tuple[float, float]]
) -> str:
    """Build a KML Placemark LineString from lat/lon coordinates.

    Args:
        name: Placemark name.
        color_abgr: KML colour as ``aabbggrr`` hex (see :func:`rgb_to_kml_color`).
        coords: ``(lat, lon)`` vertices in order.

    Returns:
        The Placemark XML string (empty when fewer than two points).

    Examples:
        ```python
        linestring_placemark("car", "ff0000ff", [(25.0, 121.5)])
        # ''
        ```
    """
    if len(coords) < 2:
        return ""
    # ``name`` comes from user-entered vehicle labels; escape so ``&``/``<`` in a
    # name (e.g. "A&B car") cannot produce malformed KML that fails to import.
    return (
        f"  <Placemark><name>{escape(name)}</name>"
        f"<Style><LineStyle><color>{color_abgr}</color><width>4</width></LineStyle></Style>"
        f"<LineString><tessellate>1</tessellate>"
        f"<coordinates>{_coords_text(coords)}</coordinates></LineString></Placemark>\n"
    )


def point_placemark(name: str, latlon: tuple[float, float]) -> str:
    """Build a KML Placemark Point (used for the impact marker).

    Args:
        name: Placemark name; XML-escaped like the LineString's.
        latlon: The ``(lat, lon)`` position.

    Returns:
        The Placemark XML string.

    Examples:
        ```python
        "<coordinates>121.5000000,25.0000000,0</coordinates>" in point_placemark(
            "撞擊點", (25.0, 121.5)
        )
        # True
        ```
    """
    return (
        f"  <Placemark><name>{escape(name)}</name>"
        f"<Point><coordinates>{_coords_text([latlon])}</coordinates>"
        f"</Point></Placemark>\n"
    )


def kml_document(name: str, placemarks: Iterable[str]) -> str:
    """Wrap Placemark fragments in a complete KML document.

    Args:
        name: Document title, shown as the layer name in Google Earth / My Maps.
        placemarks: Placemark XML fragments; empty ones (a track too short to
            draw) are harmless and simply concatenate away.

    Returns:
        The full KML text, newline-terminated.
    """
    body = "".join(placemarks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
        f"  <name>{escape(name)}</name>\n"
        f"{body}</Document>\n</kml>\n"
    )


def write_kml_document(path: Path, name: str, placemarks: Iterable[str]) -> Path:
    """Write a KML document, creating the parent directory.

    Args:
        path: Destination ``.kml`` path.
        name: Document title.
        placemarks: Placemark XML fragments.

    Returns:
        The written path, so callers can report it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kml_document(name, placemarks), encoding="utf-8")
    return path
