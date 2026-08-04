"""Tests for the shared KML document assembly.

Vehicle names come from user input (``vehicle_boxes.json``); a name with ``&``
or ``<`` must not produce malformed KML. Both writers -- the road-aligned
``birdseye_manual_annotation.write_kml`` and the raw
``recognized_route.write_recognized_kml`` -- build their documents from
:mod:`accident_reconstruction.kml_export`, so these cover both.
"""

from __future__ import annotations

from accident_reconstruction.kml_export import (
    COORD_DECIMALS,
    kml_document,
    linestring_placemark,
    point_placemark,
    rgb_to_kml_color,
    write_kml_document,
)


def test_linestring_escapes_name() -> None:
    xml = linestring_placemark("A&B <car>", "ff0000ff", [(25.0, 121.5), (25.1, 121.6)])

    assert "<name>A&amp;B &lt;car&gt;</name>" in xml
    assert "A&B <car>" not in xml


def test_point_escapes_name() -> None:
    assert "<name>A&amp;B</name>" in point_placemark("A&B", (25.0, 121.5))


def test_document_escapes_name() -> None:
    # The title interpolates vehicle display names too, so it needs escaping as
    # much as the placemarks do.
    assert "<name>A&amp;B 路線</name>" in kml_document("A&B 路線", [])


def test_linestring_needs_two_points() -> None:
    assert linestring_placemark("car", "ff0000ff", [(25.0, 121.5)]) == ""
    assert linestring_placemark("car", "ff0000ff", []) == ""


def test_rgb_to_kml_color_reverses_channels() -> None:
    # KML is aabbggrr: red must land in the LAST byte, not the first.
    assert rgb_to_kml_color((255, 0, 0)) == "ff0000ff"
    assert rgb_to_kml_color((0, 0, 255)) == "ffff0000"
    assert rgb_to_kml_color((0, 128, 255)) == "ffff8000"
    assert rgb_to_kml_color((255, 0, 0), alpha=0x80) == "800000ff"


def test_coordinates_are_lon_lat_alt() -> None:
    # KML orders coordinates lon,lat -- the opposite of the (lat, lon) tuples
    # used everywhere else in this codebase.
    xml = linestring_placemark("car", "ff0000ff", [(25.0, 121.5), (25.1, 121.6)])
    assert "<coordinates>121.5000000,25.0000000,0 121.6000000,25.1000000,0" in xml


def test_coordinate_precision_is_seven_decimals() -> None:
    # Deliberately NOT the 9 decimals the CSV/JSON carry: nothing differentiates
    # a KML, so 7 (~1.1 cm) is the right map-data convention here.
    assert COORD_DECIMALS == 7
    assert "121.1234568,25.0000000,0" in point_placemark("p", (25.0, 121.12345678))


def test_document_structure() -> None:
    doc = kml_document("路線", [point_placemark("撞擊點", (25.0, 121.5))])
    assert doc.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
    assert '<kml xmlns="http://www.opengis.net/kml/2.2">' in doc
    assert doc.endswith("</Document>\n</kml>\n")
    assert "撞擊點" in doc


def test_write_creates_parent_directory(tmp_path) -> None:
    path = tmp_path / "nested" / "route.kml"
    written = write_kml_document(path, "路線", [])

    assert written == path
    assert path.read_text(encoding="utf-8") == kml_document("路線", [])
