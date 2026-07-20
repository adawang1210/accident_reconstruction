"""Regression test: KML placemark names are XML-escaped.

Vehicle names come from user input (``vehicle_boxes.json``); a name with ``&``
or ``<`` must not produce malformed KML. ``recognized_route`` reuses the same
``_kml_linestring`` helper, so this covers both KML writers.
"""

from __future__ import annotations


def test_kml_linestring_escapes_name() -> None:
    from accident_reconstruction.birdseye_manual_annotation import _kml_linestring

    xml = _kml_linestring("A&B <car>", "ff0000ff", [(25.0, 121.5), (25.1, 121.6)])

    assert "<name>A&amp;B &lt;car&gt;</name>" in xml
    assert "A&B <car>" not in xml
