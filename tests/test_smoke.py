"""Minimal smoke tests: package imports and built-in scene configs are sound.

These do not depend on ``data/`` (absent on CI / fresh clone), so they only
exercise the pure config layer. Heavy ML modules (SAM2 / ultralytics) are not
imported here, to keep CI fast.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The package itself imports cleanly."""
    import accident_reconstruction  # noqa: F401


def test_builtin_scenes_present() -> None:
    """Both built-in scenes register even without ``data/`` present."""
    from accident_reconstruction import scene_config as sc

    assert "pre_impact_motorcycle" in sc.SCENES
    assert "keelung_xinwu_yier" in sc.SCENES


def test_geo_writer_modules_import() -> None:
    """Figure/geo writers must import cleanly.

    A wrong-submodule import here does not crash the pipeline: ``run_pipeline``
    catches it and merely skips the recognised figure, so the breakage is silent.
    Importing the modules surfaces it (regression: ``recognized_route`` imported
    ``metric_to_latlon`` from the legacy annotation module instead of
    ``calibrate_homography``).
    """
    from accident_reconstruction import (
        birdseye_manual_annotation,
        recognized_route,
    )

    assert birdseye_manual_annotation is not None
    assert recognized_route is not None


def test_discover_scenes_is_safe_without_data() -> None:
    """``discover_scenes`` must not raise when ``data/`` is missing."""
    from accident_reconstruction import scene_config as sc

    sc.discover_scenes()  # must not raise
    for scene in sc.SCENES.values():
        # Paths are declarative; constructing them must not touch the disk.
        assert scene.source_video.suffix == ".mp4"
        assert scene.artifact_dir.parts[0] == "data"
