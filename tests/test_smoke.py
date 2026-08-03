"""Smoke test so CI has something real to run from day one."""

from codrona_mind import __version__


def test_version_is_present() -> None:
    assert __version__
