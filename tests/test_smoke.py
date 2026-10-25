"""Smoke tests: the package installs and exposes its version."""

import re

import fitrank


def test_version_is_exposed() -> None:
    assert fitrank.__version__ == "0.1.0"


def test_version_follows_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?", fitrank.__version__)
