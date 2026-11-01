"""Smoke tests: the package installs and exposes its version."""

import re
from importlib.metadata import version

import fitrank


def test_version_matches_the_installed_package() -> None:
    assert fitrank.__version__ == version("fitrank")


def test_version_follows_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?", fitrank.__version__)
