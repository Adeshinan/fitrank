"""Exceptions raised by FitRank for user errors."""

from collections.abc import Sequence


class FitRankError(Exception):
    """Base class for every error FitRank raises on purpose."""


class ConfigError(FitRankError):
    """The configuration file is missing, unreadable or invalid.

    The message names the file, then one line per problem with the YAML key
    at fault and how to fix it. ``details`` holds those lines without the
    leading arrow, for programmatic use.
    """

    def __init__(self, message: str, details: Sequence[str] = ()) -> None:
        self.details = list(details)
        super().__init__("\n".join([message, *(f"  -> {line}" for line in self.details)]))
