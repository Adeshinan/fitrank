"""Exceptions raised by FitRank for user errors."""

from collections.abc import Sequence


class FitRankError(Exception):
    """Base class for every error FitRank raises on purpose.

    The message says what is wrong, then one line per problem (where it is
    and how to fix it), then an optional hint, e.g. a YAML snippet.
    ``details`` holds the problem lines without the leading arrow, for
    programmatic use.
    """

    def __init__(self, message: str, details: Sequence[str] = (), hint: str | None = None) -> None:
        self.details = list(details)
        self.hint = hint
        lines = [message, *(f"  -> {line}" for line in self.details)]
        if hint:
            first, *rest = hint.splitlines()
            lines += [f"  Hint: {first}", *(f"        {line}" for line in rest)]
        super().__init__("\n".join(lines))


class ConfigError(FitRankError):
    """The configuration file is missing, unreadable or invalid."""


class DataError(FitRankError):
    """The data cannot be read, or cannot be used as declared in the configuration."""
