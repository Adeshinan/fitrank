"""Reference leaderboards ("golden files") recorded with FitRank 1.1.2.

The configs in ``configs/`` are frozen copies of the shipped examples: editing an
example never moves the reference. Scores depend on the library versions, so the
references are only compared with the versions pinned in ``constraints.txt``.

Regenerate them only when a change of results is intended, and say why in the PR::

    python tests/regression/golden.py
"""

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from fitrank import Benchmark

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIGS = HERE / "configs"
GOLDEN = HERE / "golden"
CONSTRAINTS = HERE / "constraints.txt"
EXAMPLES = ["iris", "diabetes", "titanic"]

# Compared exactly; every other column but fit_s (a timing) is a score.
EXACT = ["rank", "model", "reduction", "status", "best_params", "error", "notes"]


def leaderboard(name: str) -> pd.DataFrame:
    """Run a frozen config (from the repository root) and return its comparable leaderboard."""
    cwd = Path.cwd()
    os.chdir(REPO)  # the configs read examples/data/...
    try:
        board = Benchmark.from_yaml(CONFIGS / f"{name}.yaml").run(export=False).leaderboard()
    finally:
        os.chdir(cwd)
    board["best_params"] = [json.dumps(p, sort_keys=True, default=str) for p in board.best_params]
    return board.drop(columns="fit_s")


def score_columns(board: pd.DataFrame) -> list[str]:
    return [column for column in board.columns if column not in EXACT]


def pinned_versions() -> dict[str, str]:
    """The ``package==version`` lines of constraints.txt."""
    lines = CONSTRAINTS.read_text(encoding="utf-8").splitlines()
    pairs = (line.split("#")[0].strip().split("==") for line in lines)
    return {pair[0]: pair[1] for pair in pairs if len(pair) == 2}


def version_mismatches() -> list[str]:
    """The pinned packages whose installed version differs, e.g. 'numpy 2.4.0 (pinned 2.5.4)'."""
    return [
        f"{package} {version(package)} (pinned {pinned})"
        for package, pinned in pinned_versions().items()
        if version(package) != pinned
    ]


def main() -> None:
    if mismatches := version_mismatches():
        sys.exit(
            "install the pinned versions first: "
            'pip install -e ".[dev]" -c tests/regression/constraints.txt\n' + "\n".join(mismatches)
        )
    for name in EXAMPLES:
        board = leaderboard(name)
        board.to_csv(GOLDEN / f"{name}.csv", index=False)
        print(f"{name}: {len(board)} candidates written to {GOLDEN / f'{name}.csv'}")


if __name__ == "__main__":
    main()
