"""End-to-end tests: the shipped examples, run by the installed fitrank command.

Each test starts real processes, as a user would, in a copy of the examples
folder. They take a few seconds each; skip them with ``pytest -m "not e2e"``.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = ["iris", "diabetes"]

pytestmark = pytest.mark.e2e


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A copy of examples/, so that the runs/ folders are written in tmp_path."""
    shutil.copytree(REPO / "examples", tmp_path / "examples")
    return tmp_path


def fitrank(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    command = shutil.which("fitrank")
    assert command is not None, "the fitrank command is not installed: pip install -e ."
    return subprocess.run(
        [command, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # Windows consoles are not always UTF-8
        check=False,
    )


def run_example(example: str, cwd: Path) -> Path:
    """Run an example and return its results folder."""
    before = set((cwd / "runs").glob("*")) if (cwd / "runs").exists() else set()
    result = fitrank("run", f"examples/{example}.yaml", cwd=cwd)
    assert result.returncode == 0, result.stderr
    assert "Best model:" in result.stdout
    (run_dir,) = set((cwd / "runs").glob("*")) - before
    return run_dir


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_validates(example: str, workdir: Path) -> None:
    result = fitrank("validate", f"examples/{example}.yaml", cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert "Config OK" in result.stdout and "Data OK" in result.stdout


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs_end_to_end(example: str, workdir: Path) -> None:
    run_dir = run_example(example, workdir)
    files = {path.name for path in run_dir.iterdir()}
    assert files == {
        "config.yaml",
        "leaderboard.csv",
        "leaderboard.json",
        "leaderboard.md",
        "best_model.joblib",
        "metadata.json",
    }
    board = pd.read_csv(run_dir / "leaderboard.csv")
    assert (board["status"] == "ok").all(), board[["model", "error"]]

    # The saved model predicts on the original file, target column included.
    data = f"examples/data/{example}.csv"
    result = fitrank(
        "predict", str(run_dir / "best_model.joblib"), data, "-o", "out.csv", cwd=workdir
    )
    assert result.returncode == 0, result.stderr
    out = pd.read_csv(workdir / "out.csv")
    assert len(out) == len(pd.read_csv(workdir / data))
    assert out.columns[-1] == "prediction"


def test_two_runs_give_the_same_leaderboard(workdir: Path) -> None:
    """ENF-06: same config and random_state, same results (timings aside)."""
    first, second = (pd.read_csv(run_example("iris", workdir) / "leaderboard.csv") for _ in "12")
    pd.testing.assert_frame_equal(first.drop(columns="fit_s"), second.drop(columns="fit_s"))
