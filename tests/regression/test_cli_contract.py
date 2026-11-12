"""The fitrank command and its files still look like FitRank 1.1 for scripts that use them.

Same commands and options, same exit codes, same files in a run folder, same
columns in leaderboard.csv / leaderboard.json and the same keys in
metadata.json. New commands, options, columns or keys are fine; removing or
renaming one fails here.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import typer
import typer.main
import yaml
from typer.testing import CliRunner

from fitrank import Benchmark, cli
from fitrank.cli import app

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()

# command -> {parameter name: option strings, or "argument"}, recorded from FitRank 1.1.2.
V1_1_COMMANDS: dict[str, dict[str, Any]] = {
    "": {"verbose": ["--verbose", "-v"], "quiet": ["--quiet", "-q"], "version": ["--version"]},
    "run": {"config": "argument"},
    "validate": {"config": "argument"},
    "init": {
        "output": ["--output", "-o"],
        "task": ["--task"],
        "data": ["--data"],
        "target": ["--target"],
        "force": ["--force"],
    },
    "predict": {"model": "argument", "data": "argument", "output": ["--output", "-o"]},
    "list": {},
    "list models": {"task": ["--task"]},
    "list reductions": {},
}
V1_1_EXIT_CODES = {"EXIT_CONFIG": 1, "EXIT_DATA": 2, "EXIT_ALL_FAILED": 3}

V1_1_RUN_FILES = {
    "config.yaml",
    "leaderboard.csv",
    "leaderboard.json",
    "leaderboard.md",
    "best_model.joblib",
    "metadata.json",
}
# With primary_metric f1_weighted and metrics [accuracy].
V1_1_COLUMNS = [
    "rank",
    "model",
    "reduction",
    "status",
    "cv_f1_weighted",
    "cv_std",
    "test_f1_weighted",
    "test_accuracy",
    "fit_s",
    "best_params",
    "error",
    "notes",
]
V1_1_METADATA = {
    "created_at": None,
    "duration_seconds": None,
    "task": None,
    "primary_metric": None,
    "best": {"model", "reduction", "cv"},
    "versions": {"python", "fitrank", "scikit-learn", "pandas", "numpy", "joblib"},
}
V1_1_RESULTS_ATTRIBUTES = [
    "config",
    "metric",
    "results",
    "duration_seconds",
    "best",
    "run_dir",
    "leaderboard",
    "best_model",
    "export",
]

CONFIG = """
project: {name: iris}
data: {path: iris.csv, target: species}
task: classification
models:
  - {name: logistic_regression, params: {C: [1]}}
  - {name: knn, params: {n_neighbors: [5]}}
evaluation: {cv: 3, metrics: [accuracy]}
"""


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copy(REPO / "examples" / "data" / "iris.csv", tmp_path / "iris.csv")
    (tmp_path / "iris.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commands() -> dict[str, dict[str, Any]]:
    """Every command of the fitrank CLI, with its parameters."""
    found: dict[str, dict[str, Any]] = {}

    def walk(command: Any, path: str, context: Any) -> None:
        found[path] = {
            param.name: "argument" if param.param_type_name == "argument" else list(param.opts)
            for param in command.params
        }
        if hasattr(command, "list_commands"):
            for name in command.list_commands(context):
                child = command.get_command(context, name)
                walk(child, f"{path} {name}".strip(), typer.Context(child, parent=context))

    root = typer.main.get_command(app)
    walk(root, "", typer.Context(root))
    return found


def test_commands_and_options_are_unchanged() -> None:
    commands = _commands()
    for name, params in V1_1_COMMANDS.items():
        label = f"fitrank {name}".strip()
        assert name in commands, f"{label} was removed or renamed"
        for param, opts in params.items():
            assert param in commands[name], f"{label}: {param!r} was removed or renamed"
            got = commands[name][param]
            if opts == "argument":
                assert got == "argument", f"{label}: {param!r} is no longer an argument"
            else:
                assert got != "argument" and set(opts) <= set(got), (
                    f"{label}: {param!r} options changed from {opts} to {got}"
                )


def test_exit_code_values_are_unchanged() -> None:
    for name, code in V1_1_EXIT_CODES.items():
        assert getattr(cli, name) == code, f"{name} changed from {code}"


def test_exit_codes(workdir: Path) -> None:
    assert runner.invoke(app, ["validate", "iris.yaml"]).exit_code == 0

    (workdir / "bad.yaml").write_text(CONFIG.replace("knn", "knnn"), encoding="utf-8")
    assert runner.invoke(app, ["validate", "bad.yaml"]).exit_code == 1  # config error

    (workdir / "data.yaml").write_text(CONFIG.replace("species", "nope"), encoding="utf-8")
    assert runner.invoke(app, ["validate", "data.yaml"]).exit_code == 2  # data error

    failing = CONFIG.replace("C: [1]", "C: [-1]").replace("n_neighbors: [5]", "n_neighbors: [0]")
    (workdir / "failing.yaml").write_text(failing, encoding="utf-8")
    assert runner.invoke(app, ["run", "failing.yaml"]).exit_code == 3  # every model failed


def test_run_folder_and_exports_are_unchanged(workdir: Path) -> None:
    result = runner.invoke(app, ["run", "iris.yaml"])
    assert result.exit_code == 0, result.output
    (run_dir,) = (workdir / "runs").iterdir()
    assert {path.name for path in run_dir.iterdir()} >= V1_1_RUN_FILES

    csv = pd.read_csv(run_dir / "leaderboard.csv")
    assert list(csv.columns) == V1_1_COLUMNS, "leaderboard.csv columns changed"
    records = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert list(records[0]) == V1_1_COLUMNS, "leaderboard.json keys changed"
    assert isinstance(records[0]["best_params"], dict)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    for key, subkeys in V1_1_METADATA.items():
        assert key in metadata, f"metadata.json: {key!r} was removed"
        if subkeys:
            assert subkeys <= set(metadata[key]), f"metadata.json: {key} keys changed"

    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert set(saved) >= {"project", "data", "task", "models", "evaluation", "output"}

    predict = runner.invoke(
        app, ["predict", str(run_dir / "best_model.joblib"), "iris.csv", "-o", "out.csv"]
    )
    assert predict.exit_code == 0, predict.output
    assert pd.read_csv(workdir / "out.csv").columns[-1] == "prediction"


def test_results_attributes_are_unchanged(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run()
    for attribute in V1_1_RESULTS_ATTRIBUTES:
        assert hasattr(results, attribute), f"Results.{attribute} was removed"
    assert list(results.leaderboard().columns) == V1_1_COLUMNS
