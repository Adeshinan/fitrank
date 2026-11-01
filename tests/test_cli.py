"""Tests for the fitrank command, run in-process with typer's CliRunner."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fitrank import Benchmark, __version__
from fitrank.cli import app

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()

IRIS_YAML = """
project: {name: iris}
data: {path: iris.csv, target: species}
task: classification
preprocessing:
  numeric:
    columns: [sepal_length, sepal_width, petal_length, petal_width]
    steps: [scaler]
models:
  - {name: logistic_regression, params: {C: [1]}}
  - {name: knn, params: {n_neighbors: [5]}}
evaluation: {cv: 3}
"""


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder with iris.csv and iris.yaml, used as the current directory."""
    shutil.copy(REPO / "examples" / "data" / "iris.csv", tmp_path / "iris.csv")
    (tmp_path / "iris.yaml").write_text(IRIS_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def only_run(workdir: Path) -> Path:
    (run_dir,) = (workdir / "runs").iterdir()
    return run_dir


# --- General ---------------------------------------------------------------------------------


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"fitrank {__version__}"


def test_no_command_shows_the_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "run" in result.output and "predict" in result.output


def test_verbose_and_quiet_are_exclusive(workdir: Path) -> None:
    result = runner.invoke(app, ["--verbose", "--quiet", "validate", "iris.yaml"])
    assert result.exit_code != 0
    assert "not both" in result.output


# --- list ------------------------------------------------------------------------------------


def test_list_models_classification() -> None:
    result = runner.invoke(app, ["list", "models", "--task", "classification"])
    assert result.exit_code == 0
    assert "random_forest" in result.output
    assert "ridge" not in result.output


def test_list_models_for_both_tasks() -> None:
    result = runner.invoke(app, ["list", "models"])
    assert "classification (" in result.output and "regression (" in result.output
    assert "ridge" in result.output


def test_list_models_unknown_task() -> None:
    result = runner.invoke(app, ["list", "models", "--task", "clustering"])
    assert result.exit_code != 0
    assert "classification, regression" in result.output


def test_list_reductions() -> None:
    result = runner.invoke(app, ["list", "reductions"])
    assert result.exit_code == 0
    assert result.output.split() == sorted(result.output.split())
    assert {"none", "pca"} <= set(result.output.split())


# --- validate --------------------------------------------------------------------------------


def test_validate_does_not_train(workdir: Path) -> None:
    result = runner.invoke(app, ["validate", "iris.yaml"])
    assert result.exit_code == 0, result.output
    assert "Config OK: iris.yaml" in result.output
    assert "Data OK: 150 rows, 4 features, target 'species'" in result.output
    assert "2 candidates: 1 reduction(s) x 2 model(s), 3-fold cross-validation" in result.output
    assert not (workdir / "runs").exists()


# --- run -------------------------------------------------------------------------------------


def test_run_iris_succeeds(workdir: Path) -> None:
    result = runner.invoke(app, ["run", "iris.yaml"])
    assert result.exit_code == 0, result.output
    assert "logistic_regression" in result.output and "knn" in result.output
    assert "Best model: " in result.output
    run_dir = only_run(workdir)
    assert f"Results saved to runs/{run_dir.name}/" in result.output
    assert (run_dir / "leaderboard.csv").is_file()


def test_quiet_shows_only_the_leaderboard(workdir: Path) -> None:
    result = runner.invoke(app, ["--quiet", "run", "iris.yaml"])
    assert result.exit_code == 0
    assert "knn" in result.output
    assert "Best model" not in result.output and "Results saved" not in result.output


def test_verbose_shows_each_candidate(workdir: Path) -> None:
    result = runner.invoke(app, ["--verbose", "run", "iris.yaml"])
    assert result.exit_code == 0
    assert "ok      none + logistic_regression: cv " in result.output
    assert "ok      none + knn: cv " in result.output


def test_api_and_cli_give_same_leaderboard(workdir: Path) -> None:
    assert runner.invoke(app, ["run", "iris.yaml"]).exit_code == 0
    from_cli = pd.read_csv(only_run(workdir) / "leaderboard.csv")
    from_api = Benchmark.from_yaml("iris.yaml").run(export=False).leaderboard()
    from_api["best_params"] = [json.dumps(p) for p in from_api["best_params"]]
    pd.testing.assert_frame_equal(
        from_cli.drop(columns=["fit_s", "error", "notes"]),
        from_api.drop(columns=["fit_s", "error", "notes"]),
        check_dtype=False,
    )


# --- Exit codes and errors -------------------------------------------------------------------


def test_bad_config_exits_1_without_traceback(workdir: Path) -> None:
    bad = write_config(workdir / "bad.yaml", IRIS_YAML.replace("task: classification", "task: x"))
    result = runner.invoke(app, ["run", bad])
    assert result.exit_code == 1
    assert "Error: invalid configuration" in result.output
    assert "task: must be 'classification' or 'regression'" in result.output
    assert "Traceback" not in result.output


def test_verbose_shows_the_traceback(workdir: Path) -> None:
    result = runner.invoke(app, ["--verbose", "run", "missing.yaml"])
    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "config file not found: missing.yaml" in result.output


def test_data_error_exits_2(workdir: Path) -> None:
    bad = write_config(workdir / "bad.yaml", IRIS_YAML.replace("target: species", "target: colour"))
    result = runner.invoke(app, ["run", bad])
    assert result.exit_code == 2
    assert "colour" in result.output
    assert "Traceback" not in result.output


def test_every_candidate_failed_exits_3(workdir: Path) -> None:
    text = IRIS_YAML.replace("{C: [1]}", "{C: [-1]}").replace(
        "{n_neighbors: [5]}", "{n_neighbors: [-1]}"
    )
    result = runner.invoke(app, ["--verbose", "run", write_config(workdir / "fail.yaml", text)])
    assert result.exit_code == 3
    assert "failed  none + knn: ValueError" in result.output  # one line per candidate
    assert "every candidate failed" in result.output
    assert (only_run(workdir) / "leaderboard.csv").is_file()  # the failures are still saved


# --- predict ---------------------------------------------------------------------------------


@pytest.fixture
def model_path(workdir: Path) -> Path:
    assert runner.invoke(app, ["--quiet", "run", "iris.yaml"]).exit_code == 0
    return only_run(workdir) / "best_model.joblib"


def test_predict_writes_a_csv(workdir: Path, model_path: Path) -> None:
    result = runner.invoke(app, ["predict", str(model_path), "iris.csv", "-o", "out.csv"])
    assert result.exit_code == 0, result.output
    assert "150 predictions written to out.csv" in result.output
    out = pd.read_csv("out.csv")
    assert out.columns[-1] == "prediction"
    assert (out["prediction"] == out["species"]).mean() > 0.9  # the target column is ignored


def test_predict_to_stdout(workdir: Path, model_path: Path) -> None:
    pd.read_csv("iris.csv").drop(columns="species").head(3).to_csv("new.csv", index=False)
    result = runner.invoke(app, ["predict", str(model_path), "new.csv"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0].endswith(",prediction") and len(lines) == 4


def test_predict_missing_column_exits_2(workdir: Path, model_path: Path) -> None:
    pd.read_csv("iris.csv").drop(columns="petal_width").to_csv("new.csv", index=False)
    result = runner.invoke(app, ["predict", str(model_path), "new.csv"])
    assert result.exit_code == 2
    assert "missing column(s) the model was trained on: petal_width" in result.output


def test_predict_bad_value_exits_2(workdir: Path, model_path: Path) -> None:
    frame = pd.read_csv("iris.csv").astype({"petal_width": object})
    frame.loc[0, "petal_width"] = "wide"
    frame.to_csv("new.csv", index=False)
    result = runner.invoke(app, ["predict", str(model_path), "new.csv"])
    assert result.exit_code == 2
    assert "cannot predict on new.csv" in result.output
    assert "Traceback" not in result.output


def test_predict_missing_model_exits_1(workdir: Path) -> None:
    result = runner.invoke(app, ["predict", "nope.joblib", "iris.csv"])
    assert result.exit_code == 1
    assert "model file not found" in result.output


def test_predict_shows_version_warnings(workdir: Path, model_path: Path) -> None:
    (model_path.parent / "metadata.json").unlink()
    result = runner.invoke(app, ["predict", str(model_path), "iris.csv", "-o", "out.csv"])
    assert result.exit_code == 0
    assert "Warning: cannot check the scikit-learn version" in result.output
