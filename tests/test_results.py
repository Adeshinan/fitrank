"""Tests for the results: leaderboard, run folder, exports and reloading the best model."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sklearn
import yaml
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fitrank.config import FitRankConfig
from fitrank.evaluation import CandidateResult, evaluate, split_data
from fitrank.exceptions import FitRankError, FitRankWarning
from fitrank.pipeline import build_candidates
from fitrank.results import Results, load_model

IRIS = load_iris(as_frame=True).frame.rename(columns={"target": "species"})


def make_config(task: str = "classification", **output: Any) -> FitRankConfig:
    return FitRankConfig.model_validate({
        "project": {"name": "iris"},
        "data": {"path": "data/iris.csv", "target": "species"},
        "task": task,
        "reduction": ["none", {"pca": {"n_components": [2]}}],
        "models": [
            {"name": "logistic_regression", "params": {"C": [0.1, 1]}},
            {"name": "knn", "params": {"n_neighbors": [3, 5]}},
        ],
        "evaluation": {"cv": 3},
        "output": output,
    })  # fmt: skip


def fake(model: str, cv: float | None, status: str = "ok", **fields: Any) -> CandidateResult:
    """A result built by hand: tests the ranking without training anything."""
    values: dict[str, Any] = {
        "model_name": model,
        "reduction_name": "none",
        "status": status,
        "cv_mean": cv,
        "cv_std": None if cv is None else 0.01,
        "test_score": None if cv is None else cv - 0.02,
        "secondary": {},
        "best_params": {},
        "fit_seconds": 0.5,
        "error": None,
        "best_estimator": Pipeline([("model", DummyClassifier())]) if status == "ok" else None,
    }
    return CandidateResult(**{**values, **fields})


@pytest.fixture
def fake_results() -> list[CandidateResult]:
    return [
        fake("knn", 0.90),
        fake("always_fails", None, status="failed", error="ValueError: boom"),
        fake("svm", 0.97, secondary={"accuracy": 0.96}),
        fake("decision_tree", 0.85),
    ]


@pytest.fixture(scope="module")
def iris_run() -> tuple[Results, pd.DataFrame]:
    """A real (small) run on Iris, shared by the export tests; returns it with X_test."""
    config = make_config()
    X, y = IRIS.drop(columns="species"), IRIS["species"]
    results = Results(evaluate(build_candidates(config), X, y, config), config)
    _, X_test, _, _ = split_data(X, y, config)
    return results, X_test


# --- Leaderboard -----------------------------------------------------------------------------


def test_leaderboard_is_sorted_and_failed_last(fake_results: list[CandidateResult]) -> None:
    board = Results(fake_results, make_config()).leaderboard()
    assert board["model"].tolist() == ["svm", "knn", "decision_tree", "always_fails"]
    assert board["rank"].tolist() == [1, 2, 3, 4]
    assert board["status"].tolist() == ["ok", "ok", "ok", "failed"]
    assert board.iloc[-1]["error"] == "ValueError: boom"
    assert math.isnan(board.iloc[-1]["cv_f1_weighted"])


def test_columns_are_named_after_the_metrics(fake_results: list[CandidateResult]) -> None:
    board = Results(fake_results, make_config()).leaderboard()
    assert board.columns.tolist() == [
        "rank", "model", "reduction", "status", "cv_f1_weighted", "cv_std", "test_f1_weighted",
        "test_accuracy", "test_precision_weighted", "test_recall_weighted",
        "fit_s", "best_params", "error", "notes",
    ]  # fmt: skip
    assert board.iloc[0]["test_accuracy"] == 0.96
    assert math.isnan(board.iloc[1]["test_accuracy"])  # not computed for this candidate


def test_errors_are_ranked_lowest_first() -> None:
    results = [fake("ridge", 55.0), fake("lasso", 48.0), fake("decision_tree", 70.0)]
    board = Results(results, make_config("regression")).leaderboard()
    assert board["model"].tolist() == ["lasso", "ridge", "decision_tree"]
    assert "cv_root_mean_squared_error" in board.columns


def test_ties_keep_the_declaration_order() -> None:
    results = [fake("knn", 0.9), fake("svm", 0.9)]
    assert Results(results, make_config()).leaderboard()["model"].tolist() == ["knn", "svm"]


def test_best_params_lose_their_step_prefix() -> None:
    params = {"model__C": 1, "reduction__n_components": 2}
    board = Results([fake("svm", 0.9, best_params=params)], make_config()).leaderboard()
    assert board.iloc[0]["best_params"] == {"C": 1, "n_components": 2}
    # Two steps with the same parameter name keep their prefix
    clash = {"model__n_components": 1, "reduction__n_components": 2}
    board = Results([fake("svm", 0.9, best_params=clash)], make_config()).leaderboard()
    assert board.iloc[0]["best_params"] == clash


def test_notes_are_joined() -> None:
    result = fake("svm", 0.9, notes=["a skipped", "b skipped"])
    assert Results([result], make_config()).leaderboard().iloc[0]["notes"] == "a skipped; b skipped"


def test_empty_leaderboard_keeps_its_columns() -> None:
    board = Results([], make_config()).leaderboard()
    assert board.empty and board.columns[0] == "rank"


# --- Best model ------------------------------------------------------------------------------


def test_best_model_is_the_first_ranked(iris_run: tuple[Results, pd.DataFrame]) -> None:
    results, _ = iris_run
    best = results.best
    assert best is not None and best is results.results[0]
    assert results.best_model() is best.best_estimator


def test_best_model_when_everything_failed() -> None:
    results = Results([fake("svm", None, status="failed", error="ValueError: boom")], make_config())
    assert results.best is None
    with pytest.raises(FitRankError, match="every candidate failed") as info:
        results.best_model()
    assert info.value.details == ["none + svm: ValueError: boom"]


# --- Run folder and exports ------------------------------------------------------------------


def test_export_writes_all_files(tmp_path: Path, iris_run: tuple[Results, pd.DataFrame]) -> None:
    results, _ = iris_run
    run_dir = results.export(tmp_path, ["csv", "json", "markdown"])
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("iris_20")
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "best_model.joblib", "config.yaml", "leaderboard.csv", "leaderboard.json",
        "leaderboard.md", "metadata.json",
    ]  # fmt: skip


def test_export_follows_the_output_section(tmp_path: Path) -> None:
    config = make_config(formats=["json"], save_best_model=False)
    results = Results([fake("svm", 0.9)], config)
    run_dir = results.export(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.yaml",
        "leaderboard.json",
        "metadata.json",
    ]


def test_export_defaults_to_project_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Results([fake("svm", 0.9)], make_config()).export()
    assert run_dir.parent == Path("runs")


def test_two_runs_never_share_a_folder(tmp_path: Path) -> None:
    results = Results([fake("svm", 0.9)], make_config())
    first, second = results.export(tmp_path), results.export(tmp_path)
    assert first != second and first.is_dir() and second.is_dir()


def test_unknown_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown format"):
        Results([fake("svm", 0.9)], make_config()).export(tmp_path, ["xlsx"])


def test_saved_config_reloads_to_the_same_config(tmp_path: Path) -> None:
    config = make_config()
    run_dir = Results([fake("svm", 0.9)], config).export(tmp_path)
    text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "path: data/iris.csv" in text  # forward slashes, even on Windows
    assert FitRankConfig.model_validate(yaml.safe_load(text)) == config


def test_leaderboard_files_hold_the_same_rows(
    tmp_path: Path, fake_results: list[CandidateResult]
) -> None:
    run_dir = Results(fake_results, make_config()).export(tmp_path)

    csv = pd.read_csv(run_dir / "leaderboard.csv")
    assert csv["model"].tolist() == ["svm", "knn", "decision_tree", "always_fails"]

    rows = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["cv_f1_weighted"] is None  # NaN becomes null

    markdown = (run_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert "Ranked on the cross-validation f1_weighted (higher is better)" in markdown
    assert "| 1 | svm | none | ok | 0.9700 |" in markdown
    assert "| 4 | always_fails | none | failed | — |" in markdown


def test_metadata_records_versions_and_best(tmp_path: Path) -> None:
    results = Results([fake("knn", 0.9), fake("svm", 0.95)], make_config(), duration_seconds=12.5)
    metadata = json.loads((results.export(tmp_path) / "metadata.json").read_text("utf-8"))
    assert metadata["duration_seconds"] == 12.5
    assert metadata["primary_metric"] == "f1_weighted"
    assert metadata["best"] == {"model": "svm", "reduction": "none", "cv": 0.95}
    assert metadata["versions"]["scikit-learn"] == sklearn.__version__
    assert set(metadata["versions"]) == {
        "python", "fitrank", "scikit-learn", "pandas", "numpy", "joblib"
    }  # fmt: skip


def test_export_when_everything_failed(tmp_path: Path) -> None:
    results = Results([fake("svm", None, status="failed")], make_config())
    run_dir = results.export(tmp_path)
    assert not (run_dir / "best_model.joblib").exists()
    assert json.loads((run_dir / "metadata.json").read_text("utf-8"))["best"] is None


# --- Reloading the best model ----------------------------------------------------------------


def test_saved_model_predicts_like_original(
    tmp_path: Path, iris_run: tuple[Results, pd.DataFrame]
) -> None:
    results, X_test = iris_run
    run_dir = results.export(tmp_path, ["csv"])
    reloaded = load_model(run_dir / "best_model.joblib")
    assert (reloaded.predict(X_test) == results.best_model().predict(X_test)).all()


def test_load_model_warns_on_another_scikit_learn(
    tmp_path: Path, iris_run: tuple[Results, pd.DataFrame]
) -> None:
    run_dir = iris_run[0].export(tmp_path, ["csv"])
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["versions"]["scikit-learn"] = "1.4.0"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.warns(FitRankWarning, match="saved with scikit-learn 1.4.0"):
        load_model(run_dir / "best_model.joblib")


def test_load_model_warns_without_metadata(
    tmp_path: Path, iris_run: tuple[Results, pd.DataFrame]
) -> None:
    run_dir = iris_run[0].export(tmp_path, ["csv"])
    (run_dir / "metadata.json").unlink()
    with pytest.warns(FitRankWarning, match="cannot check the scikit-learn version"):
        load_model(run_dir / "best_model.joblib")


def test_load_model_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FitRankError, match="model file not found"):
        load_model(tmp_path / "nope.joblib")
