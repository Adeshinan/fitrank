"""Tests for models[].calibrate: a model wrapped in CalibratedClassifierCV."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification

from fitrank import Benchmark, ConfigError, Results
from fitrank.config import validate_config
from fitrank.pipeline import build_candidates

REPO = Path(__file__).resolve().parents[1]
IRIS = pd.read_csv(REPO / "examples" / "data" / "iris.csv")
RF = {"n_estimators": [50], "max_depth": [None, 5]}


def _synthetic() -> pd.DataFrame:
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=0)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(20)])
    return data.assign(y=y)


@pytest.fixture(scope="module")
def forests() -> Results:
    """A raw and two calibrated random forests, side by side, with the Brier score."""
    models = [
        {"name": "random_forest", "params": RF},
        {"name": "random_forest", "params": RF, "calibrate": "isotonic"},
        {"name": "random_forest", "params": RF, "calibrate": "sigmoid"},
    ]
    return Benchmark(
        data=_synthetic(),
        target="y",
        task="classification",
        models=models,
        evaluation={"cv": 3, "metrics": ["neg_brier_score", "accuracy"]},
    ).run(export=False)


def _config(models: list[Any], task: str = "classification") -> Any:
    raw = {"data": {"path": "x.csv", "target": "y"}, "task": task, "models": models}
    return validate_config(raw, "test")


def test_calibrated_model_appears_with_suffix(forests: Results) -> None:
    board = forests.leaderboard()

    assert sorted(board["model"]) == [
        "random_forest",
        "random_forest (calibrated: isotonic)",
        "random_forest (calibrated: sigmoid)",
    ]
    assert (board["status"] == "ok").all()


def test_grid_is_rewritten_for_calibrated_model() -> None:
    raw, calibrated = build_candidates(
        _config([{"name": "random_forest", "params": RF},
                 {"name": "random_forest", "params": RF, "calibrate": "isotonic"}])
    )  # fmt: skip

    assert raw.param_grid == {"model__n_estimators": [50], "model__max_depth": [None, 5]}
    assert calibrated.param_grid == {
        "model__estimator__n_estimators": [50],
        "model__estimator__max_depth": [None, 5],
    }
    model = calibrated.pipeline.named_steps["model"]
    assert isinstance(model, CalibratedClassifierCV)
    assert (model.method, model.cv) == ("isotonic", 3)
    assert model.estimator.get_params()["random_state"] == 42  # set before wrapping


def test_leaderboard_shows_the_model_parameters(forests: Results) -> None:
    board = forests.leaderboard().set_index("model")
    params = board.loc["random_forest (calibrated: sigmoid)", "best_params"]
    assert set(params) == {"n_estimators", "max_depth"}  # not estimator__n_estimators


def test_calibration_improves_brier_on_random_forest(forests: Results) -> None:
    brier = forests.leaderboard().set_index("model")["test_brier_score"]

    assert brier["random_forest (calibrated: isotonic)"] < brier["random_forest"]
    assert brier["random_forest (calibrated: sigmoid)"] < brier["random_forest"]


def test_brier_score_as_secondary_metric(forests: Results) -> None:
    board = forests.leaderboard()
    assert "test_brier_score" in board.columns  # shown positive, lower is better
    assert board["test_brier_score"].between(0, 1).all()


def test_brier_score_as_primary_metric_ranks_lowest_first() -> None:
    models = [{"name": "knn", "params": {"n_neighbors": [1]}},  # 0/1 probabilities
              {"name": "logistic_regression", "params": {"C": [1]}}]  # fmt: skip
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models,
        evaluation={"cv": 3, "primary_metric": "neg_brier_score"},
    ).run(export=False)  # fmt: skip
    board = run.leaderboard()

    assert "cv_brier_score" in board.columns
    assert board["cv_brier_score"].is_monotonic_increasing


def test_calibrate_on_regression_is_an_error() -> None:
    config = _config([{"name": "ridge", "calibrate": "sigmoid"}], task="regression")
    with pytest.raises(ConfigError) as caught:
        build_candidates(config)
    assert "models[0].calibrate: calibration is for classification only" in str(caught.value)
    assert "remove calibrate" in str(caught.value)


def test_unknown_method_is_an_error() -> None:
    with pytest.raises(ConfigError, match=r"models\[0\]\.calibrate: must be 'sigmoid' or"):
        _config([{"name": "random_forest", "calibrate": "platt"}])


def test_without_calibrate_names_and_grids_are_unchanged() -> None:
    (candidate,) = build_candidates(_config([{"name": "random_forest", "params": RF}]))

    assert candidate.model_name == "random_forest"
    assert candidate.param_grid == {"model__n_estimators": [50], "model__max_depth": [None, 5]}
    assert not isinstance(candidate.pipeline.named_steps["model"], CalibratedClassifierCV)


def test_calibrated_svm_gets_predict_proba_for_oof() -> None:
    models = [{"name": "svm", "params": {"C": [1]}, "calibrate": "sigmoid"}]
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models,
        evaluation={"cv": 3, "metrics": ["neg_brier_score"]},
    ).run(export=False)  # fmt: skip
    proba = run.oof_proba("svm (calibrated: sigmoid)")

    assert proba.shape == (120, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert run.leaderboard()["notes"].isna().all()  # Brier not skipped: predict_proba exists


def test_oof_proba_default_params_keeps_calibration() -> None:
    models = [{"name": "knn", "params": {"n_neighbors": [3, 15]}, "calibrate": "sigmoid"}]
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models, evaluation={"cv": 3}
    ).run(export=False)
    pipeline = run._pipeline(run._candidate("knn (calibrated: sigmoid)", None), "default")

    assert isinstance(pipeline.named_steps["model"], CalibratedClassifierCV)
    assert pipeline.get_params()["model__estimator__n_neighbors"] == 5  # searched: default
    assert run.oof_proba("knn (calibrated: sigmoid)", params="default").shape == (120, 3)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_max_iter_is_counted_through_calibration() -> None:
    models = [{"name": "logistic_regression", "params": {"C": [100], "max_iter": [1]},
               "calibrate": "sigmoid"}]  # fmt: skip
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models, evaluation={"cv": 3}
    ).run(export=False)
    notes = run.leaderboard()["notes"][0]

    assert notes == "4 of 4 fits stopped at max_iter before converging (ConvergenceWarning)"


def test_class_path_can_be_calibrated() -> None:
    models = [{"class": "sklearn.svm.LinearSVC", "calibrate": "sigmoid"}]
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models, evaluation={"cv": 3}
    ).run(export=False)

    assert run.leaderboard()["model"][0] == "LinearSVC (calibrated: sigmoid)"
    assert run.oof_proba("LinearSVC (calibrated: sigmoid)").shape == (120, 3)


def test_saved_config_keeps_calibrate_only_when_set(tmp_path: Path) -> None:
    models = [{"name": "knn", "params": {"n_neighbors": [5]}},
              {"name": "knn", "params": {"n_neighbors": [5]}, "calibrate": "sigmoid"}]  # fmt: skip
    run = Benchmark(
        data=IRIS, target="species", task="classification", models=models, evaluation={"cv": 3}
    ).run(export=False)
    saved = yaml.safe_load((run.export(tmp_path) / "config.yaml").read_text(encoding="utf-8"))

    assert "calibrate" not in saved["models"][0]  # a v1.1 model, key for key
    assert saved["models"][1]["calibrate"] == "sigmoid"
