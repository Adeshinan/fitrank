"""Tests for the evaluation: split, shared CV folds, search, metrics and failures."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import KFold, StratifiedKFold

from fitrank import registry
from fitrank.config import FitRankConfig
from fitrank.evaluation import (
    CandidateResult,
    display_name,
    evaluate,
    higher_is_better,
    make_folds,
    primary_metric,
    secondary_metrics,
    split_data,
)
from fitrank.exceptions import DataError
from fitrank.pipeline import build_candidates
from fitrank.registry import Grid, register_model

IRIS = load_iris(as_frame=True).frame.rename(columns={"target": "species"})
DIABETES = load_diabetes(as_frame=True).frame


def iris_config(**evaluation: Any) -> FitRankConfig:
    return FitRankConfig.model_validate({
        "data": {"path": "iris.csv", "target": "species"},
        "task": "classification",
        "reduction": ["none", {"pca": {"n_components": [2]}}],
        "models": [
            {"name": "logistic_regression", "params": {"C": [0.1, 1]}},
            {"name": "knn", "params": {"n_neighbors": [3, 5]}},
        ],
        "evaluation": {"cv": 3, **evaluation},
    })  # fmt: skip


def diabetes_config(**evaluation: Any) -> FitRankConfig:
    return FitRankConfig.model_validate({
        "data": {"path": "diabetes.csv", "target": "target"},
        "task": "regression",
        "models": [{"name": "ridge"}, {"name": "decision_tree", "params": {"max_depth": [3]}}],
        "evaluation": {"cv": 3, **evaluation},
    })  # fmt: skip


def run(config: FitRankConfig, df: pd.DataFrame) -> list[CandidateResult]:
    X, y = df.drop(columns=config.data.target), df[config.data.target]
    return evaluate(build_candidates(config), X, y, config)


class FailingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: Any, y: Any) -> "FailingClassifier":
        raise ValueError("this model always fails")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def failing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "MODELS", dict(registry.MODELS))

    @register_model("always_fails", task="classification")
    def always_fails() -> tuple[FailingClassifier, Grid]:
        return FailingClassifier(), {}


# --- Results ---------------------------------------------------------------------------------


def test_every_candidate_is_evaluated_on_iris() -> None:
    results = run(iris_config(), IRIS)
    assert [(r.reduction_name, r.model_name) for r in results] == [
        ("none", "logistic_regression"), ("none", "knn"),
        ("pca", "logistic_regression"), ("pca", "knn"),
    ]  # fmt: skip
    for result in results:
        assert result.status == "ok"
        assert result.error is None
        assert 0.8 < result.cv_mean <= 1  # type: ignore[operator]
        assert result.cv_std is not None and result.cv_std >= 0
        assert 0.8 < result.test_score <= 1  # type: ignore[operator]
        assert result.fit_seconds > 0
        assert result.best_estimator is not None


def test_best_params_come_from_the_grid() -> None:
    result = run(iris_config(), IRIS)[0]
    assert set(result.best_params) == {"model__C"}
    assert result.best_params["model__C"] in (0.1, 1)


def test_secondary_metrics_default_to_the_task_suggestions() -> None:
    result = run(iris_config(), IRIS)[0]
    assert set(result.secondary) == {"accuracy", "precision_weighted", "recall_weighted"}
    assert all(0 <= value <= 1 for value in result.secondary.values())


def test_declared_metrics_are_used() -> None:
    config = iris_config(primary_metric="accuracy", metrics=["accuracy", "f1_macro"])
    assert primary_metric(config) == "accuracy"
    assert secondary_metrics(config) == ["f1_macro"]  # the primary is not repeated
    assert set(run(config, IRIS)[0].secondary) == {"f1_macro"}


def test_results_are_reproducible() -> None:
    first = run(iris_config(), IRIS)
    second = run(iris_config(), IRIS)
    assert [(r.cv_mean, r.cv_std, r.test_score) for r in first] == [
        (r.cv_mean, r.cv_std, r.test_score) for r in second
    ]


def test_random_search_samples_the_grid() -> None:
    config = iris_config(search="random", n_iter=1)
    results = run(config, IRIS)
    assert all(r.status == "ok" for r in results)
    # n_iter above the grid size is capped instead of warning
    assert all(r.status == "ok" for r in run(iris_config(search="random", n_iter=50), IRIS))


# --- Regression ------------------------------------------------------------------------------


def test_regression_rmse_is_reported_positive() -> None:
    config = diabetes_config()
    assert primary_metric(config) == "neg_root_mean_squared_error"
    results = run(config, DIABETES)
    for result in results:
        assert result.status == "ok"
        assert result.cv_mean is not None and 30 < result.cv_mean < 100  # RMSE, positive
        assert result.test_score is not None and result.test_score > 0
        assert result.secondary["neg_mean_absolute_error"] > 0
        assert -1 < result.secondary["r2"] < 1


def test_metric_direction_and_display_name() -> None:
    assert higher_is_better("r2")
    assert not higher_is_better("neg_root_mean_squared_error")
    assert display_name("neg_root_mean_squared_error") == "root_mean_squared_error"
    assert display_name("f1_weighted") == "f1_weighted"


# --- Same conditions for every candidate -----------------------------------------------------


def test_test_set_is_put_aside_and_stratified() -> None:
    config = iris_config()
    X, y = IRIS.drop(columns="species"), IRIS["species"]
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    assert len(X_test) == 30 and len(X_train) == 120
    assert y_test.value_counts().tolist() == [10, 10, 10]  # stratified: 1/3 of each class
    assert not set(X_train.index) & set(X_test.index)


def test_regression_split_is_not_stratified() -> None:
    X, y = DIABETES.drop(columns="target"), DIABETES["target"]
    X_train, X_test, _, _ = split_data(X, y, diabetes_config())
    assert len(X_test) == math.ceil(len(X) * 0.2)  # scikit-learn rounds the test set up


def test_folds_depend_on_the_task_and_are_seeded() -> None:
    folds = make_folds(iris_config())
    assert isinstance(folds, StratifiedKFold)
    assert (folds.n_splits, folds.shuffle, folds.random_state) == (3, True, 42)
    assert type(make_folds(diabetes_config())) is KFold


def test_every_candidate_sees_the_same_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[list[int]]] = []
    real_split = StratifiedKFold.split

    def recording_split(self: StratifiedKFold, X: Any, y: Any = None, groups: Any = None) -> Any:
        splits = list(real_split(self, X, y, groups))
        seen.append([test.tolist() for _, test in splits])
        return iter(splits)

    monkeypatch.setattr(StratifiedKFold, "split", recording_split)
    run(iris_config(), IRIS)
    assert len(seen) == 4
    assert all(folds == seen[0] for folds in seen)


# --- Failures --------------------------------------------------------------------------------


@pytest.mark.usefixtures("failing_model")
def test_failing_model_does_not_stop_others() -> None:
    config = iris_config()
    config.models.insert(0, config.models[0].model_copy(update={"name": "always_fails"}))
    config.models[0].params = None
    results = run(config, IRIS)
    failed = [r for r in results if r.status == "failed"]
    assert len(failed) == 2  # one per reduction
    for result in failed:
        assert result.model_name == "always_fails"
        assert result.error is not None
        assert result.error.startswith("ValueError: All the 3 fits failed.")
        assert "this model always fails" in result.error
        assert result.cv_mean is None and result.best_estimator is None
    assert sum(r.status == "ok" for r in results) == 4


def test_secondary_metric_the_model_cannot_compute_is_skipped() -> None:
    config = FitRankConfig.model_validate({
        "data": {"path": "iris.csv", "target": "species"},
        "task": "classification",
        "models": [{"name": "svm", "params": {"C": [1]}}, {"name": "logistic_regression"}],
        "evaluation": {"cv": 3, "metrics": ["accuracy", "roc_auc_ovr"]},
    })  # fmt: skip
    svm, logistic = run(config, IRIS)
    assert svm.status == "ok"
    assert set(svm.secondary) == {"accuracy"}
    assert svm.notes == ["roc_auc_ovr skipped: svm has no predict_proba"]
    assert set(logistic.secondary) == {"accuracy", "roc_auc_ovr"}
    assert logistic.notes == []


def test_primary_metric_the_model_cannot_compute_fails_clearly() -> None:
    config = FitRankConfig.model_validate({
        "data": {"path": "iris.csv", "target": "species"},
        "task": "classification",
        "models": [{"name": "svm", "params": {"C": [1]}}],
        "evaluation": {"cv": 3, "primary_metric": "roc_auc_ovr"},
    })  # fmt: skip
    (result,) = run(config, IRIS)
    assert result.status == "failed"
    assert result.error == (
        "ValueError: primary metric 'roc_auc_ovr' needs predict_proba, which svm does not provide"
    )


# --- Data that cannot be split ---------------------------------------------------------------


def test_class_with_a_single_row_is_refused() -> None:
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"])
    with pytest.raises(DataError, match="too rare") as info:
        split_data(X, y, iris_config())
    assert info.value.details == ["class 'c' has only 1 row"]


def test_too_few_rows_for_the_folds_is_refused() -> None:
    X = pd.DataFrame({"a": np.arange(5.0)})
    y = pd.Series(np.arange(5.0))
    with pytest.raises(DataError, match="not enough rows for 5-fold"):
        split_data(X, y, diabetes_config(cv=5))


def test_impossible_split_is_explained() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    with pytest.raises(DataError, match="cannot put the test set aside"):
        split_data(X, y, iris_config(test_size=0.1))
