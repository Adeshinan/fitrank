"""Evaluate every candidate under exactly the same conditions.

The test set is put aside first and only used for the final score. The same
CV folds and the same metrics are used for every candidate, and a candidate
that fails is reported as ``failed`` without stopping the others.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from fitrank.config import FitRankConfig
from fitrank.exceptions import DataError
from fitrank.pipeline import Candidate
from fitrank.registry import Task

DEFAULT_PRIMARY: dict[Task, str] = {
    "classification": "f1_weighted",
    "regression": "neg_root_mean_squared_error",
}
DEFAULT_SECONDARY: dict[Task, list[str]] = {
    "classification": ["accuracy", "precision_weighted", "recall_weighted"],
    "regression": ["r2", "neg_mean_absolute_error"],
}


@dataclass
class CandidateResult:
    """The outcome of one candidate. Scores are shown positive (RMSE = 3.21)."""

    model_name: str
    reduction_name: str
    status: Literal["ok", "failed"]
    cv_mean: float | None
    cv_std: float | None
    test_score: float | None
    secondary: dict[str, float]  # test-set score of each secondary metric
    best_params: dict[str, Any]
    fit_seconds: float
    error: str | None
    best_estimator: Pipeline | None
    notes: list[str] = field(default_factory=list)


def primary_metric(config: FitRankConfig) -> str:
    """The metric that ranks candidates: the declared one, or the task default."""
    return config.evaluation.primary_metric or DEFAULT_PRIMARY[config.task]


def secondary_metrics(config: FitRankConfig) -> list[str]:
    """The other metrics reported: the declared ones, or the task suggestions."""
    declared = config.evaluation.metrics or DEFAULT_SECONDARY[config.task]
    primary = primary_metric(config)
    return [name for name in dict.fromkeys(declared) if name != primary]


def higher_is_better(metric: str) -> bool:
    """False for errors such as RMSE, which are shown positive (lower is better)."""
    return not metric.startswith("neg_")


def display_name(metric: str) -> str:
    """``root_mean_squared_error`` for ``neg_root_mean_squared_error``."""
    return metric.removeprefix("neg_")


def split_data(
    X: pd.DataFrame, y: pd.Series, config: FitRankConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Put the test set aside, stratified by class for classification.

    Raises:
        DataError: too few rows for the split or the CV folds.
    """
    classification = config.task == "classification"
    if classification:
        counts = y.value_counts()
        rare = counts[counts < 2]
        if len(rare):
            raise DataError(
                "some classes are too rare to split the data",
                [f"class {label!r} has only {n} row" for label, n in rare.items()],
                hint="each class needs at least 2 rows; remove or merge the rare classes",
            )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.evaluation.test_size,
            stratify=y if classification else None,
            random_state=config.project.random_state,
        )
    except ValueError as exc:
        raise DataError("cannot put the test set aside", [str(exc)]) from None

    folds = config.evaluation.cv
    if len(X_train) < folds:
        raise DataError(
            f"not enough rows for {folds}-fold cross-validation",
            [f"the training set has {len(X_train)} rows; lower evaluation.cv or add data"],
        )
    return X_train, X_test, y_train, y_test


def make_folds(config: FitRankConfig) -> KFold | StratifiedKFold:
    """The CV splitter shared by every candidate (same folds for all)."""
    kind = StratifiedKFold if config.task == "classification" else KFold
    return kind(
        n_splits=config.evaluation.cv, shuffle=True, random_state=config.project.random_state
    )


def evaluate(
    candidates: list[Candidate],
    X: pd.DataFrame,
    y: pd.Series,
    config: FitRankConfig,
    on_result: Callable[[CandidateResult], None] | None = None,
) -> list[CandidateResult]:
    """Search, cross-validate and test every candidate, in order.

    ``on_result`` is called after each candidate, e.g. to advance a progress bar.

    Raises:
        DataError: the data cannot be split (see ``split_data``). A failing
            candidate never raises: it is returned with ``status="failed"``.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    folds = make_folds(config)
    results = []
    for candidate in candidates:
        result = _evaluate_one(candidate, X_train, X_test, y_train, y_test, folds, config)
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


def _evaluate_one(
    candidate: Candidate,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    folds: KFold | StratifiedKFold,
    config: FitRankConfig,
) -> CandidateResult:
    primary = primary_metric(config)
    secondary, notes = [], []
    for metric in secondary_metrics(config):
        missing = _missing_method(candidate.pipeline, metric)
        if missing:
            notes.append(f"{metric} skipped: {candidate.model_name} has no {missing}")
        else:
            secondary.append(metric)

    start = time.perf_counter()
    try:
        missing = _missing_method(candidate.pipeline, primary)
        if missing:
            raise ValueError(
                f"primary metric {primary!r} needs {missing},"
                f" which {candidate.model_name} does not provide"
            )
        search = _make_search(candidate, folds, [primary, *secondary], config)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        sign = 1 if higher_is_better(primary) else -1
        index = search.best_index_
        result = CandidateResult(
            model_name=candidate.model_name,
            reduction_name=candidate.reduction_name,
            status="ok",
            cv_mean=sign * float(search.cv_results_[f"mean_test_{primary}"][index]),
            cv_std=float(search.cv_results_[f"std_test_{primary}"][index]),
            test_score=_score(best, primary, X_test, y_test),
            secondary={m: _score(best, m, X_test, y_test) for m in secondary},
            best_params=dict(search.best_params_),
            fit_seconds=0.0,
            error=None,
            best_estimator=best,
            notes=notes,
        )
    except Exception as exc:  # a failing candidate must never stop the others
        result = CandidateResult(
            model_name=candidate.model_name,
            reduction_name=candidate.reduction_name,
            status="failed",
            cv_mean=None,
            cv_std=None,
            test_score=None,
            secondary={},
            best_params={},
            fit_seconds=0.0,
            error=_describe_failure(exc),
            best_estimator=None,
            notes=notes,
        )
    result.fit_seconds = time.perf_counter() - start
    return result


def _make_search(
    candidate: Candidate, folds: KFold | StratifiedKFold, metrics: list[str], config: FitRankConfig
) -> GridSearchCV | RandomizedSearchCV:
    common: dict[str, Any] = {
        "scoring": {metric: metric for metric in metrics},
        "refit": metrics[0],
        "cv": folds,
        "error_score": np.nan,  # a failing combination is skipped, the others still count
    }
    if config.evaluation.search == "grid":
        return GridSearchCV(candidate.pipeline, candidate.param_grid, **common)
    size = len(ParameterGrid(candidate.param_grid))
    return RandomizedSearchCV(
        candidate.pipeline,
        candidate.param_grid,
        n_iter=min(config.evaluation.n_iter, size),
        random_state=config.project.random_state,
        **common,
    )


def _score(estimator: Pipeline, metric: str, X: pd.DataFrame, y: pd.Series) -> float:
    value = float(get_scorer(metric)(estimator, X, y))
    return value if higher_is_better(metric) else -value


def _missing_method(pipeline: Pipeline, metric: str) -> str | None:
    """Name the prediction method ``metric`` needs and ``pipeline`` lacks, if any.

    Without this check, one such metric (e.g. roc_auc_ovr on an SVM, which has
    no predict_proba) makes the whole search fail.
    """
    # Private but stable since scikit-learn 1.4; "predict" if it ever goes away.
    methods = getattr(get_scorer(metric), "_response_method", "predict")
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    if any(hasattr(pipeline, method) for method in methods):
        return None
    return " or ".join(methods)


def _describe_failure(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    text = f"{type(exc).__name__}: {lines[0]}"
    if "fits failed" in lines[0]:  # scikit-learn's summary; its last line is the cause
        text += f" Last error: {lines[-1]}"
    return text
