"""Tests for Results.oof_proba: out-of-fold class probabilities on the training rows."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fitrank import Benchmark, FitRankError, Results

REPO = Path(__file__).resolve().parents[1]
IRIS = pd.read_csv(REPO / "examples" / "data" / "iris.csv")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SETUP: dict[str, Any] = {
    "target": "species",
    "task": "classification",
    "preprocessing": {"numeric": {"columns": FEATURES, "steps": ["scaler"]}},
    "evaluation": {"cv": 3},
}
MODELS: list[Any] = [
    {"name": "logistic_regression", "params": {"C": [0.01, 1, 100]}},
    {"name": "knn", "params": {"n_neighbors": [3, 15], "weights": ["distance"]}},
    {"name": "svm", "params": {"C": [1]}},  # no probability=True: no predict_proba
    {"name": "random_forest", "params": {"n_estimators": [20], "max_depth": [3]}},
]


@pytest.fixture(scope="module")
def results() -> Results:
    """One small Iris run (test_size split), shared by the tests that only read it."""
    return Benchmark(data=IRIS, models=MODELS, **SETUP).run(export=False)


def test_oof_proba_shape_and_sums(results: Results) -> None:
    proba = results.oof_proba("logistic_regression")

    assert results.n_train == 120
    assert proba.shape == (120, 3)
    assert list(proba.columns) == ["setosa", "versicolor", "virginica"]
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_index_matches_training_data(results: Results) -> None:
    proba = results.oof_proba("random_forest")

    assert results.X_train is not None and results.y_train is not None
    assert proba.index.equals(results.X_train.index)
    assert proba.index.isin(IRIS.index).all()  # the original row labels: y.loc[proba.index]
    assert results.y_train.index.equals(proba.index)


def test_oof_uses_out_of_fold_predictions(results: Results) -> None:
    """Each row is predicted by a model that did not see it, unlike the refitted best model."""
    proba = results.oof_proba("knn")  # weights=distance: a seen row predicts itself at 100 %
    knn = next(r for r in results.results if r.model_name == "knn").best_estimator
    assert knn is not None and results.X_train is not None
    in_sample = knn.predict_proba(results.X_train)  # the refitted model saw every row

    assert np.allclose(in_sample.max(axis=1), 1)
    assert not np.allclose(proba.to_numpy(), in_sample)
    assert (proba.max(axis=1) < 1).any()  # out of fold: some rows are uncertain


def test_same_folds_and_best_params_as_the_run(results: Results) -> None:
    proba = results.oof_proba("logistic_regression")
    best = next(r for r in results.results if r.model_name == "logistic_regression")
    assert best.best_estimator is not None and results.X_train is not None
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # the run's folds
    expected = cross_val_predict(
        clone(best.best_estimator), results.X_train, results.y_train, cv=folds,
        method="predict_proba",
    )  # fmt: skip

    np.testing.assert_allclose(proba.to_numpy(), expected)
    assert best.best_params["model__C"] in (0.01, 1, 100)


def test_same_result_twice(results: Results) -> None:
    again = Benchmark(data=IRIS, models=MODELS, **SETUP).run(export=False)
    first = results.oof_proba("random_forest")

    pd.testing.assert_frame_equal(first, results.oof_proba("random_forest"))
    pd.testing.assert_frame_equal(first, again.oof_proba("random_forest"))


def test_a_copy_is_returned(results: Results) -> None:
    proba = results.oof_proba("logistic_regression")
    proba.iloc[:, :] = 0
    assert results.oof_proba("logistic_regression").sum().sum() == pytest.approx(120)


def test_run_computes_nothing_until_called() -> None:
    fresh = Benchmark(data=IRIS, models=MODELS[:1], **SETUP).run(export=False)
    assert fresh._oof == {}
    fresh.oof_proba("logistic_regression")
    assert list(fresh._oof) == [("logistic_regression", "none", "best")]


def test_model_without_predict_proba_is_explained(results: Results) -> None:
    with pytest.raises(FitRankError) as caught:
        results.oof_proba("svm")
    assert "svm + none has no predict_proba" in str(caught.value)
    assert "calibrate it in the config (calibrate: sigmoid)" in str(caught.value)
    assert "or choose a model that has it" in str(caught.value)


@pytest.mark.filterwarnings("ignore:The `probability` parameter was deprecated:FutureWarning")
def test_svm_with_probability_works() -> None:
    models = [{"name": "svm", "params": {"C": [1], "probability": [True]}}]
    proba = Benchmark(data=IRIS, models=models, **SETUP).run(export=False).oof_proba("svm")
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_default_params_keep_fixed_values_only(results: Results) -> None:
    """params="default": searched values fall back to the defaults, single values stay."""
    knn = results._pipeline(results._candidate("knn", None), "default")
    assert knn.get_params()["model__n_neighbors"] == 5  # searched [3, 15]: scikit-learn default
    assert knn.get_params()["model__weights"] == "distance"  # one value: fixed in advance

    strict = results.oof_proba("knn", params="default")
    assert strict.shape == (120, 3)
    assert not np.allclose(strict.to_numpy(), results.oof_proba("knn").to_numpy())


def test_reduction_chooses_among_several() -> None:
    bench = Benchmark(
        data=IRIS, models=MODELS[:1], reduction=["none", {"pca": {"n_components": [2]}}], **SETUP
    )
    several = bench.run(export=False)
    with pytest.raises(FitRankError) as caught:
        several.oof_proba("logistic_regression")
    assert "'logistic_regression' ran with 2 reductions: choose one" in str(caught.value)
    assert "oof_proba('logistic_regression', reduction='pca')" in str(caught.value)

    proba = several.oof_proba("logistic_regression", reduction="pca")
    assert proba.shape == (120, 3)


def test_unknown_candidate_is_explained(results: Results) -> None:
    with pytest.raises(FitRankError) as caught:
        results.oof_proba("gradient_boosting")
    assert "no candidate 'gradient_boosting' in these results" in str(caught.value)
    assert "logistic_regression + none" in str(caught.value)
    with pytest.raises(FitRankError, match="no candidate 'knn \\+ pca'"):
        results.oof_proba("knn", reduction="pca")


@pytest.mark.filterwarnings("ignore")  # the failing fits warn, on purpose
def test_failed_candidate_is_explained() -> None:
    models = [{"name": "logistic_regression", "params": {"C": [1]}},
              {"name": "knn", "params": {"n_neighbors": [500]}}]  # fmt: skip
    run = Benchmark(data=IRIS, models=models, **SETUP).run(export=False)
    with pytest.raises(FitRankError, match="knn \\+ none failed: no model to predict with"):
        run.oof_proba("knn")


def test_test_data_uses_every_training_row() -> None:
    train, test = IRIS.drop(index=IRIS.index[::5]), IRIS.iloc[::5]
    run = Benchmark(data=train, test_data=test, models=MODELS[:1], **SETUP).run(export=False)
    proba = run.oof_proba("logistic_regression")

    assert proba.index.equals(train.index)  # all of data, in its order


def test_regression_is_refused() -> None:
    diabetes = pd.read_csv(REPO / "examples" / "data" / "diabetes.csv")
    run = Benchmark(
        data=diabetes, target="progression", task="regression",
        models=[{"name": "ridge", "params": {"alpha": [1]}}], evaluation={"cv": 3},
    ).run(export=False)  # fmt: skip
    with pytest.raises(FitRankError, match="oof_proba is for classification only"):
        run.oof_proba("ridge")


def test_results_without_training_rows(results: Results) -> None:
    bare = Results(results.results, results.config)
    with pytest.raises(FitRankError, match="no training rows in these results"):
        bare.oof_proba("logistic_regression")


def test_params_must_be_best_or_default(results: Results) -> None:
    with pytest.raises(ValueError, match="params must be 'best' or 'default'"):
        results.oof_proba("knn", params="tuned")  # type: ignore[arg-type]
