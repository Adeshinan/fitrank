"""Tests for the model registry and the built-in models."""

import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import ParameterGrid

from fitrank import registry
from fitrank.exceptions import ConfigError
from fitrank.registry import TASKS, Grid, Task, get_model, list_models, register_model

BUILTINS = [(name, task) for task in TASKS for name in list_models(task)]


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a test change the registry without leaking into other tests."""
    monkeypatch.setattr(registry, "MODELS", dict(registry.MODELS))


def test_sixteen_builtin_models() -> None:
    assert list_models("classification") == [
        "decision_tree", "gradient_boosting", "knn", "logistic_regression",
        "naive_bayes", "random_forest", "svm",
    ]  # fmt: skip
    assert list_models("regression") == [
        "decision_tree", "gradient_boosting", "knn", "lasso", "linear_regression",
        "polynomial_regression", "random_forest", "ridge", "svr",
    ]  # fmt: skip
    assert len(BUILTINS) == 16


@pytest.mark.parametrize(("name", "task"), BUILTINS)
def test_every_builtin_model_fits_with_every_grid_value(name: str, task: Task) -> None:
    if task == "classification":
        X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    else:
        X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    estimator, grid = get_model(name, task).factory()

    assert set(grid) <= set(estimator.get_params()), "the grid names an unknown parameter"
    for key, values in grid.items():
        for value in values:  # one value at a time, so that a bad one is caught
            clone(estimator).set_params(**{key: value}).fit(X, y)


@pytest.mark.parametrize(("name", "task"), BUILTINS)
def test_default_grids_are_small(name: str, task: Task) -> None:
    _, grid = get_model(name, task).factory()
    assert all(grid.values()), "a grid entry has no value"
    assert 2 <= len(ParameterGrid(grid)) <= 12


def test_imbalance_is_searched_by_models_that_support_it() -> None:
    for name in ["logistic_regression", "svm", "decision_tree", "random_forest"]:
        _, grid = get_model(name, "classification").factory()
        assert grid["class_weight"] == [None, "balanced"]


def test_factory_returns_a_fresh_estimator_each_time() -> None:
    spec = get_model("random_forest", "classification")
    assert spec.factory()[0] is not spec.factory()[0]


def test_yaml_params_replace_the_default_grid() -> None:
    _, grid = get_model("random_forest", "classification").build({"n_estimators": [50]})
    assert grid == {"n_estimators": [50]}


def test_default_grid_is_used_without_params() -> None:
    _, grid = get_model("ridge", "regression").build()
    assert grid == {"alpha": [0.1, 1, 10]}


def test_random_state_is_set_when_supported() -> None:
    forest, _ = get_model("random_forest", "regression").build(random_state=7)
    assert forest.random_state == 7
    knn, _ = get_model("knn", "regression").build(random_state=7)
    assert "random_state" not in knn.get_params()


def test_same_name_in_both_tasks_gives_different_models() -> None:
    classifier, _ = get_model("knn", "classification").factory()
    regressor, _ = get_model("knn", "regression").factory()
    assert type(classifier).__name__ == "KNeighborsClassifier"
    assert type(regressor).__name__ == "KNeighborsRegressor"


@pytest.mark.usefixtures("clean_registry")
def test_custom_model_registration() -> None:
    @register_model("extra_trees", task="classification")
    def extra_trees() -> tuple[ExtraTreesClassifier, Grid]:
        return ExtraTreesClassifier(), {"n_estimators": [50]}

    assert "extra_trees" in list_models("classification")
    assert "extra_trees" not in list_models("regression")
    assert get_model("extra_trees", "classification").factory is extra_trees


@pytest.mark.usefixtures("clean_registry")
def test_custom_model_does_not_leak_into_other_tests() -> None:
    assert "extra_trees" not in list_models("classification")


@pytest.mark.usefixtures("clean_registry")
def test_registering_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="'svm' is already registered for classification"):
        register_model("svm", task="classification")(lambda: (ExtraTreesClassifier(), {}))


def test_unknown_model_suggests_available() -> None:
    with pytest.raises(ConfigError) as info:
        get_model("random_forrest", "classification")
    assert str(info.value) == (
        "unknown model 'random_forrest' for task 'classification'\n"
        "  -> available: decision_tree, gradient_boosting, knn, logistic_regression,"
        " naive_bayes, random_forest, svm\n"
        "  -> did you mean 'random_forest'?"
    )


@pytest.mark.usefixtures("clean_registry")
def test_task_without_models_says_none() -> None:
    registry.MODELS.clear()
    with pytest.raises(ConfigError, match="available: none"):
        get_model("anything", "regression")
