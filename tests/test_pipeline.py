"""Tests for preprocessing, reductions and candidate pipelines."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, f_regression
from sklearn.model_selection import GridSearchCV

from fitrank import reduction as reduction_module
from fitrank.config import FitRankConfig
from fitrank.data import check_data, load_dataframe, split_features_target
from fitrank.exceptions import ConfigError
from fitrank.pipeline import build_candidates
from fitrank.preprocessing import build_preprocessing, register_step
from fitrank.reduction import build_reduction, list_reductions, register_reduction

FIXTURES = Path(__file__).parent / "fixtures"

TITANIC: dict[str, Any] = {
    "data": {
        "path": str(FIXTURES / "titanic_small.csv"),
        "target": "Survived",
        "features": ["Age", "Fare", "Sex", "Pclass", "Embarked"],
    },
    "task": "classification",
    "preprocessing": {
        "numeric": {
            "columns": ["Age", "Fare"],
            "steps": [{"imputer": {"strategy": "median"}}, {"scaler": "standard"}],
        },
        "categorical": {
            "columns": ["Sex", "Pclass", "Embarked"],
            "steps": [{"imputer": {"strategy": "most_frequent"}}, {"encoder": "onehot"}],
        },
    },
    "reduction": ["none", {"pca": {"n_components": [2, 3]}}],
    "models": [
        {"name": "logistic_regression", "params": {"C": [0.1, 1]}},
        {"name": "random_forest", "params": {"n_estimators": [10], "max_depth": [None, 3]}},
        {"name": "svm"},
        {"name": "decision_tree"},
    ],
}


def make_config(**overrides: Any) -> FitRankConfig:
    return FitRankConfig.model_validate(TITANIC | overrides)


def titanic_xy(config: FitRankConfig) -> tuple[pd.DataFrame, pd.Series]:
    X, y = split_features_target(load_dataframe(config.data.path), config.data)
    check_data(X, y, config)  # the fixture is valid for the declared preprocessing
    return X, y


def config_error(config: FitRankConfig) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        build_candidates(config)
    return info.value


# --- Candidates ------------------------------------------------------------------------------


def test_two_reductions_four_models_give_eight_candidates() -> None:
    candidates = build_candidates(make_config())
    assert [(c.reduction_name, c.model_name) for c in candidates] == [
        ("none", "logistic_regression"), ("none", "random_forest"),
        ("none", "svm"), ("none", "decision_tree"),
        ("pca", "logistic_regression"), ("pca", "random_forest"),
        ("pca", "svm"), ("pca", "decision_tree"),
    ]  # fmt: skip


def test_grid_keys_are_prefixed_and_merged() -> None:
    candidate = build_candidates(make_config())[5]  # pca + random_forest
    assert candidate.param_grid == {
        "reduction__n_components": [2, 3],
        "model__n_estimators": [10],
        "model__max_depth": [None, 3],
    }


def test_default_grids_are_used_without_options() -> None:
    config = make_config(reduction=["select_k_best"], models=[{"name": "knn"}])
    (candidate,) = build_candidates(config)
    assert candidate.param_grid == {
        "reduction__k": [5, 10],
        "model__n_neighbors": [3, 5, 11],
        "model__weights": ["uniform", "distance"],
    }


def test_every_grid_key_is_a_pipeline_parameter() -> None:
    for candidate in build_candidates(make_config()):
        assert set(candidate.param_grid) <= set(candidate.pipeline.get_params())


def test_every_candidate_fits_and_predicts_on_mixed_data() -> None:
    config = make_config()
    X, y = titanic_xy(config)
    for candidate in build_candidates(config):
        candidate.pipeline.fit(X, y)
        assert len(candidate.pipeline.predict(X)) == len(y)


@pytest.mark.parametrize("reduction", ["none", "pca", "select_k_best", "variance_threshold",
                                       "polynomial_features"])  # fmt: skip
def test_every_reduction_runs_a_grid_search(reduction: str) -> None:
    config = make_config(reduction=[reduction], models=[{"name": "logistic_regression"}])
    X, y = titanic_xy(config)
    (candidate,) = build_candidates(config)
    search = GridSearchCV(candidate.pipeline, candidate.param_grid, cv=2, error_score="raise")
    search.fit(X, y)
    assert search.best_params_.keys() == candidate.param_grid.keys()


def test_regression_candidates_fit() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    df["y"] = 2 * df["a"] - df["b"]
    config = FitRankConfig.model_validate({
        "data": {"path": "unused.csv", "target": "y"},
        "task": "regression",
        "reduction": ["none", "select_k_best"],
        "models": [{"name": "ridge"}, {"name": "polynomial_regression"}],
    })  # fmt: skip
    X, y = split_features_target(df, config.data)
    candidates = build_candidates(config)
    assert len(candidates) == 4
    for candidate in candidates:
        assert len(candidate.pipeline.fit(X, y).predict(X)) == 30


def test_random_state_reaches_the_model_and_the_reduction() -> None:
    config = make_config(project={"random_state": 7})
    candidate = build_candidates(config)[5]  # pca + random_forest
    assert candidate.pipeline.named_steps["model"].random_state == 7
    assert candidate.pipeline.named_steps["reduction"].random_state == 7


def test_select_k_best_scores_by_task() -> None:
    _, classifier, _ = build_reduction("select_k_best", "classification")
    _, regressor, _ = build_reduction("select_k_best", "regression")
    assert classifier.score_func is f_classif
    assert regressor.score_func is f_regression


# --- No data leakage (EF-24) -----------------------------------------------------------------


def test_preprocessing_is_inside_the_pipeline_and_not_fitted() -> None:
    for candidate in build_candidates(make_config()):
        name, preprocessing = candidate.pipeline.steps[0]
        assert name == "preprocessing"
        assert not hasattr(preprocessing, "transformers_")


def test_candidates_do_not_share_steps() -> None:
    first, second = build_candidates(make_config())[4:6]  # pca + two models
    assert first.pipeline.steps[0][1] is not second.pipeline.steps[0][1]
    assert first.pipeline.steps[1][1] is not second.pipeline.steps[1][1]


def test_imputer_learns_from_training_rows_only() -> None:
    config = make_config(models=[{"name": "decision_tree"}], reduction=["none"])
    (candidate,) = build_candidates(config)
    X, y = titanic_xy(config)
    train = X.iloc[:5]
    candidate.pipeline.fit(train, y.iloc[:5])
    imputer = candidate.pipeline.named_steps["preprocessing"].named_transformers_["numeric"][0]
    assert imputer.statistics_[0] == train["Age"].median()
    assert imputer.statistics_[0] != X["Age"].median()


# --- Preprocessing ---------------------------------------------------------------------------


def test_declared_groups_become_a_column_transformer() -> None:
    transformer = build_preprocessing(make_config().preprocessing)
    assert isinstance(transformer, ColumnTransformer)
    assert transformer.remainder == "passthrough"
    numeric = transformer.transformers[0]
    assert numeric[0] == "numeric"
    assert [name for name, _ in numeric[1].steps] == ["imputer", "scaler"]
    assert numeric[1].steps[0][1].strategy == "median"
    assert numeric[2] == ["Age", "Fare"]


def test_undeclared_columns_pass_through_unchanged() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
    config = make_config(preprocessing={"num": {"columns": ["a"], "steps": ["scaler"]}})
    out = build_preprocessing(config.preprocessing).fit_transform(df)
    assert out[:, 1].tolist() == [10, 20, 30]


def test_no_preprocessing_passes_everything_through() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = build_preprocessing({}).fit_transform(df)
    assert out.tolist() == [[1.0, 3.0], [2.0, 4.0]]


@pytest.mark.parametrize(
    ("step", "kind", "attributes"),
    [
        ("imputer", "SimpleImputer", {"strategy": "mean"}),
        ({"imputer": "constant"}, "SimpleImputer", {"strategy": "constant"}),
        ({"imputer": {"strategy": "constant", "fill_value": 0}}, "SimpleImputer",
         {"fill_value": 0}),
        ("scaler", "StandardScaler", {}),
        ({"scaler": "minmax"}, "MinMaxScaler", {}),
        ({"scaler": "robust"}, "RobustScaler", {}),
        ("encoder", "OneHotEncoder", {"handle_unknown": "ignore"}),
        ({"encoder": "ordinal"}, "OrdinalEncoder", {"handle_unknown": "use_encoded_value"}),
    ],
)  # fmt: skip
def test_step_variants(step: Any, kind: str, attributes: dict[str, Any]) -> None:
    config = make_config(preprocessing={"g": {"columns": ["Age"], "steps": [step]}})
    (_, pipeline, _), *_ = build_preprocessing(config.preprocessing).transformers
    transformer = pipeline.steps[0][1]
    assert type(transformer).__name__ == kind
    assert attributes.items() <= transformer.get_params().items()


def test_unseen_category_does_not_crash() -> None:
    config = make_config(preprocessing={"cat": {"columns": ["c"], "steps": ["encoder"]}})
    transformer = build_preprocessing(config.preprocessing)
    transformer.fit(pd.DataFrame({"c": ["a", "b"]}))
    assert transformer.transform(pd.DataFrame({"c": ["z"]})).tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (["normalizer"], "unknown step 'normalizer' (available: encoder, imputer, scaler)"),
        ([{"scaler": "standrad"}], "unknown scaler 'standrad' (available: minmax, robust,"),
        ([{"encoder": "binary"}], "unknown encoder 'binary' (available: onehot, ordinal)"),
        ([{"imputer": "average"}], "unknown imputer strategy 'average' (available: constant,"),
        ([{"imputer": {"strategy": "mean", "fill": 0}}], "unknown option 'fill' (allowed: "),
        ([{"imputer": [1]}], "must be a strategy name or a mapping, e.g. {strategy: median}"),
        (["scaler", {"scaler": "minmax"}], "steps[1]: step 'scaler' is declared twice"),
    ],
)  # fmt: skip
def test_bad_steps_are_explained(steps: list[Any], message: str) -> None:
    config = make_config(preprocessing={"g": {"columns": ["Age"], "steps": steps}})
    (detail,) = config_error(config).details
    assert detail.startswith("preprocessing.g.steps[")
    assert message in detail


def test_registering_a_step_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="'scaler' is already registered"):
        register_step("scaler")(lambda options: None)


def test_column_in_two_groups_is_refused() -> None:
    config = make_config(preprocessing={
        "a": {"columns": ["Age"], "steps": ["scaler"]},
        "b": {"columns": ["Age", "Fare"], "steps": ["imputer"]},
    })  # fmt: skip
    assert config_error(config).details == [
        "preprocessing.b.columns: 'Age' is already in group 'a'; a column belongs to one group only"
    ]


# --- Reductions ------------------------------------------------------------------------------


def test_builtin_reductions() -> None:
    assert list_reductions() == [
        "none", "pca", "polynomial_features", "select_k_best", "variance_threshold",
    ]  # fmt: skip


def test_none_is_passthrough() -> None:
    assert build_reduction("none", "classification") == ("none", "passthrough", {})


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ("lda", "reduction[0]: unknown reduction 'lda' (available: none, pca,"),
        ({"none": {"x": [1]}}, "reduction[0]: none takes no options"),
        ({"pca": 5}, "reduction[0]: pca: options must be a mapping of lists"),
        ({"pca": {"n_componentz": [2]}}, "reduction[0]: pca.n_componentz: unknown option"),
        ({"pca": {"n_components": 2}}, "reduction[0]: pca.n_components: must be a non-empty list"),
        ({"pca": {"n_components": []}}, "reduction[0]: pca.n_components: must be a non-empty list"),
    ],
)  # fmt: skip
def test_bad_reductions_are_explained(step: Any, message: str) -> None:
    (detail,) = config_error(make_config(reduction=[step])).details
    assert detail.startswith(message)


def test_custom_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reduction_module, "REDUCTIONS", dict(reduction_module.REDUCTIONS))

    @register_reduction("pca_whiten")
    def pca_whiten(task: Any) -> tuple[PCA, dict[str, list[Any]]]:
        return PCA(whiten=True), {"n_components": [2]}

    candidates = build_candidates(make_config(reduction=["pca_whiten"]))
    assert {c.reduction_name for c in candidates} == {"pca_whiten"}
    with pytest.raises(ValueError, match="already registered"):
        register_reduction("pca")(pca_whiten)


# --- All problems at once --------------------------------------------------------------------


def test_every_problem_is_reported_at_once() -> None:
    config = make_config(
        preprocessing={"g": {"columns": ["Age"], "steps": ["normalizer"]}},
        reduction=["lda"],
        models=[{"name": "random_forrest"}],
    )
    error = config_error(config)
    assert str(error).startswith("invalid configuration\n")
    assert [detail.split(":")[0] for detail in error.details] == [
        "models[0].name",
        "preprocessing.g.steps[0]",
        "reduction[0]",
    ]
