"""Tests for loading and validating the YAML configuration."""

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from fitrank.config import FitRankConfig, check_models, load_config, step_name
from fitrank.exceptions import ConfigError, FitRankError

MINIMAL = """
data: {path: data.csv, target: y}
task: classification
models: [{name: random_forest}]
"""

# The Titanic example from the specification (section 5), without `exclude`
# and `n_jobs`, which ship in v1.1.
TITANIC = """
project:
  name: titanic-survie
  output_dir: runs/
  random_state: 42

data:
  path: data/titanic.csv
  target: Survived
  features: [Age, Fare, Sex, Pclass, Embarked]

task: classification

preprocessing:
  numeric:
    columns: [Age, Fare]
    steps:
      - imputer: {strategy: median}
      - scaler: standard
  categorical:
    columns: [Sex, Pclass, Embarked]
    steps:
      - imputer: {strategy: most_frequent}
      - encoder: onehot

reduction:
  - none
  - pca: {n_components: [3, 5]}

models:
  - name: logistic_regression
    params: {C: [0.1, 1, 10]}
  - name: random_forest
    params: {n_estimators: [100, 300], max_depth: [null, 10]}
  - name: svm
    params: {C: [1, 10], kernel: [linear, rbf]}
  - name: decision_tree        # default grid

evaluation:
  test_size: 0.2
  cv: 5
  search: grid
  primary_metric: f1_weighted
  metrics: [accuracy, precision_weighted, recall_weighted, roc_auc_ovr]

output:
  formats: [csv, json, markdown]
  save_best_model: true
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(dedent(text), encoding="utf-8")
    return path


def load_error(tmp_path: Path, text: str) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        load_config(write(tmp_path, text))
    return info.value


def with_extra(yaml_text: str) -> str:
    return MINIMAL + yaml_text


# --- valid configurations ---------------------------------------------------


def test_minimal_config_gets_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, MINIMAL))

    assert isinstance(cfg, FitRankConfig)
    assert cfg.project.name == "fitrank"
    assert cfg.project.output_dir == Path("runs")
    assert cfg.project.random_state == 42
    assert cfg.data.path == Path("data.csv")
    assert cfg.data.features == "all"
    assert cfg.preprocessing == {}
    assert cfg.reduction == ["none"]
    assert cfg.models[0].name == "random_forest"
    assert cfg.models[0].params is None
    assert cfg.evaluation.test_size == 0.2
    assert cfg.evaluation.cv == 5
    assert cfg.evaluation.search == "grid"
    assert cfg.evaluation.n_iter == 20
    assert cfg.evaluation.primary_metric is None
    assert cfg.evaluation.metrics == []
    assert cfg.output.formats == ["csv", "json", "markdown"]
    assert cfg.output.save_best_model is True


def test_defaults_are_not_shared_between_configs(tmp_path: Path) -> None:
    first = load_config(write(tmp_path, MINIMAL))
    first.reduction.append("pca")
    first.evaluation.metrics.append("accuracy")

    second = load_config(write(tmp_path, MINIMAL))

    assert second.reduction == ["none"]
    assert second.evaluation.metrics == []


def test_full_titanic_config(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, TITANIC))

    assert cfg.project.name == "titanic-survie"
    assert cfg.data.features == ["Age", "Fare", "Sex", "Pclass", "Embarked"]
    assert list(cfg.preprocessing) == ["numeric", "categorical"]
    assert cfg.preprocessing["numeric"].steps == [
        {"imputer": {"strategy": "median"}},
        {"scaler": "standard"},
    ]
    assert cfg.reduction == ["none", {"pca": {"n_components": [3, 5]}}]
    assert [m.name for m in cfg.models] == [
        "logistic_regression",
        "random_forest",
        "svm",
        "decision_tree",
    ]
    assert cfg.models[1].params == {"n_estimators": [100, 300], "max_depth": [None, 10]}
    assert cfg.models[3].params is None
    assert cfg.evaluation.primary_metric == "f1_weighted"
    assert cfg.output.save_best_model is True


def test_regression_task_and_random_search(tmp_path: Path) -> None:
    cfg = load_config(
        write(
            tmp_path,
            """
            data: {path: d.csv, target: price}
            task: regression
            models: [{name: ridge}]
            evaluation: {search: random, n_iter: 50}
            """,
        )
    )

    assert cfg.task == "regression"
    assert cfg.evaluation.search == "random"
    assert cfg.evaluation.n_iter == 50


def test_accepts_str_path(tmp_path: Path) -> None:
    assert load_config(str(write(tmp_path, MINIMAL))).task == "classification"


# --- file and YAML problems -------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError, match="config file not found") as info:
        load_config(missing)

    assert str(missing) in str(info.value)


def test_unreadable_path_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path)  # a directory


def test_malformed_yaml_reports_line(tmp_path: Path) -> None:
    error = load_error(tmp_path, "data: {path: data.csv\ntask: classification\n")

    assert str(error).startswith("invalid YAML in")
    assert "line " in error.details[0]


def test_unsafe_yaml_tag_is_refused(tmp_path: Path) -> None:
    error = load_error(tmp_path, "data: !!python/object/apply:os.system ['echo hacked']\n")

    assert str(error).startswith("invalid YAML in")


def test_empty_file(tmp_path: Path) -> None:
    assert load_error(tmp_path, "").details == ["the file is empty"]


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    error = load_error(tmp_path, "- data\n- task\n")

    assert "top level must be a mapping" in error.details[0]


# --- schema problems --------------------------------------------------------


def test_bad_test_size_is_explained(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("evaluation: {test_size: 1.5}"))

    assert str(error) == (
        f"invalid configuration in {tmp_path / 'config.yaml'}\n"
        "  -> evaluation.test_size: must be between 0 and 1 (got 1.5)"
    )


def test_unknown_top_level_key_lists_allowed_keys(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("evalution: {cv: 3}"))

    assert error.details == [
        "evalution: unknown key (allowed: project, data, task, preprocessing, "
        "reduction, models, evaluation, output)"
    ]


def test_unknown_nested_key_lists_section_keys(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("evaluation: {folds: 3}"))

    assert error.details == [
        "evaluation.folds: unknown key "
        "(allowed: test_size, cv, search, n_iter, primary_metric, metrics)"
    ]


def test_unknown_key_in_a_model(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        """
        data: {path: d.csv, target: y}
        task: classification
        models: [{name: svm, grid: {C: [1]}}]
        """,
    )

    assert error.details == ["models[0].grid: unknown key (allowed: name, params)"]


def test_unknown_key_in_a_preprocessing_group(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        with_extra(
            "preprocessing: {numeric: {columns: [a], steps: [scaler: standard], cols: [b]}}"
        ),
    )

    assert error.details == ["preprocessing.numeric.cols: unknown key (allowed: columns, steps)"]


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ("project: {n_jobs: -1}", "project.n_jobs"),
        (None, "data.exclude"),
    ],
)
def test_keys_planned_for_v1_1_are_refused(tmp_path: Path, extra: str | None, key: str) -> None:
    text = (
        with_extra(extra)
        if extra
        else """
        data: {path: d.csv, target: y, exclude: [id]}
        task: classification
        models: [{name: svm}]
        """
    )

    assert load_error(tmp_path, text).details == [
        f"{key}: not supported yet (planned for v1.1), remove this key"
    ]


def test_invalid_task(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        """
        data: {path: d.csv, target: y}
        task: clasification
        models: [{name: svm}]
        """,
    )

    assert error.details == ["task: must be 'classification' or 'regression' (got 'clasification')"]


def test_empty_models(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        """
        data: {path: d.csv, target: y}
        task: classification
        models: []
        """,
    )

    assert error.details == ["models: must contain at least 1 item(s)"]


def test_missing_required_sections(tmp_path: Path) -> None:
    error = load_error(tmp_path, "task: regression\n")

    assert error.details == [
        "data: required key is missing",
        "models: required key is missing",
    ]


def test_all_problems_are_reported_at_once(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("evaluation: {cv: 1, search: bayes}"))

    assert error.details == [
        "evaluation.cv: must be at least 2 (got 1)",
        "evaluation.search: must be 'grid' or 'random' (got 'bayes')",
    ]


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ("evaluation: {test_size: 0}", "evaluation.test_size: must be between 0 and 1 (got 0)"),
        ("evaluation: {test_size: big}", "evaluation.test_size: must be a number (got 'big')"),
        ("evaluation: {cv: 2.5}", "evaluation.cv: must be a whole number (got 2.5)"),
        ("evaluation: {n_iter: 0}", "evaluation.n_iter: must be at least 1 (got 0)"),
        (
            "evaluation: {metrics: accuracy}",
            "evaluation.metrics: must be a list, e.g. [a, b] (got 'accuracy')",
        ),
        ("evaluation: 5", "evaluation: must be a mapping, e.g. {key: value} (got 5)"),
        (
            "output: {save_best_model: maybe}",
            "output.save_best_model: must be true or false (got 'maybe')",
        ),
        (
            "output: {formats: [csv, pdf]}",
            "output.formats[1]: must be 'csv', 'json' or 'markdown' (got 'pdf')",
        ),
        ("project: {name: [a]}", "project.name: must be text"),
        ("project: {name: true}", "project.name: must be text (got true)"),
        ("project: null", "project: must be a mapping, e.g. {key: value} (got null)"),
        ("reduction: []", "reduction: must contain at least 1 item(s)"),
        ("preprocessing: [numeric]", "preprocessing: must be a mapping, e.g. {key: value}"),
    ],
)
def test_type_and_range_errors(tmp_path: Path, extra: str, detail: str) -> None:
    assert load_error(tmp_path, with_extra(extra)).details == [detail]


def test_features_must_be_all_or_a_list_of_names(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        """
        data: {path: d.csv, target: y, features: some}
        task: classification
        models: [{name: svm}]
        """,
    )

    assert error.details == ["data.features: must be 'all' or a list of column names (got 'some')"]


def test_reduction_item_must_be_a_name_or_one_mapping(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("reduction: [none, {pca: {}, kbest: {}}]"))

    assert len(error.details) == 1
    assert error.details[0].startswith(
        "reduction[1]: must be a name, or a single 'name: options' mapping"
    )


def test_preprocessing_step_must_be_a_name_or_one_mapping(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("preprocessing: {num: {columns: [a], steps: [42]}}"))

    assert len(error.details) == 1
    assert error.details[0].startswith("preprocessing.num.steps[0]: must be a name")
    assert error.details[0].endswith("(got 42)")


def test_preprocessing_group_needs_columns_and_steps(tmp_path: Path) -> None:
    error = load_error(tmp_path, with_extra("preprocessing: {num: {columns: []}}"))

    assert error.details == [
        "preprocessing.num.columns: must contain at least 1 item(s)",
        "preprocessing.num.steps: required key is missing",
    ]


def test_model_params_values_must_be_lists(tmp_path: Path) -> None:
    error = load_error(
        tmp_path,
        """
        data: {path: d.csv, target: y}
        task: classification
        models: [{name: svm, params: {C: 1}}]
        """,
    )

    assert error.details == ["models[0].params.C: must be a list, e.g. [a, b] (got 1)"]


def test_error_message_has_no_pydantic_noise(tmp_path: Path) -> None:
    message = str(load_error(tmp_path, with_extra("evaluation: {test_size: 1.5}")))

    assert "pydantic" not in message
    assert "Value error" not in message
    assert "errors.pydantic.dev" not in message


def test_config_error_is_a_fitrank_error() -> None:
    assert issubclass(ConfigError, FitRankError)
    assert ConfigError("bad", ["a", "b"]).details == ["a", "b"]
    assert str(ConfigError("bad")) == "bad"


# --- Models and task (checked against the model registry) ------------------------------------

CLASSIFIERS = (
    "decision_tree, gradient_boosting, knn, logistic_regression, naive_bayes, random_forest, svm"
)
REGRESSORS = (
    "decision_tree, gradient_boosting, knn, lasso, linear_regression, polynomial_regression,"
    " random_forest, ridge, svr"
)


def models_config(task: str, *models: str | dict[str, Any]) -> FitRankConfig:
    return FitRankConfig.model_validate({
        "data": {"path": "data.csv", "target": "y"},
        "task": task,
        "models": [{"name": m} if isinstance(m, str) else m for m in models],
    })  # fmt: skip


def test_models_available_for_the_task_pass() -> None:
    check_models(models_config("regression", "ridge", "random_forest", "polynomial_regression"))


def test_model_of_the_other_task_is_refused() -> None:
    with pytest.raises(ConfigError) as info:
        check_models(models_config("regression", "ridge", "logistic_regression"))
    assert str(info.value) == (
        "some models cannot be used for regression\n"
        "  -> models[1].name: 'logistic_regression' is a classification model,"
        f" but task is 'regression' (available for regression: {REGRESSORS})"
    )


def test_unknown_model_is_refused_with_a_suggestion() -> None:
    with pytest.raises(ConfigError) as info:
        check_models(models_config("classification", "random_forrest", "svm"))
    assert info.value.details == [
        "models[0].name: unknown model 'random_forrest'"
        f" (did you mean 'random_forest'? available for classification: {CLASSIFIERS})"
    ]


def test_valid_params_pass() -> None:
    params = {"poly__degree": [2], "linear__fit_intercept": [True]}
    check_models(models_config("regression", {"name": "polynomial_regression", "params": params}))


def test_unknown_param_is_refused_with_a_suggestion() -> None:
    model = {"name": "random_forest", "params": {"n_estimator": [100], "max_depth": [5]}}
    with pytest.raises(ConfigError) as info:
        check_models(models_config("classification", model))
    (detail,) = info.value.details
    assert detail.startswith(
        "models[0].params.n_estimator: unknown parameter for random_forest"
        " (did you mean 'n_estimators'? allowed: bootstrap, ccp_alpha,"
    )


def test_step_name() -> None:
    assert step_name("none") == "none"
    assert step_name({"pca": {"n_components": [5]}}) == "pca"


def test_hint_is_indented_under_the_details() -> None:
    error = ConfigError("bad", ["a"], hint="first\nsecond")
    assert str(error) == "bad\n  -> a\n  Hint: first\n        second"
