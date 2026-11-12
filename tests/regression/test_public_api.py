"""The Python API of FitRank 1.1 still works for code written against it.

Adding a name, a keyword parameter with a default, a model or a config key is
fine. Removing or renaming one, turning it from keyword to positional, or
changing a default fails here: that would break someone's script. If such a
change is really intended, it needs a new major version and a CHANGELOG entry.
"""

import inspect
from typing import Any

import pytest

import fitrank
from fitrank import Benchmark, Results
from fitrank.config import (
    DataConfig,
    EvaluationConfig,
    FitRankConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
)
from fitrank.evaluation import primary_metric
from fitrank.preprocessing import STEPS
from fitrank.reduction import list_reductions
from fitrank.registry import list_models

REQUIRED = inspect.Parameter.empty
POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD = inspect.Parameter.KEYWORD_ONLY

V1_1_EXPORTS = {
    "Benchmark",
    "ConfigError",
    "DataError",
    "FitRankError",
    "FitRankWarning",
    "Results",
    "__version__",
    "load_model",
    "register_model",
    "register_reduction",
}

# callable -> {parameter: (kind, default)}, recorded from FitRank 1.1.2.
V1_1_SIGNATURES: dict[str, tuple[Any, dict[str, tuple[Any, Any]]]] = {
    "Benchmark()": (
        Benchmark.__init__,
        {
            "data": (POSITIONAL, REQUIRED),
            "target": (POSITIONAL, REQUIRED),
            "task": (POSITIONAL, REQUIRED),
            "models": (POSITIONAL, REQUIRED),
            "features": (KEYWORD, "all"),
            "exclude": (KEYWORD, ()),
            "preprocessing": (KEYWORD, None),
            "reduction": (KEYWORD, None),
            "evaluation": (KEYWORD, None),
            "output": (KEYWORD, None),
            "project": (KEYWORD, None),
        },
    ),
    "Benchmark.from_yaml": (Benchmark.from_yaml, {"path": (POSITIONAL, REQUIRED)}),
    "Benchmark.from_config": (
        Benchmark.from_config,
        {"config": (POSITIONAL, REQUIRED), "data": (POSITIONAL, None)},
    ),
    "Benchmark.validate": (Benchmark.validate, {}),
    "Benchmark.run": (Benchmark.run, {"export": (KEYWORD, True), "on_result": (KEYWORD, None)}),
    "Results.leaderboard": (Results.leaderboard, {}),
    "Results.best_model": (Results.best_model, {}),
    "Results.export": (
        Results.export,
        {"output_dir": (POSITIONAL, None), "formats": (POSITIONAL, None)},
    ),
    "load_model": (fitrank.load_model, {"path": (POSITIONAL, REQUIRED)}),
    "register_model": (
        fitrank.register_model,
        {"name": (POSITIONAL, REQUIRED), "task": (POSITIONAL, REQUIRED)},
    ),
    "register_reduction": (fitrank.register_reduction, {"name": (POSITIONAL, REQUIRED)}),
    "FitRankError()": (
        fitrank.FitRankError.__init__,
        {
            "message": (POSITIONAL, REQUIRED),
            "details": (POSITIONAL, ()),
            "hint": (POSITIONAL, None),
        },
    ),
}

# YAML key -> default, per config section (required keys: REQUIRED).
V1_1_CONFIG_DEFAULTS: dict[type, dict[str, Any]] = {
    ProjectConfig: {
        "name": "fitrank",
        "output_dir": "runs",
        "random_state": 42,
        "n_jobs": 1,
        "allowed_modules": [],
    },
    DataConfig: {"path": REQUIRED, "target": REQUIRED, "features": "all", "exclude": []},
    ModelConfig: {"name": None, "class": None, "params": None},
    EvaluationConfig: {
        "test_size": 0.2,
        "cv": 5,
        "search": "grid",
        "n_iter": 20,
        "primary_metric": None,
        "metrics": [],
    },
    OutputConfig: {"formats": ["csv", "json", "markdown"], "save_best_model": True},
}
V1_1_SECTIONS = [
    "project",
    "data",
    "task",
    "preprocessing",
    "reduction",
    "models",
    "evaluation",
    "output",
]

V1_1_MODELS = {
    "classification": [
        "decision_tree",
        "gradient_boosting",
        "knn",
        "logistic_regression",
        "naive_bayes",
        "random_forest",
        "svm",
    ],
    "regression": [
        "decision_tree",
        "gradient_boosting",
        "knn",
        "lasso",
        "linear_regression",
        "polynomial_regression",
        "random_forest",
        "ridge",
        "svr",
    ],
}
V1_1_REDUCTIONS = ["none", "pca", "polynomial_features", "select_k_best", "variance_threshold"]
V1_1_STEPS = ["imputer", "scaler", "encoder"]


def test_every_v1_1_name_is_still_exported() -> None:
    missing = V1_1_EXPORTS - set(fitrank.__all__)
    assert not missing, f"no longer exported by fitrank: {sorted(missing)}"
    for name in V1_1_EXPORTS:
        assert hasattr(fitrank, name), f"fitrank.{name} is in __all__ but does not exist"


@pytest.mark.parametrize("name", sorted(V1_1_SIGNATURES))
def test_signature_is_backward_compatible(name: str) -> None:
    function, expected = V1_1_SIGNATURES[name]
    params = {p: v for p, v in inspect.signature(function).parameters.items() if p != "self"}
    for param, (kind, default) in expected.items():
        assert param in params, f"{name}: parameter {param!r} was removed or renamed"
        got = params[param]
        assert got.kind == kind, f"{name}: {param!r} is now {got.kind.description}"
        assert got.default == default, (
            f"{name}: the default of {param!r} changed from {default!r} to {got.default!r}"
        )
    for param, got in params.items():
        if param not in expected and got.kind not in (got.VAR_POSITIONAL, got.VAR_KEYWORD):
            assert got.default is not REQUIRED, (
                f"{name}: new parameter {param!r} has no default, so v1.1 calls would break"
            )


def test_exceptions_keep_their_hierarchy() -> None:
    assert issubclass(fitrank.ConfigError, fitrank.FitRankError)
    assert issubclass(fitrank.DataError, fitrank.FitRankError)
    assert issubclass(fitrank.FitRankError, Exception)
    assert issubclass(fitrank.FitRankWarning, UserWarning)


@pytest.mark.parametrize("section", list(V1_1_CONFIG_DEFAULTS), ids=lambda s: s.__name__)
def test_config_keys_and_defaults_are_unchanged(section: type) -> None:
    fields = {field.alias or name: field for name, field in section.model_fields.items()}
    for key, default in V1_1_CONFIG_DEFAULTS[section].items():
        assert key in fields, f"{section.__name__}: YAML key {key!r} was removed or renamed"
        field = fields[key]
        if default is REQUIRED:
            assert field.is_required(), f"{section.__name__}: {key!r} is no longer required"
            continue
        got = field.get_default(call_default_factory=True)
        got = str(got) if key == "output_dir" else got  # a Path
        assert got == default, (
            f"{section.__name__}: the default of {key!r} changed from {default!r} to {got!r}"
        )
    for key, field in fields.items():
        if key not in V1_1_CONFIG_DEFAULTS[section]:
            assert not field.is_required(), (
                f"{section.__name__}: new key {key!r} is required, so v1.1 configs would fail"
            )


def test_config_sections_are_unchanged() -> None:
    missing = set(V1_1_SECTIONS) - set(FitRankConfig.model_fields)
    assert not missing, f"config sections removed: {sorted(missing)}"


@pytest.mark.parametrize(
    ("task", "metric"),
    [("classification", "f1_weighted"), ("regression", "neg_root_mean_squared_error")],
)
def test_default_primary_metric_is_unchanged(task: str, metric: str) -> None:
    config = FitRankConfig.model_validate(
        {"data": {"path": "x.csv", "target": "y"}, "task": task, "models": [{"name": "knn"}]}
    )
    assert primary_metric(config) == metric


@pytest.mark.parametrize("task", sorted(V1_1_MODELS))
def test_every_v1_1_model_is_still_registered(task: str) -> None:
    missing = set(V1_1_MODELS[task]) - set(list_models(task))
    assert not missing, f"{task} models removed: {sorted(missing)}"


def test_every_v1_1_reduction_and_step_is_still_registered() -> None:
    assert not set(V1_1_REDUCTIONS) - set(list_reductions()), "a reduction was removed"
    assert not set(V1_1_STEPS) - set(STEPS), "a preprocessing step was removed"
