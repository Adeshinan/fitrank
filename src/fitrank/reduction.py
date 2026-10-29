"""Registry of dimension reductions, applied between preprocessing and the model.

A reduction declares a transformer and its default grid. In the YAML, options
(``pca: {n_components: [3, 5]}``) replace the default grid, like model params::

    @register_reduction("pca")
    def pca(task):
        return PCA(), {"n_components": [0.95]}
"""

from collections.abc import Callable
from typing import Any

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures

from fitrank.config import step_name, step_options
from fitrank.registry import Grid, Task

# Task -> (transformer, or "passthrough" for no reduction; default grid).
ReductionFactory = Callable[[Task], tuple[Any, Grid]]

REDUCTIONS: dict[str, ReductionFactory] = {}


def register_reduction(name: str) -> Callable[[ReductionFactory], ReductionFactory]:
    """Register a reduction factory under ``name``."""

    def decorator(factory: ReductionFactory) -> ReductionFactory:
        if name in REDUCTIONS:
            raise ValueError(f"reduction {name!r} is already registered")
        REDUCTIONS[name] = factory
        return factory

    return decorator


def list_reductions() -> list[str]:
    """Return the names of the registered reductions, sorted."""
    return sorted(REDUCTIONS)


def build_reduction(
    step: str | dict[str, Any], task: Task, random_state: int | None = None
) -> tuple[str, Any, Grid]:
    """Return the name, a fresh transformer and the grid of a declared reduction.

    Raises:
        ValueError: unknown reduction or option, or an option that is not a list.
    """
    name = step_name(step)
    if name not in REDUCTIONS:
        raise ValueError(f"unknown reduction {name!r} (available: {', '.join(list_reductions())})")
    transformer, grid = REDUCTIONS[name](task)
    options = step_options(step)
    if isinstance(transformer, str):  # "passthrough"
        if options is not None:
            raise ValueError(f"{name} takes no options")
        return name, transformer, {}

    if options is None:
        options = grid
    if not isinstance(options, dict):
        raise ValueError(
            f"{name}: options must be a mapping of lists, e.g. {{n_components: [3, 5]}}"
        )
    known = transformer.get_params()
    for key, values in options.items():
        if key not in known:
            raise ValueError(f"{name}.{key}: unknown option (allowed: {', '.join(sorted(known))})")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}.{key}: must be a non-empty list, e.g. [a, b]")
    if random_state is not None and "random_state" in known:
        transformer.set_params(random_state=random_state)
    return name, transformer, dict(options)


@register_reduction("none")
def none(task: Task) -> tuple[str, Grid]:
    return "passthrough", {}


@register_reduction("pca")
def pca(task: Task) -> tuple[PCA, Grid]:
    # A share of variance rather than a count, so it fits any number of features.
    return PCA(), {"n_components": [0.95]}


@register_reduction("select_k_best")
def select_k_best(task: Task) -> tuple[SelectKBest, Grid]:
    # k above the number of features keeps them all (with a warning).
    score = f_classif if task == "classification" else f_regression
    return SelectKBest(score_func=score), {"k": [5, 10]}


@register_reduction("variance_threshold")
def variance_threshold(task: Task) -> tuple[VarianceThreshold, Grid]:
    return VarianceThreshold(), {"threshold": [0.0, 0.01]}


@register_reduction("polynomial_features")
def polynomial_features(task: Task) -> tuple[PolynomialFeatures, Grid]:
    return PolynomialFeatures(include_bias=False), {"degree": [2]}
