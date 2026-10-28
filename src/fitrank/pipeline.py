"""Turn a configuration into the candidate pipelines to evaluate.

One candidate per (reduction, model) pair::

    Pipeline([("preprocessing", ColumnTransformer), ("reduction", ...), ("model", ...)])

Preprocessing lives inside the pipeline, so during cross-validation it is
fitted on the training folds only (no data leakage).
"""

from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from fitrank.config import FitRankConfig, check_models
from fitrank.exceptions import ConfigError
from fitrank.preprocessing import build_preprocessing
from fitrank.reduction import build_reduction
from fitrank.registry import Grid, get_model


@dataclass
class Candidate:
    """An unfitted pipeline and the grid to search, with prefixed parameter names."""

    model_name: str
    reduction_name: str
    pipeline: Pipeline
    param_grid: dict[str, list[Any]]


def build_candidates(config: FitRankConfig) -> list[Candidate]:
    """Return one candidate per declared reduction and model, in declaration order.

    Raises:
        ConfigError: every problem found in the models, the preprocessing and
            the reductions, reported at once.
    """
    problems = []
    preprocessing = None
    try:
        check_models(config)
    except ConfigError as exc:
        problems += exc.details
    try:
        preprocessing = build_preprocessing(config.preprocessing)
    except ConfigError as exc:
        problems += exc.details

    random_state = config.project.random_state
    reductions = []
    for i, step in enumerate(config.reduction):
        try:
            reductions.append(build_reduction(step, config.task, random_state))
        except ValueError as exc:
            problems.append(f"reduction[{i}]: {exc}")

    if problems or preprocessing is None:
        raise ConfigError("invalid configuration", problems)

    candidates = []
    for reduction_name, reduction, reduction_grid in reductions:
        for model in config.models:
            estimator, model_grid = get_model(model.name, config.task).build(
                model.params, random_state
            )
            pipeline = Pipeline(
                [
                    ("preprocessing", clone(preprocessing)),
                    ("reduction", reduction if isinstance(reduction, str) else clone(reduction)),
                    ("model", estimator),
                ]
            )
            grid = _prefixed("reduction", reduction_grid) | _prefixed("model", model_grid)
            candidates.append(Candidate(model.name, reduction_name, pipeline, grid))
    return candidates


def _prefixed(step: str, grid: Grid) -> Grid:
    return {f"{step}__{key}": values for key, values in grid.items()}
