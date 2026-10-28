"""Registry of the models FitRank can train.

Each model is declared once, with its task and its default grid, through
``@register_model``. The engine only talks to the registry, so adding a model
never touches the engine::

    @register_model("extra_trees", task="classification")
    def extra_trees():
        return ExtraTreesClassifier(), {"n_estimators": [100, 300]}
"""

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from sklearn.base import BaseEstimator

from fitrank.exceptions import ConfigError

Task = Literal["classification", "regression"]
TASKS: tuple[Task, ...] = get_args(Task)

Grid = dict[str, list[Any]]
Factory = Callable[[], tuple[BaseEstimator, Grid]]


@dataclass(frozen=True)
class ModelSpec:
    """A registered model: ``factory()`` returns a fresh estimator and its default grid."""

    name: str
    task: Task
    factory: Factory

    def build(
        self, params: Mapping[str, list[Any]] | None = None, random_state: int | None = None
    ) -> tuple[BaseEstimator, Grid]:
        """Return a fresh estimator and the grid to search.

        ``params`` (from the YAML) replaces the default grid, it is never merged
        with it. ``random_state`` is set on estimators that accept one.
        """
        estimator, grid = self.factory()
        if random_state is not None and "random_state" in estimator.get_params():
            estimator.set_params(random_state=random_state)
        return estimator, dict(grid if params is None else params)


MODELS: dict[tuple[str, Task], ModelSpec] = {}


def register_model(name: str, task: Task) -> Callable[[Factory], Factory]:
    """Register a model factory under ``(name, task)``.

    Raises:
        ValueError: a model with this name is already registered for this task.
    """

    def decorator(factory: Factory) -> Factory:
        if (name, task) in MODELS:
            raise ValueError(f"model {name!r} is already registered for {task}")
        MODELS[(name, task)] = ModelSpec(name, task, factory)
        return factory

    return decorator


def get_model(name: str, task: Task) -> ModelSpec:
    """Return the model registered under ``(name, task)``.

    Raises:
        ConfigError: no such model for this task; the message lists the
            available ones.
    """
    _load_builtins()
    spec = MODELS.get((name, task))
    if spec is None:
        available = list_models(task)
        details = [f"available: {', '.join(available) or 'none'}"]
        close = difflib.get_close_matches(name, available, n=1)
        if close:
            details.append(f"did you mean {close[0]!r}?")
        raise ConfigError(f"unknown model {name!r} for task {task!r}", details)
    return spec


def list_models(task: Task) -> list[str]:
    """Return the names of the models registered for ``task``, sorted."""
    _load_builtins()
    return sorted(name for name, model_task in MODELS if model_task == task)


def _load_builtins() -> None:
    import fitrank.models  # noqa: F401  (registers the built-in models on first import)
