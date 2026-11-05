"""Registry of the models FitRank can train.

Each model is declared once, with its task and its default grid, through
``@register_model``. The engine only talks to the registry, so adding a model
never touches the engine::

    @register_model("extra_trees", task="classification")
    def extra_trees():
        return ExtraTreesClassifier(), {"n_estimators": [100, 300]}

Any scikit-learn estimator can also be used by its class path, without
registering it: see ``import_model``.
"""

import difflib
import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from sklearn.base import BaseEstimator, is_classifier, is_regressor

from fitrank.exceptions import ConfigError

Task = Literal["classification", "regression"]
TASKS: tuple[Task, ...] = get_args(Task)

Grid = dict[str, list[Any]]
Factory = Callable[[], tuple[BaseEstimator, Grid]]

# Packages whose classes a config may import without being allowed explicitly.
TRUSTED_MODULES = ("sklearn",)
MODULE_PATH = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


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


def import_model(path: str, task: Task, allowed_modules: Sequence[str] = ()) -> ModelSpec:
    """Return a model for the estimator class at ``path``, e.g. ``sklearn.svm.LinearSVC``.

    The path is checked before anything is imported: it must be inside
    ``sklearn`` or one of ``allowed_modules``, so a config can never run
    ``os.system`` or any other function. The model's default grid is empty:
    the estimator is cross-validated with its default parameters.

    Raises:
        ValueError: the path is not allowed, cannot be imported, is not an
            estimator for ``task``, or the class needs constructor arguments.
            The message does not say where the path comes from.
    """
    if "." not in path or not MODULE_PATH.fullmatch(path):
        raise ValueError(
            f"{path!r} is not a class path; write it in full,"
            " e.g. sklearn.ensemble.ExtraTreesClassifier"
        )
    prefixes = (*TRUSTED_MODULES, *allowed_modules)
    if not any(path.startswith(f"{prefix}.") for prefix in prefixes):
        raise ValueError(
            f"{path!r} is outside the allowed modules ({', '.join(prefixes)});"
            " if you trust its package, add it to project.allowed_modules"
        )
    module_name, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # any error raised while importing a third-party module
        raise ValueError(
            f"cannot import module {module_name!r} ({type(exc).__name__}: {exc});"
            " check the spelling, or install the package"
        ) from None
    cls = getattr(module, class_name, None)
    if cls is None:
        names = [name for name in dir(module) if not name.startswith("_")]
        close = difflib.get_close_matches(class_name, names, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"module {module_name!r} has no class {class_name!r}{hint}")
    if not (isinstance(cls, type) and issubclass(cls, BaseEstimator)):
        raise ValueError(f"{path!r} is not a scikit-learn estimator class")
    try:
        estimator = cls()
    except Exception as exc:  # a constructor with required arguments
        raise ValueError(f"{class_name}() cannot be created without arguments ({exc})") from None
    expected = is_classifier if task == "classification" else is_regressor
    if not expected(estimator):
        kind = "classifier" if task == "classification" else "regressor"
        raise ValueError(f"{class_name} is not a {kind}, but task is {task!r}")
    return ModelSpec(class_name, task, lambda: (cls(), {}))


def _load_builtins() -> None:
    import fitrank.models  # noqa: F401  (registers the built-in models on first import)
