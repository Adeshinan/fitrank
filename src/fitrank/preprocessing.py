"""Preprocessing steps, and the ColumnTransformer built from the declared groups.

Nothing is applied unless declared: columns outside every group are passed
through unchanged. Steps live in a registry, like models::

    @register_step("scaler")
    def scaler(options):
        return choose("scaler", options, {"standard": StandardScaler, ...}, "standard")
"""

from collections.abc import Callable, Mapping
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from fitrank.config import PreprocessingGroup, step_name, step_options
from fitrank.exceptions import ConfigError

# Options written after the step name (None, a variant name or a mapping) -> transformer.
# A factory raises ValueError with a readable message when the options are wrong.
StepFactory = Callable[[Any], Any]

STEPS: dict[str, StepFactory] = {}


def register_step(name: str) -> Callable[[StepFactory], StepFactory]:
    """Register a preprocessing step factory under ``name``."""

    def decorator(factory: StepFactory) -> StepFactory:
        if name in STEPS:
            raise ValueError(f"preprocessing step {name!r} is already registered")
        STEPS[name] = factory
        return factory

    return decorator


def choose(kind: str, options: Any, variants: Mapping[str, Callable[[], Any]], default: str) -> Any:
    """Build the variant named by ``options`` (``None`` means ``default``)."""
    name = default if options is None else options
    if isinstance(name, str) and name in variants:
        return variants[name]()
    raise ValueError(f"unknown {kind} {name!r} (available: {', '.join(sorted(variants))})")


def set_options(estimator: Any, options: Mapping[str, Any]) -> Any:
    """Set fixed options on ``estimator``, refusing names it does not have."""
    known = estimator.get_params()
    for key in options:
        if key not in known:
            raise ValueError(f"unknown option {key!r} (allowed: {', '.join(sorted(known))})")
    return estimator.set_params(**options)


_STRATEGIES = ("constant", "mean", "median", "most_frequent")


@register_step("imputer")
def imputer(options: Any) -> SimpleImputer:
    if options is None or isinstance(options, str):
        options = {} if options is None else {"strategy": options}
    if not isinstance(options, dict):
        raise ValueError("must be a strategy name or a mapping, e.g. {strategy: median}")
    strategy = options.get("strategy", "mean")
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown imputer strategy {strategy!r} (available: {', '.join(_STRATEGIES)})"
        )
    return set_options(SimpleImputer(), options)


@register_step("scaler")
def scaler(options: Any) -> Any:
    variants = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    return choose("scaler", options, variants, "standard")


@register_step("encoder")
def encoder(options: Any) -> Any:
    variants: dict[str, Callable[[], Any]] = {
        # Categories unseen during training (e.g. in a CV fold) must not crash.
        "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
        "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    }
    return choose("encoder", options, variants, "onehot")


def build_preprocessing(groups: Mapping[str, PreprocessingGroup]) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer, one sub-pipeline per declared group.

    Raises:
        ConfigError: unknown step or option, a step repeated in a group, or a
            column declared in two groups.
    """
    problems = []
    transformers = []
    owner: dict[str, str] = {}
    for group, settings in groups.items():
        steps: list[tuple[str, Any]] = []
        for i, step in enumerate(settings.steps):
            where = f"preprocessing.{group}.steps[{i}]"
            name = step_name(step)
            if name not in STEPS:
                problems.append(
                    f"{where}: unknown step {name!r} (available: {', '.join(sorted(STEPS))})"
                )
                continue
            if any(name == done for done, _ in steps):
                problems.append(f"{where}: step {name!r} is declared twice in this group")
                continue
            try:
                steps.append((name, STEPS[name](step_options(step))))
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
        for column in settings.columns:
            if column in owner:
                problems.append(
                    f"preprocessing.{group}.columns: {column!r} is already in group"
                    f" {owner[column]!r}; a column belongs to one group only"
                )
            owner.setdefault(column, group)
        transformers.append((group, Pipeline(steps) if steps else "passthrough", settings.columns))

    if problems:
        raise ConfigError("invalid preprocessing", problems)
    # Dense output, so that every reduction (e.g. PCA) accepts it.
    return ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)
