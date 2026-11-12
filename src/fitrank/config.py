"""Load and validate a FitRank YAML configuration.

Only the shape of the configuration is checked here. Model, reduction,
preprocessing step and metric names are checked against their registries
when the pipelines are built.
"""

import difflib
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails
from sklearn.metrics import get_scorer_names

from fitrank.exceptions import ConfigError
from fitrank.registry import (
    MODULE_PATH,
    TASKS,
    ModelSpec,
    Task,
    get_model,
    import_model,
    list_models,
)

Loc = tuple[int | str, ...]

# Keys from the specification whose feature ships in a later release.
_PLANNED_KEYS: dict[Loc, str] = {}

_MESSAGES = {
    "greater_than": "must be greater than {gt}",
    "greater_than_equal": "must be at least {ge}",
    "less_than": "must be less than {lt}",
    "less_than_equal": "must be at most {le}",
    "literal_error": "must be {expected}",
    "too_short": "must contain at least {min_length} item(s)",
    "int_type": "must be a whole number",
    "int_parsing": "must be a whole number",
    "int_from_float": "must be a whole number",
    "float_type": "must be a number",
    "float_parsing": "must be a number",
    "bool_type": "must be true or false",
    "bool_parsing": "must be true or false",
    "string_type": "must be text",
    "list_type": "must be a list, e.g. [a, b]",
    "dict_type": "must be a mapping, e.g. {{key: value}}",
    "model_type": "must be a mapping, e.g. {{key: value}}",
    "path_type": "must be a file path",
}


def _check_step(value: Any) -> Any:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and len(value) == 1:
        (name,) = value
        if isinstance(name, str) and name:
            return value
    raise ValueError(
        "must be a name, or a single 'name: options' mapping (e.g. pca: {n_components: [5, 10]})"
    )


# A preprocessing step or a reduction: `none`, `scaler: standard`, `pca: {n_components: [5]}`.
Step = Annotated[str | dict[str, Any], BeforeValidator(_check_step)]


def _check_metric(name: str) -> str:
    names = get_scorer_names()
    if name in names:
        return name
    close = difflib.get_close_matches(name, names, n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    raise ValueError(f"unknown metric{hint}; see sklearn.metrics.get_scorer_names()")


# A scikit-learn scorer name, e.g. f1_weighted or neg_root_mean_squared_error.
Metric = Annotated[str, AfterValidator(_check_metric)]


def step_name(step: str | dict[str, Any]) -> str:
    """Return the name of a step: ``imputer`` for ``{imputer: {strategy: median}}``."""
    return step if isinstance(step, str) else next(iter(step))


def step_options(step: str | dict[str, Any]) -> Any:
    """Return what follows the name: ``None`` for ``scaler``, ``robust`` for ``scaler: robust``."""
    return None if isinstance(step, str) else next(iter(step.values()))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Section):
    name: str = "fitrank"
    output_dir: Path = Path("runs")
    random_state: int = 42
    n_jobs: int = 1  # CPU cores for the search; -1 = all of them
    allowed_modules: list[str] = []  # packages models.class may import, besides sklearn

    @field_validator("n_jobs")
    @classmethod
    def _check_n_jobs(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("must be a positive number of cores, or -1 for all of them")
        return value

    @field_validator("allowed_modules")
    @classmethod
    def _check_allowed_modules(cls, value: list[str]) -> list[str]:
        for module in value:
            if not MODULE_PATH.fullmatch(module):
                raise ValueError(
                    f"{module!r} is not a module name, e.g. xgboost or my_package.models"
                )
        return value


class DataConfig(_Section):
    path: Path
    target: str
    features: list[str] | Literal["all"] = "all"
    exclude: list[str] = []  # with features: all, the columns to leave out

    @field_validator("features", mode="before")
    @classmethod
    def _check_features(cls, value: Any) -> Any:
        if value == "all" or (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            return value
        raise ValueError("must be 'all' or a list of column names")

    @model_validator(mode="after")
    def _check_exclude(self) -> "DataConfig":
        if self.exclude and self.features != "all":
            raise ValueError(
                "use features or exclude, not both: list the columns to keep in features,"
                " or set features: all and list the columns to leave out in exclude"
            )
        if self.target in self.exclude:
            raise ValueError(f"exclude contains the target {self.target!r}: remove it from exclude")
        return self


class PreprocessingGroup(_Section):
    columns: list[str] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)


class ModelConfig(_Section):
    name: str | None = None  # a FitRank model, e.g. random_forest
    class_: str | None = Field(default=None, alias="class")  # or a class path
    params: dict[str, list[Any]] | None = None  # None = the model's default grid

    @model_validator(mode="after")
    def _check_name_or_class(self) -> "ModelConfig":
        if self.name is not None and self.class_ is not None:
            raise ValueError("give name or class, not both")
        if self.name is None and self.class_ is None:
            raise ValueError(
                "give a model name (e.g. name: random_forest) or a class path"
                " (e.g. class: sklearn.ensemble.ExtraTreesClassifier)"
            )
        return self

    @property
    def label(self) -> str:
        """The name shown in the leaderboard: ``ExtraTreesClassifier`` for a class path."""
        return self.name or str(self.class_).rsplit(".", 1)[-1]


class EvaluationConfig(_Section):
    test_size: float = 0.2
    test_data: Path | None = None  # your own test set: no split, all of data.path trains
    cv: int = Field(default=5, ge=2)
    search: Literal["grid", "random"] = "grid"
    n_iter: int = Field(default=20, ge=1)
    primary_metric: Metric | None = None  # None = default for the task
    metrics: list[Metric] = []  # empty = suggested metrics for the task

    @field_validator("test_size")
    @classmethod
    def _check_test_size(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _check_test_choice(self) -> "EvaluationConfig":
        if self.test_data is not None and "test_size" in self.model_fields_set:
            raise ValueError(
                "choose test_size or test_data, not both: test_size splits data.path,"
                " test_data is a separate test set; remove one of the two"
            )
        return self


class OutputConfig(_Section):
    formats: list[Literal["csv", "json", "markdown"]] = ["csv", "json", "markdown"]
    save_best_model: bool = True


class FitRankConfig(_Section):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig
    task: Task
    preprocessing: dict[str, PreprocessingGroup] = {}
    reduction: list[Step] = Field(default=["none"], min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    evaluation: EvaluationConfig = EvaluationConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> FitRankConfig:
    """Read a YAML file and return the validated configuration.

    Raises:
        ConfigError: the file is missing, is not valid YAML, or does not
            match the expected schema.
    """
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"config file not found: {source}",
            ["check the path, or run the command from the folder that contains the file"],
        ) from None
    except OSError as exc:
        raise ConfigError(
            f"cannot read config file: {source}", [exc.strerror or str(exc)]
        ) from None

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}", [_describe_yaml_error(exc)]) from None

    if raw is None:
        raise ConfigError(f"invalid configuration in {source}", ["the file is empty"])
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid configuration in {source}",
            [f"the top level must be a mapping of sections ({_allowed_keys(FitRankConfig)})"],
        )
    return validate_config(raw, source)


def validate_config(raw: dict[str, Any], source: str) -> FitRankConfig:
    """Validate a configuration given as a dict; ``source`` names it in errors.

    Raises:
        ConfigError: the dict does not match the expected schema.
    """
    try:
        return FitRankConfig.model_validate(raw)
    except ValidationError as exc:
        details = [_describe(error) for error in exc.errors(include_url=False)]
        raise ConfigError(f"invalid configuration in {source}", details) from None


def check_models(config: FitRankConfig) -> None:
    """Check every declared model against the model registry, or import its class.

    Raises:
        ConfigError: a model is unknown, only exists for the other task, has
            a class path that is not allowed or not an estimator for the task,
            ``params`` names a parameter the model does not have, or sets
            ``n_jobs`` while ``project.n_jobs`` already runs in parallel.
    """
    allowed = list_models(config.task)
    choices = f"available for {config.task}: {', '.join(allowed) or 'none'}"
    problems = []
    for i, model in enumerate(config.models):
        if model.class_ is not None:
            try:
                spec = model_spec(model, config)
            except ValueError as exc:
                problems.append(f"models[{i}].class: {exc}")
                continue
        elif model.name not in allowed:
            other = [task for task in TASKS if model.name in list_models(task)]
            if other:
                problems.append(
                    f"models[{i}].name: {model.name!r} is a {other[0]} model,"
                    f" but task is {config.task!r} ({choices})"
                )
            else:
                close = difflib.get_close_matches(model.label, allowed, n=1)
                hint = f"did you mean {close[0]!r}? " if close else ""
                problems.append(f"models[{i}].name: unknown model {model.name!r} ({hint}{choices})")
            continue
        else:
            spec = model_spec(model, config)
        if model.params:
            estimator, _ = spec.factory()
            known = sorted(estimator.get_params())
            if "n_jobs" in model.params and config.project.n_jobs != 1:
                # Cores used twice: each search worker would start its own workers.
                problems.append(
                    f"models[{i}].params.n_jobs: project.n_jobs already runs the search"
                    " in parallel; remove one of the two"
                )
            for key in model.params:
                if key not in known:
                    close = difflib.get_close_matches(key, known, n=1)
                    hint = f"did you mean {close[0]!r}? " if close else ""
                    problems.append(
                        f"models[{i}].params.{key}: unknown parameter for {model.label}"
                        f" ({hint}allowed: {', '.join(known)})"
                    )
    if problems:
        raise ConfigError(f"some models cannot be used for {config.task}", problems)


def model_spec(model: ModelConfig, config: FitRankConfig) -> ModelSpec:
    """The registered model named by ``model``, or the one imported from its class path.

    Raises:
        ConfigError: no model with this name for the task.
        ValueError: the class path cannot be used (see ``import_model``).
    """
    if model.class_ is not None:
        return import_model(model.class_, config.task, config.project.allowed_modules)
    return get_model(str(model.name), config.task)


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    if mark is None:
        return problem
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def _describe(error: ErrorDetails) -> str:
    loc = error["loc"]
    where = _format_loc(loc)
    kind = error["type"]

    if kind == "extra_forbidden":
        release = _PLANNED_KEYS.get(loc)
        if release:
            return f"{where}: not supported yet (planned for {release}), remove this key"
        section = _section_at(loc[:-1])
        allowed = f" (allowed: {_allowed_keys(section)})" if section else ""
        return f"{where}: unknown key{allowed}"
    if kind == "missing":
        return f"{where}: required key is missing"

    ctx = error.get("ctx", {})
    if kind == "value_error":
        text = str(ctx["error"])
    elif kind in _MESSAGES:
        text = _MESSAGES[kind].format(**ctx)
    else:
        text = error["msg"][:1].lower() + error["msg"][1:]

    value = error["input"]
    if not isinstance(value, dict | list):
        text += f" (got {_show(value)})"
    return f"{where}: {text}"


def _format_loc(loc: Loc) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else f".{part}" if out else part
    return out


def _show(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value) if isinstance(value, str) else str(value)


def _allowed_keys(model: type[BaseModel]) -> str:
    return ", ".join(field.alias or name for name, field in model.model_fields.items())


def _section_at(loc: Loc) -> type[BaseModel] | None:
    """Return the section model reached by following ``loc`` from the root."""
    annotation: Any = FitRankConfig
    for part in loc:
        annotation = _child(annotation, part)
        if annotation is None:
            return None
    return _as_model(annotation)


def _child(annotation: Any, part: int | str) -> Any:
    model = _as_model(annotation)
    if model is not None:
        fields = {field.alias or name: field for name, field in model.model_fields.items()}
        field = fields.get(str(part))
        return None if field is None else field.annotation
    origin = get_origin(annotation)
    if origin is list:
        return get_args(annotation)[0] if isinstance(part, int) else None
    if origin is dict:
        return get_args(annotation)[1]
    return None


def _as_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None
