"""Load the data and check that it can be used as declared, before any training.

FitRank reports data problems, it never fixes them: missing values need an
imputer and text columns need an encoder, both declared in the configuration.
"""

import difflib
import hashlib
import warnings
import zipfile
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path

import pandas as pd

from fitrank.config import DataConfig, FitRankConfig, step_name
from fitrank.exceptions import DataError, FitRankWarning
from fitrank.registry import Task

# File extension -> (reader, optional extra that provides the reader's engine).
_READERS: dict[str, tuple[Callable[[Path], pd.DataFrame], str | None]] = {
    ".csv": (lambda path: pd.read_csv(path), None),
    ".xlsx": (lambda path: pd.read_excel(path), "excel"),
    ".parquet": (lambda path: pd.read_parquet(path), "parquet"),
}

_MAX_COLUMNS_SHOWN = 20

# A numeric target with more distinct values than both limits looks continuous.
_CONTINUOUS_MIN_VALUES = 20
_CONTINUOUS_MIN_SHARE = 0.05


def load_dataframe(source: str | Path | pd.DataFrame, key: str = "data.path") -> pd.DataFrame:
    """Read a CSV, Excel (.xlsx) or Parquet file, or return a DataFrame as is.

    ``key`` is the YAML key that gave the path, named in error messages.

    Raises:
        DataError: the file is missing, has an unsupported extension, needs
            an extra that is not installed, or cannot be parsed.
    """
    if isinstance(source, pd.DataFrame):
        return source

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix not in _READERS:
        raise DataError(
            f"unsupported data file: {path}",
            [f"use one of: {', '.join(_READERS)}, or pass a pandas DataFrame"],
        )
    if not path.is_file():
        raise DataError(
            f"data file not found: {path}",
            [f"check {key}; a relative path starts from the folder the command runs in"],
        )

    reader, extra = _READERS[suffix]
    try:
        return reader(path)
    except ImportError:
        if extra is None:
            raise
        raise DataError(
            f"reading {suffix} files needs an extra dependency",
            [f'pip install "fitrank[{extra}]"'],
        ) from None
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise DataError(f"cannot read data file: {path}", [reason]) from None


def split_features_target(
    df: pd.DataFrame, data_cfg: DataConfig, source: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``X`` (the declared features) and ``y`` (the target column).

    ``source`` names the data in error messages; it defaults to ``data.path``.

    Raises:
        DataError: a declared column does not exist, the target is also a
            feature, or no feature is left.
    """
    where = source or data_cfg.path.as_posix()  # as written in the YAML, on every OS
    target = data_cfg.target
    if data_cfg.features == "all":
        left_out = {target, *data_cfg.exclude}
        features = [column for column in df.columns if column not in left_out]
    else:
        features = list(data_cfg.features)

    missing = [(target, "data.target")] if target not in df.columns else []
    missing += [(column, "data.features") for column in features if column not in df.columns]
    missing += [(column, "data.exclude") for column in data_cfg.exclude if column not in df.columns]
    if missing:
        raise _not_found(missing, df.columns, where)
    if target in features:
        raise DataError(
            f"column {target!r} is both the target and a feature",
            ["remove it from data.features"],
        )
    if not features:
        fix = "use fewer columns in data.exclude" if data_cfg.exclude else "use features: all"
        raise DataError(
            "no feature to train on",
            [f"list at least one column in data.features, or {fix}"],
        )
    return df[features], df[target]


def split_test_data(
    df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, where: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``X_test`` and ``y_test``: the training columns, read from the test set.

    ``where`` names the test set in error messages (its path, or "the test
    DataFrame"). Extra columns in the test set are ignored.

    Raises:
        DataError: the test set has no row, or lacks the target or a feature.
    """
    target = y_train.name
    if target not in df.columns:
        raise DataError(
            f"test_data has no target column {target!r}",
            [f"column {target!r} (data.target) not found in {where}"],
            hint="the test set needs the true answers to be scored",
        )
    missing = [column for column in X_train.columns if column not in df.columns]
    if missing:
        n = len(missing)
        details = [f"column {column!r} not found in {where}" for column in missing]
        names = [str(column) for column in df.columns]
        for column in missing:
            close = difflib.get_close_matches(str(column), names, n=1)
            if close:
                details.append(f"did you mean {close[0]!r} instead of {column!r}?")
        raise DataError(
            f"test_data is missing {n} column{'s' if n > 1 else ''} used for training", details
        )
    if len(df) == 0:
        raise DataError("test_data has no rows", [f"check that {where} is the right file"])
    return df[list(X_train.columns)], df[target]


def fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """SHA-256 of the rows and columns used: the same test set gives the same value.

    Computed on the values, row by row in order, whatever the file format or
    the index; another row, value, order or dtype gives another fingerprint.
    """
    rows = pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=False)
    return hashlib.sha256(rows.to_numpy().tobytes()).hexdigest()


def check_test_classes(y_train: pd.Series, y_test: pd.Series, where: str) -> None:
    """Compare the classes of the training data and of the test set.

    Raises:
        DataError: a test class never appears in the training data: no model
            can predict it.

    Warns (``FitRankWarning``) when a training class has no test row: the test
    scores are still computed, but roc_auc and macro averages can mislead.
    """
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    unseen = [label for label in test_counts.index if label not in train_counts.index]
    if unseen:
        n = len(unseen)
        raise DataError(
            f"test_data has {n} class{'es' if n > 1 else ''} not seen in training",
            [
                f"class {label!r} ({test_counts[label]} test rows) is not in data.path"
                for label in unseen
            ],
            hint="no model can predict a class it never saw: add rows of it to the"
            " training data, or remove them from the test set",
        )
    absent = [label for label in train_counts.index if label not in test_counts.index]
    if absent:
        listed = ", ".join(repr(label) for label in absent)
        warnings.warn(
            f"{where} has no row of class {listed}, which the training data has;"
            " roc_auc and macro averages on the test set can be misleading",
            FitRankWarning,
            stacklevel=3,
        )


def check_data(
    X: pd.DataFrame, y: pd.Series, config: FitRankConfig, name: str = "the data"
) -> None:
    """Check that the data can be trained on with the declared preprocessing.

    ``name`` says which data the errors are about, e.g. "test_data".

    Raises:
        DataError: one line per problem, with a YAML hint to fix them.
    """
    if len(X) == 0:
        raise DataError(f"{name} has no rows", ["check data.path points to the right file"])

    with_imputer = _columns_with_step(config, "imputer")
    with_encoder = _columns_with_step(config, "encoder")
    problems = []

    for group, settings in config.preprocessing.items():
        problems += [
            f"preprocessing.{group}.columns: {column!r} is not a feature"
            + (
                " (it is listed in data.exclude)"
                if column in config.data.exclude
                else " (check the spelling, or add it to data.features)"
            )
            for column in settings.columns
            if column not in X.columns
        ]

    need_imputer, need_encoder = [], []
    for column in X.columns:
        n_missing = int(X[column].isna().sum())
        if n_missing and column not in with_imputer:
            need_imputer.append(column)
            problems.append(
                f"column {column!r} has {n_missing} missing values"
                " but no imputer is declared for it"
            )
        if is_text(X[column]) and column not in with_encoder:
            need_encoder.append(column)
            problems.append(f"column {column!r} contains text but no encoder is declared for it")

    n_missing = int(y.isna().sum())
    if n_missing:
        problems.append(
            f"target {y.name!r} has {n_missing} missing values"
            " (remove those rows from the data: FitRank never drops rows)"
        )

    if problems:
        count = f"{len(problems)} problem{'s' if len(problems) > 1 else ''}"
        where = "" if name == "the data" else f" in {name}"
        raise DataError(
            f"{count} found{where} before training",
            problems,
            hint=_preprocessing_hint(X, need_imputer, need_encoder),
        )


def check_task(y: pd.Series, task: Task) -> None:
    """Warn when the target does not look like the declared task; never blocks.

    Regression on text, true/false or two values, and classification on a
    number with many distinct values (more than 20 and more than 5 % of the
    rows), are most likely a wrong ``task``.
    """
    n_values = int(y.nunique())
    if task == "regression":
        categorical = is_text(y) or pd.api.types.is_bool_dtype(y) or n_values <= 2
        if not categorical:
            return
        message = (
            f"target {y.name!r} looks categorical ({n_values} distinct values)"
            " but task is 'regression'; use task: classification to predict a category"
        )
    else:
        continuous = (
            pd.api.types.is_numeric_dtype(y)
            and not pd.api.types.is_bool_dtype(y)
            and n_values > _CONTINUOUS_MIN_VALUES
            and n_values > _CONTINUOUS_MIN_SHARE * len(y)
        )
        if not continuous:
            return
        message = (
            f"target {y.name!r} looks continuous ({n_values} distinct values)"
            " but task is 'classification'; use task: regression to predict a number"
        )
    warnings.warn(message, FitRankWarning, stacklevel=3)


def _not_found(missing: Sequence[tuple[Hashable, str]], columns: pd.Index, where: str) -> DataError:
    names = [str(column) for column in columns]
    if len(missing) == 1:
        column, key = missing[0]
        message = f"column {column!r} ({key}) not found in {where}"
    else:
        listed = ", ".join(f"{column!r} ({key})" for column, key in missing)
        message = f"{len(missing)} columns not found in {where}: {listed}"

    shown = ", ".join(names[:_MAX_COLUMNS_SHOWN])
    if len(names) > _MAX_COLUMNS_SHOWN:
        shown += f", ... ({len(names) - _MAX_COLUMNS_SHOWN} more)"
    details = [f"available columns: {shown}"]
    for column, _ in missing:
        close = difflib.get_close_matches(str(column), names, n=1)
        if close:
            details.append(f"did you mean {close[0]!r} instead of {column!r}?")
    return DataError(message, details)


def _columns_with_step(config: FitRankConfig, name: str) -> set[str]:
    return {
        column
        for group in config.preprocessing.values()
        if any(step_name(step) == name for step in group.steps)
        for column in group.columns
    }


def is_text(column: pd.Series) -> bool:
    """True for a column FitRank cannot train on without an encoder: text or categories."""
    return (
        pd.api.types.is_object_dtype(column)
        or pd.api.types.is_string_dtype(column)
        or isinstance(column.dtype, pd.CategoricalDtype)
    )


def _preprocessing_hint(
    X: pd.DataFrame, need_imputer: Sequence[Hashable], need_encoder: Sequence[Hashable]
) -> str | None:
    numeric = [column for column in need_imputer if not is_text(X[column])]
    text = list(dict.fromkeys([*need_encoder, *(c for c in need_imputer if is_text(X[c]))]))
    if not numeric and not text:
        return None

    lines = ["add them to a preprocessing group, e.g.", "preprocessing:"]
    if numeric:
        steps = "{imputer: {strategy: median}}"
        lines.append(f"  numeric: {{columns: [{_join(numeric)}], steps: [{steps}]}}")
    if text:
        steps = "{encoder: onehot}"
        if any(column in need_imputer for column in text):
            steps = "{imputer: {strategy: most_frequent}}, " + steps
        lines.append(f"  categorical: {{columns: [{_join(text)}], steps: [{steps}]}}")
    return "\n".join(lines)


def _join(columns: Sequence[Hashable]) -> str:
    return ", ".join(str(column) for column in columns)
