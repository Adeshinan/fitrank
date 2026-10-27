"""Load the data and check that it can be used as declared, before any training.

FitRank reports data problems, it never fixes them: missing values need an
imputer and text columns need an encoder, both declared in the configuration.
"""

import difflib
import zipfile
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path

import pandas as pd

from fitrank.config import DataConfig, FitRankConfig, step_name
from fitrank.exceptions import DataError

# File extension -> (reader, optional extra that provides the reader's engine).
_READERS: dict[str, tuple[Callable[[Path], pd.DataFrame], str | None]] = {
    ".csv": (lambda path: pd.read_csv(path), None),
    ".xlsx": (lambda path: pd.read_excel(path), "excel"),
    ".parquet": (lambda path: pd.read_parquet(path), "parquet"),
}

_MAX_COLUMNS_SHOWN = 20


def load_dataframe(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a CSV, Excel (.xlsx) or Parquet file, or return a DataFrame as is.

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
            ["check data.path; a relative path starts from the folder the command runs in"],
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
        features = [column for column in df.columns if column != target]
    else:
        features = list(data_cfg.features)

    missing = [(target, "data.target")] if target not in df.columns else []
    missing += [(column, "data.features") for column in features if column not in df.columns]
    if missing:
        raise _not_found(missing, df.columns, where)
    if target in features:
        raise DataError(
            f"column {target!r} is both the target and a feature",
            ["remove it from data.features"],
        )
    if not features:
        raise DataError(
            "no feature to train on",
            ["list at least one column in data.features, or use features: all"],
        )
    return df[features], df[target]


def check_data(X: pd.DataFrame, y: pd.Series, config: FitRankConfig) -> None:
    """Check that the data can be trained on with the declared preprocessing.

    Raises:
        DataError: one line per problem, with a YAML hint to fix them.
    """
    if len(X) == 0:
        raise DataError("the data has no rows", ["check data.path points to the right file"])

    with_imputer = _columns_with_step(config, "imputer")
    with_encoder = _columns_with_step(config, "encoder")
    problems = []

    for group, settings in config.preprocessing.items():
        problems += [
            f"preprocessing.{group}.columns: {column!r} is not a feature"
            " (check the spelling, or add it to data.features)"
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
        if _is_text(X[column]) and column not in with_encoder:
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
        raise DataError(
            f"{count} found before training",
            problems,
            hint=_preprocessing_hint(X, need_imputer, need_encoder),
        )


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


def _is_text(column: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(column)
        or pd.api.types.is_string_dtype(column)
        or isinstance(column.dtype, pd.CategoricalDtype)
    )


def _preprocessing_hint(
    X: pd.DataFrame, need_imputer: Sequence[Hashable], need_encoder: Sequence[Hashable]
) -> str | None:
    numeric = [column for column in need_imputer if not _is_text(X[column])]
    text = list(dict.fromkeys([*need_encoder, *(c for c in need_imputer if _is_text(X[c]))]))
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
