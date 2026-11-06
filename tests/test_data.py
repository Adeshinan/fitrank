"""Tests for loading the data and checking it before training."""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fitrank.config import DataConfig, FitRankConfig
from fitrank.data import check_data, check_task, load_dataframe, split_features_target
from fitrank.exceptions import DataError, FitRankError, FitRankWarning
from fitrank.registry import Task

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


def make_config(**overrides: Any) -> FitRankConfig:
    raw: dict[str, Any] = {
        "data": {"path": "data.csv", "target": "Survived"},
        "task": "classification",
        "models": [{"name": "random_forest"}],
    }
    raw.update(overrides)
    return FitRankConfig.model_validate(raw)


def data_error(X: pd.DataFrame, y: pd.Series, config: FitRankConfig) -> DataError:
    with pytest.raises(DataError) as info:
        check_data(X, y, config)
    return info.value


def load_xy(name: str, config: FitRankConfig) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(load_dataframe(FIXTURES / name), config.data)


# --- Reading files ---------------------------------------------------------------------------


def test_csv_is_read() -> None:
    df = load_dataframe(FIXTURES / "clean.csv")
    assert df.shape == (10, 4)
    assert list(df.columns) == ["Age", "Fare", "Pclass", "Survived"]


def test_str_path_is_accepted() -> None:
    assert len(load_dataframe(str(FIXTURES / "clean.csv"))) == 10


def test_dataframe_is_returned_as_is() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    assert load_dataframe(df) is df


def test_excel_is_read(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "data.xlsx"
    pd.read_csv(FIXTURES / "clean.csv").to_excel(path, index=False)
    assert load_dataframe(path).shape == (10, 4)


def test_parquet_is_read(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    pd.read_csv(FIXTURES / "clean.csv").to_parquet(path)
    assert load_dataframe(path).shape == (10, 4)


def test_extension_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "DATA.CSV"
    path.write_text("a,y\n1,0\n", encoding="utf-8")
    assert load_dataframe(path).shape == (1, 2)


@pytest.mark.parametrize(("suffix", "extra"), [(".xlsx", "excel"), (".parquet", "parquet")])
def test_missing_extra_says_what_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, extra: str
) -> None:
    def no_engine(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise ImportError("Missing optional dependency")

    monkeypatch.setattr(pd, "read_excel", no_engine)
    monkeypatch.setattr(pd, "read_parquet", no_engine)
    path = tmp_path / f"data{suffix}"
    path.write_bytes(b"")
    with pytest.raises(DataError) as info:
        load_dataframe(path)
    assert str(info.value) == (
        f'reading {suffix} files needs an extra dependency\n  -> pip install "fitrank[{extra}]"'
    )


def test_missing_file_is_explained(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="data file not found") as info:
        load_dataframe(tmp_path / "nope.csv")
    assert "check data.path" in str(info.value)


def test_unsupported_extension_lists_formats(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(DataError, match="unsupported data file") as info:
        load_dataframe(path)
    assert ".csv, .xlsx, .parquet" in str(info.value)


def test_empty_csv_cannot_be_read(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="cannot read data file"):
        load_dataframe(path)


def test_corrupt_excel_cannot_be_read(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "data.xlsx"
    path.write_text("not a spreadsheet", encoding="utf-8")
    with pytest.raises(DataError, match="cannot read data file"):
        load_dataframe(path)


def test_data_error_is_a_fitrank_error() -> None:
    assert issubclass(DataError, FitRankError)


# --- Features and target ---------------------------------------------------------------------


def test_dataframe_input_is_accepted() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [0, 1]})
    X, y = split_features_target(df, DataConfig(path=Path("unused"), target="y"))
    assert list(X.columns) == ["a", "b"]
    assert y.tolist() == [0, 1]


def test_declared_features_keep_their_order() -> None:
    X, y = load_xy("clean.csv", make_config(data={
        "path": "clean.csv", "target": "Survived", "features": ["Pclass", "Age"]
    }))  # fmt: skip
    assert list(X.columns) == ["Pclass", "Age"]
    assert y.name == "Survived"


def test_unknown_target_lists_columns_and_suggests() -> None:
    config = make_config(data={"path": "data/titanic.csv", "target": "Survive"})
    with pytest.raises(DataError) as info:
        load_xy("clean.csv", config)
    assert str(info.value) == (
        "column 'Survive' (data.target) not found in data/titanic.csv\n"
        "  -> available columns: Age, Fare, Pclass, Survived\n"
        "  -> did you mean 'Survived' instead of 'Survive'?"
    )


def test_all_unknown_columns_are_reported_at_once() -> None:
    config = make_config(data={
        "path": "clean.csv", "target": "Survived", "features": ["Agee", "Fare", "Cabin"]
    })  # fmt: skip
    with pytest.raises(DataError, match="2 columns not found") as info:
        load_xy("clean.csv", config)
    assert "'Agee' (data.features), 'Cabin' (data.features)" in str(info.value)
    assert "did you mean 'Age' instead of 'Agee'?" in info.value.details


def test_source_names_the_data_in_messages() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataError, match="not found in the DataFrame"):
        split_features_target(df, DataConfig(path=Path("unused"), target="y"), "the DataFrame")


def test_long_column_lists_are_shortened() -> None:
    df = pd.DataFrame({f"c{i}": [0] for i in range(25)})
    with pytest.raises(DataError, match=r"c19, \.\.\. \(5 more\)"):
        split_features_target(df, DataConfig(path=Path("x.csv"), target="zzz"))


def test_target_cannot_be_a_feature() -> None:
    config = make_config(data={
        "path": "clean.csv", "target": "Survived", "features": ["Age", "Survived"]
    })  # fmt: skip
    with pytest.raises(DataError, match="both the target and a feature"):
        load_xy("clean.csv", config)


@pytest.mark.parametrize("features", [[], "all"])
def test_no_feature_left_is_refused(features: Any) -> None:
    df = pd.DataFrame({"y": [0, 1]})
    with pytest.raises(DataError, match="no feature to train on"):
        split_features_target(df, DataConfig(path=Path("x.csv"), target="y", features=features))


def test_excluded_columns_are_left_out() -> None:
    X, y = load_xy("clean.csv", make_config(data={
        "path": "clean.csv", "target": "Survived", "exclude": ["Fare"]
    }))  # fmt: skip
    assert list(X.columns) == ["Age", "Pclass"]
    assert y.name == "Survived"


def test_unknown_excluded_column_lists_columns_and_suggests() -> None:
    config = make_config(data={"path": "clean.csv", "target": "Survived", "exclude": ["Fair"]})
    with pytest.raises(DataError) as info:
        load_xy("clean.csv", config)
    assert str(info.value) == (
        "column 'Fair' (data.exclude) not found in clean.csv\n"
        "  -> available columns: Age, Fare, Pclass, Survived\n"
        "  -> did you mean 'Fare' instead of 'Fair'?"
    )


def test_excluding_every_feature_is_refused() -> None:
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    with pytest.raises(DataError) as info:
        split_features_target(df, DataConfig(path=Path("x.csv"), target="y", exclude=["a"]))
    assert info.value.details == [
        "list at least one column in data.features, or use fewer columns in data.exclude"
    ]


# --- Checks before training ------------------------------------------------------------------


def test_clean_data_passes() -> None:
    config = make_config()
    check_data(*load_xy("clean.csv", config), config)


def test_missing_values_without_imputer_is_reported() -> None:
    config = make_config()
    error = data_error(*load_xy("with_missing.csv", config), config)
    assert str(error) == (
        "1 problem found before training\n"
        "  -> column 'Age' has 2 missing values but no imputer is declared for it\n"
        "  Hint: add them to a preprocessing group, e.g.\n"
        "        preprocessing:\n"
        "          numeric: {columns: [Age], steps: [{imputer: {strategy: median}}]}"
    )


def test_text_without_encoder_is_reported() -> None:
    config = make_config(preprocessing={
        "num": {"columns": ["Age"], "steps": ["scaler"]},
    })  # fmt: skip
    error = data_error(*load_xy("with_text.csv", config), config)
    assert error.details == [
        "column 'Sex' contains text but no encoder is declared for it",
        "column 'Embarked' has 1 missing values but no imputer is declared for it",
        "column 'Embarked' contains text but no encoder is declared for it",
    ]
    assert error.hint == (
        "add them to a preprocessing group, e.g.\n"
        "preprocessing:\n"
        "  categorical: {columns: [Sex, Embarked],"
        " steps: [{imputer: {strategy: most_frequent}}, {encoder: onehot}]}"
    )


def test_declared_imputer_and_encoder_cover_the_columns() -> None:
    config = make_config(preprocessing={
        "numeric": {"columns": ["Age"], "steps": [{"imputer": {"strategy": "median"}}]},
        "categorical": {
            "columns": ["Sex", "Embarked"],
            "steps": [{"imputer": {"strategy": "most_frequent"}}, {"encoder": "onehot"}],
        },
    })  # fmt: skip
    check_data(*load_xy("with_text.csv", config), config)


def test_imputer_alone_covers_numeric_missing_values() -> None:
    config = make_config(preprocessing={
        "numeric": {"columns": ["Age"], "steps": [{"imputer": {"strategy": "median"}}, "scaler"]},
    })  # fmt: skip
    check_data(*load_xy("with_missing.csv", config), config)


def test_category_dtype_counts_as_text() -> None:
    X = pd.DataFrame({"color": pd.Categorical(["red", "blue"])})
    error = data_error(X, pd.Series([0, 1], name="y"), make_config())
    assert error.details == ["column 'color' contains text but no encoder is declared for it"]


def test_bool_and_numbers_are_not_text() -> None:
    X = pd.DataFrame({"flag": [True, False], "n": [1, 2], "x": [0.5, 1.5]})
    check_data(X, pd.Series([0, 1], name="y"), make_config())


def test_missing_target_values_are_reported() -> None:
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, None, 1], name="Survived")
    error = data_error(X, y, make_config())
    assert error.details == [
        "target 'Survived' has 1 missing values (remove those rows from the data:"
        " FitRank never drops rows)"
    ]
    assert error.hint is None


def test_preprocessing_column_must_be_a_feature() -> None:
    config = make_config(preprocessing={
        "numeric": {"columns": ["Agee"], "steps": ["scaler"]},
    })  # fmt: skip
    error = data_error(*load_xy("clean.csv", config), config)
    assert error.details == [
        "preprocessing.numeric.columns: 'Agee' is not a feature"
        " (check the spelling, or add it to data.features)"
    ]


def test_preprocessing_an_excluded_column_says_so() -> None:
    config = make_config(
        data={"path": "clean.csv", "target": "Survived", "exclude": ["Fare"]},
        preprocessing={"numeric": {"columns": ["Fare"], "steps": ["scaler"]}},
    )
    error = data_error(*load_xy("clean.csv", config), config)
    assert error.details == [
        "preprocessing.numeric.columns: 'Fare' is not a feature (it is listed in data.exclude)"
    ]


def test_every_problem_is_reported_at_once() -> None:
    X = pd.DataFrame({"a": [1.0, None], "s": ["x", "y"]})
    y = pd.Series([None, 1], name="y")
    error = data_error(X, y, make_config())
    assert str(error).startswith("3 problems found before training\n")
    assert len(error.details) == 3


def test_empty_data_is_refused() -> None:
    X = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(DataError, match="the data has no rows"):
        check_data(X, pd.Series([], dtype=float, name="y"), make_config())


# --- Task and target (warnings only) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "n_values"),
    [(["yes", "no", "yes", "maybe"], 3), ([1, 0, 0, 1], 2), ([True, False, True, True], 2)],
)
def test_regression_on_a_categorical_target_warns(values: list[Any], n_values: int) -> None:
    with pytest.warns(UserWarning) as record:
        check_task(pd.Series(values, name="Survived"), "regression")
    (warning,) = record
    assert warning.category is FitRankWarning
    assert str(warning.message) == (
        f"target 'Survived' looks categorical ({n_values} distinct values) but task is"
        " 'regression'; use task: classification to predict a category"
    )


def test_classification_on_a_continuous_target_warns() -> None:
    y = pd.Series([i * 1.5 for i in range(100)], name="price")
    with pytest.warns(FitRankWarning) as record:
        check_task(y, "classification")
    assert str(record[0].message) == (
        "target 'price' looks continuous (100 distinct values) but task is 'classification';"
        " use task: regression to predict a number"
    )


@pytest.mark.parametrize(
    ("values", "task"),
    [
        (list(range(20)) * 5, "classification"),  # 20 distinct values: not more than 20
        (list(range(25)) * 40, "classification"),  # 25 values but 2.5 % of 1000 rows
        ([f"c{i}" for i in range(100)], "classification"),  # many labels, but text
        ([0, 1, 2, 3.5], "regression"),
    ],
)
def test_plausible_target_does_not_warn(values: list[Any], task: Task) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_task(pd.Series(values, name="y"), task)


@pytest.mark.parametrize(
    ("name", "target", "task"),
    [
        ("iris.csv", "species", "classification"),
        ("diabetes.csv", "progression", "regression"),
        ("titanic.csv", "Survived", "classification"),
    ],
)
def test_examples_do_not_warn(name: str, target: str, task: Task) -> None:
    y = load_dataframe(EXAMPLES / name)[target]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_task(y, task)
