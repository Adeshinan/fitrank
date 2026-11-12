"""Tests for evaluation.test_data: score every model on a test set of your own."""

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from fitrank import Benchmark, ConfigError, DataError, FitRankWarning, Results
from fitrank.cli import app
from fitrank.config import load_config
from fitrank.data import fingerprint
from fitrank.evaluation import evaluate, split_data
from fitrank.pipeline import build_candidates

REPO = Path(__file__).resolve().parents[1]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
ARGS: dict[str, Any] = {
    "target": "species",
    "task": "classification",
    "models": [
        {"name": "logistic_regression", "params": {"C": [1]}},
        {"name": "knn", "params": {"n_neighbors": [5]}},
    ],
    "preprocessing": {"numeric": {"columns": FEATURES, "steps": ["scaler"]}},
    "evaluation": {"cv": 3},
}
YAML = """
project: {name: iris}
data: {path: train.csv, target: species}
task: classification
preprocessing:
  numeric:
    columns: [sepal_length, sepal_width, petal_length, petal_width]
    steps: [scaler]
models:
  - {name: logistic_regression, params: {C: [1]}}
evaluation: {test_data: test.csv, cv: 3}
"""


@pytest.fixture
def iris() -> pd.DataFrame:
    return pd.read_csv(REPO / "examples" / "data" / "iris.csv")


@pytest.fixture
def train_test(iris: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every 5th row is the test set: 120 training rows, 30 test rows, 10 per class."""
    test = iris.iloc[::5].reset_index(drop=True)
    train = iris.drop(index=iris.index[::5]).reset_index(drop=True)
    return train, test


@pytest.fixture
def workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_test: tuple[pd.DataFrame, pd.DataFrame]
) -> Path:
    """train.csv, test.csv and iris.yaml (with test_data), used as the current directory."""
    train, test = train_test
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "iris.yaml").write_text(YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def scores(results: Results) -> pd.DataFrame:
    return results.leaderboard().drop(columns="fit_s")


# --- the same test set for every version of the training data ------------------------------


def test_same_test_set_for_all_training_versions(
    train_test: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = train_test
    noisy = train.copy()
    noisy.loc[::4, "species"] = noisy["species"].sample(frac=1, random_state=0).to_numpy()[::4]

    r1 = Benchmark(data=train, test_data=test, **ARGS).run(export=False)
    r2 = Benchmark(data=noisy, test_data=test, **ARGS).run(export=False)

    assert r1.n_test == r2.n_test == len(test)  # the same test set, never split again
    assert r1.n_train == r2.n_train == len(train)  # all of data trains
    assert r1.test_data == r2.test_data
    assert r1.test_data == {
        "source": "<DataFrame>",
        "sha256": fingerprint(test[FEATURES], test.species),
    }


def test_test_data_equal_to_the_split_gives_the_same_scores(iris: pd.DataFrame) -> None:
    """Giving the rows test_size would put aside changes nothing: same folds, same scores."""
    split = Benchmark(data=iris, **ARGS)
    plan = split.validate()
    X_train, X_test, y_train, y_test = split_data(plan.X, plan.y, split.config)
    train = X_train.assign(species=y_train)
    test = X_test.assign(species=y_test)

    given = Benchmark(data=train, test_data=test, **ARGS).run(export=False)
    pd.testing.assert_frame_equal(scores(given), scores(split.run(export=False)))


def test_without_test_data_the_v1_1_path_is_unchanged(iris: pd.DataFrame) -> None:
    bench = Benchmark(data=iris, **ARGS)
    results = bench.run(export=False)
    # The v1.1 call: evaluate() splits test_size of the rows itself.
    plan = bench.validate()
    old = Results(
        evaluate(build_candidates(bench.config), plan.X, plan.y, bench.config), bench.config
    )

    pd.testing.assert_frame_equal(scores(results), scores(old))
    assert (results.n_train, results.n_test) == (120, 30)
    assert results.test_data is None


# --- YAML and run folder ---------------------------------------------------------------------


def test_yaml_test_data_path(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run()

    assert (results.n_train, results.n_test) == (120, 30)
    assert results.run_dir is not None
    metadata = json.loads((results.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rows"] == {"train": 120, "test": 30}
    assert metadata["test_data"]["source"] == "test.csv"
    assert len(metadata["test_data"]["sha256"]) == 64


def test_saved_config_can_be_run_again(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run()
    assert results.run_dir is not None
    saved = results.run_dir / "config.yaml"

    assert "test_size" not in saved.read_text(encoding="utf-8")  # it would be refused
    again = Benchmark.from_yaml(saved).run(export=False)
    pd.testing.assert_frame_equal(scores(again), scores(results))


def test_metadata_without_test_data(tmp_path: Path, iris: pd.DataFrame) -> None:
    run_dir = Benchmark(data=iris, **ARGS).run(export=False).export(tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    saved = (run_dir / "config.yaml").read_text(encoding="utf-8")

    assert metadata["rows"] == {"train": 120, "test": 30}
    assert metadata["test_data"] is None
    assert "test_size: 0.2" in saved and "test_data" not in saved  # a v1.1 config.yaml


def test_cli_validate_names_the_test_set(workdir: Path) -> None:
    result = CliRunner().invoke(app, ["validate", "iris.yaml"])

    assert result.exit_code == 0, result.output
    assert "Data OK: 120 rows" in result.output
    assert "Test data OK: 30 rows from test.csv" in result.output


# --- test_size and test_data -----------------------------------------------------------------


def test_test_size_and_test_data_together_is_an_error(workdir: Path) -> None:
    Path("both.yaml").write_text(
        YAML.replace("{test_data: test.csv,", "{test_size: 0.2, test_data: test.csv,"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as caught:
        load_config("both.yaml")
    assert "choose test_size or test_data, not both" in str(caught.value)
    assert caught.value.details[0].startswith("evaluation: ")


def test_test_size_and_test_data_together_in_python(
    train_test: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = train_test
    args = ARGS | {"evaluation": {"cv": 3, "test_size": 0.3}}
    with pytest.raises(ConfigError, match="choose test_size or test_data"):
        Benchmark(data=train, test_data=test, **args)


def test_test_data_given_twice_is_an_error(train_test: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = train_test
    args = ARGS | {"evaluation": {"cv": 3, "test_data": "test.csv"}}
    with pytest.raises(ConfigError, match="test_data: given twice"):
        Benchmark(data=train, test_data=test, **args)


def test_from_config_with_a_test_frame_needs_the_key(
    train_test: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = train_test
    config = Benchmark(data=train, **ARGS).config
    with pytest.raises(ConfigError, match="no evaluation.test_data"):
        Benchmark.from_config(config, train, test_data=test)

    with_key = Benchmark(data=train, test_data=test, **ARGS).config
    results = Benchmark.from_config(with_key, train, test_data=test).run(export=False)
    assert results.n_test == len(test)


# --- checks before training ------------------------------------------------------------------


def test_missing_column_in_test_data_is_explained(workdir: Path, train_test: Any) -> None:
    _, test = train_test
    test.drop(columns="petal_width").rename(columns={"sepal_width": "sepal_widht"}).to_csv(
        "test.csv", index=False
    )
    with pytest.raises(DataError) as caught:
        Benchmark.from_yaml("iris.yaml").validate()

    message = str(caught.value)
    assert message.startswith("test_data is missing 2 columns used for training")
    assert "column 'petal_width' not found in test.csv" in message
    assert "did you mean 'sepal_widht' instead of 'sepal_width'?" in message


def test_missing_target_in_test_data_is_explained(train_test: Any) -> None:
    train, test = train_test
    with pytest.raises(DataError) as caught:
        Benchmark(data=train, test_data=test.drop(columns="species"), **ARGS).validate()
    assert "test_data has no target column 'species'" in str(caught.value)
    assert "not found in the test DataFrame" in str(caught.value)


def test_unseen_class_in_test_data_is_explained(train_test: Any) -> None:
    train, test = train_test
    test.loc[:2, "species"] = "unknown"
    with pytest.raises(DataError) as caught:
        Benchmark(data=train, test_data=test, **ARGS).validate()
    assert "test_data has 1 class not seen in training" in str(caught.value)
    assert "class 'unknown' (3 test rows) is not in data.path" in str(caught.value)


def test_training_class_missing_from_test_data_only_warns(train_test: Any) -> None:
    train, test = train_test
    test = test[test.species != "setosa"]
    with pytest.warns(FitRankWarning, match="no row of class 'setosa'.*roc_auc"):
        results = Benchmark(data=train, test_data=test, **ARGS).run(export=False)
    assert results.n_test == 20
    assert results.best is not None


def test_missing_values_in_test_data_need_an_imputer(train_test: Any) -> None:
    train, test = train_test
    test.loc[0, "sepal_length"] = None
    with pytest.raises(DataError) as caught:
        Benchmark(data=train, test_data=test, **ARGS).validate()
    assert str(caught.value).startswith("1 problem found in test_data before training")
    assert "'sepal_length' has 1 missing values but no imputer" in str(caught.value)


def test_missing_test_file_names_the_yaml_key(workdir: Path) -> None:
    Path("test.csv").unlink()
    with pytest.raises(DataError) as caught:
        Benchmark.from_yaml("iris.yaml").validate()
    assert "data file not found: test.csv" in str(caught.value)
    assert "check evaluation.test_data" in str(caught.value)


def test_empty_test_data_is_an_error(train_test: Any) -> None:
    train, test = train_test
    with pytest.raises(DataError, match="test_data has no rows"):
        Benchmark(data=train, test_data=test.iloc[:0], **ARGS).validate()


def test_too_few_training_rows_for_the_folds(train_test: Any) -> None:
    train, test = train_test
    one_per_class = train.iloc[::40]  # 3 rows: setosa, versicolor, virginica
    args = ARGS | {"evaluation": {"cv": 5}}
    with pytest.raises(DataError, match="not enough rows for 5-fold"):
        Benchmark(data=one_per_class, test_data=test, **args).validate()


def test_extra_test_columns_are_ignored_and_features_apply(train_test: Any) -> None:
    train, test = train_test
    args = ARGS | {"preprocessing": {"numeric": {"columns": FEATURES[:2], "steps": ["scaler"]}}}
    test = test.assign(comment="extra")[["comment", "species", *reversed(FEATURES)]]
    plan = Benchmark(data=train, test_data=test, features=FEATURES[:2], **args).validate()

    assert list(plan.X_test.columns) == FEATURES[:2]  # the training columns, in their order
    assert plan.test_source == "<DataFrame>"


def test_regression_with_test_data(tmp_path: Path) -> None:
    data = pd.read_csv(REPO / "examples" / "data" / "diabetes.csv")
    train, test = data.iloc[:350], data.iloc[350:]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FitRankWarning)  # no class checks in regression
        results = Benchmark(
            data=train, test_data=test, target="progression", task="regression",
            models=[{"name": "ridge", "params": {"alpha": [1]}}], evaluation={"cv": 3},
        ).run(export=False)  # fmt: skip
    assert (results.n_train, results.n_test) == (350, len(test))
    assert results.best is not None


# --- fingerprint -----------------------------------------------------------------------------


def test_fingerprint_follows_the_rows_not_the_index(train_test: Any) -> None:
    _, test = train_test
    X, y = test[FEATURES], test.species
    same = fingerprint(X.set_axis(range(100, 130)), y.set_axis(range(100, 130)))
    changed = X.copy()
    changed.iloc[0, 0] += 0.1

    assert fingerprint(X, y) == same
    assert fingerprint(changed, y) != fingerprint(X, y)
    assert fingerprint(X.iloc[::-1], y.iloc[::-1]) != fingerprint(X, y)
