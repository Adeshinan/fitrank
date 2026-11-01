"""Tests for the Benchmark facade: YAML and Python arguments share one code path."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import fitrank
from fitrank import Benchmark, ConfigError, DataError
from fitrank.evaluation import CandidateResult

REPO = Path(__file__).resolve().parents[1]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_YAML = """
project: {name: iris}
data: {path: iris.csv, target: species}
task: classification
preprocessing:
  numeric:
    columns: [sepal_length, sepal_width, petal_length, petal_width]
    steps: [scaler]
models:
  - {name: logistic_regression, params: {C: [1]}}
  - {name: knn, params: {n_neighbors: [5]}}
evaluation: {cv: 3}
"""
# The same experiment as IRIS_YAML, as Python arguments.
IRIS_ARGS = {
    "target": "species",
    "task": "classification",
    "models": [
        {"name": "logistic_regression", "params": {"C": [1]}},
        {"name": "knn", "params": {"n_neighbors": [5]}},
    ],
    "preprocessing": {"numeric": {"columns": FEATURES, "steps": ["scaler"]}},
    "evaluation": {"cv": 3},
    "project": {"name": "iris"},
}


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder with iris.csv and iris.yaml, used as the current directory."""
    shutil.copy(REPO / "examples" / "data" / "iris.csv", tmp_path / "iris.csv")
    (tmp_path / "iris.yaml").write_text(IRIS_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def scores(results: fitrank.Results) -> pd.DataFrame:
    """The leaderboard without the timing column, which changes on every run."""
    return results.leaderboard().drop(columns="fit_s")


def test_public_api_is_importable() -> None:
    for name in ["Benchmark", "register_model", "register_reduction", "load_model",
                 "ConfigError", "DataError", "__version__"]:  # fmt: skip
        assert name in fitrank.__all__
        assert hasattr(fitrank, name)


def test_run_from_yaml(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run()
    assert [r.status for r in results.results] == ["ok", "ok"]
    assert results.run_dir is not None
    assert results.run_dir.parent == Path("runs")
    assert (workdir / results.run_dir / "best_model.joblib").is_file()


def test_yaml_and_dataframe_give_the_same_leaderboard(workdir: Path) -> None:
    from_yaml = Benchmark.from_yaml("iris.yaml").run(export=False)
    frame = pd.read_csv("iris.csv")
    from_frame = Benchmark(data=frame, **IRIS_ARGS).run(export=False)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(scores(from_yaml), scores(from_frame))


def test_data_can_be_a_file_path(workdir: Path) -> None:
    bench = Benchmark(data="iris.csv", **IRIS_ARGS)  # type: ignore[arg-type]
    assert bench.config.data.path == Path("iris.csv")
    assert len(bench.validate().X) == 150


def test_dataframe_is_recorded_in_the_config() -> None:
    bench = Benchmark(data=pd.DataFrame({"x": [1.0], "y": [0]}), target="y",
                      task="classification", models=["svm"])  # fmt: skip
    assert bench.config.data.path == Path("<DataFrame>")


def test_models_by_name() -> None:
    frame = pd.DataFrame({"x": [1.0]})
    bench = Benchmark(data=frame, target="x", task="regression", models=["ridge", "lasso"])
    assert [m.name for m in bench.config.models] == ["ridge", "lasso"]
    assert [m.params for m in bench.config.models] == [None, None]
    single = Benchmark(data=frame, target="x", task="regression", models="ridge")
    assert [m.name for m in single.config.models] == ["ridge"]


def test_bad_arguments_raise_config_error() -> None:
    with pytest.raises(
        ConfigError, match="invalid configuration in the Benchmark arguments"
    ) as info:
        Benchmark(data=pd.DataFrame(), target="y", task="clasification", models=["svm"])  # type: ignore[arg-type]
    assert info.value.details[0].startswith("task: must be 'classification' or 'regression'")


def test_unknown_model_is_reported_before_reading_the_data(workdir: Path) -> None:
    bench = Benchmark(data="missing.csv", target="y", task="classification", models=["svn"])
    with pytest.raises(ConfigError, match="svn"):
        bench.validate()


def test_data_problems_are_reported_before_training(workdir: Path) -> None:
    bench = Benchmark(data="iris.csv", target="colour", task="classification", models=["knn"])
    with pytest.raises(DataError, match="colour"):
        bench.run()
    assert not (workdir / "runs").exists()


def test_validate_returns_the_plan(workdir: Path) -> None:
    bench = Benchmark.from_yaml("iris.yaml")
    plan = bench.validate()
    assert plan.X.columns.tolist() == FEATURES
    assert plan.y.name == "species"
    assert len(plan.candidates) == bench.n_candidates == 2


def test_run_without_export(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run(export=False)
    assert results.run_dir is None
    assert not (workdir / "runs").exists()


def test_on_result_is_called_after_each_candidate(workdir: Path) -> None:
    seen: list[CandidateResult] = []
    results = Benchmark.from_yaml("iris.yaml").run(export=False, on_result=seen.append)
    assert [r.model_name for r in seen] == ["logistic_regression", "knn"]  # declaration order
    assert {id(r) for r in seen} == {id(r) for r in results.results}


def test_duration_covers_the_whole_run(workdir: Path) -> None:
    results = Benchmark.from_yaml("iris.yaml").run(export=False)
    assert results.duration_seconds >= sum(r.fit_seconds for r in results.results)
