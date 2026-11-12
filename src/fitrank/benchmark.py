"""The Python entry point: compare models on one dataset.

A benchmark is built from a YAML file or from Python arguments. Both ways
produce the same ``FitRankConfig``, so they share one code path and give the
same results::

    results = Benchmark.from_yaml("examples/iris.yaml").run()

    results = Benchmark(
        data=df, target="species", task="classification",
        models=["logistic_regression", "random_forest", "svm"],
    ).run()
    print(results.leaderboard())

With ``test_data``, every candidate is scored on that test set and all of
``data`` is used for training and cross-validation: nothing is split.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fitrank.config import FitRankConfig, load_config, validate_config
from fitrank.data import (
    check_data,
    check_task,
    check_test_classes,
    fingerprint,
    load_dataframe,
    split_features_target,
    split_test_data,
)
from fitrank.evaluation import CandidateResult, check_folds, evaluate, split_data
from fitrank.exceptions import ConfigError
from fitrank.pipeline import Candidate, build_candidates
from fitrank.registry import Task
from fitrank.results import Results

# data.path / evaluation.test_data recorded in the config for a DataFrame.
DATAFRAME_PATH = "<DataFrame>"


@dataclass
class Plan:
    """What a run will use, once the config and the data have been checked.

    ``X`` and ``y`` hold every row of ``data``; the ``_train`` / ``_test``
    parts are what the candidates are trained and scored on.
    """

    X: pd.DataFrame
    y: pd.Series
    candidates: list[Candidate]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_source: str | None = None  # with test_data: its path, or "<DataFrame>"


class Benchmark:
    """Train, compare and rank models on one dataset.

    Only ``data``, ``target``, ``task`` and ``models`` are required. The
    keyword arguments are the YAML sections with the same name, e.g.
    ``evaluation={"cv": 3, "primary_metric": "f1_macro"}``. Models are given
    by name, or as ``{"name": ..., "params": {...}}`` to choose the grid.

    ``test_data`` (a DataFrame or a file path) is a test set of your own:
    every candidate is scored on it, and all of ``data`` trains. It is
    ``evaluation.test_data`` in YAML; ``test_size`` cannot be set with it.

    Raises:
        ConfigError: the arguments do not form a valid configuration.
    """

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        target: str,
        task: Task,
        models: Sequence[str | Mapping[str, Any]],
        *,
        features: Sequence[str] | Literal["all"] = "all",
        exclude: Sequence[str] = (),
        preprocessing: Mapping[str, Any] | None = None,
        reduction: Sequence[Any] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        project: Mapping[str, Any] | None = None,
        test_data: pd.DataFrame | str | Path | None = None,
    ) -> None:
        frame = data if isinstance(data, pd.DataFrame) else None
        test_frame = test_data if isinstance(test_data, pd.DataFrame) else None
        if isinstance(models, str):  # Benchmark(..., models="svm")
            models = [models]
        if test_data is not None:
            if evaluation is not None and "test_data" in evaluation:
                raise ConfigError(
                    "invalid configuration in the Benchmark arguments",
                    ["test_data: given twice; pass it as test_data=... or in evaluation, not both"],
                )
            path = DATAFRAME_PATH if test_frame is not None else str(test_data)
            evaluation = {**(evaluation or {}), "test_data": path}
        raw: dict[str, Any] = {
            "data": {
                "path": DATAFRAME_PATH if frame is not None else str(data),
                "target": target,
                "features": features if features == "all" else list(features),
                "exclude": list(exclude),
            },
            "task": task,
            "models": [{"name": m} if isinstance(m, str) else dict(m) for m in models],
        }
        sections = {
            "preprocessing": preprocessing,
            "reduction": reduction,
            "evaluation": evaluation,
            "output": output,
            "project": project,
        }
        raw |= {name: value for name, value in sections.items() if value is not None}
        self.config = validate_config(raw, "the Benchmark arguments")
        self._frame = frame
        self._test_frame = test_frame

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Benchmark":
        """Build a benchmark from a YAML file; the data is read from ``data.path``.

        Raises:
            ConfigError: the file is missing or invalid.
        """
        return cls.from_config(load_config(path))

    @classmethod
    def from_config(
        cls,
        config: FitRankConfig,
        data: pd.DataFrame | None = None,
        *,
        test_data: pd.DataFrame | None = None,
    ) -> "Benchmark":
        """Build a benchmark from a validated config, optionally with the data in memory.

        ``test_data`` replaces the file of ``evaluation.test_data``, which must be set.

        Raises:
            ConfigError: ``test_data`` is given but ``evaluation.test_data`` is not set.
        """
        if test_data is not None and config.evaluation.test_data is None:
            raise ConfigError(
                "a test DataFrame was given, but the config has no evaluation.test_data",
                [f"set evaluation.test_data (e.g. to {DATAFRAME_PATH!r}) to use it"],
            )
        bench = cls.__new__(cls)
        bench.config = config
        bench._frame = data
        bench._test_frame = test_data
        return bench

    @property
    def n_candidates(self) -> int:
        """One candidate per declared reduction and model."""
        return len(self.config.reduction) * len(self.config.models)

    def validate(self) -> Plan:
        """Check the config and the data without training anything.

        Raises:
            ConfigError: unknown model, step or reduction, or a bad option.
            DataError: the data cannot be read, or cannot be used as declared;
                with ``test_data``, the test set lacks a training column or
                has a class the training data does not have.
        """
        config = self.config
        candidates = build_candidates(config)  # checks models, steps and reductions
        if self._frame is not None:
            frame, source = self._frame, "the DataFrame"
        else:
            frame, source = load_dataframe(config.data.path), None
        X, y = split_features_target(frame, config.data, source=source)
        check_task(y, config.task)  # a warning only: the task is the user's choice
        check_data(X, y, config)

        test_path = config.evaluation.test_data
        if test_path is None:  # rare classes, too few rows for the folds
            X_train, X_test, y_train, y_test = split_data(X, y, config)
            return Plan(X, y, candidates, X_train, X_test, y_train, y_test)

        if self._test_frame is not None:
            test_frame, where = self._test_frame, "the test DataFrame"
        else:
            test_frame = load_dataframe(test_path, key="evaluation.test_data")
            where = test_path.as_posix()
        X_test, y_test = split_test_data(test_frame, X, y, where)
        check_data(X_test, y_test, config, name="test_data")
        if config.task == "classification":
            check_test_classes(y, y_test, where)
        check_folds(len(X), config)
        test_source = DATAFRAME_PATH if self._test_frame is not None else where
        return Plan(X, y, candidates, X, X_test, y, y_test, test_source=test_source)

    def run(
        self,
        *,
        export: bool = True,
        on_result: Callable[[CandidateResult], None] | None = None,
    ) -> Results:
        """Check everything, evaluate every candidate and rank them.

        With ``export`` (the default), the run folder is written as the
        ``project`` and ``output`` sections say; its path is ``results.run_dir``.
        ``on_result`` is called after each candidate.

        Raises:
            ConfigError, DataError: see ``validate``. A failing candidate
                never raises: it is ranked last with ``status="failed"``.
        """
        start = time.perf_counter()
        plan = self.validate()
        evaluated = evaluate(
            plan.candidates,
            plan.X_train,
            plan.y_train,
            self.config,
            on_result=on_result,
            test=(plan.X_test, plan.y_test),
        )
        test_data = None
        if plan.test_source is not None:
            test_data = {
                "source": plan.test_source,
                "sha256": fingerprint(plan.X_test, plan.y_test),
            }
        results = Results(
            evaluated,
            self.config,
            duration_seconds=time.perf_counter() - start,
            n_train=len(plan.X_train),
            n_test=len(plan.X_test),
            test_data=test_data,
        )
        if export:
            results.export()
        return results
