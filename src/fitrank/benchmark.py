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
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fitrank.config import FitRankConfig, load_config, validate_config
from fitrank.data import check_data, load_dataframe, split_features_target
from fitrank.evaluation import CandidateResult, evaluate, split_data
from fitrank.pipeline import Candidate, build_candidates
from fitrank.registry import Task
from fitrank.results import Results

# data.path recorded in the config when the data is given as a DataFrame.
DATAFRAME_PATH = "<DataFrame>"


@dataclass
class Plan:
    """What a run will use, once the config and the data have been checked."""

    X: pd.DataFrame
    y: pd.Series
    candidates: list[Candidate]


class Benchmark:
    """Train, compare and rank models on one dataset.

    Only ``data``, ``target``, ``task`` and ``models`` are required. The
    keyword arguments are the YAML sections with the same name, e.g.
    ``evaluation={"cv": 3, "primary_metric": "f1_macro"}``. Models are given
    by name, or as ``{"name": ..., "params": {...}}`` to choose the grid.

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
        preprocessing: Mapping[str, Any] | None = None,
        reduction: Sequence[Any] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        project: Mapping[str, Any] | None = None,
    ) -> None:
        frame = data if isinstance(data, pd.DataFrame) else None
        if isinstance(models, str):  # Benchmark(..., models="svm")
            models = [models]
        raw: dict[str, Any] = {
            "data": {
                "path": DATAFRAME_PATH if frame is not None else str(data),
                "target": target,
                "features": features if features == "all" else list(features),
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Benchmark":
        """Build a benchmark from a YAML file; the data is read from ``data.path``.

        Raises:
            ConfigError: the file is missing or invalid.
        """
        return cls.from_config(load_config(path))

    @classmethod
    def from_config(cls, config: FitRankConfig, data: pd.DataFrame | None = None) -> "Benchmark":
        """Build a benchmark from a validated config, optionally with the data in memory."""
        bench = cls.__new__(cls)
        bench.config = config
        bench._frame = data
        return bench

    @property
    def n_candidates(self) -> int:
        """One candidate per declared reduction and model."""
        return len(self.config.reduction) * len(self.config.models)

    def validate(self) -> Plan:
        """Check the config and the data without training anything.

        Raises:
            ConfigError: unknown model, step or reduction, or a bad option.
            DataError: the data cannot be read, or cannot be used as declared.
        """
        candidates = build_candidates(self.config)  # checks models, steps and reductions
        if self._frame is not None:
            frame, source = self._frame, "the DataFrame"
        else:
            frame, source = load_dataframe(self.config.data.path), None
        X, y = split_features_target(frame, self.config.data, source=source)
        check_data(X, y, self.config)
        split_data(X, y, self.config)  # rare classes, too few rows for the folds
        return Plan(X, y, candidates)

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
        evaluated = evaluate(plan.candidates, plan.X, plan.y, self.config, on_result=on_result)
        results = Results(evaluated, self.config, duration_seconds=time.perf_counter() - start)
        if export:
            results.export()
        return results
