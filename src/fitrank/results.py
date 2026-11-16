"""Rank the evaluation results, save each run to disk and reload the best model.

Each run gets its own folder, so nothing is ever overwritten::

    runs/iris_2026-11-03_14-32-05/
    ├── config.yaml          the configuration as FitRank read it, defaults included
    ├── leaderboard.csv      \\
    ├── leaderboard.json      > the formats listed in output.formats
    ├── leaderboard.md       /
    ├── best_model.joblib    if output.save_best_model is true
    └── metadata.json        versions, date, duration, best candidate, rows, test set
"""

import json
import math
import platform
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from fitrank.config import FitRankConfig
from fitrank.evaluation import (
    CandidateResult,
    display_name,
    higher_is_better,
    make_folds,
    primary_metric,
    secondary_metrics,
)
from fitrank.exceptions import FitRankError, FitRankWarning
from fitrank.pipeline import build_candidates

FORMATS = {"csv": "leaderboard.csv", "json": "leaderboard.json", "markdown": "leaderboard.md"}
CONFIG_FILE = "config.yaml"
MODEL_FILE = "best_model.joblib"
METADATA_FILE = "metadata.json"


class Results:
    """The candidates of a run, best first (ranked on the CV score), failures last."""

    def __init__(
        self,
        results: Sequence[CandidateResult],
        config: FitRankConfig,
        duration_seconds: float | None = None,
        *,
        n_train: int | None = None,
        n_test: int | None = None,
        test_data: dict[str, str] | None = None,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
    ) -> None:
        self.config = config
        self.n_train = n_train  # rows used for the search and the CV
        self.n_test = n_test  # rows of the test set: test_size share, or all of test_data
        self.test_data = test_data  # with test_data: {"source": ..., "sha256": ...}
        self.X_train = X_train  # the training rows (search and CV), kept for oof_proba
        self.y_train = y_train
        self._oof: dict[tuple[str, str, str], pd.DataFrame] = {}  # oof_proba, once computed
        self.metric = primary_metric(config)
        ok = sorted(
            (r for r in results if r.status == "ok"),
            key=lambda r: r.cv_mean if r.cv_mean is not None else -math.inf,
            reverse=higher_is_better(self.metric),
        )  # sorted() is stable: ties keep the declaration order
        self.results = ok + [r for r in results if r.status != "ok"]
        if duration_seconds is None:
            duration_seconds = sum(r.fit_seconds for r in results)
        self.duration_seconds = duration_seconds
        self.run_dir: Path | None = None  # set by export()

    @property
    def best(self) -> CandidateResult | None:
        """The first-ranked candidate, or None when every candidate failed."""
        return next((r for r in self.results if r.status == "ok"), None)

    def leaderboard(self) -> pd.DataFrame:
        """One row per candidate, ranked; scores are shown positive (RMSE = 53.8).

        Score columns are named after the metrics: ``cv_f1_weighted``,
        ``test_f1_weighted``, ``test_accuracy``... Secondary metrics are
        test-set scores.
        """
        name = display_name(self.metric)
        extra = {f"test_{display_name(m)}": m for m in secondary_metrics(self.config)}
        rows = [
            {
                "rank": rank,
                "model": r.model_name,
                "reduction": r.reduction_name,
                "status": r.status,
                f"cv_{name}": _number(r.cv_mean),
                "cv_std": _number(r.cv_std),
                f"test_{name}": _number(r.test_score),
                **{column: _number(r.secondary.get(m)) for column, m in extra.items()},
                "fit_s": r.fit_seconds,
                "best_params": _short_params(
                    r.best_params, calibrated="(calibrated: " in r.model_name
                ),
                "error": r.error,
                "notes": "; ".join(r.notes) or None,
            }
            for rank, r in enumerate(self.results, start=1)
        ]
        columns = [
            "rank", "model", "reduction", "status", f"cv_{name}", "cv_std", f"test_{name}",
            *extra, "fit_s", "best_params", "error", "notes",
        ]  # fmt: skip
        board: pd.DataFrame = pd.DataFrame(rows, columns=columns)  # columns: even with no rows
        return board

    def best_model(self) -> Pipeline:
        """The first-ranked pipeline, refitted on the whole training set.

        Raises:
            FitRankError: every candidate failed.
        """
        best = self.best
        if best is None or best.best_estimator is None:
            raise FitRankError(
                "no model to return: every candidate failed",
                [f"{r.reduction_name} + {r.model_name}: {r.error}" for r in self.results],
            )
        return best.best_estimator

    def oof_proba(
        self,
        model: str,
        reduction: str | None = None,
        *,
        params: Literal["best", "default"] = "best",
    ) -> pd.DataFrame:
        """Out-of-fold class probabilities of one candidate, for every training row.

        Each row is predicted by a model fitted on the other CV folds, never on
        itself: the same pipeline (preprocessing, reduction, model), the same
        folds as the run's cross-validation and, by default, the best
        parameters found. Nothing is computed until this is called; the result
        is then kept, and a copy is returned.

        The rows are not fully out-of-sample: the best parameters were chosen
        with these same folds, so with every row (common practice, e.g. for
        cleanlab). ``params="default"`` is stricter: the parameters given a
        single value in ``params`` (fixed in advance) are kept, the searched
        ones fall back to the scikit-learn defaults.

        Args:
            model: the model name, as in the leaderboard.
            reduction: which reduction, when the model ran with several.
            params: ``"best"`` (the search's choice) or ``"default"``.

        Returns:
            One row per training row (same index as ``X_train``), one column
            per class (sorted), each row summing to 1.

        Raises:
            FitRankError: not a classification run, no training rows kept, an
                unknown, ambiguous or failed candidate, or a model without
                ``predict_proba`` (e.g. ``svm``: add ``calibrate: sigmoid``).
        """
        if self.config.task != "classification":
            raise FitRankError(
                "oof_proba is for classification only",
                [f"task is {self.config.task!r}: a regression model has no class probabilities"],
            )
        if params not in ("best", "default"):
            raise ValueError(f"params must be 'best' or 'default', got {params!r}")
        if self.X_train is None or self.y_train is None:
            raise FitRankError(
                "no training rows in these results",
                ["oof_proba needs the Results returned by Benchmark.run(), which keeps them"],
            )
        result = self._candidate(model, reduction)
        label = f"{result.model_name} + {result.reduction_name}"
        key = (result.model_name, result.reduction_name, params)
        if key not in self._oof:
            pipeline = self._pipeline(result, params)
            if not hasattr(pipeline, "predict_proba"):
                fixes = [
                    "calibrate it in the config (calibrate: sigmoid): a calibrated model"
                    " has predict_proba",
                    "or choose a model that has it, e.g. logistic_regression or random_forest",
                ]
                raise FitRankError(
                    f"{label} has no predict_proba: no probabilities to return", fixes
                )
            with warnings.catch_warnings():  # counted in the leaderboard notes already
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                proba = cross_val_predict(
                    pipeline,
                    self.X_train,
                    self.y_train,
                    cv=make_folds(self.config),
                    method="predict_proba",
                    n_jobs=self.config.project.n_jobs,
                )
            classes = np.unique(self.y_train)  # the column order of predict_proba
            self._oof[key] = pd.DataFrame(proba, index=self.X_train.index, columns=classes)
        copy: pd.DataFrame = self._oof[key].copy()  # the caller may change it
        return copy

    def _candidate(self, model: str, reduction: str | None) -> CandidateResult:
        """The one candidate named by ``model`` (and ``reduction``)."""
        matches = [
            r
            for r in self.results
            if r.model_name == model and reduction in (None, r.reduction_name)
        ]
        named = f"{model} + {reduction}" if reduction else model
        if not matches:
            names = ", ".join(f"{r.model_name} + {r.reduction_name}" for r in self.results)
            raise FitRankError(f"no candidate {named!r} in these results", [f"candidates: {names}"])
        if len(matches) > 1:
            reductions = [r.reduction_name for r in matches]
            if reduction is None and len(set(reductions)) == len(reductions):
                raise FitRankError(
                    f"{model!r} ran with {len(matches)} reductions: choose one",
                    [f"oof_proba({model!r}, reduction={name!r})" for name in reductions],
                )
            raise FitRankError(
                f"{named!r} is declared more than once in models",
                ["oof_proba finds a candidate by its model and reduction names"],
            )
        result = matches[0]
        if result.status != "ok" or result.best_estimator is None:
            raise FitRankError(
                f"{result.model_name} + {result.reduction_name} failed: no model to predict with",
                [str(result.error)],
            )
        return result

    def _pipeline(self, result: CandidateResult, params: str) -> Pipeline:
        """An unfitted copy of the candidate's pipeline, with the chosen parameters."""
        if params == "best":
            best: Pipeline = clone(result.best_estimator)
            return best
        candidate = next(
            c
            for c in build_candidates(self.config)
            if (c.model_name, c.reduction_name) == (result.model_name, result.reduction_name)
        )
        fixed = {
            name: values[0] for name, values in candidate.param_grid.items() if len(values) == 1
        }
        default: Pipeline = candidate.pipeline.set_params(**fixed)
        return default

    def export(
        self, output_dir: str | Path | None = None, formats: Sequence[str] | None = None
    ) -> Path:
        """Write the run to a new timestamped folder inside ``output_dir`` and return it.

        ``output_dir`` and ``formats`` default to ``project.output_dir`` and
        ``output.formats``. The best model is saved when
        ``output.save_best_model`` is true and at least one candidate succeeded.
        """
        chosen = list(self.config.output.formats if formats is None else formats)
        unknown = [f for f in chosen if f not in FORMATS]
        if unknown:
            raise ValueError(f"unknown format(s) {unknown}; choose from {', '.join(FORMATS)}")

        now = datetime.now().astimezone()
        root = Path(self.config.project.output_dir if output_dir is None else output_dir)
        run_dir = _new_run_dir(root, self.config.project.name, now)

        _write(run_dir / CONFIG_FILE, yaml.safe_dump(_config_dict(self.config), sort_keys=False))
        table = self.leaderboard()
        if "csv" in chosen:
            flat = table.assign(best_params=[json.dumps(p, default=str) for p in table.best_params])
            flat.to_csv(run_dir / FORMATS["csv"], index=False)
        if "json" in chosen:
            table.to_json(run_dir / FORMATS["json"], orient="records", indent=2, force_ascii=False)
        if "markdown" in chosen:
            _write(run_dir / FORMATS["markdown"], self._markdown(table))
        if self.config.output.save_best_model and self.best is not None:
            joblib.dump(self.best_model(), run_dir / MODEL_FILE)
        _write(run_dir / METADATA_FILE, json.dumps(self._metadata(now), indent=2) + "\n")
        self.run_dir = run_dir
        return run_dir

    def _markdown(self, table: pd.DataFrame) -> str:
        direction = "higher" if higher_is_better(self.metric) else "lower"
        lines = [
            f"# FitRank leaderboard: {self.config.project.name}",
            "",
            f"Task: {self.config.task}. Ranked on the cross-validation"
            f" {display_name(self.metric)} ({direction} is better).",
            "",
            "| " + " | ".join(table.columns) + " |",
            "|" + "---|" * len(table.columns),
        ]
        for row in table.itertuples(index=False):
            lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
        return "\n".join(lines) + "\n"

    def _metadata(self, now: datetime) -> dict[str, Any]:
        from fitrank import __version__  # not at the top: fitrank/__init__.py imports this module

        best = self.best
        return {
            "created_at": now.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration_seconds, 3),
            "task": self.config.task,
            "primary_metric": self.metric,
            "best": None
            if best is None
            else {"model": best.model_name, "reduction": best.reduction_name, "cv": best.cv_mean},
            "rows": {"train": self.n_train, "test": self.n_test},
            "test_data": self.test_data,  # null when the test set was split from data.path
            "versions": {
                "python": platform.python_version(),
                "fitrank": __version__,
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        }


def load_model(path: str | Path) -> Pipeline:
    """Load a model saved by a run, ready to ``predict`` on new data.

    Warns (``FitRankWarning``) when the scikit-learn version differs from the
    one recorded in the ``metadata.json`` next to the model. Only load files
    you trust: loading a joblib file can run code.

    Raises:
        FitRankError: the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FitRankError(
            f"model file not found: {path.as_posix()}",
            [f"a run saves it as <run folder>/{MODEL_FILE} when output.save_best_model is true"],
        )
    _check_version(path)
    model: Pipeline = joblib.load(path)
    return model


def _check_version(model_path: Path) -> None:
    try:
        metadata = json.loads(model_path.with_name(METADATA_FILE).read_text(encoding="utf-8"))
        saved = metadata["versions"]["scikit-learn"]
    except (OSError, ValueError, KeyError, TypeError):
        warnings.warn(
            f"cannot check the scikit-learn version of {model_path.name}:"
            f" no readable {METADATA_FILE} next to it",
            FitRankWarning,
            stacklevel=3,
        )
        return
    if saved != sklearn.__version__:
        warnings.warn(
            f"{model_path.name} was saved with scikit-learn {saved}, but {sklearn.__version__}"
            f" is installed; predictions may differ or fail."
            f" Install scikit-learn=={saved}, or train the model again.",
            FitRankWarning,
            stacklevel=3,
        )


def _new_run_dir(root: Path, name: str, now: datetime) -> Path:
    safe_name = re.sub(r"[^\w.-]+", "-", name).strip("-") or "run"
    stem = f"{safe_name}_{now:%Y-%m-%d_%H-%M-%S}"
    attempt = 1
    while True:  # two runs in the same second get _2, _3...
        run_dir = root / (stem if attempt == 1 else f"{stem}_{attempt}")
        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:
            attempt += 1
        except OSError as exc:
            raise FitRankError(
                f"cannot create the run folder in {root.as_posix()}",
                [exc.strerror or str(exc)],
                hint="change project.output_dir in the config",
            ) from None


def _config_dict(config: FitRankConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json", by_alias=True)  # models[].class, not class_
    for model in data["models"]:  # keep only the key that names the model
        for key in ("name", "class", "calibrate"):  # calibrate: a v1.1 model, key for key
            if model[key] is None:
                del model[key]
    data["data"]["path"] = config.data.path.as_posix()  # same file on every OS
    data["project"]["output_dir"] = config.project.output_dir.as_posix()
    test_data = config.evaluation.test_data
    if test_data is not None:  # test_size does not apply, and both together are refused
        del data["evaluation"]["test_size"]
        data["evaluation"]["test_data"] = test_data.as_posix()
    else:
        del data["evaluation"]["test_data"]  # a v1.1 config.yaml, key for key
    return data


def _short_params(params: dict[str, Any], *, calibrated: bool = False) -> dict[str, Any]:
    """``{"model__C": 1}`` -> ``{"C": 1}``, unless two steps share a parameter name.

    For a calibrated model, ``model__estimator__C`` -> ``C`` too.
    """
    short = {}
    for key, value in params.items():
        if calibrated:
            key = key.replace("model__estimator__", "model__", 1)
        short[key.split("__", 1)[-1]] = value
    return short if len(short) == len(params) else dict(params)


def _number(value: float | None) -> float:
    return math.nan if value is None else value


def _cell(value: Any) -> str:
    if isinstance(value, dict):
        text = ", ".join(f"{key}={val}" for key, val in value.items())
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        text = "—"
    elif isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)
    return text.replace("|", "\\|")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
