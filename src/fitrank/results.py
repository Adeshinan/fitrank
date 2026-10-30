"""Rank the evaluation results, save each run to disk and reload the best model.

Each run gets its own folder, so nothing is ever overwritten::

    runs/iris_2026-11-03_14-32-05/
    ├── config.yaml          the configuration as FitRank read it, defaults included
    ├── leaderboard.csv      \\
    ├── leaderboard.json      > the formats listed in output.formats
    ├── leaderboard.md       /
    ├── best_model.joblib    if output.save_best_model is true
    └── metadata.json        versions, date, duration, best candidate
"""

import json
import math
import platform
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.pipeline import Pipeline

from fitrank import __version__
from fitrank.config import FitRankConfig
from fitrank.evaluation import (
    CandidateResult,
    display_name,
    higher_is_better,
    primary_metric,
    secondary_metrics,
)
from fitrank.exceptions import FitRankError, FitRankWarning

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
    ) -> None:
        self.config = config
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
                "best_params": _short_params(r.best_params),
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
        best = self.best
        return {
            "created_at": now.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration_seconds, 3),
            "task": self.config.task,
            "primary_metric": self.metric,
            "best": None
            if best is None
            else {"model": best.model_name, "reduction": best.reduction_name, "cv": best.cv_mean},
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
    data = config.model_dump(mode="json")
    data["data"]["path"] = config.data.path.as_posix()  # same file on every OS
    data["project"]["output_dir"] = config.project.output_dir.as_posix()
    return data


def _short_params(params: dict[str, Any]) -> dict[str, Any]:
    """``{"model__C": 1}`` -> ``{"C": 1}``, unless two steps share a parameter name."""
    short = {key.split("__", 1)[-1]: value for key, value in params.items()}
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
