"""Measure how much two FitRank rankings agree.

For example the ranking on clean data and the ranking on noisy data::

    cmp = compare_rankings(clean_results, noisy_results)
    cmp.kendall_tau   # 1 = same order, 0 = unrelated, -1 = reversed
    cmp.top1_agree    # the same best candidate on both sides?
    cmp.regret        # test score lost, in the reference, by taking the other's winner
    cmp.table         # one row per candidate: both ranks and the shift

Candidates are matched by (model, reduction). Either side can be a
``Results``, a leaderboard ``DataFrame`` or a ``leaderboard.csv`` /
``leaderboard.json`` file of a run folder.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import get_scorer_names

from fitrank.evaluation import display_name
from fitrank.exceptions import FitRankError
from fitrank.results import Results

Ranking = Results | pd.DataFrame | str | Path

_REQUIRED = ["model", "reduction", "status"]


@dataclass
class RankingComparison:
    """How much two rankings agree; see ``compare_rankings``.

    Attributes:
        kendall_tau: Kendall's tau-b between the two orders of the common
            candidates: 1 = same order, -1 = reversed. Ties count as ties.
            NaN with fewer than 2 common candidates, or when every candidate
            ties on one side.
        top1_agree: True when both sides rank the same candidate first.
        regret: test score of the reference's winner minus the reference
            test score of the other side's winner, in the metric's unit and
            always "positive = lost" (an RMSE regret of 0.5 means 0.5 more
            error). 0 when the winners agree. It can be negative: winners are
            chosen on the CV score, so the other's winner may test better.
        table: one row per common candidate, in the reference order: model,
            reduction, reference_rank, other_rank, rank_shift (other minus
            reference: positive = ranked lower by the other side) and the two
            scores the ranks come from.
        metric: the primary metric both rankings use, e.g. ``f1_weighted``.
        by: ``"cv"`` or ``"test"``, the scores that were ranked.
        reference_best: the reference winner, e.g. ``"svm + none"``.
        other_best: the other side's winner.
        notes: candidates left out, ties at the top, why tau is NaN...
    """

    kendall_tau: float
    top1_agree: bool
    regret: float
    table: pd.DataFrame
    metric: str
    by: Literal["cv", "test"]
    reference_best: str
    other_best: str
    notes: list[str] = field(default_factory=list)


def compare_rankings(
    reference: Ranking, other: Ranking, *, by: Literal["cv", "test"] = "cv"
) -> RankingComparison:
    """Compare two rankings of the same candidates.

    Each side is a ``Results``, a leaderboard ``DataFrame``, or the path of
    a ``leaderboard.csv`` / ``leaderboard.json`` file. Candidates are matched
    by (model, reduction); one present on a single side, or ``failed`` on
    either side, is left out with a note.

    By default the candidates are ranked as in the leaderboard, on the
    cross-validation score; ``by="test"`` ranks them on the test score. A tie
    at the top is broken by the leaderboard order (a note says so).

    With 5 to 8 models, tau takes few values and moves from one random seed
    to another: before reading a change of tau as an effect of the data,
    compare it with the tau between two seeds on the same data.

    Raises:
        FitRankError: a side is not a leaderboard, the two sides use
            different primary metrics, a candidate appears twice on one
            side, or no candidate is usable on both sides.
    """
    if by not in ("cv", "test"):
        raise ValueError(f"by must be 'cv' or 'test', got {by!r}")
    ref_board, ref_metric = _leaderboard(reference, "reference")
    other_board, other_metric = _leaderboard(other, "other")
    if ref_metric != other_metric:
        raise FitRankError(
            "the two rankings use different primary metrics",
            [f"reference: {ref_metric}", f"other: {other_metric}"],
            hint="rank both with the same evaluation.primary_metric",
        )
    metric = ref_metric
    higher = _higher_is_better(metric)

    notes: list[str] = []
    ref_ok = _usable(ref_board, "reference", notes)
    other_ok = _usable(other_board, "other", notes)
    for key in ref_ok.index.difference(other_ok.index, sort=False):
        if key not in other_board.index:
            notes.append(f"{_label(key)}: only in reference, left out")
    for key in other_ok.index.difference(ref_ok.index, sort=False):
        if key not in ref_board.index:
            notes.append(f"{_label(key)}: only in other, left out")
    common = [key for key in ref_ok.index if key in other_ok.index]
    if not common:
        raise FitRankError(
            "no candidate to compare: none is ranked on both sides",
            notes or ["both leaderboards are empty"],
            hint="compare runs of the same models and reductions",
        )

    column = f"{by}_{metric}"
    ref_scores = ref_ok.loc[common, column].astype(float)
    other_scores = other_ok.loc[common, column].astype(float)
    ref_best = _winner(ref_scores, higher, "reference", notes)
    other_best = _winner(other_scores, higher, "other", notes)

    if len(common) < 2:
        tau = math.nan
        notes.append("kendall_tau is NaN: it needs at least 2 common candidates")
    elif ref_scores.nunique() == 1 or other_scores.nunique() == 1:
        tau = math.nan
        notes.append("kendall_tau is NaN: every candidate ties on one side")
    else:
        tau = float(kendalltau(ref_scores.to_numpy(), other_scores.to_numpy())[0])  # tau-b

    test = ref_ok.loc[common, f"test_{metric}"].astype(float)
    lost = test[ref_best] - test[other_best]
    regret = float(lost if higher else -lost)

    table = pd.DataFrame(
        {
            "model": [key[0] for key in common],
            "reduction": [key[1] for key in common],
            "reference_rank": _ranks(ref_scores, higher),
            "other_rank": _ranks(other_scores, higher),
            f"reference_{column}": ref_scores.to_numpy(),
            f"other_{column}": other_scores.to_numpy(),
        }
    )
    table.insert(4, "rank_shift", table.other_rank - table.reference_rank)
    return RankingComparison(
        kendall_tau=tau,
        top1_agree=ref_best == other_best,
        regret=regret,
        table=table,
        metric=metric,
        by=by,
        reference_best=_label(ref_best),
        other_best=_label(other_best),
        notes=notes,
    )


def _leaderboard(source: Ranking, side: str) -> tuple[pd.DataFrame, str]:
    """The leaderboard indexed by (model, reduction), in its order, and its metric."""
    if isinstance(source, Results):
        board, metric = source.leaderboard(), display_name(source.metric)
    else:
        board = source if isinstance(source, pd.DataFrame) else _read(Path(source), side)
        metric = _metric_of(board, side)
    missing = [c for c in [*_REQUIRED, f"cv_{metric}", f"test_{metric}"] if c not in board]
    if missing:
        raise FitRankError(
            f"the {side} ranking is not a FitRank leaderboard",
            [f"missing column(s): {', '.join(missing)}"],
            hint="pass a Results, results.leaderboard() or a run's leaderboard.csv",
        )
    if "rank" in board:
        board = board.sort_values("rank", kind="stable")
    board = board.set_index(["model", "reduction"])
    twice = board.index[board.index.duplicated()].unique()
    if len(twice):
        raise FitRankError(
            f"the {side} ranking has the same candidate twice",
            [f"{_label(key)} appears more than once" for key in twice],
            hint="compare_rankings matches candidates by (model, reduction)",
        )
    return board, metric


def _read(path: Path, side: str) -> pd.DataFrame:
    if not path.is_file():
        raise FitRankError(
            f"{side} leaderboard not found: {path.as_posix()}",
            ["give a run's leaderboard.csv or leaderboard.json"],
        )
    if path.suffix.lower() == ".json":
        return pd.read_json(path, orient="records")
    return pd.read_csv(path)


def _metric_of(board: pd.DataFrame, side: str) -> str:
    """``f1_weighted`` for a leaderboard with a ``cv_f1_weighted`` column."""
    names = [str(c)[3:] for c in board.columns if str(c).startswith("cv_") and c != "cv_std"]
    if len(names) != 1:
        raise FitRankError(
            f"the {side} ranking is not a FitRank leaderboard",
            ["expected one cv_<metric> column, e.g. cv_f1_weighted"],
            hint="pass a Results, results.leaderboard() or a run's leaderboard.csv",
        )
    return names[0]


def _higher_is_better(metric: str) -> bool:
    """False for errors such as root_mean_squared_error (neg_ scorers, shown positive)."""
    return f"neg_{metric}" not in get_scorer_names()


def _usable(board: pd.DataFrame, side: str, notes: list[str]) -> pd.DataFrame:
    ok = board["status"] == "ok"
    notes += [f"{_label(key)}: failed in {side}, left out" for key in board.index[~ok]]
    usable: pd.DataFrame = board[ok]
    return usable


def _winner(scores: pd.Series, higher: bool, side: str, notes: list[str]) -> tuple[str, str]:
    """The first candidate with the best score, in the leaderboard order."""
    best = scores.max() if higher else scores.min()
    tied: list[Any] = [key for key, value in scores.items() if value == best]
    if len(tied) > 1:
        names = ", ".join(_label(key) for key in tied)
        notes.append(
            f"{len(tied)} candidates tie first in {side} ({names}): {_label(tied[0])} is used"
        )
    winner: tuple[str, str] = tied[0]
    return winner


def _ranks(scores: pd.Series, higher: bool) -> list[int]:
    """1 = best; tied candidates share the best of their ranks (1, 2, 2, 4)."""
    return [int(r) for r in scores.rank(method="min", ascending=not higher)]


def _label(key: Any) -> str:
    """``svm + pca`` for the (model, reduction) key of a candidate."""
    model, reduction = key
    return f"{model} + {reduction}"
