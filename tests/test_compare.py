"""Tests for compare_rankings, on leaderboards built by hand: no training."""

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scipy.stats import kendalltau
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fitrank import FitRankError, RankingComparison, Results, compare_rankings
from fitrank.config import FitRankConfig
from fitrank.evaluation import CandidateResult

# (model, cv score, test score), best first; reduction "none", status "ok".
REFERENCE = [("svm", 0.95, 0.94), ("random_forest", 0.93, 0.92), ("knn", 0.90, 0.91),
             ("logistic_regression", 0.88, 0.86)]  # fmt: skip


def board(rows: list[tuple[Any, ...]], metric: str = "f1_weighted") -> pd.DataFrame:
    """A leaderboard like Results.leaderboard(), from (model, cv, test[, status, reduction])."""
    records = []
    for rank, (model, cv, test, *rest) in enumerate(rows, start=1):
        status = rest[0] if rest else "ok"
        reduction = rest[1] if len(rest) > 1 else "none"
        records.append({
            "rank": rank, "model": model, "reduction": reduction, "status": status,
            f"cv_{metric}": cv, "cv_std": 0.01, f"test_{metric}": test, "fit_s": 0.1,
            "best_params": {}, "error": None, "notes": None,
        })  # fmt: skip
    return pd.DataFrame(records)


def shuffled(order: list[str]) -> pd.DataFrame:
    """The REFERENCE candidates ranked in ``order``, with decreasing cv scores."""
    tests = {model: test for model, _, test in REFERENCE}
    return board([(m, 0.9 - 0.01 * i, tests[m]) for i, m in enumerate(order)])


# --- the three numbers -----------------------------------------------------------------------


def test_identical_rankings_give_tau_1() -> None:
    cmp = compare_rankings(board(REFERENCE), board(REFERENCE))

    assert isinstance(cmp, RankingComparison)
    assert cmp.kendall_tau == pytest.approx(1.0)
    assert cmp.top1_agree is True
    assert cmp.regret == 0
    assert cmp.table.rank_shift.tolist() == [0, 0, 0, 0]
    assert cmp.reference_best == cmp.other_best == "svm + none"
    assert cmp.notes == []


def test_reversed_rankings_give_tau_minus_1() -> None:
    order = [model for model, *_ in reversed(REFERENCE)]
    cmp = compare_rankings(board(REFERENCE), shuffled(order))

    assert cmp.kendall_tau == pytest.approx(-1.0)
    assert cmp.top1_agree is False
    assert cmp.other_best == "logistic_regression + none"
    assert cmp.regret == pytest.approx(0.94 - 0.86)  # svm's test score - logistic's, in ref
    assert cmp.table.rank_shift.tolist() == [3, 1, -1, -3]


def test_one_swap_gives_the_kendall_value() -> None:
    cmp = compare_rankings(
        board(REFERENCE), shuffled(["svm", "knn", "random_forest", "logistic_regression"])
    )

    assert cmp.kendall_tau == pytest.approx(4 / 6)  # 5 concordant pairs, 1 discordant
    assert cmp.top1_agree is True
    assert cmp.regret == 0


def test_regret_is_zero_when_same_winner() -> None:
    cmp = compare_rankings(
        board(REFERENCE), shuffled(["svm", "logistic_regression", "knn", "random_forest"])
    )
    assert cmp.top1_agree is True
    assert cmp.regret == 0
    assert cmp.kendall_tau < 1


def test_regret_uses_the_reference_test_scores() -> None:
    """The other side's own test scores do not matter: the loss is measured in the reference."""
    other = board([("random_forest", 0.99, 0.10), ("svm", 0.50, 0.99), ("knn", 0.4, 0.5),
                   ("logistic_regression", 0.3, 0.5)])  # fmt: skip
    cmp = compare_rankings(board(REFERENCE), other)
    assert cmp.regret == pytest.approx(0.94 - 0.92)


def test_regret_is_positive_for_an_error_metric() -> None:
    metric = "root_mean_squared_error"  # lower is better
    ref = board([("ridge", 3.0, 3.1), ("lasso", 3.5, 3.6)], metric)
    other = board([("lasso", 2.9, 3.0), ("ridge", 3.2, 3.3)], metric)
    cmp = compare_rankings(ref, other)

    assert cmp.regret == pytest.approx(0.5)  # 0.5 more error by taking lasso
    assert cmp.kendall_tau == pytest.approx(-1.0)
    assert cmp.table.reference_rank.tolist() == [1, 2]


# --- ties --------------------------------------------------------------------------------------


def test_ties_are_handled() -> None:
    """Iris: decision_tree and naive_bayes tie; tau-b counts the tie, without an error."""
    ref = board([("svm", 0.97, 0.96), ("decision_tree", 0.9496, 0.93),
                 ("naive_bayes", 0.9496, 0.93), ("knn", 0.94, 0.95)])  # fmt: skip
    other = board([("svm", 0.96, 0.96), ("naive_bayes", 0.95, 0.93), ("knn", 0.94, 0.95),
                   ("decision_tree", 0.93, 0.93)])  # fmt: skip
    cmp = compare_rankings(ref, other)

    expected = kendalltau([0.97, 0.9496, 0.9496, 0.94], [0.96, 0.93, 0.95, 0.94])[0]
    assert cmp.kendall_tau == pytest.approx(expected)
    assert cmp.table.reference_rank.tolist() == [1, 2, 2, 4]  # tied candidates share a rank


def test_tie_at_the_top_uses_the_leaderboard_order_with_a_note() -> None:
    ref = board([("knn", 0.95, 0.90), ("svm", 0.95, 0.94), ("logistic_regression", 0.90, 0.88)])
    other = board([("svm", 0.96, 0.94), ("knn", 0.95, 0.90), ("logistic_regression", 0.90, 0.88)])
    cmp = compare_rankings(ref, other)

    assert cmp.reference_best == "knn + none"  # first of the tie in the reference leaderboard
    assert cmp.top1_agree is False
    assert cmp.regret == pytest.approx(0.90 - 0.94)  # negative: the other's winner tests better
    assert cmp.notes == [
        "2 candidates tie first in reference (knn + none, svm + none): knn + none is used"
    ]


def test_every_candidate_tied_gives_nan_with_a_note() -> None:
    ref = board([("knn", 0.9, 0.9), ("svm", 0.9, 0.8)])
    cmp = compare_rankings(ref, board(REFERENCE[:1] + [("knn", 0.5, 0.5)]))
    assert math.isnan(cmp.kendall_tau)
    assert "kendall_tau is NaN: every candidate ties on one side" in cmp.notes


# --- matching candidates -------------------------------------------------------------------


def test_failed_candidates_are_ignored_with_a_note() -> None:
    other = board([("svm", 0.95, 0.94), ("random_forest", None, None, "failed"),
                   ("knn", 0.90, 0.91), ("logistic_regression", 0.88, 0.86)])  # fmt: skip
    cmp = compare_rankings(board(REFERENCE), other)

    assert cmp.table.model.tolist() == ["svm", "knn", "logistic_regression"]
    assert cmp.kendall_tau == pytest.approx(1.0)
    assert cmp.notes == ["random_forest + none: failed in other, left out"]


def test_candidates_on_one_side_only_are_ignored_with_a_note() -> None:
    other = board([("svm", 0.95, 0.94), ("knn", 0.90, 0.91), ("gradient_boosting", 0.89, 0.9)])
    cmp = compare_rankings(board(REFERENCE), other)

    assert cmp.table.model.tolist() == ["svm", "knn"]
    assert cmp.notes == [
        "random_forest + none: only in reference, left out",
        "logistic_regression + none: only in reference, left out",
        "gradient_boosting + none: only in other, left out",
    ]


def test_candidates_are_matched_by_model_and_reduction() -> None:
    ref = board([("svm", 0.95, 0.94, "ok", "pca"), ("svm", 0.93, 0.92, "ok", "none")])
    other = board([("svm", 0.96, 0.92, "ok", "none"), ("svm", 0.91, 0.94, "ok", "pca")])
    cmp = compare_rankings(ref, other)

    assert cmp.table[["model", "reduction"]].values.tolist() == [["svm", "pca"], ["svm", "none"]]
    assert cmp.other_best == "svm + none"
    assert cmp.kendall_tau == pytest.approx(-1.0)


def test_a_single_common_candidate_gives_nan_with_a_note() -> None:
    cmp = compare_rankings(board(REFERENCE[:1]), board(REFERENCE[:1]))
    assert math.isnan(cmp.kendall_tau)
    assert cmp.top1_agree is True
    assert "kendall_tau is NaN: it needs at least 2 common candidates" in cmp.notes


# --- inputs ------------------------------------------------------------------------------------


def test_by_test_ranks_on_the_test_scores() -> None:
    cmp = compare_rankings(board(REFERENCE), board(REFERENCE), by="test")
    assert cmp.by == "test"
    assert cmp.table.reference_rank.tolist() == [1, 2, 3, 4]  # cv order...
    assert cmp.table["reference_test_f1_weighted"].tolist() == [0.94, 0.92, 0.91, 0.86]

    swapped = board([("svm", 0.95, 0.94), ("random_forest", 0.93, 0.90), ("knn", 0.90, 0.91),
                     ("logistic_regression", 0.88, 0.86)])  # fmt: skip
    assert compare_rankings(board(REFERENCE), swapped).kendall_tau == pytest.approx(1.0)
    assert compare_rankings(board(REFERENCE), swapped, by="test").kendall_tau < 1


def test_accepts_results_and_exported_leaderboards(tmp_path: Path) -> None:
    config = FitRankConfig.model_validate({
        "data": {"path": "iris.csv", "target": "species"}, "task": "classification",
        "models": [{"name": "svm"}, {"name": "knn"}, {"name": "random_forest"}],
    })  # fmt: skip
    results = Results([_result(m, cv) for m, cv, _ in REFERENCE[:3]], config)
    run_dir = results.export(tmp_path)

    for other in [
        results.leaderboard(),
        run_dir / "leaderboard.csv",
        str(run_dir / "leaderboard.json"),
    ]:
        cmp = compare_rankings(results, other)
        assert cmp.kendall_tau == pytest.approx(1.0), other
        assert cmp.metric == "f1_weighted"
        assert cmp.table.model.tolist() == ["svm", "random_forest", "knn"]


def test_different_metrics_are_an_error() -> None:
    with pytest.raises(FitRankError) as caught:
        compare_rankings(board(REFERENCE), board(REFERENCE, "accuracy"))
    assert "the two rankings use different primary metrics" in str(caught.value)
    assert caught.value.details == ["reference: f1_weighted", "other: accuracy"]


def test_not_a_leaderboard_is_an_error() -> None:
    with pytest.raises(FitRankError, match="the other ranking is not a FitRank leaderboard"):
        compare_rankings(board(REFERENCE), pd.DataFrame({"model": ["svm"], "score": [0.9]}))
    with pytest.raises(FitRankError, match="missing column"):
        compare_rankings(board(REFERENCE).drop(columns="status"), board(REFERENCE))


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FitRankError, match="other leaderboard not found"):
        compare_rankings(board(REFERENCE), tmp_path / "leaderboard.csv")


def test_same_candidate_twice_is_an_error() -> None:
    with pytest.raises(FitRankError, match="has the same candidate twice"):
        compare_rankings(board(REFERENCE + [("svm", 0.5, 0.5)]), board(REFERENCE))


def test_no_common_candidate_is_an_error() -> None:
    with pytest.raises(FitRankError, match="no candidate to compare"):
        compare_rankings(board(REFERENCE[:2]), board(REFERENCE[2:]))


def test_by_must_be_cv_or_test() -> None:
    with pytest.raises(ValueError, match="by must be 'cv' or 'test'"):
        compare_rankings(board(REFERENCE), board(REFERENCE), by="train")  # type: ignore[arg-type]


def _result(model: str, cv: float) -> CandidateResult:
    return CandidateResult(
        model_name=model, reduction_name="none", status="ok", cv_mean=cv, cv_std=0.01,
        test_score=cv - 0.01, secondary={}, best_params={}, fit_seconds=0.1, error=None,
        best_estimator=Pipeline([("model", DummyClassifier())]),
    )  # fmt: skip
