"""The three examples still give the leaderboards recorded with FitRank 1.1.2.

Same order of candidates, same parameters, same notes, and the same scores to
1e-6. Run with the pinned versions of ``constraints.txt`` only, on Linux (the
"Non-regression" CI job): another platform or version can move the last digits.
Set ``FITRANK_GOLDEN=1`` to run them anyway, or to make a version mismatch fail
instead of skipping.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from golden import EXACT, EXAMPLES, GOLDEN, leaderboard, score_columns, version_mismatches

FORCED = os.environ.get("FITRANK_GOLDEN") == "1"
ATOL = 1e-6

pytestmark = pytest.mark.golden


def _skip_reason() -> str | None:
    mismatches = version_mismatches()
    if FORCED:
        assert not mismatches, "FITRANK_GOLDEN=1 but the versions are not pinned: " + ", ".join(
            mismatches
        )
        return None
    if mismatches:
        return "golden leaderboards need the pinned versions: " + ", ".join(mismatches)
    if sys.platform != "linux":
        return "golden leaderboards are compared on Linux (set FITRANK_GOLDEN=1 to force)"
    return None


def _table(board: pd.DataFrame, scores: list[str]) -> str:
    return board[["model", "reduction", *scores]].to_string(index=False)


@pytest.mark.parametrize("name", EXAMPLES)
def test_leaderboard_unchanged(name: str) -> None:
    if reason := _skip_reason():
        pytest.skip(reason)
    expected = pd.read_csv(GOLDEN / f"{name}.csv")
    got = leaderboard(name)
    scores = score_columns(expected)

    assert list(got.columns) == list(expected.columns), "the leaderboard columns changed"
    assert list(zip(got.model, got.reduction, strict=True)) == list(
        zip(expected.model, expected.reduction, strict=True)
    ), (
        f"the {name} ranking changed.\nexpected:\n{_table(expected, scores)}"
        f"\ngot:\n{_table(got, scores)}"
    )
    for column in EXACT:
        pd.testing.assert_series_equal(
            got[column].astype(object).where(got[column].notna(), None),
            expected[column].astype(object).where(expected[column].notna(), None),
            check_dtype=False,
            obj=f"{name} leaderboard, column {column!r}",
        )
    for column in scores:
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            atol=ATOL,
            rtol=0,
            err_msg=f"{name} leaderboard, column {column!r}",
        )
