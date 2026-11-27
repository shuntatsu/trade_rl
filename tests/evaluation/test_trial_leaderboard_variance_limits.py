"""Synthetic public leaderboard oracles at binary64 variance boundaries."""

from __future__ import annotations

from fractions import Fraction

import pytest

from trade_rl.evaluation.comparison.leaderboard import (
    VarianceSource,
    build_family_leaderboard,
)
from trade_rl.evaluation.comparison.multiple_testing import SharpeMoments
from trade_rl.evaluation.comparison.trial_ledger import (
    TrialLedger,
    TrialLedgerError,
    TrialRecord,
    TrialStatus,
)


def _ledger(values: list[float]) -> TrialLedger:
    # These are synthetic moment summaries, not measured returns. Use n=2,
    # skew=0, kurtosis=1 to avoid unrelated overflow in the PSR radicand.
    return TrialLedger(
        [
            TrialRecord(
                str(index),
                "family",
                "synthetic",
                "b" * 64,
                TrialStatus.COMPLETED,
                performance=SharpeMoments(value, 2, 0.0, 1.0),
            )
            for index, value in enumerate(values)
        ]
        + [
            TrialRecord(
                "failed",
                "family",
                "synthetic",
                "b" * 64,
                TrialStatus.FAILED,
                reason="synthetic failure",
            )
        ]
    )


def _exact_variance(values: list[float]) -> Fraction:
    exact = [Fraction(value) for value in values]
    mean = sum(exact) / len(exact)
    return sum((value - mean) ** 2 for value in exact) / (len(exact) - 1)


@pytest.mark.parametrize(
    "values",
    [
        [-1e154, 0.0, 1e154],
        [1e308, 1e308],
        [0.05, 0.10, 0.15],
        [0.0, 0.0],
        [0.0, 3e-162],
        [-1e-160, 1e-160],
    ],
)
def test_completed_variance_matches_exact_rational_oracle(values: list[float]) -> None:
    ledger = _ledger(values)
    original = ledger.digest()
    board = build_family_leaderboard(ledger, "family")
    assert board.trial_sr_variance == float(_exact_variance(values))
    assert board.variance_source is VarianceSource.COMPLETED_TRIALS
    assert board.n_trials == len(values) + 1
    assert board.ledger_digest == ledger.digest() == original
    assert len(board.entries) == len(values)
    if len(set(values)) > 1:
        assert board.trial_sr_variance > 0.0
        assert board.expected_max_sharpe > 0.0
    else:
        assert board.expected_max_sharpe == 0.0


@pytest.mark.parametrize("values", [[-1e308, 1e308], [-1e-200, 1e-200]])
def test_unrepresentable_completed_variance_refuses(values: list[float]) -> None:
    assert _exact_variance(values) > 0
    with pytest.raises(TrialLedgerError, match="trial_sr_variance"):
        build_family_leaderboard(_ledger(values), "family")


@pytest.mark.parametrize("values", [[], [0.1]])
@pytest.mark.parametrize(
    "supplied", [10**400, Fraction(1, 10**400), Fraction(-1, 10**400)]
)
def test_unrepresentable_supplied_variance_refuses(
    values: list[float], supplied: int | Fraction
) -> None:
    with pytest.raises(TrialLedgerError, match="trial_sr_variance"):
        build_family_leaderboard(
            _ledger(values),
            "family",
            trial_sr_variance=supplied,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("values", [[], [0.1]])
@pytest.mark.parametrize("supplied", [0, Fraction(1, 10)])
def test_representable_supplied_variance_unchanged(
    values: list[float], supplied: int | Fraction
) -> None:
    board = build_family_leaderboard(
        _ledger(values),
        "family",
        trial_sr_variance=supplied,  # type: ignore[arg-type]
    )
    assert board.variance_source is VarianceSource.SUPPLIED
    assert board.trial_sr_variance == float(supplied)
    assert board.n_trials == len(values) + 1
