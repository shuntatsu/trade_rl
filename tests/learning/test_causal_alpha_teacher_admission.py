from __future__ import annotations

import pytest

from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)


def _metric(symbol: str, gross: float, *, net: float | None = None):
    return CausalAlphaTeacherHoldoutMetric(
        symbol=symbol,
        gross_return=gross,
        net_return=gross if net is None else net,
        turnover_per_day=1.0,
        total_execution_cost=0.1,
        trade_count=2,
        maximum_drawdown=-0.05,
    )


def test_teacher_admission_accepts_nonnegative_aggregate_without_majority_losses() -> (
    None
):
    metrics = tuple(
        _metric(f"S{index}", gross)
        for index, gross in enumerate(
            (-0.01, -0.01, -0.01, -0.01, 0.10, 0.10, 0.10, 0.10, 0.10)
        )
    )
    evidence = evaluate_causal_alpha_teacher_admission(metrics)

    assert evidence.passed is True
    assert evidence.negative_gross_symbol_count == 4
    assert evidence.aggregate_gross_return == pytest.approx(0.46)
    assert evidence.rejection_reasons == ()
    assert len(evidence.digest) == 64


def test_teacher_admission_rejects_negative_aggregate_gross() -> None:
    metrics = tuple(_metric(f"S{index}", -0.01) for index in range(9))
    evidence = evaluate_causal_alpha_teacher_admission(metrics)

    assert evidence.passed is False
    assert "negative_aggregate_gross_return" in evidence.rejection_reasons
    assert "majority_negative_gross_holdouts" in evidence.rejection_reasons


def test_teacher_admission_rejects_one_winner_carrying_majority_losers() -> None:
    metrics = tuple(
        _metric(f"S{index}", gross)
        for index, gross in enumerate(
            (-0.01, -0.01, -0.01, -0.01, -0.01, 0.10, 0.10, 0.10, 0.10)
        )
    )
    evidence = evaluate_causal_alpha_teacher_admission(metrics)

    assert evidence.aggregate_gross_return > 0.0
    assert evidence.negative_gross_symbol_count == 5
    assert evidence.passed is False
    assert evidence.rejection_reasons == ("majority_negative_gross_holdouts",)


def test_teacher_admission_requires_unique_symbol_holdouts() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_causal_alpha_teacher_admission(
            (_metric("AAAUSDT", 0.1), _metric("AAAUSDT", 0.2))
        )
