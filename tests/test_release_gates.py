import math

import pytest

from mars_lite.pipeline.gates import (
    diagnostic_baseline,
    evaluate_baseline_only_gate,
    evaluate_direct_gate2,
    evaluate_residual_gate2,
)


def _metrics(total_return: float, max_drawdown: float = 0.1) -> dict[str, float]:
    return {"total_return": total_return, "max_drawdown": max_drawdown}


def test_oracles_are_diagnostic_only() -> None:
    assert diagnostic_baseline("oracle_dp") is True
    assert diagnostic_baseline("oracle_ic0.05") is True
    assert diagnostic_baseline("trend_following") is False


def test_direct_gate2_ignores_oracle_and_alternative_strategy_losses() -> None:
    result = evaluate_direct_gate2(
        agent=_metrics(0.12),
        baselines={
            "flat": _metrics(0.0),
            "trend_following": _metrics(0.08),
            "trend_v2": _metrics(0.20),
            "oracle_dp": _metrics(99_999.0),
        },
    )

    assert result["passed"] is True
    assert result["mandatory_comparisons"] == ("flat", "trend_following")
    assert result["details"]["oracle_dp"]["mandatory"] is False
    assert result["details"]["trend_v2"]["diagnostic_only"] is True


def test_residual_gate2_ignores_oracle_results() -> None:
    result = evaluate_residual_gate2(
        hybrid=_metrics(0.12, 0.14),
        shadow=_metrics(0.08, 0.10),
        flat=_metrics(0.0, 0.0),
        paired_p_value=0.01,
        diagnostic_results={"oracle_dp": _metrics(99_999.0, 0.0)},
        max_drawdown_slack=0.05,
    )

    assert result["passed"] is True
    assert result["mandatory_comparisons"] == ("flat", "shadow")
    assert result["diagnostic_results"]["oracle_dp"]["mandatory"] is False


def test_residual_gate2_requires_hybrid_to_beat_shadow() -> None:
    result = evaluate_residual_gate2(
        hybrid=_metrics(0.07),
        shadow=_metrics(0.08),
        flat=_metrics(0.0),
        paired_p_value=0.01,
    )

    assert result["passed"] is False
    assert result["checks"]["beats_shadow"] is False


def test_baseline_only_gate_does_not_compare_baseline_with_itself() -> None:
    result = evaluate_baseline_only_gate(
        trend_development_gate={"passed": True},
        holdout=_metrics(0.09, 0.12),
        cost2x_holdout=_metrics(0.01, 0.16),
        positive_return_p_value=0.02,
        max_drawdown_limit=0.20,
    )

    assert result["passed"] is True
    assert "beats_shadow" not in result["checks"]
    assert result["candidate_mode"] == "baseline_only"


def test_gate_rejects_non_finite_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_residual_gate2(
            hybrid=_metrics(math.nan),
            shadow=_metrics(0.1),
            flat=_metrics(0.0),
            paired_p_value=0.01,
        )
