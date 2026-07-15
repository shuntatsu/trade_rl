from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.research_gate import (
    ResearchReturnGate,
)
from trade_rl.evaluation.research_gate import (
    evaluate_research_return_gate as _evaluate_research_return_gate,
)

RL_POLICY_DIGESTS = ("a" * 64, "b" * 64)


def evaluate_research_return_gate(**evidence: object) -> ResearchReturnGate:
    evidence.setdefault("selected_policy_digests", RL_POLICY_DIGESTS)
    return _evaluate_research_return_gate(**evidence)


def test_research_return_gate_passes_profitable_cost_adjusted_evidence() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.03,
        maximum_fold_drawdown=0.12,
    )

    assert result.thresholds == {
        "selected_mean_return_exclusive_minimum": 0.0,
        "baseline_uplift_minimum": 0.0,
        "maximum_independently_reset_fold_drawdown": 0.20,
    }
    assert result.observed == {
        "selected_mean_return": 0.08,
        "baseline_mean_return": 0.03,
        "baseline_uplift": pytest.approx(0.05),
        "maximum_independently_reset_fold_drawdown": 0.12,
        "selected_policy_digests": RL_POLICY_DIGESTS,
    }
    assert result.conditions == {
        "selected_mean_return_positive": True,
        "baseline_uplift_nonnegative": True,
        "maximum_fold_drawdown_within_limit": True,
        "rl_policy_selected_all_folds": True,
        "evidence_valid": True,
    }
    assert result.passed is True
    assert result.evidence_errors == ()


def test_research_return_gate_rejects_all_baseline_fold_fallbacks() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.08,
        maximum_fold_drawdown=0.12,
        selected_policy_digests=(None, None),
    )

    assert result.observed["selected_policy_digests"] == (None, None)
    assert result.conditions["rl_policy_selected_all_folds"] is False
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (
        "selected_policy_digests[0] must be a non-empty string",
        "selected_policy_digests[1] must be a non-empty string",
    )


def test_research_return_gate_rejects_one_baseline_fold_among_rl_folds() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.03,
        maximum_fold_drawdown=0.12,
        selected_policy_digests=("a" * 64, None, "c" * 64),
    )

    assert result.conditions["rl_policy_selected_all_folds"] is False
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (
        "selected_policy_digests[1] must be a non-empty string",
    )


@pytest.mark.parametrize("selected_mean_return", [0.0, -0.01])
def test_research_return_gate_requires_strictly_positive_selected_return(
    selected_mean_return: float,
) -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=selected_mean_return,
        baseline_mean_return=-0.01,
        maximum_fold_drawdown=0.10,
    )

    assert result.conditions["selected_mean_return_positive"] is False
    assert result.passed is False


def test_research_return_gate_rejects_negative_baseline_uplift() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.03,
        baseline_mean_return=0.04,
        maximum_fold_drawdown=0.10,
    )

    assert result.observed["baseline_uplift"] == pytest.approx(-0.01)
    assert result.conditions["baseline_uplift_nonnegative"] is False
    assert result.passed is False


def test_research_return_gate_rejects_drawdown_above_limit() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=0.01,
        maximum_fold_drawdown=0.200_001,
    )

    assert result.conditions["maximum_fold_drawdown_within_limit"] is False
    assert result.passed is False


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("selected_mean_return", math.nan),
        ("selected_mean_return", math.inf),
        ("baseline_mean_return", -math.inf),
        ("maximum_fold_drawdown", math.nan),
    ],
)
def test_research_return_gate_fails_closed_on_non_finite_evidence(
    field_name: str,
    value: float,
) -> None:
    evidence: dict[str, object] = {
        "selected_mean_return": 0.05,
        "baseline_mean_return": 0.01,
        "maximum_fold_drawdown": 0.10,
    }
    evidence[field_name] = value

    result = evaluate_research_return_gate(**evidence)

    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (f"{field_name} must be a finite number",)


def test_research_return_gate_accepts_nonnegative_uplift_and_drawdown_boundary() -> (
    None
):
    result = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=0.05,
        maximum_fold_drawdown=0.20,
    )

    assert result.conditions["baseline_uplift_nonnegative"] is True
    assert result.conditions["maximum_fold_drawdown_within_limit"] is True
    assert result.passed is True


@pytest.mark.parametrize("malformed", [None, "0.1", True, [], {}])
def test_research_return_gate_fails_closed_on_malformed_evidence(
    malformed: object,
) -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=malformed,
        baseline_mean_return=0.01,
        maximum_fold_drawdown=0.10,
    )

    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.observed["selected_mean_return"] is None


@pytest.mark.parametrize("drawdown", [-0.01, 1.01])
def test_research_return_gate_rejects_drawdown_outside_financial_range(
    drawdown: float,
) -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=0.01,
        maximum_fold_drawdown=drawdown,
    )

    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == ("maximum_fold_drawdown must be between 0 and 1",)


def test_research_return_gate_rejects_extreme_baseline_before_uplift() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=1e308,
        baseline_mean_return=-1e308,
        maximum_fold_drawdown=0.10,
    )

    assert result.observed["baseline_uplift"] is None
    assert result.conditions["baseline_uplift_nonnegative"] is False
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (
        "baseline_mean_return must be greater than or equal to -1",
    )


def test_research_return_gate_fails_closed_on_oversized_summary_integer() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=10**400,
        baseline_mean_return=0.01,
        maximum_fold_drawdown=0.10,
    )

    assert result.observed["selected_mean_return"] is None
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == ("selected_mean_return must be a finite number",)


def test_research_return_gate_rejects_selected_return_below_total_loss() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=-2.0,
        baseline_mean_return=-1.0,
        maximum_fold_drawdown=0.10,
    )

    assert result.observed["selected_mean_return"] is None
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (
        "selected_mean_return must be greater than or equal to -1",
    )


def test_research_return_gate_rejects_baseline_return_below_total_loss() -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=-2.0,
        maximum_fold_drawdown=0.10,
    )

    assert result.observed["baseline_mean_return"] is None
    assert result.observed["baseline_uplift"] is None
    assert result.conditions["evidence_valid"] is False
    assert result.passed is False
    assert result.evidence_errors == (
        "baseline_mean_return must be greater than or equal to -1",
    )


def test_research_return_gate_accepts_total_loss_return_boundaries_as_evidence() -> (
    None
):
    selected_boundary = evaluate_research_return_gate(
        selected_mean_return=-1.0,
        baseline_mean_return=-1.0,
        maximum_fold_drawdown=0.0,
    )
    baseline_boundary = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=-1.0,
        maximum_fold_drawdown=0.0,
    )

    assert selected_boundary.conditions["evidence_valid"] is True
    assert selected_boundary.passed is False
    assert selected_boundary.evidence_errors == ()
    assert baseline_boundary.conditions["evidence_valid"] is True
    assert baseline_boundary.passed is True
    assert baseline_boundary.evidence_errors == ()


@pytest.mark.parametrize("drawdown", [0.0, 1.0])
def test_research_return_gate_accepts_drawdown_domain_boundaries_as_evidence(
    drawdown: float,
) -> None:
    result = evaluate_research_return_gate(
        selected_mean_return=0.05,
        baseline_mean_return=0.01,
        maximum_fold_drawdown=drawdown,
    )

    assert result.conditions["evidence_valid"] is True
    assert result.evidence_errors == ()
