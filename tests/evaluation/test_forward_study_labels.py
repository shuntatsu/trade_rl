"""FS1 label rule on fabricated numbers: every falsifier and every INVALID path."""

from __future__ import annotations

import math
from typing import Any

import pytest

from trade_rl.evaluation.forward_study.labels import (
    CandidateEvidence,
    classify_candidate,
)
from trade_rl.evaluation.forward_study.registry import FS1_REGISTRY


def evidence(**overrides: Any) -> CandidateEvidence:
    fields: dict[str, Any] = {
        "valid": True,
        "invalid_reasons": (),
        "max_drawdown_pct": {"base": 12.0, "cost_x2": 14.0},
        "excess_vs_cash_ci_low": 0.02,
        "excess_vs_cash_bonferroni_low": 0.01,
        "excess_vs_passive_ci_low": 0.03,
    }
    fields.update(overrides)
    return CandidateEvidence(**fields)


def label(item: CandidateEvidence, *, controls_valid: bool = True):  # type: ignore[no-untyped-def]
    return classify_candidate(
        item, controls_valid=controls_valid, registry=FS1_REGISTRY
    )


def test_pass_needs_drawdown_and_a_positive_cash_lower_bound() -> None:
    result = label(evidence())
    assert result.label == "FORWARD_PASS"
    assert result.failed_falsifiers == ()
    assert result.passive_status == "BETTER_THAN_PASSIVE"
    assert result.edge_claim_supported is True
    assert result.survives_bonferroni_k2 is True


@pytest.mark.parametrize("scenario", ["base", "cost_x2"])
def test_drawdown_above_twenty_percent_in_base_or_cost_x2_fails(scenario: str) -> None:
    drawdowns = {"base": 12.0, "cost_x2": 14.0, scenario: 20.0001}
    result = label(evidence(max_drawdown_pct=drawdowns))
    assert result.label == "FORWARD_FAIL"
    assert result.failed_falsifiers == (f"max_drawdown_above_limit_{scenario}",)
    assert result.edge_claim_supported is False


def test_exactly_twenty_percent_drawdown_is_not_a_failure() -> None:
    result = label(evidence(max_drawdown_pct={"base": 20.0, "cost_x2": 20.0}))
    assert result.label == "FORWARD_PASS"


@pytest.mark.parametrize("low", [0.0, -0.01])
def test_a_non_positive_cash_lower_bound_at_cost_x2_fails(low: float) -> None:
    result = label(evidence(excess_vs_cash_ci_low=low))
    assert result.label == "FORWARD_FAIL"
    assert result.failed_falsifiers == ("excess_vs_cash_ci_low_not_positive_cost_x2",)


def test_latency_drawdown_is_reported_but_is_not_a_falsifier() -> None:
    result = label(
        evidence(max_drawdown_pct={"base": 5.0, "cost_x2": 6.0, "latency_1bar": 55.0})
    )
    assert result.label == "FORWARD_PASS"


def test_not_better_than_passive_is_a_label_not_a_pass_of_the_edge_claim() -> None:
    result = label(evidence(excess_vs_passive_ci_low=0.0))
    assert result.label == "FORWARD_PASS"
    assert result.passive_status == "NO_BETTER_THAN_PASSIVE"
    assert result.edge_claim_supported is False


def test_bonferroni_is_reported_separately_from_the_unadjusted_label() -> None:
    result = label(evidence(excess_vs_cash_bonferroni_low=-0.001))
    assert result.label == "FORWARD_PASS"
    assert result.survives_bonferroni_k2 is False
    # The headline flag needs the family-wise bound too: PASS + BETTER + Bonferroni.
    assert result.passive_status == "BETTER_THAN_PASSIVE"
    assert result.edge_claim_supported is False
    missing = label(evidence(excess_vs_cash_bonferroni_low=None))
    assert missing.label == "FORWARD_PASS" and missing.survives_bonferroni_k2 is None
    assert missing.edge_claim_supported is False


def test_the_headline_edge_claim_needs_pass_passive_and_bonferroni_together() -> None:
    assert label(evidence()).edge_claim_supported is True
    assert label(evidence(excess_vs_passive_ci_low=-0.1)).edge_claim_supported is False
    assert (
        label(evidence(excess_vs_cash_bonferroni_low=-0.1)).edge_claim_supported
        is False
    )
    failing = label(evidence(excess_vs_cash_ci_low=-0.1))
    assert failing.label == "FORWARD_FAIL" and failing.edge_claim_supported is False


def test_both_falsifiers_are_listed_together() -> None:
    result = label(
        evidence(
            max_drawdown_pct={"base": 30.0, "cost_x2": 31.0},
            excess_vs_cash_ci_low=-0.5,
        )
    )
    assert result.label == "FORWARD_FAIL"
    assert len(result.failed_falsifiers) == 3


def test_an_invalid_arm_is_invalid_not_fail() -> None:
    result = label(
        evidence(
            valid=False,
            invalid_reasons=("ledger_not_reconciled_base",),
            max_drawdown_pct={"base": 99.0, "cost_x2": 99.0},
        )
    )
    assert result.label == "FORWARD_INVALID"
    assert result.invalid_reasons == ("ledger_not_reconciled_base",)
    assert result.failed_falsifiers == ()
    assert result.edge_claim_supported is False


def test_an_invalid_control_makes_every_candidate_invalid() -> None:
    result = label(evidence(), controls_valid=False)
    assert result.label == "FORWARD_INVALID"
    assert "control_invalid" in result.invalid_reasons


@pytest.mark.parametrize(
    "field",
    [
        "excess_vs_cash_ci_low",
        "excess_vs_passive_ci_low",
    ],
)
@pytest.mark.parametrize("bad", [None, math.nan, math.inf])
def test_a_missing_or_non_finite_number_is_invalid_never_pass(
    field: str, bad: float | None
) -> None:
    result = label(evidence(**{field: bad}))
    assert result.label == "FORWARD_INVALID"


def test_a_missing_drawdown_is_invalid() -> None:
    result = label(evidence(max_drawdown_pct={"base": 5.0, "cost_x2": None}))
    assert result.label == "FORWARD_INVALID"
    assert "missing_max_drawdown_cost_x2" in result.invalid_reasons


def test_the_label_vocabulary_has_no_winner() -> None:
    mapping = label(evidence()).to_mapping()
    assert mapping["label"] in FS1_REGISTRY.labels
    assert "WINNER" not in str(mapping)
