from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.ppo_interleaved_evaluation_prereg import (
    canonical_ppo_interleaved_evaluation_protocol,
    classify_ppo_interleaved_evaluation,
)


def test_protocol_binds_exact_calibrated_studyplan_ppo_identity() -> None:
    protocol = canonical_ppo_interleaved_evaluation_protocol()

    assert protocol.feature_names == (
        "1h__log_return_1bar",
        "1h__log_return_4bar",
        "1h__log_return_24bar",
        "1h__realized_volatility_24bar",
        "1h__volume_zscore_24bar",
        "1h__funding_bps",
        "1h__rsi_14bar",
        "1h__macd_histogram_12_26_9",
        "4h__log_return_4bar",
        "4h__realized_volatility_24bar",
        "1d__log_return_1bar",
        "1d__realized_volatility_24bar",
    )
    assert protocol.feature_indices == (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
    assert protocol.fit_symbol_names == protocol.symbols
    assert protocol.gross_budget == 0.5
    assert protocol.initial_capital == 100_000.0
    assert protocol.slippage_std == 0.0

    with pytest.raises(ValueError, match="feature_indices"):
        replace(protocol, feature_indices=(0, 1))
    with pytest.raises(ValueError, match="gross_budget"):
        replace(protocol, gross_budget=0.6)
    with pytest.raises(ValueError, match="slippage_std"):
        replace(protocol, slippage_std=0.0001)


def _valid_counts() -> dict[str, bool | int | float]:
    return {
        "evidence_valid": True,
        "positive_factor_effect_symbols": 5,
        "cross_symbol_median_factor_effect": 0.01,
        "positive_seed_median_excess": 5,
        "new_termination_cells": 0,
        "cross_symbol_median_candidate_return": 0.02,
        "positive_seed_candidate_medians": 5,
        "drawdown_nonworse_symbols": 5,
    }


def test_decision_rule_uses_strict_stage_a_zero_boundary() -> None:
    values = _valid_counts()
    values["cross_symbol_median_factor_effect"] = 0.0
    assert classify_ppo_interleaved_evaluation(**values) == "KEEP_SEQUENTIAL_BASELINE"


def test_any_new_termination_fails_robust_factor_gate() -> None:
    values = _valid_counts()
    values["new_termination_cells"] = 1
    assert classify_ppo_interleaved_evaluation(**values) == "KEEP_SEQUENTIAL_BASELINE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cross_symbol_median_candidate_return", 0.0),
        ("positive_seed_candidate_medians", 3),
        ("drawdown_nonworse_symbols", 3),
    ],
)
def test_stage_b_boundaries_retain_research_reference_only(
    field: str, value: int | float
) -> None:
    values = _valid_counts()
    values[field] = value
    assert (
        classify_ppo_interleaved_evaluation(**values)
        == "PROMOTE_RESEARCH_REFERENCE_ONLY"
    )


def test_decision_rule_validates_all_inputs_even_when_evidence_is_invalid() -> None:
    values = _valid_counts()
    values["evidence_valid"] = False
    values["positive_seed_candidate_medians"] = True
    with pytest.raises(ValueError, match="positive_seed_candidate_medians"):
        classify_ppo_interleaved_evaluation(**values)
