from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_evaluation import (
    MeanReversionEconomicGateEvaluation,
    MeanReversionEconomicGateSymbolResult,
    canonical_mean_reversion_economic_gate_evaluation_spec,
    research_status_from_counts,
)


def _symbol_result(symbol: str) -> MeanReversionEconomicGateSymbolResult:
    return MeanReversionEconomicGateSymbolResult(
        symbol=symbol,
        baseline_total_return=0.0,
        candidate_total_return=0.1,
        excess_total_return=0.1,
        baseline_total_cost=1.0,
        candidate_total_cost=0.5,
        baseline_turnover_total=2.0,
        candidate_turnover_total=1.0,
        baseline_max_drawdown=0.2,
        candidate_max_drawdown=0.1,
        baseline_termination_count=0,
        candidate_termination_count=0,
        baseline_termination_reasons=(),
        candidate_termination_reasons=(),
        new_termination=False,
        baseline_n_periods=10,
        candidate_n_periods=10,
        baseline_return_sha256="0" * 64,
        candidate_return_sha256="1" * 64,
    )


def _evaluation() -> MeanReversionEconomicGateEvaluation:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    by_symbol = tuple(_symbol_result(symbol) for symbol in spec.symbols)
    return MeanReversionEconomicGateEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        by_symbol=by_symbol,
        positive_effect_symbols=5,
        median_excess_total_return=0.1,
        cost_reduction_symbols=5,
        turnover_reduction_symbols=5,
        drawdown_nonworse_symbols=5,
        new_termination_symbols=0,
        candidate_positive_total_return_symbols=5,
        research_status="PROMOTE_RESEARCH_REFERENCE",
    )


def test_evaluation_spec_rejects_bool_integer_alias() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()

    with pytest.raises(ValueError, match="frozen evaluation"):
        replace(spec, production_eligible=0)
    with pytest.raises(ValueError, match="frozen evaluation"):
        replace(spec, final_test_authorized=0)


def test_research_status_rejects_counts_outside_frozen_roster() -> None:
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        research_status_from_counts(
            positive_effect_symbols=6,
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=0,
        )


def test_symbol_result_rejects_arithmetic_and_termination_inconsistency() -> None:
    result = _symbol_result("BTCUSDT")

    with pytest.raises(ValueError, match="excess_total_return"):
        replace(result, excess_total_return=0.2)
    with pytest.raises(ValueError, match="new_termination"):
        replace(
            result,
            candidate_termination_count=1,
            candidate_termination_reasons=("economic_insolvency",),
            new_termination=False,
        )


def test_evaluation_result_rejects_authority_and_aggregate_inconsistency() -> None:
    result = _evaluation()

    with pytest.raises(ValueError, match="spec_digest"):
        replace(result, spec_digest="0" * 64)
    with pytest.raises(ValueError, match="dataset_id"):
        replace(result, dataset_id="0" * 64)
    with pytest.raises(ValueError, match="positive_effect_symbols"):
        replace(result, positive_effect_symbols=4)
    with pytest.raises(ValueError, match="median_excess_total_return"):
        replace(result, median_excess_total_return=0.01)
    with pytest.raises(ValueError, match="research_status"):
        replace(result, research_status="INCONCLUSIVE")
