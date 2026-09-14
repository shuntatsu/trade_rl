from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_evaluation import (
    MeanReversionEconomicGateEvaluationSpec,
    canonical_mean_reversion_economic_gate_evaluation_spec,
    evaluate_mean_reversion_economic_gate,
    research_status_from_counts,
)


def _dataset() -> MarketDataset:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    start = np.datetime64(spec.evaluation_start, "ns")
    stop = np.datetime64(spec.evaluation_stop_exclusive, "ns")
    n_bars = int((stop - start) / np.timedelta64(1, "h")) + 1
    timestamps = start + np.arange(n_bars, dtype=np.int64) * np.timedelta64(1, "h")
    n_symbols = len(spec.symbols)

    features = np.zeros((n_bars, n_symbols, 3), dtype=np.float32)
    signal = np.where((np.arange(n_bars) // 48) % 2 == 0, -0.02, 0.02)
    features[:, :, spec.signal_index] = signal[:, None]
    feature_available = np.ones_like(features, dtype=np.bool_)
    prices = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    tradable = np.ones((n_bars, n_symbols), dtype=np.bool_)
    active = np.ones((n_bars, n_symbols), dtype=np.bool_)

    return MarketDataset(
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices,
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full((n_bars, n_symbols), 1_000_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=tradable,
        feature_available=feature_available,
        feature_names=("feature_0", "feature_1", spec.signal_name),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, n_symbols), spec.fee_rate),
        taker_fee_rate=np.full((n_bars, n_symbols), spec.taker_fee_rate),
        spread_rate=np.full((n_bars, n_symbols), spec.spread_rate),
        max_participation_rate=np.tile(
            np.asarray(spec.capacity_caps, dtype=np.float64),
            (n_bars, 1),
        ),
        asset_active=active,
    )


def test_canonical_evaluation_spec_freezes_all_bound_authorities() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()

    assert isinstance(spec, MeanReversionEconomicGateEvaluationSpec)
    assert spec.issue_number == 549
    assert spec.dataset_id == (
        "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
    )
    assert spec.study_digest == (
        "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert spec.protocol_digest == (
        "c4d6b6f4627dcc58160454f706afc176a37c150e4b2ed731fca929450a76328f"
    )
    assert spec.calibration_result_digest == (
        "df341ed67e87453a887a0889a82a1831a75843a58c7b25f0e98046af03e0814f"
    )
    assert spec.beta_gate == -0.026993016001905932
    assert spec.one_way_explicit_cost == 0.0007
    assert spec.gross_budget == 0.5
    assert spec.initial_capital == 100_000.0
    assert spec.signal_index == 2
    assert spec.signal_name == "1h__log_return_24bar"
    assert spec.rule_entry_threshold == 0.01
    assert spec.rule_exit_threshold == 0.0025
    assert spec.evaluation_start == "2023-01-01T00:00:00.000000000"
    assert spec.evaluation_stop_exclusive == "2025-01-01T00:00:00.000000000"
    assert spec.cost_gate_artifact_id == 10341675711
    assert spec.plan_metadata_artifact_id == 10341661866
    assert spec.production_eligible is False
    assert spec.final_test_authorized is False
    assert spec.shared_cash_profitability_established is False
    assert spec.live_trading_authorized is False
    assert len(spec.digest) == 64


def test_evaluation_spec_rejects_semantic_drift() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    mutations: tuple[dict[str, object], ...] = (
        {"dataset_id": "0" * 64},
        {"study_digest": "0" * 64},
        {"beta_gate": -0.02},
        {"one_way_explicit_cost": 0.0006},
        {"gross_budget": 0.4},
        {"initial_capital": 50_000.0},
        {"signal_index": 1},
        {"signal_name": "wrong"},
        {"rule_entry_threshold": 0.02},
        {"rule_exit_threshold": 0.001},
        {"evaluation_start": "2023-02-01T00:00:00.000000000"},
        {"evaluation_stop_exclusive": "2024-01-01T00:00:00.000000000"},
        {"execution_overlay": "zero_overlay_dataset_fields_authoritative"},
        {"production_eligible": True},
        {"final_test_authorized": True},
        {"shared_cash_profitability_established": True},
        {"live_trading_authorized": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="frozen evaluation"):
            replace(spec, **mutation)


def test_evaluation_uses_common_replay_and_promotes_cost_avoiding_synthetic_case() -> (
    None
):
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    dataset = _dataset()

    first = evaluate_mean_reversion_economic_gate(dataset, spec)
    second = evaluate_mean_reversion_economic_gate(dataset, spec)

    assert first == second
    assert first.digest == second.digest
    assert first.research_status == "PROMOTE_RESEARCH_REFERENCE"
    assert first.positive_effect_symbols == 5
    assert first.cost_reduction_symbols == 5
    assert first.turnover_reduction_symbols == 5
    assert first.drawdown_nonworse_symbols == 5
    assert first.new_termination_symbols == 0
    assert first.candidate_positive_total_return_symbols == 0
    assert first.median_excess_total_return > 0.0
    assert first.production_eligible is False
    assert first.final_test_authorized is False
    assert first.shared_cash_profitability_established is False
    assert first.live_trading_authorized is False

    for symbol_result in first.by_symbol:
        assert symbol_result.baseline_total_return < 0.0
        assert symbol_result.candidate_total_return == pytest.approx(0.0, abs=1e-15)
        assert symbol_result.excess_total_return > 0.0
        assert symbol_result.candidate_total_cost < symbol_result.baseline_total_cost
        assert (
            symbol_result.candidate_turnover_total
            < symbol_result.baseline_turnover_total
        )
        assert (
            symbol_result.candidate_max_drawdown <= symbol_result.baseline_max_drawdown
        )
        assert symbol_result.baseline_n_periods == symbol_result.candidate_n_periods
        assert len(symbol_result.baseline_return_sha256) == 64
        assert len(symbol_result.candidate_return_sha256) == 64


def test_evaluation_fails_closed_on_dataset_cost_or_signal_identity_drift() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    dataset = _dataset()

    fee = np.asarray(dataset.fee_rate).copy()
    fee[10, 0] = 0.0006
    with pytest.raises(ValueError, match="evaluation cost drift"):
        evaluate_mean_reversion_economic_gate(replace(dataset, fee_rate=fee), spec)

    names = list(dataset.feature_names)
    names[spec.signal_index] = "wrong_signal"
    with pytest.raises(ValueError, match="signal identity"):
        evaluate_mean_reversion_economic_gate(
            replace(dataset, feature_names=tuple(names)), spec
        )

    with pytest.raises(ValueError, match="dataset identity"):
        evaluate_mean_reversion_economic_gate(
            replace(dataset, dataset_id="0" * 64), spec
        )


def test_evaluation_requires_exact_stop_boundary() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    dataset = _dataset()
    shortened = replace(
        dataset,
        timestamps=np.asarray(dataset.timestamps[:-1]),
        features=np.asarray(dataset.features[:-1]),
        global_features=np.asarray(dataset.global_features[:-1]),
        open=np.asarray(dataset.open[:-1]),
        high=np.asarray(dataset.high[:-1]),
        low=np.asarray(dataset.low[:-1]),
        close=np.asarray(dataset.close[:-1]),
        volume=np.asarray(dataset.volume[:-1]),
        funding_rate=np.asarray(dataset.funding_rate[:-1]),
        tradable=np.asarray(dataset.tradable[:-1]),
        feature_available=np.asarray(dataset.feature_available[:-1]),
        fee_rate=np.asarray(dataset.fee_rate[:-1]),
        taker_fee_rate=np.asarray(dataset.taker_fee_rate[:-1]),
        spread_rate=np.asarray(dataset.spread_rate[:-1]),
        max_participation_rate=np.asarray(dataset.max_participation_rate[:-1]),
        asset_active=np.asarray(dataset.asset_active[:-1]),
    )
    with pytest.raises(ValueError, match="evaluation_stop_exclusive"):
        evaluate_mean_reversion_economic_gate(shortened, spec)


def test_research_status_boundaries_are_frozen() -> None:
    assert (
        research_status_from_counts(
            positive_effect_symbols=4,
            median_excess_total_return=0.01,
            cost_reduction_symbols=4,
            turnover_reduction_symbols=4,
            drawdown_nonworse_symbols=4,
            new_termination_symbols=0,
        )
        == "PROMOTE_RESEARCH_REFERENCE"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=3,
            median_excess_total_return=0.01,
            cost_reduction_symbols=3,
            turnover_reduction_symbols=3,
            drawdown_nonworse_symbols=3,
            new_termination_symbols=0,
        )
        == "INCONCLUSIVE"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=5,
            median_excess_total_return=0.0,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=0,
        )
        == "REJECT_MECHANISM"
    )
    assert (
        research_status_from_counts(
            positive_effect_symbols=5,
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=1,
        )
        == "REJECT_MECHANISM"
    )
