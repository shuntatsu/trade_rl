from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_prereg import (
    RidgeEconomicGateProtocol,
    canonical_ridge_economic_gate_protocol,
    load_ridge_economic_gate_protocol,
)

_FEATURE_NAMES = (
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
_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)


def test_protocol_freezes_bound_plan_and_ridge_semantics() -> None:
    protocol = canonical_ridge_economic_gate_protocol()

    assert protocol.schema_version == "ridge_economic_gate_prereg_v1"
    assert protocol.issue_number == 616
    assert protocol.parent_roadmap_issue == 604
    assert protocol.diagnosis_issue == 543
    assert protocol.diagnosis_pr == 544
    assert protocol.diagnosis_head_sha == "9f3fcc3c84820b7aca25719777427aa37b90256e"
    assert protocol.diagnosis_verification_run_id == 34812564402
    assert protocol.successor_bundle_run_id == 34803217815
    assert protocol.successor_bundle_artifact_id == 10331899302
    assert protocol.successor_bundle_artifact_digest == (
        "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
    )
    assert protocol.successor_dataset_id == (
        "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
    )
    assert protocol.successor_dataset_artifact_digest == (
        "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
    )
    assert protocol.successor_study_digest == (
        "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert protocol.successor_execution_overlay == (
        "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
    )
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert protocol.feature_names == _FEATURE_NAMES
    assert protocol.feature_indices == _FEATURE_INDICES
    assert protocol.fit_symbol_names == protocol.symbols
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_stop_exclusive == datetime(2025, 1, 1, tzinfo=UTC)
    assert protocol.ridge_horizon_hours == 24
    assert protocol.ridge_alpha == 1.0
    assert protocol.forecast_entry_threshold == 0.0025
    assert protocol.forecast_exit_threshold == 0.0005
    assert protocol.gross_budget == 0.5
    assert protocol.initial_capital == 100_000.0
    assert len(protocol.digest) == 64


def test_protocol_freezes_single_transition_gate_without_calibration() -> None:
    protocol = canonical_ridge_economic_gate_protocol()

    assert protocol.target_strategy == "ridge24"
    assert protocol.proposal_model == "canonical_pooled_ridge24"
    assert protocol.additional_training_calibration_required is False
    assert protocol.position_values == (("LONG", 1), ("FLAT", 0), ("SHORT", -1))
    assert protocol.edge_formula == "delta_position * ridge_forecast"
    assert protocol.one_way_cost_formula == "fee_rate + taker_fee_rate + spread_rate"
    assert protocol.transition_cost_formula == "abs(delta_position) * one_way_cost"
    assert protocol.gate_operator == ">"
    assert protocol.equality_action == "HOLD_CURRENT"
    assert protocol.unavailable_feature_action == "FLAT_BYPASS_GATE"
    assert protocol.hard_risk_overrides_gate is True
    assert protocol.model_unchanged is True
    assert protocol.features_unchanged is True
    assert protocol.thresholds_unchanged is True
    assert protocol.horizon_unchanged is True
    assert protocol.symbol_scope_unchanged is True
    assert protocol.impact_slippage_invented_by_gate is False

    assert protocol.market_order_fee_rate == 0.0005
    assert protocol.market_order_taker_fee_rate == 0.0
    assert protocol.market_order_spread_rate == 0.0002
    assert protocol.nominal_one_way_explicit_cost == pytest.approx(0.0007)
    assert protocol.require_full_evaluation_cost_constancy is True
    assert protocol.future_row_economics_read_by_strategy is False


def test_protocol_freezes_decision_and_production_boundaries() -> None:
    protocol = canonical_ridge_economic_gate_protocol()

    assert protocol.research_promote_status == "PROMOTE_RESEARCH_REFERENCE"
    assert protocol.reject_status == "REJECT_MECHANISM"
    assert protocol.inconclusive_status == "INCONCLUSIVE"
    assert protocol.invalid_cost_drift_status == "INVALID_EVALUATION_COST_DRIFT"
    assert protocol.promote_min_positive_effect_symbols == 4
    assert protocol.promote_requires_positive_median_excess is True
    assert protocol.promote_min_cost_reduction_symbols == 4
    assert protocol.promote_min_turnover_reduction_symbols == 4
    assert protocol.promote_min_drawdown_nonworse_symbols == 4
    assert protocol.promote_requires_no_new_termination is True
    assert protocol.reject_max_positive_effect_symbols == 2
    assert protocol.reject_on_nonpositive_median_excess is True
    assert protocol.reject_max_cost_reduction_symbols == 2
    assert protocol.reject_max_turnover_reduction_symbols == 2
    assert protocol.reject_max_drawdown_nonworse_symbols == 2
    assert protocol.absolute_positive_symbol_count_is_decision_input is False
    assert protocol.one_slot_only is True
    assert protocol.post_result_variant_allowed is False

    assert protocol.development_data_already_used is True
    assert protocol.final_test_accessed is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.operational_eligibility_established is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False
    assert protocol.merge_authorized is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.evaluation_execution_authorized is False


def test_protocol_rejects_semantic_drift() -> None:
    protocol = canonical_ridge_economic_gate_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"successor_bundle_artifact_id": 1},
        {"successor_dataset_id": "0" * 64},
        {"feature_names": tuple(reversed(_FEATURE_NAMES))},
        {"feature_indices": tuple(reversed(_FEATURE_INDICES))},
        {"fit_symbol_names": ("BTCUSDT",)},
        {"ridge_horizon_hours": 12},
        {"ridge_alpha": 2.0},
        {"forecast_entry_threshold": 0.003},
        {"forecast_exit_threshold": 0.001},
        {"edge_formula": "abs(ridge_forecast)"},
        {"gate_operator": ">="},
        {"equality_action": "TRADE"},
        {"unavailable_feature_action": "HOLD_CURRENT"},
        {"additional_training_calibration_required": True},
        {"hard_risk_overrides_gate": False},
        {"model_unchanged": False},
        {"features_unchanged": False},
        {"thresholds_unchanged": False},
        {"horizon_unchanged": False},
        {"symbol_scope_unchanged": False},
        {"future_row_economics_read_by_strategy": True},
        {"one_slot_only": False},
        {"post_result_variant_allowed": True},
        {"absolute_positive_symbol_count_is_decision_input": True},
        {"final_test_accessed": True},
        {"final_test_authorized": True},
        {"shared_cash_profitability_established": True},
        {"operational_eligibility_established": True},
        {"production_eligible": True},
        {"live_trading_authorized": True},
        {"merge_authorized": True},
        {"evaluation_pnl_inspected": True},
        {"evaluation_execution_authorized": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_payload_is_strict_and_result_blind(tmp_path) -> None:
    protocol = canonical_ridge_economic_gate_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "candidate_total_returns",
        "baseline_total_returns",
        "candidate_result",
        "evaluation_result",
        "successor_pnl",
        "pnl",
        "positive_effect_symbols",
        "median_excess_total_return",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "ridge-economic-gate-prereg.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_ridge_economic_gate_protocol(path)
    assert isinstance(loaded, RidgeEconomicGateProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["candidate_total_returns"] = [1.0] * 5
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_ridge_economic_gate_protocol(path)


def test_loader_rejects_nonfinite_and_naive_time(tmp_path) -> None:
    protocol = canonical_ridge_economic_gate_protocol()
    path = tmp_path / "ridge-economic-gate-prereg.json"

    broken = protocol.to_payload()
    broken["market_order_fee_rate"] = float("nan")
    path.write_text(json.dumps(broken, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_ridge_economic_gate_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, fit_cutoff=datetime(2023, 1, 1))
