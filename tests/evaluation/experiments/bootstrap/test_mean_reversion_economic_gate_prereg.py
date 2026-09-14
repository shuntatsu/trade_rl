from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    MeanReversionEconomicGateProtocol,
    canonical_mean_reversion_economic_gate_protocol,
    load_mean_reversion_economic_gate_protocol,
)


def test_protocol_freezes_bound_authority_and_training_clock() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()

    assert protocol.schema_version == "mean_reversion_economic_gate_prereg_v1"
    assert protocol.issue_number == 545
    assert protocol.diagnosis_issue == 543
    assert protocol.diagnosis_pr == 544
    assert protocol.diagnosis_head_sha == "9f3fcc3c84820b7aca25719777427aa37b90256e"
    assert protocol.diagnosis_report_digest == (
        "9b300bf8c6d6ac4b8230a6179c4988d8c7bac2f3988e9c0d823a9ed95361fef0"
    )
    assert protocol.diagnosis_verification_run_id == 34812564402
    assert protocol.diagnosis_verification_artifact_id == 10335845094
    assert protocol.diagnosis_verification_artifact_digest == (
        "e722c27e2b3903ae91ad81b1d244fdbb85e7deba093e0a0db0dcaf644a05d6e9"
    )

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
    assert protocol.capacity_caps == (
        0.0021629560553901974,
        0.002044685341258238,
        0.002184898995567895,
        0.0020480213652913385,
        0.002346308308284808,
    )

    assert protocol.signal_name == "1h__log_return_24bar"
    assert protocol.signal_index == 2
    assert protocol.fit_start == datetime(2021, 1, 1, 1, tzinfo=UTC)
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_stop_exclusive == datetime(2025, 1, 1, tzinfo=UTC)
    assert protocol.label_execution_offset_bars == 1
    assert protocol.label_horizon_bars == 24
    assert protocol.label_formula == "log(open[t+25] / open[t+1])"
    assert protocol.minimum_eligible_observations_per_symbol == 8760

    assert protocol.rule_entry_threshold == 0.01
    assert protocol.rule_exit_threshold == 0.0025
    assert protocol.market_order_fee_rate == 0.0005
    assert protocol.market_order_taker_fee_rate == 0.0
    assert protocol.market_order_spread_rate == 0.0002
    assert protocol.nominal_one_way_explicit_cost == pytest.approx(0.0007)
    assert len(protocol.digest) == 64


def test_protocol_freezes_calibration_and_gate_formula() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()

    assert protocol.calibration_method == "per_symbol_no_intercept_fixed_order_fsum"
    assert protocol.calibration_formula == "beta_i = fsum(s_t*y_t) / fsum(s_t*s_t)"
    assert protocol.required_negative_symbol_slopes == 4
    assert protocol.beta_gate_order_statistic == 4
    assert protocol.beta_gate_description == "fourth_smallest_weakest_required_negative"
    assert protocol.invalid_coverage_status == "INVALID_CALIBRATION_COVERAGE"
    assert protocol.invalid_training_edge_status == "INVALID_NO_TRAINING_EDGE"
    assert protocol.no_calibration_fallback is True

    assert protocol.position_values == (("LONG", 1), ("FLAT", 0), ("SHORT", -1))
    assert protocol.edge_formula == "delta_position * beta_gate * signal"
    assert protocol.one_way_cost_formula == "fee_rate + taker_fee_rate + spread_rate"
    assert protocol.transition_cost_formula == "abs(delta_position) * one_way_cost"
    assert protocol.gate_operator == ">"
    assert protocol.equality_action == "HOLD_CURRENT"
    assert protocol.unavailable_signal_action == "FLAT_BYPASS_GATE"
    assert protocol.hard_risk_overrides_gate is True
    assert protocol.thresholds_unchanged is True
    assert protocol.impact_slippage_invented_by_gate is False


def test_protocol_separates_research_promotion_from_production() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()

    assert protocol.target_strategy == "mean_reversion"
    assert protocol.unaffected_strategies == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "ridge24",
        "lightgbm24",
        "ppo",
    )
    assert protocol.research_promote_status == "PROMOTE_RESEARCH_REFERENCE"
    assert protocol.reject_status == "REJECT_MECHANISM"
    assert protocol.inconclusive_status == "INCONCLUSIVE"
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

    assert protocol.development_data_already_used is True
    assert protocol.production_eligible is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.live_trading_authorized is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.calibration_executed is False
    assert protocol.evaluation_execution_authorized is False


def test_protocol_rejects_any_semantic_drift() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"diagnosis_head_sha": "0" * 40},
        {"diagnosis_report_digest": "0" * 64},
        {"successor_bundle_artifact_id": 1},
        {"successor_dataset_id": "0" * 64},
        {"successor_execution_overlay": "zero_overlay_dataset_fields_authoritative"},
        {"capacity_caps": (0.05,) * 5},
        {"signal_name": "1h__log_return_1bar"},
        {"signal_index": 0},
        {"label_execution_offset_bars": 0},
        {"label_horizon_bars": 12},
        {"label_formula": "log(close[t+24] / close[t])"},
        {"minimum_eligible_observations_per_symbol": 100},
        {"rule_entry_threshold": 0.02},
        {"rule_exit_threshold": 0.001},
        {"calibration_method": "pooled_ols"},
        {"required_negative_symbol_slopes": 3},
        {"beta_gate_order_statistic": 3},
        {"no_calibration_fallback": False},
        {"edge_formula": "abs(beta_gate * signal)"},
        {"gate_operator": ">="},
        {"equality_action": "TRADE"},
        {"unavailable_signal_action": "HOLD_CURRENT"},
        {"hard_risk_overrides_gate": False},
        {"thresholds_unchanged": False},
        {"absolute_positive_symbol_count_is_decision_input": True},
        {"production_eligible": True},
        {"final_test_authorized": True},
        {"shared_cash_profitability_established": True},
        {"live_trading_authorized": True},
        {"evaluation_pnl_inspected": True},
        {"calibration_executed": True},
        {"evaluation_execution_authorized": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_payload_is_strict_and_contains_no_calibration_or_evaluation_result(
    tmp_path,
) -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "beta_by_symbol",
        "beta_gate",
        "eligible_observations_by_symbol",
        "calibration_result",
        "candidate_total_returns",
        "baseline_total_returns",
        "candidate_result",
        "evaluation_result",
        "successor_pnl",
        "pnl",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "mean-reversion-economic-gate-prereg.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_mean_reversion_economic_gate_protocol(path)
    assert isinstance(loaded, MeanReversionEconomicGateProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["beta_gate"] = -0.5
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_mean_reversion_economic_gate_protocol(path)


def test_loader_rejects_nonfinite_and_naive_time(tmp_path) -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    path = tmp_path / "mean-reversion-economic-gate-prereg.json"

    broken = protocol.to_payload()
    broken["market_order_fee_rate"] = float("nan")
    path.write_text(json.dumps(broken, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_mean_reversion_economic_gate_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, fit_cutoff=datetime(2023, 1, 1))
