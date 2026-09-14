from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    SignedTakerFlowProtocol,
    canonical_signed_taker_flow_protocol,
    load_signed_taker_flow_protocol,
)


def test_protocol_freezes_source_feature_and_training_clock() -> None:
    protocol = canonical_signed_taker_flow_protocol()

    assert protocol.schema_version == "signed_taker_flow_prereg_v1"
    assert protocol.issue_number == 558
    assert protocol.source_issue == 556
    assert protocol.source_publisher_run_id == 34840494198
    assert protocol.source_artifact_id == 10345533785
    assert protocol.source_artifact_api_digest == (
        "689188893dec90b41819972e1cb151920ab198bf6bf27b1abeae2ab18e2fac0f"
    )
    assert protocol.source_report_content_digest == (
        "e3b627cc8efa63623135512d344a883c1b1be8a0e1bc169c24605ec63abaa693"
    )
    assert protocol.source_fresh_run_id == 34840645375
    assert protocol.source_fresh_artifact_id == 10346140972
    assert protocol.source_fresh_artifact_api_digest == (
        "16f91dcaab5f1914d19ee93934643cf27ad3ab94e1787cb04123b8dc3ad68a60"
    )
    assert protocol.source_status == "PASS"
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )

    assert protocol.source_quote_volume_field == "quote_volume"
    assert protocol.source_taker_buy_quote_volume_field == "taker_buy_quote_volume"
    assert protocol.feature_name == "1h__signed_taker_quote_flow_24bar"
    assert protocol.feature_kind == "signed_taker_quote_flow"
    assert protocol.feature_lookback_bars == 24
    assert protocol.feature_formula == (
        "(2*fsum(taker_buy_quote_volume[t-23:t+1])"
        "-fsum(quote_volume[t-23:t+1]))/fsum(quote_volume[t-23:t+1])"
    )
    assert protocol.feature_quote_denominator_must_be_positive is True
    assert protocol.feature_lower_bound == -1.0
    assert protocol.feature_upper_bound == 1.0

    assert protocol.fit_start == datetime(2021, 1, 1, 1, tzinfo=UTC)
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.label_execution_offset_bars == 1
    assert protocol.label_endpoint_offset_bars == 25
    assert protocol.label_horizon_bars == 24
    assert protocol.label_formula == "log(open[t+25] / open[t+1])"
    assert protocol.minimum_eligible_observations_per_symbol == 8760
    assert len(protocol.digest) == 64


def test_protocol_freezes_causality_and_continuation_gate() -> None:
    protocol = canonical_signed_taker_flow_protocol()

    assert protocol.feature_window_start_offset_bars == -23
    assert protocol.feature_window_stop_offset_bars_inclusive == 0
    assert protocol.require_all_feature_rows_present is True
    assert protocol.require_all_feature_rows_information_available is True
    assert protocol.require_all_feature_rows_active is True
    assert protocol.require_all_feature_rows_tradable is True
    assert protocol.require_quote_volume_finite_nonnegative is True
    assert protocol.require_taker_volume_finite_nonnegative is True
    assert protocol.require_taker_not_above_quote is True
    assert protocol.zero_quote_denominator_action == "UNAVAILABLE"
    assert protocol.future_feature_rows_forbidden is True
    assert protocol.require_label_end_strictly_before_fit_cutoff is True
    assert protocol.require_execution_and_label_rows_present is True
    assert protocol.require_execution_and_label_rows_contiguous is True
    assert protocol.require_execution_and_label_rows_information_available is True
    assert protocol.require_execution_and_label_rows_tradable is True
    assert protocol.require_execution_and_label_rows_active is True
    assert protocol.require_label_open_finite_positive is True

    assert protocol.calibration_method == "per_symbol_no_intercept_fixed_order_fsum"
    assert protocol.calibration_formula == "beta_i = fsum(x_t*y_t) / fsum(x_t*x_t)"
    assert protocol.require_calibration_denominator_finite_positive is True
    assert protocol.require_calibration_beta_finite is True
    assert protocol.expected_effect_direction == "CONTINUATION"
    assert protocol.required_positive_symbol_slopes == 4
    assert protocol.valid_status == "VALID_FLOW_HYPOTHESIS"
    assert protocol.reject_status == "REJECT_FLOW_HYPOTHESIS"
    assert protocol.invalid_coverage_status == "INVALID_FLOW_COVERAGE"
    assert protocol.no_sign_flip_fallback is True
    assert protocol.no_magnitude_threshold_after_results is True
    assert protocol.calibration_slope_used_as_strategy_coefficient is False


def test_protocol_freezes_no_hidden_transform_contract() -> None:
    protocol = canonical_signed_taker_flow_protocol()

    assert protocol.feature_transform == "identity"
    assert protocol.winsorization_allowed is False
    assert protocol.fitted_normalization_allowed is False
    assert protocol.clipping_allowed is False
    assert protocol.log_transform_allowed is False
    assert protocol.ema_allowed is False
    assert protocol.alternate_lookback_allowed is False
    assert protocol.symbol_specific_normalization_allowed is False
    assert protocol.missing_value_imputation_allowed is False
    assert protocol.feature_threshold_allowed is False


def test_protocol_freezes_legacy_compatibility_and_evaluation_boundary() -> None:
    protocol = canonical_signed_taker_flow_protocol()

    assert protocol.raw_field_optional_for_legacy_sources is True
    assert protocol.feature_requires_raw_taker_field is True
    assert protocol.missing_raw_taker_field_action == "FEATURE_UNAVAILABLE"
    assert protocol.legacy_dataset_behavior_unchanged_when_feature_omitted is True
    assert protocol.existing_feature_bytes_unchanged_when_feature_omitted is True
    assert protocol.portable_fixed_order_reduction_required is True
    assert protocol.prefix_causality_required is True

    assert protocol.training_relation_executed is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.evaluation_execution_authorized is False
    assert protocol.production_eligible is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.live_trading_authorized is False


def test_protocol_rejects_semantic_drift_and_bool_integer_alias() -> None:
    protocol = canonical_signed_taker_flow_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"source_artifact_id": 1},
        {"source_report_content_digest": "0" * 64},
        {"source_status": "PARTIAL_SOURCE"},
        {"symbols": tuple(reversed(protocol.symbols))},
        {"feature_name": "1h__signed_taker_quote_flow_1bar"},
        {"feature_lookback_bars": 1},
        {"feature_formula": "taker_buy_quote_volume / quote_volume"},
        {"feature_lower_bound": 0.0},
        {"feature_window_start_offset_bars": -24},
        {"feature_window_stop_offset_bars_inclusive": 1},
        {"require_all_feature_rows_information_available": False},
        {"require_all_feature_rows_tradable": False},
        {"future_feature_rows_forbidden": False},
        {"feature_transform": "rank"},
        {"winsorization_allowed": True},
        {"fitted_normalization_allowed": True},
        {"clipping_allowed": True},
        {"log_transform_allowed": True},
        {"ema_allowed": True},
        {"alternate_lookback_allowed": True},
        {"symbol_specific_normalization_allowed": True},
        {"missing_value_imputation_allowed": True},
        {"feature_threshold_allowed": True},
        {"label_execution_offset_bars": 0},
        {"label_endpoint_offset_bars": 24},
        {"label_formula": "log(close[t+24]/close[t])"},
        {"require_execution_and_label_rows_present": False},
        {"require_execution_and_label_rows_contiguous": False},
        {"require_execution_and_label_rows_information_available": False},
        {"require_label_open_finite_positive": False},
        {"minimum_eligible_observations_per_symbol": 100},
        {"calibration_method": "pooled_ols"},
        {"require_calibration_denominator_finite_positive": False},
        {"require_calibration_beta_finite": False},
        {"expected_effect_direction": "REVERSAL"},
        {"required_positive_symbol_slopes": 3},
        {"no_sign_flip_fallback": False},
        {"calibration_slope_used_as_strategy_coefficient": True},
        {"training_relation_executed": True},
        {"evaluation_pnl_inspected": True},
        {"evaluation_execution_authorized": True},
        {"production_eligible": True},
        {"final_test_authorized": True},
        {"shared_cash_profitability_established": True},
        {"live_trading_authorized": True},
        {"raw_field_optional_for_legacy_sources": 1},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_payload_is_strict_and_contains_no_training_or_evaluation_result(
    tmp_path,
) -> None:
    protocol = canonical_signed_taker_flow_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "feature_values",
        "beta_by_symbol",
        "eligible_observations_by_symbol",
        "training_result",
        "candidate_total_returns",
        "baseline_total_returns",
        "evaluation_result",
        "pnl",
        "ic",
        "correlation",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "signed-taker-flow-prereg.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_signed_taker_flow_protocol(path)
    assert isinstance(loaded, SignedTakerFlowProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["beta_by_symbol"] = [1.0] * 5
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_signed_taker_flow_protocol(path)


def test_loader_rejects_nonfinite_and_naive_time(tmp_path) -> None:
    protocol = canonical_signed_taker_flow_protocol()
    path = tmp_path / "signed-taker-flow-prereg.json"

    broken = protocol.to_payload()
    broken["feature_lower_bound"] = float("nan")
    path.write_text(json.dumps(broken, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_signed_taker_flow_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, fit_cutoff=datetime(2023, 1, 1))
