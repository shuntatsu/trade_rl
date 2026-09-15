from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.evaluation.experiments.bootstrap.premium_pressure_prereg import (
    PremiumPressureProtocol,
    canonical_premium_pressure_protocol,
    canonical_premium_pressure_protocol_bytes,
    load_premium_pressure_protocol_bytes,
)


def test_canonical_protocol_freezes_exact_one_slot_contract() -> None:
    protocol = canonical_premium_pressure_protocol()

    assert protocol.schema_version == "premium_pressure_prereg_v1"
    assert protocol.issue_number == 600
    assert protocol.multiplicity_issue == 599
    assert protocol.source_issue == 570
    assert protocol.source_publisher_run_id == 34863522941
    assert protocol.source_probe_head == "80b65a773c622af530f074888eb01fff2f7f98df"
    assert protocol.source_artifact_id == 10355913123
    assert protocol.source_artifact_api_digest == (
        "0076698414d3ac8197a676aecc0155853befc3828a2a9c62e92b76576ad2dc04"
    )
    assert protocol.source_fresh_artifact_id == 10355333076
    assert protocol.source_fresh_artifact_api_digest == (
        "2212693404afefedc041b869a03dbe664d3b5cb1e25e28c6c3cdc04a89c1bd3f"
    )
    assert protocol.source_report_sha256 == (
        "059987f3692c0f249f2db5ef2e14f8cbaaefa5ca24dccf4c45bae009ca19ebe4"
    )
    assert protocol.source_report_content_digest == (
        "fc756962aa997cfb8beafdafe06e70e02602e23e09a5dce5a36d229ddb558f88"
    )
    assert protocol.source_status == "PASS_SPARSE_SOURCE"
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )

    assert protocol.feature_name == "1h__premium_index_close_bps"
    assert protocol.feature_kind == "premium_index_close_bps"
    assert protocol.premium_price_field == "close"
    assert protocol.feature_formula == "10000*premium_index_close(raw_open_time=t-1h)"
    assert protocol.feature_scale_bps == 10_000.0
    assert protocol.feature_transform == "fixed_scale_only"
    assert protocol.source_family == "premiumIndexKlines"
    assert protocol.source_interval == "1h"
    assert protocol.source_market == "USD_M"
    assert (
        protocol.premium_dataset_timestamp_semantics == "completed_bar_close_boundary"
    )
    assert (
        protocol.premium_dataset_timestamp_formula
        == "dataset_timestamp=raw_open_time+1h"
    )
    assert protocol.premium_decision_raw_open_time_offset_minutes == -60

    assert protocol.fit_start == datetime(2021, 1, 1, 1, tzinfo=UTC)
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.last_candidate_decision == datetime(2022, 12, 30, 23, tzinfo=UTC)
    assert protocol.nominal_candidate_decisions_per_symbol == 17_495
    assert protocol.minimum_eligible_observations_per_symbol == 16_621

    assert protocol.target_source_market == "USD_M"
    assert protocol.target_source_family == "klines"
    assert protocol.target_interval == "1h"
    assert protocol.target_dataset_timestamp_semantics == "completed_bar_close_boundary"
    assert (
        protocol.target_dataset_timestamp_formula
        == "dataset_timestamp=raw_open_time+1h"
    )
    assert protocol.target_decision_raw_open_time_offset_minutes == -60
    assert protocol.execution_raw_open_time_offset_minutes == 0
    assert protocol.endpoint_raw_open_time_offset_minutes == 1_440
    assert protocol.label_execution_offset_bars == 1
    assert protocol.label_endpoint_offset_bars == 25
    assert protocol.label_horizon_bars == 24
    assert protocol.label_formula == "log(open[t+25] / open[t+1])"

    assert protocol.calibration_method == "per_symbol_ols_intercept_fixed_order_fsum"
    assert protocol.calibration_formula == (
        "x_bar=fsum(x)/n;y_bar=fsum(y)/n;"
        "beta=fsum((x-x_bar)*(y-y_bar))/fsum((x-x_bar)^2);"
        "alpha=y_bar-beta*x_bar"
    )
    assert protocol.expected_effect_direction == "REVERSAL"
    assert protocol.required_negative_symbol_slopes == 4
    assert protocol.valid_status == "VALID_PREMIUM_PRESSURE_REVERSAL"
    assert protocol.reject_status == "REJECT_PREMIUM_PRESSURE_HYPOTHESIS"
    assert protocol.invalid_coverage_status == "INVALID_PREMIUM_PRESSURE_COVERAGE"


def test_fit_clock_counts_are_derived_from_raw_endpoint_semantics() -> None:
    protocol = canonical_premium_pressure_protocol()
    one_hour = timedelta(hours=1)
    endpoint_offset = timedelta(minutes=protocol.endpoint_raw_open_time_offset_minutes)
    decisions: list[datetime] = []
    decision = protocol.fit_start
    while decision + endpoint_offset < protocol.fit_cutoff:
        decisions.append(decision)
        decision += one_hour

    assert decisions[-1] == protocol.last_candidate_decision
    assert len(decisions) == protocol.nominal_candidate_decisions_per_symbol == 17_495
    assert (
        math.ceil(0.95 * len(decisions))
        == protocol.minimum_eligible_observations_per_symbol
        == 16_621
    )
    assert decisions[-1] + 24 * one_hour < protocol.fit_cutoff
    assert decisions[-1] + 25 * one_hour == protocol.fit_cutoff
    assert protocol.label_endpoint_offset_bars == protocol.label_horizon_bars + 1


def test_canonical_protocol_freezes_causal_and_stop_rule_boundaries() -> None:
    protocol = canonical_premium_pressure_protocol()

    true_fields = (
        "require_exact_native_timestamp_match",
        "require_premium_row_present",
        "require_premium_information_available",
        "future_feature_rows_forbidden",
        "prefix_causality_required",
        "require_label_end_strictly_before_fit_cutoff",
        "require_execution_and_label_rows_present",
        "require_execution_and_label_rows_contiguous",
        "require_execution_and_label_rows_information_available",
        "require_execution_and_label_rows_tradable",
        "require_execution_and_label_rows_active",
        "require_label_open_finite_positive",
        "require_calibration_x_mean_finite",
        "require_calibration_y_mean_finite",
        "require_calibration_numerator_finite",
        "require_calibration_denominator_finite_positive",
        "require_calibration_alpha_finite",
        "require_calibration_beta_finite",
        "calibration_intercept_required",
        "calibration_centering_required",
        "target_source_authority_required",
        "target_source_structural_preflight_required",
        "no_sign_flip_fallback",
        "no_second_premium_hypothesis",
        "portable_fixed_order_reduction_required",
    )
    for field in true_fields:
        assert getattr(protocol, field) is True

    false_fields = (
        "stale_carry_allowed",
        "nearest_or_asof_alignment_allowed",
        "interpolation_allowed",
        "forward_fill_allowed",
        "source_substitution_allowed",
        "rolling_window_allowed",
        "feature_centering_allowed",
        "fitted_feature_normalization_allowed",
        "zscore_allowed",
        "clipping_allowed",
        "winsorization_allowed",
        "ema_allowed",
        "absolute_value_allowed",
        "cross_sectional_normalization_allowed",
        "rank_transform_allowed",
        "symbol_specific_transform_allowed",
        "feature_threshold_allowed",
        "alternate_feature_field_allowed",
        "alternate_feature_formula_allowed",
        "alternate_horizon_allowed",
        "alternate_sign_allowed",
        "no_intercept_regression_allowed",
        "funding_reconstruction_allowed",
        "current_funding_formula_backcast_allowed",
        "weighted_regression_allowed",
        "robust_regression_fallback_allowed",
        "calibration_slope_used_as_strategy_coefficient",
        "training_relation_executed",
        "evaluation_pnl_inspected",
        "evaluation_execution_authorized",
        "final_test_authorized",
        "shared_cash_profitability_established",
        "production_eligible",
        "live_trading_authorized",
    )
    for field in false_fields:
        assert getattr(protocol, field) is False

    assert protocol.missing_native_row_action == "FEATURE_UNAVAILABLE"
    assert protocol.premium_zero_is_valid is True
    assert protocol.premium_signed_values_preserved is True


def test_protocol_bytes_are_deterministic_canonical_and_roundtrip() -> None:
    protocol = canonical_premium_pressure_protocol()
    raw = canonical_premium_pressure_protocol_bytes(protocol)

    assert raw == canonical_premium_pressure_protocol_bytes(protocol)
    assert json.loads(raw) == protocol.to_dict()
    assert load_premium_pressure_protocol_bytes(raw) == protocol
    assert len(protocol.digest) == 64
    assert protocol.digest == canonical_premium_pressure_protocol().digest


def test_loader_rejects_missing_unknown_tampered_and_noncanonical_payloads() -> None:
    protocol = canonical_premium_pressure_protocol()
    payload = protocol.to_dict()

    missing = dict(payload)
    missing.pop("feature_name")
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        )

    unknown = {**payload, "observed_beta": -1.0}
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode()
        )

    tampered = {**payload, "expected_effect_direction": "CONTINUATION"}
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
        )

    clock_shifted = {**payload, "premium_decision_raw_open_time_offset_minutes": 0}
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(
            json.dumps(clock_shifted, sort_keys=True, separators=(",", ":")).encode()
        )

    calibration_drift = {
        **payload,
        "calibration_method": "per_symbol_no_intercept_fixed_order_fsum",
    }
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(
            json.dumps(
                calibration_drift, sort_keys=True, separators=(",", ":")
            ).encode()
        )

    canonical = canonical_premium_pressure_protocol_bytes(protocol)
    with pytest.raises(ValueError):
        load_premium_pressure_protocol_bytes(b"\n" + canonical)


def test_protocol_constructor_rejects_bool_int_spoofing_and_semantic_drift() -> None:
    protocol = canonical_premium_pressure_protocol()

    with pytest.raises(ValueError):
        replace(protocol, minimum_eligible_observations_per_symbol=True)
    with pytest.raises(ValueError):
        replace(protocol, required_negative_symbol_slopes=True)
    with pytest.raises(ValueError):
        replace(protocol, feature_scale_bps=True)
    with pytest.raises(ValueError):
        replace(protocol, fit_cutoff=protocol.fit_start)
    with pytest.raises(ValueError):
        replace(protocol, last_candidate_decision=protocol.fit_cutoff)
    with pytest.raises(ValueError):
        replace(protocol, label_endpoint_offset_bars=24)
    with pytest.raises(ValueError):
        replace(protocol, label_horizon_bars=23)
    with pytest.raises(ValueError):
        replace(protocol, endpoint_raw_open_time_offset_minutes=1_380)
    with pytest.raises(ValueError):
        replace(protocol, premium_decision_raw_open_time_offset_minutes=0)
    with pytest.raises(ValueError):
        replace(protocol, required_negative_symbol_slopes=6)
    with pytest.raises(ValueError):
        replace(protocol, symbols=("BTCUSDT",) * 5)


def test_protocol_dataclass_has_no_observed_result_fields() -> None:
    protocol = canonical_premium_pressure_protocol()
    forbidden = {
        "alpha",
        "alphas",
        "beta",
        "betas",
        "x_bar",
        "y_bar",
        "numerator",
        "denominator",
        "negative_slope_count",
        "positive_slope_count",
        "pnl",
        "sharpe",
        "ic",
        "premium_mean",
        "premium_median",
        "premium_std",
        "premium_quantiles",
    }
    assert forbidden.isdisjoint(protocol.to_dict())
    assert forbidden.isdisjoint(PremiumPressureProtocol.__dataclass_fields__)
