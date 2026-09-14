from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    PerpIndexBasisProtocol,
    canonical_perp_index_basis_protocol,
    load_perp_index_basis_protocol,
)


def test_protocol_freezes_source_feature_and_training_clock() -> None:
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.schema_version == "perp_index_basis_prereg_v1"
    assert protocol.issue_number == 571
    assert protocol.source_issue == 569
    assert protocol.source_publisher_run_id == 34863318605
    assert protocol.source_probe_head == "3c6512059ca5515add930cd881752e466f4ec621"
    assert protocol.source_artifact_id == 10355299323
    assert protocol.source_artifact_api_digest == (
        "753406d64744b64cf6b36172b30d382c997441bed692745f9161de8f5a70de9a"
    )
    assert protocol.source_fresh_artifact_id == 10355167718
    assert protocol.source_fresh_artifact_api_digest == (
        "9a88e28b1314bb729e77adc9b78eee75d5098d6ac2c8e9d15a78038e74a881ab"
    )
    assert protocol.source_report_sha256 == (
        "29f3d3191c9e5141bbf6fd10d3e26a92093596e67e04cb9a779c3c69fb3ee955"
    )
    assert protocol.source_report_content_digest == (
        "f93d625fac21c484f2e15a7498179735295df11c25f57158d5bc0ff7a5334fd5"
    )
    assert protocol.source_status == "PASS_INDEX_SOURCE"
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )

    assert protocol.feature_name == "1h__perp_index_log_basis_bps"
    assert protocol.feature_kind == "perp_index_log_basis_bps"
    assert protocol.perpetual_price_field == "close"
    assert protocol.index_price_field == "close"
    assert protocol.feature_formula == "10000*log(perpetual_close[t]/index_close[t])"
    assert protocol.feature_scale_bps == 10_000.0
    assert protocol.feature_transform == "log_ratio_bps"

    assert protocol.fit_start == datetime(2021, 1, 1, 1, tzinfo=UTC)
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.label_execution_offset_bars == 1
    assert protocol.label_endpoint_offset_bars == 25
    assert protocol.label_horizon_bars == 24
    assert protocol.label_formula == "log(open[t+25] / open[t+1])"
    assert protocol.minimum_eligible_observations_per_symbol == 8760
    assert len(protocol.digest) == 64


def test_protocol_freezes_exact_match_causality_and_sparse_semantics() -> None:
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.require_exact_native_timestamp_match is True
    assert protocol.require_perpetual_row_present is True
    assert protocol.require_index_row_present is True
    assert protocol.require_perpetual_information_available is True
    assert protocol.require_index_information_available is True
    assert protocol.require_decision_active is True
    assert protocol.require_decision_tradable is True
    assert protocol.require_perpetual_close_finite_positive is True
    assert protocol.require_index_close_finite_positive is True
    assert protocol.missing_native_row_action == "FEATURE_UNAVAILABLE"
    assert protocol.stale_carry_allowed is False
    assert protocol.nearest_or_asof_alignment_allowed is False
    assert protocol.interpolation_allowed is False
    assert protocol.forward_fill_allowed is False
    assert protocol.source_substitution_allowed is False
    assert protocol.future_feature_rows_forbidden is True
    assert protocol.prefix_causality_required is True

    assert protocol.require_label_end_strictly_before_fit_cutoff is True
    assert protocol.require_execution_and_label_rows_present is True
    assert protocol.require_execution_and_label_rows_contiguous is True
    assert protocol.require_execution_and_label_rows_information_available is True
    assert protocol.require_execution_and_label_rows_tradable is True
    assert protocol.require_execution_and_label_rows_active is True
    assert protocol.require_label_open_finite_positive is True


def test_protocol_freezes_mean_reversion_gate_and_no_hidden_transforms() -> None:
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.rolling_window_allowed is False
    assert protocol.centering_allowed is False
    assert protocol.fitted_normalization_allowed is False
    assert protocol.zscore_allowed is False
    assert protocol.clipping_allowed is False
    assert protocol.winsorization_allowed is False
    assert protocol.ema_allowed is False
    assert protocol.absolute_value_allowed is False
    assert protocol.funding_combination_allowed is False
    assert protocol.cross_sectional_normalization_allowed is False
    assert protocol.symbol_specific_transform_allowed is False
    assert protocol.feature_threshold_allowed is False
    assert protocol.alternate_feature_formula_allowed is False
    assert protocol.alternate_horizon_allowed is False

    assert protocol.calibration_method == "per_symbol_no_intercept_fixed_order_fsum"
    assert protocol.calibration_formula == "beta_i = fsum(x_t*y_t) / fsum(x_t*x_t)"
    assert protocol.require_calibration_numerator_finite is True
    assert protocol.require_calibration_denominator_finite_positive is True
    assert protocol.require_calibration_beta_finite is True
    assert protocol.expected_effect_direction == "MEAN_REVERSION"
    assert protocol.required_negative_symbol_slopes == 4
    assert protocol.valid_status == "VALID_BASIS_MEAN_REVERSION"
    assert protocol.reject_status == "REJECT_BASIS_HYPOTHESIS"
    assert protocol.invalid_coverage_status == "INVALID_BASIS_COVERAGE"
    assert protocol.no_sign_flip_fallback is True
    assert protocol.no_magnitude_threshold_after_results is True
    assert protocol.calibration_slope_used_as_strategy_coefficient is False


def test_protocol_freezes_full_preflight_and_evaluation_boundary() -> None:
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.full_preflight_required is True
    assert protocol.full_preflight_start_month == "2021-01"
    assert protocol.full_preflight_end_month == "2022-12"
    assert protocol.full_preflight_expected_archives == 120
    assert protocol.full_preflight_source_family == "indexPriceKlines"
    assert protocol.full_preflight_interval == "1h"
    assert protocol.full_preflight_sparse_rows_remain_unavailable is True
    assert protocol.full_preflight_replacement_source_allowed is False
    assert protocol.legacy_dataset_behavior_unchanged_when_feature_omitted is True
    assert protocol.existing_feature_bytes_unchanged_when_feature_omitted is True
    assert protocol.portable_fixed_order_reduction_required is True

    assert protocol.training_relation_executed is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.evaluation_execution_authorized is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False


def test_protocol_rejects_semantic_drift_and_bool_integer_alias() -> None:
    protocol = canonical_perp_index_basis_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"source_issue": 570},
        {"source_artifact_id": 1},
        {"source_report_content_digest": "0" * 64},
        {"source_status": "PASS_SPARSE_SOURCE"},
        {"symbols": tuple(reversed(protocol.symbols))},
        {"feature_name": "1h__perp_index_basis_pct"},
        {"feature_formula": "100*(perpetual_close/index_close-1)"},
        {"feature_scale_bps": 1.0},
        {"perpetual_price_field": "open"},
        {"index_price_field": "open"},
        {"require_exact_native_timestamp_match": False},
        {"stale_carry_allowed": True},
        {"nearest_or_asof_alignment_allowed": True},
        {"interpolation_allowed": True},
        {"forward_fill_allowed": True},
        {"source_substitution_allowed": True},
        {"rolling_window_allowed": True},
        {"centering_allowed": True},
        {"fitted_normalization_allowed": True},
        {"zscore_allowed": True},
        {"clipping_allowed": True},
        {"winsorization_allowed": True},
        {"ema_allowed": True},
        {"absolute_value_allowed": True},
        {"funding_combination_allowed": True},
        {"cross_sectional_normalization_allowed": True},
        {"symbol_specific_transform_allowed": True},
        {"feature_threshold_allowed": True},
        {"alternate_feature_formula_allowed": True},
        {"alternate_horizon_allowed": True},
        {"fit_cutoff": datetime(2024, 1, 1, tzinfo=UTC)},
        {"label_execution_offset_bars": 0},
        {"label_endpoint_offset_bars": 24},
        {"label_formula": "log(close[t+24]/close[t])"},
        {"minimum_eligible_observations_per_symbol": 100},
        {"calibration_method": "centered_ols"},
        {"expected_effect_direction": "CONTINUATION"},
        {"required_negative_symbol_slopes": 3},
        {"no_sign_flip_fallback": False},
        {"no_magnitude_threshold_after_results": False},
        {"calibration_slope_used_as_strategy_coefficient": True},
        {"full_preflight_expected_archives": 100},
        {"full_preflight_replacement_source_allowed": True},
        {"training_relation_executed": True},
        {"evaluation_pnl_inspected": True},
        {"evaluation_execution_authorized": True},
        {"final_test_authorized": True},
        {"shared_cash_profitability_established": True},
        {"production_eligible": True},
        {"live_trading_authorized": True},
        {"full_preflight_required": 1},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_payload_is_result_blind_strict_and_canonical(tmp_path) -> None:
    protocol = canonical_perp_index_basis_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "basis_values",
        "basis_distribution",
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

    path = tmp_path / "perp-index-basis-prereg.json"
    path.write_bytes(canonical_json_bytes(payload))
    loaded = load_perp_index_basis_protocol(path)
    assert isinstance(loaded, PerpIndexBasisProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    path.write_text("{\n  \"schema_version\": \"pretty\"\n}", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|keys|malformed"):
        load_perp_index_basis_protocol(path)

    tampered = dict(payload)
    tampered["beta_by_symbol"] = [-1.0] * 5
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="keys differ"):
        load_perp_index_basis_protocol(path)


def test_loader_rejects_noncanonical_bytes_naive_time_and_wrong_types(tmp_path) -> None:
    protocol = canonical_perp_index_basis_protocol()
    path = tmp_path / "perp-index-basis-prereg.json"

    pretty = protocol.to_payload()
    import json

    path.write_text(json.dumps(pretty, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_perp_index_basis_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, fit_cutoff=datetime(2023, 1, 1))

    with pytest.raises(ValueError, match="preregistered"):
        replace(protocol, feature_scale_bps=10_000)
