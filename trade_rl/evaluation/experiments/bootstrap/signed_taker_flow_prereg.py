"""Result-blind preregistration for the signed taker-flow information study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "signed_taker_flow_prereg_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 558,
    "source_issue": 556,
    "source_publisher_run_id": 34840494198,
    "source_artifact_id": 10345533785,
    "source_artifact_api_digest": (
        "689188893dec90b41819972e1cb151920ab198bf6bf27b1abeae2ab18e2fac0f"
    ),
    "source_report_content_digest": (
        "e3b627cc8efa63623135512d344a883c1b1be8a0e1bc169c24605ec63abaa693"
    ),
    "source_fresh_run_id": 34840645375,
    "source_fresh_artifact_id": 10346140972,
    "source_fresh_artifact_api_digest": (
        "16f91dcaab5f1914d19ee93934643cf27ad3ab94e1787cb04123b8dc3ad68a60"
    ),
    "source_status": "PASS",
    "symbols": _SYMBOLS,
    "source_quote_volume_field": "quote_volume",
    "source_taker_buy_quote_volume_field": "taker_buy_quote_volume",
    "feature_name": "1h__signed_taker_quote_flow_24bar",
    "feature_kind": "signed_taker_quote_flow",
    "feature_lookback_bars": 24,
    "feature_formula": (
        "(2*fsum(taker_buy_quote_volume[t-23:t+1])"
        "-fsum(quote_volume[t-23:t+1]))/fsum(quote_volume[t-23:t+1])"
    ),
    "feature_quote_denominator_must_be_positive": True,
    "feature_lower_bound": -1.0,
    "feature_upper_bound": 1.0,
    "feature_transform": "identity",
    "winsorization_allowed": False,
    "fitted_normalization_allowed": False,
    "clipping_allowed": False,
    "log_transform_allowed": False,
    "ema_allowed": False,
    "alternate_lookback_allowed": False,
    "symbol_specific_normalization_allowed": False,
    "missing_value_imputation_allowed": False,
    "feature_threshold_allowed": False,
    "fit_start": datetime(2021, 1, 1, 1, tzinfo=UTC),
    "fit_cutoff": datetime(2023, 1, 1, tzinfo=UTC),
    "label_execution_offset_bars": 1,
    "label_endpoint_offset_bars": 25,
    "label_horizon_bars": 24,
    "label_formula": "log(open[t+25] / open[t+1])",
    "minimum_eligible_observations_per_symbol": 8760,
    "feature_window_start_offset_bars": -23,
    "feature_window_stop_offset_bars_inclusive": 0,
    "require_all_feature_rows_present": True,
    "require_all_feature_rows_information_available": True,
    "require_all_feature_rows_active": True,
    "require_all_feature_rows_tradable": True,
    "require_quote_volume_finite_nonnegative": True,
    "require_taker_volume_finite_nonnegative": True,
    "require_taker_not_above_quote": True,
    "zero_quote_denominator_action": "UNAVAILABLE",
    "future_feature_rows_forbidden": True,
    "require_label_end_strictly_before_fit_cutoff": True,
    "require_execution_and_label_rows_present": True,
    "require_execution_and_label_rows_contiguous": True,
    "require_execution_and_label_rows_information_available": True,
    "require_execution_and_label_rows_tradable": True,
    "require_execution_and_label_rows_active": True,
    "require_label_open_finite_positive": True,
    "calibration_method": "per_symbol_no_intercept_fixed_order_fsum",
    "calibration_formula": "beta_i = fsum(x_t*y_t) / fsum(x_t*x_t)",
    "require_calibration_denominator_finite_positive": True,
    "require_calibration_beta_finite": True,
    "expected_effect_direction": "CONTINUATION",
    "required_positive_symbol_slopes": 4,
    "valid_status": "VALID_FLOW_HYPOTHESIS",
    "reject_status": "REJECT_FLOW_HYPOTHESIS",
    "invalid_coverage_status": "INVALID_FLOW_COVERAGE",
    "no_sign_flip_fallback": True,
    "no_magnitude_threshold_after_results": True,
    "calibration_slope_used_as_strategy_coefficient": False,
    "raw_field_optional_for_legacy_sources": True,
    "feature_requires_raw_taker_field": True,
    "missing_raw_taker_field_action": "FEATURE_UNAVAILABLE",
    "legacy_dataset_behavior_unchanged_when_feature_omitted": True,
    "existing_feature_bytes_unchanged_when_feature_omitted": True,
    "portable_fixed_order_reduction_required": True,
    "prefix_causality_required": True,
    "training_relation_executed": False,
    "evaluation_pnl_inspected": False,
    "evaluation_execution_authorized": False,
    "production_eligible": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "live_trading_authorized": False,
}


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _datetime_from_text(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    try:
        resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a timezone-aware datetime") from error
    _require_aware(resolved, field=field)
    return resolved.astimezone(UTC)


def _require_hex(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character lowercase hex digest")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a 64-character lowercase hex digest")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_to_text(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SignedTakerFlowProtocol:
    """Immutable source/feature/training contract with no observed results."""

    schema_version: str
    issue_number: int
    source_issue: int
    source_publisher_run_id: int
    source_artifact_id: int
    source_artifact_api_digest: str
    source_report_content_digest: str
    source_fresh_run_id: int
    source_fresh_artifact_id: int
    source_fresh_artifact_api_digest: str
    source_status: str
    symbols: tuple[str, ...]
    source_quote_volume_field: str
    source_taker_buy_quote_volume_field: str
    feature_name: str
    feature_kind: str
    feature_lookback_bars: int
    feature_formula: str
    feature_quote_denominator_must_be_positive: bool
    feature_lower_bound: float
    feature_upper_bound: float
    feature_transform: str
    winsorization_allowed: bool
    fitted_normalization_allowed: bool
    clipping_allowed: bool
    log_transform_allowed: bool
    ema_allowed: bool
    alternate_lookback_allowed: bool
    symbol_specific_normalization_allowed: bool
    missing_value_imputation_allowed: bool
    feature_threshold_allowed: bool
    fit_start: datetime
    fit_cutoff: datetime
    label_execution_offset_bars: int
    label_endpoint_offset_bars: int
    label_horizon_bars: int
    label_formula: str
    minimum_eligible_observations_per_symbol: int
    feature_window_start_offset_bars: int
    feature_window_stop_offset_bars_inclusive: int
    require_all_feature_rows_present: bool
    require_all_feature_rows_information_available: bool
    require_all_feature_rows_active: bool
    require_all_feature_rows_tradable: bool
    require_quote_volume_finite_nonnegative: bool
    require_taker_volume_finite_nonnegative: bool
    require_taker_not_above_quote: bool
    zero_quote_denominator_action: str
    future_feature_rows_forbidden: bool
    require_label_end_strictly_before_fit_cutoff: bool
    require_execution_and_label_rows_present: bool
    require_execution_and_label_rows_contiguous: bool
    require_execution_and_label_rows_information_available: bool
    require_execution_and_label_rows_tradable: bool
    require_execution_and_label_rows_active: bool
    require_label_open_finite_positive: bool
    calibration_method: str
    calibration_formula: str
    require_calibration_denominator_finite_positive: bool
    require_calibration_beta_finite: bool
    expected_effect_direction: str
    required_positive_symbol_slopes: int
    valid_status: str
    reject_status: str
    invalid_coverage_status: str
    no_sign_flip_fallback: bool
    no_magnitude_threshold_after_results: bool
    calibration_slope_used_as_strategy_coefficient: bool
    raw_field_optional_for_legacy_sources: bool
    feature_requires_raw_taker_field: bool
    missing_raw_taker_field_action: str
    legacy_dataset_behavior_unchanged_when_feature_omitted: bool
    existing_feature_bytes_unchanged_when_feature_omitted: bool
    portable_fixed_order_reduction_required: bool
    prefix_causality_required: bool
    training_relation_executed: bool
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool
    production_eligible: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    live_trading_authorized: bool

    def __post_init__(self) -> None:
        _require_aware(self.fit_start, field="fit_start")
        _require_aware(self.fit_cutoff, field="fit_cutoff")
        if not self.fit_start < self.fit_cutoff:
            raise ValueError("preregistered training clock is invalid")

        for field_name in ("feature_lower_bound", "feature_upper_bound"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not self.feature_lower_bound < self.feature_upper_bound:
            raise ValueError("preregistered feature bounds are invalid")

        nonnegative_integer_fields = (
            "issue_number",
            "source_issue",
            "source_publisher_run_id",
            "source_artifact_id",
            "source_fresh_run_id",
            "source_fresh_artifact_id",
            "feature_lookback_bars",
            "label_execution_offset_bars",
            "label_endpoint_offset_bars",
            "label_horizon_bars",
            "minimum_eligible_observations_per_symbol",
            "feature_window_stop_offset_bars_inclusive",
            "required_positive_symbol_slopes",
        )
        for field_name in nonnegative_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if isinstance(self.feature_window_start_offset_bars, bool) or not isinstance(
            self.feature_window_start_offset_bars, int
        ):
            raise ValueError("feature_window_start_offset_bars must be an integer")
        if not 1 <= self.required_positive_symbol_slopes <= len(self.symbols):
            raise ValueError("preregistered positive-slope requirement is invalid")

        bool_fields = (
            "feature_quote_denominator_must_be_positive",
            "winsorization_allowed",
            "fitted_normalization_allowed",
            "clipping_allowed",
            "log_transform_allowed",
            "ema_allowed",
            "alternate_lookback_allowed",
            "symbol_specific_normalization_allowed",
            "missing_value_imputation_allowed",
            "feature_threshold_allowed",
            "require_all_feature_rows_present",
            "require_all_feature_rows_information_available",
            "require_all_feature_rows_active",
            "require_all_feature_rows_tradable",
            "require_quote_volume_finite_nonnegative",
            "require_taker_volume_finite_nonnegative",
            "require_taker_not_above_quote",
            "future_feature_rows_forbidden",
            "require_label_end_strictly_before_fit_cutoff",
            "require_execution_and_label_rows_present",
            "require_execution_and_label_rows_contiguous",
            "require_execution_and_label_rows_information_available",
            "require_execution_and_label_rows_tradable",
            "require_execution_and_label_rows_active",
            "require_label_open_finite_positive",
            "require_calibration_denominator_finite_positive",
            "require_calibration_beta_finite",
            "no_sign_flip_fallback",
            "no_magnitude_threshold_after_results",
            "calibration_slope_used_as_strategy_coefficient",
            "raw_field_optional_for_legacy_sources",
            "feature_requires_raw_taker_field",
            "legacy_dataset_behavior_unchanged_when_feature_omitted",
            "existing_feature_bytes_unchanged_when_feature_omitted",
            "portable_fixed_order_reduction_required",
            "prefix_causality_required",
            "training_relation_executed",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "production_eligible",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "live_trading_authorized",
        )
        for field_name in bool_fields:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"preregistered field {field_name} must be a boolean")

        if self.symbols != _SYMBOLS:
            raise ValueError("preregistered symbol roster is invalid")
        for field_name in (
            "source_artifact_api_digest",
            "source_report_content_digest",
            "source_fresh_artifact_api_digest",
        ):
            _require_hex(getattr(self, field_name), field=field_name)

        for item in fields(self):
            value = getattr(self, item.name)
            expected = _CANONICAL_FIELD_VALUES[item.name]
            if not _strict_equal(value, expected):
                raise ValueError(
                    f"preregistered field {item.name} does not match canonical protocol"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_signed_taker_flow_protocol() -> SignedTakerFlowProtocol:
    """Return the sole preregistered protocol authority."""

    return SignedTakerFlowProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def load_signed_taker_flow_protocol(path: str | Path) -> SignedTakerFlowProtocol:
    """Load one strict canonical protocol JSON and reject extra/result fields."""

    source = Path(path)
    try:
        raw: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("signed taker-flow preregistration is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("signed taker-flow preregistration must be a JSON object")

    expected_keys = {item.name for item in fields(SignedTakerFlowProtocol)}
    if set(raw) != expected_keys:
        raise ValueError(
            "signed taker-flow preregistration keys differ from canonical schema"
        )

    payload = dict(raw)
    payload["fit_start"] = _datetime_from_text(payload["fit_start"], field="fit_start")
    payload["fit_cutoff"] = _datetime_from_text(
        payload["fit_cutoff"], field="fit_cutoff"
    )
    symbols = payload["symbols"]
    if not isinstance(symbols, list) or any(
        not isinstance(item, str) for item in symbols
    ):
        raise ValueError("symbols must be a list of strings")
    payload["symbols"] = tuple(symbols)

    protocol = SignedTakerFlowProtocol(**payload)
    canonical = canonical_signed_taker_flow_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError("loaded preregistration differs from canonical protocol")
    return protocol


__all__ = [
    "SignedTakerFlowProtocol",
    "canonical_signed_taker_flow_protocol",
    "load_signed_taker_flow_protocol",
]
