"""Result-blind preregistration for the perpetual-vs-index basis study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "perp_index_basis_prereg_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 571,
    "source_issue": 569,
    "source_publisher_run_id": 34863318605,
    "source_probe_head": "3c6512059ca5515add930cd881752e466f4ec621",
    "source_artifact_id": 10355299323,
    "source_artifact_api_digest": (
        "753406d64744b64cf6b36172b30d382c997441bed692745f9161de8f5a70de9a"
    ),
    "source_fresh_artifact_id": 10355167718,
    "source_fresh_artifact_api_digest": (
        "9a88e28b1314bb729e77adc9b78eee75d5098d6ac2c8e9d15a78038e74a881ab"
    ),
    "source_report_sha256": (
        "29f3d3191c9e5141bbf6fd10d3e26a92093596e67e04cb9a779c3c69fb3ee955"
    ),
    "source_report_content_digest": (
        "f93d625fac21c484f2e15a7498179735295df11c25f57158d5bc0ff7a5334fd5"
    ),
    "source_status": "PASS_INDEX_SOURCE",
    "symbols": _SYMBOLS,
    "feature_name": "1h__perp_index_log_basis_bps",
    "feature_kind": "perp_index_log_basis_bps",
    "perpetual_price_field": "close",
    "index_price_field": "close",
    "feature_formula": "10000*log(perpetual_close[t]/index_close[t])",
    "feature_scale_bps": 10_000.0,
    "feature_transform": "log_ratio_bps",
    "require_exact_native_timestamp_match": True,
    "require_perpetual_row_present": True,
    "require_index_row_present": True,
    "require_perpetual_information_available": True,
    "require_index_information_available": True,
    "require_decision_active": True,
    "require_decision_tradable": True,
    "require_perpetual_close_finite_positive": True,
    "require_index_close_finite_positive": True,
    "missing_native_row_action": "FEATURE_UNAVAILABLE",
    "stale_carry_allowed": False,
    "nearest_or_asof_alignment_allowed": False,
    "interpolation_allowed": False,
    "forward_fill_allowed": False,
    "source_substitution_allowed": False,
    "future_feature_rows_forbidden": True,
    "prefix_causality_required": True,
    "rolling_window_allowed": False,
    "centering_allowed": False,
    "fitted_normalization_allowed": False,
    "zscore_allowed": False,
    "clipping_allowed": False,
    "winsorization_allowed": False,
    "ema_allowed": False,
    "absolute_value_allowed": False,
    "funding_combination_allowed": False,
    "cross_sectional_normalization_allowed": False,
    "symbol_specific_transform_allowed": False,
    "feature_threshold_allowed": False,
    "alternate_feature_formula_allowed": False,
    "alternate_horizon_allowed": False,
    "fit_start": datetime(2021, 1, 1, 1, tzinfo=UTC),
    "fit_cutoff": datetime(2023, 1, 1, tzinfo=UTC),
    "label_execution_offset_bars": 1,
    "label_endpoint_offset_bars": 25,
    "label_horizon_bars": 24,
    "label_formula": "log(open[t+25] / open[t+1])",
    "minimum_eligible_observations_per_symbol": 8760,
    "require_label_end_strictly_before_fit_cutoff": True,
    "require_execution_and_label_rows_present": True,
    "require_execution_and_label_rows_contiguous": True,
    "require_execution_and_label_rows_information_available": True,
    "require_execution_and_label_rows_tradable": True,
    "require_execution_and_label_rows_active": True,
    "require_label_open_finite_positive": True,
    "calibration_method": "per_symbol_no_intercept_fixed_order_fsum",
    "calibration_formula": "beta_i = fsum(x_t*y_t) / fsum(x_t*x_t)",
    "require_calibration_numerator_finite": True,
    "require_calibration_denominator_finite_positive": True,
    "require_calibration_beta_finite": True,
    "expected_effect_direction": "MEAN_REVERSION",
    "required_negative_symbol_slopes": 4,
    "valid_status": "VALID_BASIS_MEAN_REVERSION",
    "reject_status": "REJECT_BASIS_HYPOTHESIS",
    "invalid_coverage_status": "INVALID_BASIS_COVERAGE",
    "no_sign_flip_fallback": True,
    "no_magnitude_threshold_after_results": True,
    "calibration_slope_used_as_strategy_coefficient": False,
    "full_preflight_required": True,
    "full_preflight_start_month": "2021-01",
    "full_preflight_end_month": "2022-12",
    "full_preflight_expected_archives": 120,
    "full_preflight_source_family": "indexPriceKlines",
    "full_preflight_interval": "1h",
    "full_preflight_sparse_rows_remain_unavailable": True,
    "full_preflight_replacement_source_allowed": False,
    "legacy_dataset_behavior_unchanged_when_feature_omitted": True,
    "existing_feature_bytes_unchanged_when_feature_omitted": True,
    "portable_fixed_order_reduction_required": True,
    "training_relation_executed": False,
    "evaluation_pnl_inspected": False,
    "evaluation_execution_authorized": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "production_eligible": False,
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


def _require_hex(value: object, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_to_text(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PerpIndexBasisProtocol:
    """Immutable source, feature and training contract with no observed results."""

    schema_version: str
    issue_number: int
    source_issue: int
    source_publisher_run_id: int
    source_probe_head: str
    source_artifact_id: int
    source_artifact_api_digest: str
    source_fresh_artifact_id: int
    source_fresh_artifact_api_digest: str
    source_report_sha256: str
    source_report_content_digest: str
    source_status: str
    symbols: tuple[str, ...]
    feature_name: str
    feature_kind: str
    perpetual_price_field: str
    index_price_field: str
    feature_formula: str
    feature_scale_bps: float
    feature_transform: str
    require_exact_native_timestamp_match: bool
    require_perpetual_row_present: bool
    require_index_row_present: bool
    require_perpetual_information_available: bool
    require_index_information_available: bool
    require_decision_active: bool
    require_decision_tradable: bool
    require_perpetual_close_finite_positive: bool
    require_index_close_finite_positive: bool
    missing_native_row_action: str
    stale_carry_allowed: bool
    nearest_or_asof_alignment_allowed: bool
    interpolation_allowed: bool
    forward_fill_allowed: bool
    source_substitution_allowed: bool
    future_feature_rows_forbidden: bool
    prefix_causality_required: bool
    rolling_window_allowed: bool
    centering_allowed: bool
    fitted_normalization_allowed: bool
    zscore_allowed: bool
    clipping_allowed: bool
    winsorization_allowed: bool
    ema_allowed: bool
    absolute_value_allowed: bool
    funding_combination_allowed: bool
    cross_sectional_normalization_allowed: bool
    symbol_specific_transform_allowed: bool
    feature_threshold_allowed: bool
    alternate_feature_formula_allowed: bool
    alternate_horizon_allowed: bool
    fit_start: datetime
    fit_cutoff: datetime
    label_execution_offset_bars: int
    label_endpoint_offset_bars: int
    label_horizon_bars: int
    label_formula: str
    minimum_eligible_observations_per_symbol: int
    require_label_end_strictly_before_fit_cutoff: bool
    require_execution_and_label_rows_present: bool
    require_execution_and_label_rows_contiguous: bool
    require_execution_and_label_rows_information_available: bool
    require_execution_and_label_rows_tradable: bool
    require_execution_and_label_rows_active: bool
    require_label_open_finite_positive: bool
    calibration_method: str
    calibration_formula: str
    require_calibration_numerator_finite: bool
    require_calibration_denominator_finite_positive: bool
    require_calibration_beta_finite: bool
    expected_effect_direction: str
    required_negative_symbol_slopes: int
    valid_status: str
    reject_status: str
    invalid_coverage_status: str
    no_sign_flip_fallback: bool
    no_magnitude_threshold_after_results: bool
    calibration_slope_used_as_strategy_coefficient: bool
    full_preflight_required: bool
    full_preflight_start_month: str
    full_preflight_end_month: str
    full_preflight_expected_archives: int
    full_preflight_source_family: str
    full_preflight_interval: str
    full_preflight_sparse_rows_remain_unavailable: bool
    full_preflight_replacement_source_allowed: bool
    legacy_dataset_behavior_unchanged_when_feature_omitted: bool
    existing_feature_bytes_unchanged_when_feature_omitted: bool
    portable_fixed_order_reduction_required: bool
    training_relation_executed: bool
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    production_eligible: bool
    live_trading_authorized: bool

    def __post_init__(self) -> None:
        _require_aware(self.fit_start, field="fit_start")
        _require_aware(self.fit_cutoff, field="fit_cutoff")
        if not self.fit_start < self.fit_cutoff:
            raise ValueError("preregistered training clock is invalid")

        if (
            isinstance(self.feature_scale_bps, bool)
            or not isinstance(self.feature_scale_bps, (int, float))
            or not math.isfinite(float(self.feature_scale_bps))
            or float(self.feature_scale_bps) <= 0.0
        ):
            raise ValueError("feature_scale_bps must be finite and positive")

        nonnegative_integer_fields = (
            "issue_number",
            "source_issue",
            "source_publisher_run_id",
            "source_artifact_id",
            "source_fresh_artifact_id",
            "label_execution_offset_bars",
            "label_endpoint_offset_bars",
            "label_horizon_bars",
            "minimum_eligible_observations_per_symbol",
            "required_negative_symbol_slopes",
            "full_preflight_expected_archives",
        )
        for field_name in nonnegative_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not 1 <= self.required_negative_symbol_slopes <= len(self.symbols):
            raise ValueError("preregistered negative-slope requirement is invalid")

        bool_fields = (
            "require_exact_native_timestamp_match",
            "require_perpetual_row_present",
            "require_index_row_present",
            "require_perpetual_information_available",
            "require_index_information_available",
            "require_decision_active",
            "require_decision_tradable",
            "require_perpetual_close_finite_positive",
            "require_index_close_finite_positive",
            "stale_carry_allowed",
            "nearest_or_asof_alignment_allowed",
            "interpolation_allowed",
            "forward_fill_allowed",
            "source_substitution_allowed",
            "future_feature_rows_forbidden",
            "prefix_causality_required",
            "rolling_window_allowed",
            "centering_allowed",
            "fitted_normalization_allowed",
            "zscore_allowed",
            "clipping_allowed",
            "winsorization_allowed",
            "ema_allowed",
            "absolute_value_allowed",
            "funding_combination_allowed",
            "cross_sectional_normalization_allowed",
            "symbol_specific_transform_allowed",
            "feature_threshold_allowed",
            "alternate_feature_formula_allowed",
            "alternate_horizon_allowed",
            "require_label_end_strictly_before_fit_cutoff",
            "require_execution_and_label_rows_present",
            "require_execution_and_label_rows_contiguous",
            "require_execution_and_label_rows_information_available",
            "require_execution_and_label_rows_tradable",
            "require_execution_and_label_rows_active",
            "require_label_open_finite_positive",
            "require_calibration_numerator_finite",
            "require_calibration_denominator_finite_positive",
            "require_calibration_beta_finite",
            "no_sign_flip_fallback",
            "no_magnitude_threshold_after_results",
            "calibration_slope_used_as_strategy_coefficient",
            "full_preflight_required",
            "full_preflight_sparse_rows_remain_unavailable",
            "full_preflight_replacement_source_allowed",
            "legacy_dataset_behavior_unchanged_when_feature_omitted",
            "existing_feature_bytes_unchanged_when_feature_omitted",
            "portable_fixed_order_reduction_required",
            "training_relation_executed",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
        )
        for field_name in bool_fields:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"preregistered field {field_name} must be a boolean")

        if self.symbols != _SYMBOLS:
            raise ValueError("preregistered symbol roster is invalid")
        _require_hex(self.source_probe_head, length=40, field="source_probe_head")
        for field_name in (
            "source_artifact_api_digest",
            "source_fresh_artifact_api_digest",
            "source_report_sha256",
            "source_report_content_digest",
        ):
            _require_hex(getattr(self, field_name), length=64, field=field_name)

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


def canonical_perp_index_basis_protocol() -> PerpIndexBasisProtocol:
    """Return the sole preregistered protocol authority."""

    return PerpIndexBasisProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def load_perp_index_basis_protocol(path: str | Path) -> PerpIndexBasisProtocol:
    """Load the strict canonical protocol and reject result or formatting drift."""

    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        raw: Any = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("perpetual-index basis preregistration is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("perpetual-index basis preregistration must be a JSON object")
    try:
        expected_bytes = canonical_json_bytes(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("perpetual-index basis preregistration is malformed") from error
    if raw_bytes != expected_bytes:
        raise ValueError("perpetual-index basis preregistration must use canonical JSON bytes")

    expected_keys = {item.name for item in fields(PerpIndexBasisProtocol)}
    if set(raw) != expected_keys:
        raise ValueError("perpetual-index basis preregistration keys differ from canonical")

    resolved = dict(raw)
    symbols = resolved["symbols"]
    if not isinstance(symbols, list) or any(not isinstance(item, str) for item in symbols):
        raise ValueError("symbols must be a string array")
    resolved["symbols"] = tuple(symbols)
    resolved["fit_start"] = _datetime_from_text(resolved["fit_start"], field="fit_start")
    resolved["fit_cutoff"] = _datetime_from_text(
        resolved["fit_cutoff"], field="fit_cutoff"
    )
    return PerpIndexBasisProtocol(**resolved)  # type: ignore[arg-type]


__all__ = [
    "PerpIndexBasisProtocol",
    "canonical_perp_index_basis_protocol",
    "load_perp_index_basis_protocol",
]
