"""Result-blind preregistration for the one-slot premium-pressure study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "premium_pressure_prereg_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 600,
    "multiplicity_issue": 599,
    "source_issue": 570,
    "source_publisher_run_id": 34863522941,
    "source_probe_head": "80b65a773c622af530f074888eb01fff2f7f98df",
    "source_artifact_id": 10355913123,
    "source_artifact_api_digest": (
        "0076698414d3ac8197a676aecc0155853befc3828a2a9c62e92b76576ad2dc04"
    ),
    "source_fresh_artifact_id": 10355333076,
    "source_fresh_artifact_api_digest": (
        "2212693404afefedc041b869a03dbe664d3b5cb1e25e28c6c3cdc04a89c1bd3f"
    ),
    "source_report_sha256": (
        "059987f3692c0f249f2db5ef2e14f8cbaaefa5ca24dccf4c45bae009ca19ebe4"
    ),
    "source_report_content_digest": (
        "fc756962aa997cfb8beafdafe06e70e02602e23e09a5dce5a36d229ddb558f88"
    ),
    "source_status": "PASS_SPARSE_SOURCE",
    "source_market": "USD_M",
    "source_family": "premiumIndexKlines",
    "source_interval": "1h",
    "source_premium_value_distribution_inspected": False,
    "source_target_relation_computed": False,
    "source_strategy_pnl_computed": False,
    "symbols": _SYMBOLS,
    "feature_name": "1h__premium_index_close_bps",
    "feature_kind": "premium_index_close_bps",
    "premium_price_field": "close",
    "feature_formula": "10000*premium_index_close(raw_open_time=t-1h)",
    "feature_scale_bps": 10_000.0,
    "feature_transform": "fixed_scale_only",
    "premium_dataset_timestamp_semantics": "completed_bar_close_boundary",
    "premium_dataset_timestamp_formula": "dataset_timestamp=raw_open_time+1h",
    "premium_decision_raw_open_time_offset_minutes": -60,
    "require_exact_native_timestamp_match": True,
    "require_premium_row_present": True,
    "require_premium_information_available": True,
    "premium_zero_is_valid": True,
    "premium_signed_values_preserved": True,
    "missing_native_row_action": "FEATURE_UNAVAILABLE",
    "stale_carry_allowed": False,
    "nearest_or_asof_alignment_allowed": False,
    "interpolation_allowed": False,
    "forward_fill_allowed": False,
    "source_substitution_allowed": False,
    "future_feature_rows_forbidden": True,
    "prefix_causality_required": True,
    "rolling_window_allowed": False,
    "feature_centering_allowed": False,
    "fitted_feature_normalization_allowed": False,
    "zscore_allowed": False,
    "clipping_allowed": False,
    "winsorization_allowed": False,
    "ema_allowed": False,
    "absolute_value_allowed": False,
    "cross_sectional_normalization_allowed": False,
    "rank_transform_allowed": False,
    "symbol_specific_transform_allowed": False,
    "feature_threshold_allowed": False,
    "alternate_feature_field_allowed": False,
    "alternate_feature_formula_allowed": False,
    "alternate_horizon_allowed": False,
    "alternate_sign_allowed": False,
    "funding_reconstruction_allowed": False,
    "current_funding_formula_backcast_allowed": False,
    "fit_start": datetime(2021, 1, 1, 1, tzinfo=UTC),
    "fit_cutoff": datetime(2023, 1, 1, tzinfo=UTC),
    "last_candidate_decision": datetime(2022, 12, 30, 23, tzinfo=UTC),
    "nominal_candidate_decisions_per_symbol": 17_495,
    "minimum_eligible_observations_per_symbol": 16_621,
    "target_source_authority_required": True,
    "target_source_structural_preflight_required": True,
    "target_source_market": "USD_M",
    "target_source_family": "klines",
    "target_interval": "1h",
    "target_source_replacement_allowed": False,
    "target_dataset_timestamp_semantics": "completed_bar_close_boundary",
    "target_dataset_timestamp_formula": "dataset_timestamp=raw_open_time+1h",
    "target_decision_raw_open_time_offset_minutes": -60,
    "execution_raw_open_time_offset_minutes": 0,
    "endpoint_raw_open_time_offset_minutes": 1_440,
    "label_execution_offset_bars": 1,
    "label_endpoint_offset_bars": 25,
    "label_horizon_bars": 24,
    "label_formula": "log(open[t+25] / open[t+1])",
    "require_label_end_strictly_before_fit_cutoff": True,
    "require_execution_and_label_rows_present": True,
    "require_execution_and_label_rows_contiguous": True,
    "require_execution_and_label_rows_information_available": True,
    "require_execution_and_label_rows_tradable": True,
    "require_execution_and_label_rows_active": True,
    "require_label_open_finite_positive": True,
    "calibration_method": "per_symbol_ols_intercept_fixed_order_fsum",
    "calibration_formula": (
        "x_bar=fsum(x)/n;y_bar=fsum(y)/n;"
        "beta=fsum((x-x_bar)*(y-y_bar))/fsum((x-x_bar)^2);"
        "alpha=y_bar-beta*x_bar"
    ),
    "weighted_regression_allowed": False,
    "robust_regression_fallback_allowed": False,
    "no_intercept_regression_allowed": False,
    "require_calibration_x_mean_finite": True,
    "require_calibration_y_mean_finite": True,
    "require_calibration_numerator_finite": True,
    "require_calibration_denominator_finite_positive": True,
    "require_calibration_alpha_finite": True,
    "require_calibration_beta_finite": True,
    "calibration_intercept_required": True,
    "calibration_centering_required": True,
    "portable_fixed_order_reduction_required": True,
    "expected_effect_direction": "REVERSAL",
    "required_negative_symbol_slopes": 4,
    "valid_status": "VALID_PREMIUM_PRESSURE_REVERSAL",
    "reject_status": "REJECT_PREMIUM_PRESSURE_HYPOTHESIS",
    "invalid_coverage_status": "INVALID_PREMIUM_PRESSURE_COVERAGE",
    "no_sign_flip_fallback": True,
    "no_second_premium_hypothesis": True,
    "no_magnitude_threshold_after_results": True,
    "calibration_slope_used_as_strategy_coefficient": False,
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
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_to_text(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PremiumPressureProtocol:
    """Immutable one-slot premium-pressure research contract."""

    schema_version: str
    issue_number: int
    multiplicity_issue: int
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
    source_market: str
    source_family: str
    source_interval: str
    source_premium_value_distribution_inspected: bool
    source_target_relation_computed: bool
    source_strategy_pnl_computed: bool
    symbols: tuple[str, ...]
    feature_name: str
    feature_kind: str
    premium_price_field: str
    feature_formula: str
    feature_scale_bps: float
    feature_transform: str
    premium_dataset_timestamp_semantics: str
    premium_dataset_timestamp_formula: str
    premium_decision_raw_open_time_offset_minutes: int
    require_exact_native_timestamp_match: bool
    require_premium_row_present: bool
    require_premium_information_available: bool
    premium_zero_is_valid: bool
    premium_signed_values_preserved: bool
    missing_native_row_action: str
    stale_carry_allowed: bool
    nearest_or_asof_alignment_allowed: bool
    interpolation_allowed: bool
    forward_fill_allowed: bool
    source_substitution_allowed: bool
    future_feature_rows_forbidden: bool
    prefix_causality_required: bool
    rolling_window_allowed: bool
    feature_centering_allowed: bool
    fitted_feature_normalization_allowed: bool
    zscore_allowed: bool
    clipping_allowed: bool
    winsorization_allowed: bool
    ema_allowed: bool
    absolute_value_allowed: bool
    cross_sectional_normalization_allowed: bool
    rank_transform_allowed: bool
    symbol_specific_transform_allowed: bool
    feature_threshold_allowed: bool
    alternate_feature_field_allowed: bool
    alternate_feature_formula_allowed: bool
    alternate_horizon_allowed: bool
    alternate_sign_allowed: bool
    funding_reconstruction_allowed: bool
    current_funding_formula_backcast_allowed: bool
    fit_start: datetime
    fit_cutoff: datetime
    last_candidate_decision: datetime
    nominal_candidate_decisions_per_symbol: int
    minimum_eligible_observations_per_symbol: int
    target_source_authority_required: bool
    target_source_structural_preflight_required: bool
    target_source_market: str
    target_source_family: str
    target_interval: str
    target_source_replacement_allowed: bool
    target_dataset_timestamp_semantics: str
    target_dataset_timestamp_formula: str
    target_decision_raw_open_time_offset_minutes: int
    execution_raw_open_time_offset_minutes: int
    endpoint_raw_open_time_offset_minutes: int
    label_execution_offset_bars: int
    label_endpoint_offset_bars: int
    label_horizon_bars: int
    label_formula: str
    require_label_end_strictly_before_fit_cutoff: bool
    require_execution_and_label_rows_present: bool
    require_execution_and_label_rows_contiguous: bool
    require_execution_and_label_rows_information_available: bool
    require_execution_and_label_rows_tradable: bool
    require_execution_and_label_rows_active: bool
    require_label_open_finite_positive: bool
    calibration_method: str
    calibration_formula: str
    weighted_regression_allowed: bool
    robust_regression_fallback_allowed: bool
    no_intercept_regression_allowed: bool
    require_calibration_x_mean_finite: bool
    require_calibration_y_mean_finite: bool
    require_calibration_numerator_finite: bool
    require_calibration_denominator_finite_positive: bool
    require_calibration_alpha_finite: bool
    require_calibration_beta_finite: bool
    calibration_intercept_required: bool
    calibration_centering_required: bool
    portable_fixed_order_reduction_required: bool
    expected_effect_direction: str
    required_negative_symbol_slopes: int
    valid_status: str
    reject_status: str
    invalid_coverage_status: str
    no_sign_flip_fallback: bool
    no_second_premium_hypothesis: bool
    no_magnitude_threshold_after_results: bool
    calibration_slope_used_as_strategy_coefficient: bool
    training_relation_executed: bool
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    production_eligible: bool
    live_trading_authorized: bool

    def __post_init__(self) -> None:
        for field_name in ("fit_start", "fit_cutoff", "last_candidate_decision"):
            _require_aware(getattr(self, field_name), field=field_name)
        if not self.fit_start < self.last_candidate_decision < self.fit_cutoff:
            raise ValueError("preregistered training clock is invalid")
        if (
            isinstance(self.feature_scale_bps, bool)
            or not isinstance(self.feature_scale_bps, (int, float))
            or not math.isfinite(float(self.feature_scale_bps))
            or float(self.feature_scale_bps) <= 0.0
        ):
            raise ValueError("feature_scale_bps must be finite and positive")

        integer_fields = (
            "issue_number",
            "multiplicity_issue",
            "source_issue",
            "source_publisher_run_id",
            "source_artifact_id",
            "source_fresh_artifact_id",
            "premium_decision_raw_open_time_offset_minutes",
            "nominal_candidate_decisions_per_symbol",
            "minimum_eligible_observations_per_symbol",
            "target_decision_raw_open_time_offset_minutes",
            "execution_raw_open_time_offset_minutes",
            "endpoint_raw_open_time_offset_minutes",
            "label_execution_offset_bars",
            "label_endpoint_offset_bars",
            "label_horizon_bars",
            "required_negative_symbol_slopes",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        for field_name in (
            "issue_number",
            "multiplicity_issue",
            "source_issue",
            "source_publisher_run_id",
            "source_artifact_id",
            "source_fresh_artifact_id",
            "nominal_candidate_decisions_per_symbol",
            "minimum_eligible_observations_per_symbol",
            "label_execution_offset_bars",
            "label_endpoint_offset_bars",
            "label_horizon_bars",
            "required_negative_symbol_slopes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if not 1 <= self.required_negative_symbol_slopes <= len(self.symbols):
            raise ValueError("negative-slope gate is invalid")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbol roster must be unique")
        if self.label_endpoint_offset_bars - self.label_execution_offset_bars != (
            self.label_horizon_bars
        ):
            raise ValueError("label offsets do not match frozen horizon")
        if self.endpoint_raw_open_time_offset_minutes != self.label_horizon_bars * 60:
            raise ValueError("endpoint raw-open offset does not match frozen horizon")
        if self.execution_raw_open_time_offset_minutes != 0:
            raise ValueError("execution raw-open offset must be the decision boundary")
        if self.premium_decision_raw_open_time_offset_minutes != -60:
            raise ValueError("premium decision row must be the completed prior hour")
        if self.target_decision_raw_open_time_offset_minutes != -60:
            raise ValueError("target decision row must be the completed prior hour")
        if self.minimum_eligible_observations_per_symbol > (
            self.nominal_candidate_decisions_per_symbol
        ):
            raise ValueError("minimum eligible observations exceed nominal decisions")

        endpoint_raw_open_offset = timedelta(
            minutes=self.endpoint_raw_open_time_offset_minutes
        )
        one_hour = timedelta(hours=1)
        if not (
            self.last_candidate_decision + endpoint_raw_open_offset < self.fit_cutoff
        ):
            raise ValueError("last candidate endpoint raw open must precede fit cutoff")
        if (
            self.last_candidate_decision + endpoint_raw_open_offset + one_hour
            != self.fit_cutoff
        ):
            raise ValueError(
                "last candidate is not the maximal eligible hourly decision"
            )
        span = self.last_candidate_decision - self.fit_start
        if span.total_seconds() < 0 or span.total_seconds() % 3600 != 0:
            raise ValueError("candidate clock must be an exact hourly grid")
        expected_nominal = int(span.total_seconds() // 3600) + 1
        if self.nominal_candidate_decisions_per_symbol != expected_nominal:
            raise ValueError("nominal candidate count differs from frozen clock")
        expected_minimum = math.ceil(0.95 * expected_nominal)
        if self.minimum_eligible_observations_per_symbol != expected_minimum:
            raise ValueError("minimum eligible count differs from frozen 95% gate")

        for field_name, expected in _CANONICAL_FIELD_VALUES.items():
            if not _strict_equal(getattr(self, field_name), expected):
                raise ValueError(f"{field_name} differs from frozen Issue 600 contract")

        _require_hex(self.source_probe_head, length=40, field="source_probe_head")
        for field_name in (
            "source_artifact_api_digest",
            "source_fresh_artifact_api_digest",
            "source_report_sha256",
            "source_report_content_digest",
        ):
            _require_hex(getattr(self, field_name), length=64, field=field_name)

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: _json_value(getattr(self, field.name)) for field in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PremiumPressureProtocol:
        expected_fields = {field.name for field in fields(cls)}
        if set(payload) != expected_fields:
            raise ValueError(
                "premium-pressure protocol fields differ from frozen schema"
            )
        resolved = dict(payload)
        for field_name in ("fit_start", "fit_cutoff", "last_candidate_decision"):
            resolved[field_name] = _datetime_from_text(
                resolved[field_name], field=field_name
            )
        symbols = resolved["symbols"]
        if not isinstance(symbols, list) or any(
            not isinstance(item, str) for item in symbols
        ):
            raise ValueError("symbols must be a JSON string array")
        resolved["symbols"] = tuple(symbols)
        try:
            return cls(**resolved)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "premium-pressure protocol payload is malformed"
            ) from error


def canonical_premium_pressure_protocol() -> PremiumPressureProtocol:
    return PremiumPressureProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def canonical_premium_pressure_protocol_bytes(
    protocol: PremiumPressureProtocol | None = None,
) -> bytes:
    resolved = protocol or canonical_premium_pressure_protocol()
    if resolved != canonical_premium_pressure_protocol():
        raise ValueError("only the frozen Issue 600 protocol is canonical")
    return canonical_json_bytes(resolved.to_dict())


def load_premium_pressure_protocol_bytes(payload: bytes) -> PremiumPressureProtocol:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        raw: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("premium-pressure protocol is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("premium-pressure protocol must be a JSON object")
    protocol = PremiumPressureProtocol.from_dict(raw)
    if payload != canonical_premium_pressure_protocol_bytes(protocol):
        raise ValueError("premium-pressure protocol bytes are not canonical")
    return protocol
