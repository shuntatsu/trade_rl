"""Sealed pre-result contract for the training-only signed taker-flow gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "signed_taker_flow_prereg_v1"
_ISSUE_NUMBER = 558
_SOURCE_ISSUE = 556
_SOURCE_PUBLISHER_RUN_ID = 34840494198
_SOURCE_ARTIFACT_ID = 10345533785
_SOURCE_ARTIFACT_API_DIGEST = (
    "689188893dec90b41819972e1cb151920ab198bf6bf27b1abeae2ab18e2fac0f"
)
_SOURCE_REPORT_CONTENT_DIGEST = (
    "e3b627cc8efa63623135512d344a883c1b1be8a0e1bc169c24605ec63abaa693"
)
_SOURCE_FRESH_RUN_ID = 34840645375
_SOURCE_FRESH_ARTIFACT_ID = 10346140972
_SOURCE_FRESH_ARTIFACT_API_DIGEST = (
    "16f91dcaab5f1914d19ee93934643cf27ad3ab94e1787cb04123b8dc3ad68a60"
)
_SOURCE_STATUS = "PASS"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_SOURCE_QUOTE_VOLUME_FIELD = "quote_volume"
_SOURCE_TAKER_BUY_QUOTE_VOLUME_FIELD = "taker_buy_quote_volume"
_FEATURE_NAME = "1h__signed_taker_quote_flow_24bar"
_FEATURE_KIND = "signed_taker_quote_flow"
_FEATURE_LOOKBACK_BARS = 24
_FEATURE_FORMULA = (
    "(2*fsum(taker_buy_quote_volume[t-23:t+1])"
    "-fsum(quote_volume[t-23:t+1]))/fsum(quote_volume[t-23:t+1])"
)
_FEATURE_MINIMUM_QUOTE_VOLUME = math.nextafter(0.0, math.inf)
_FEATURE_LOWER_BOUND = -1.0
_FEATURE_UPPER_BOUND = 1.0
_FIT_START = datetime(2021, 1, 1, 1, tzinfo=UTC)
_FIT_CUTOFF = datetime(2023, 1, 1, tzinfo=UTC)
_LABEL_EXECUTION_OFFSET_BARS = 1
_LABEL_ENDPOINT_OFFSET_BARS = 25
_LABEL_HORIZON_BARS = 24
_LABEL_FORMULA = "log(open[t+25] / open[t+1])"
_MINIMUM_ELIGIBLE_OBSERVATIONS_PER_SYMBOL = 8760
_FEATURE_WINDOW_START_OFFSET_BARS = -23
_FEATURE_WINDOW_STOP_OFFSET_BARS_INCLUSIVE = 0
_REQUIRE_ALL_FEATURE_ROWS_PRESENT = True
_REQUIRE_ALL_FEATURE_ROWS_INFORMATION_AVAILABLE = True
_REQUIRE_ALL_FEATURE_ROWS_ACTIVE = True
_REQUIRE_QUOTE_VOLUME_FINITE_NONNEGATIVE = True
_REQUIRE_TAKER_VOLUME_FINITE_NONNEGATIVE = True
_REQUIRE_TAKER_NOT_ABOVE_QUOTE = True
_ZERO_QUOTE_DENOMINATOR_ACTION = "UNAVAILABLE"
_FUTURE_FEATURE_ROWS_FORBIDDEN = True
_REQUIRE_LABEL_END_STRICTLY_BEFORE_FIT_CUTOFF = True
_REQUIRE_EXECUTION_AND_LABEL_ROWS_TRADABLE = True
_REQUIRE_EXECUTION_AND_LABEL_ROWS_ACTIVE = True
_CALIBRATION_METHOD = "per_symbol_no_intercept_fixed_order_fsum"
_CALIBRATION_FORMULA = "beta_i = fsum(x_t*y_t) / fsum(x_t*x_t)"
_EXPECTED_EFFECT_DIRECTION = "CONTINUATION"
_REQUIRED_POSITIVE_SYMBOL_SLOPES = 4
_VALID_STATUS = "VALID_FLOW_HYPOTHESIS"
_REJECT_STATUS = "REJECT_FLOW_HYPOTHESIS"
_INVALID_COVERAGE_STATUS = "INVALID_FLOW_COVERAGE"
_NO_SIGN_FLIP_FALLBACK = True
_NO_MAGNITUDE_THRESHOLD_AFTER_RESULTS = True
_CALIBRATION_SLOPE_USED_AS_STRATEGY_COEFFICIENT = False
_RAW_FIELD_OPTIONAL_FOR_LEGACY_SOURCES = True
_FEATURE_REQUIRES_RAW_TAKER_FIELD = True
_MISSING_RAW_TAKER_FIELD_ACTION = "FEATURE_UNAVAILABLE"
_LEGACY_DATASET_BEHAVIOR_UNCHANGED_WHEN_FEATURE_OMITTED = True
_EXISTING_FEATURE_BYTES_UNCHANGED_WHEN_FEATURE_OMITTED = True
_PORTABLE_FIXED_ORDER_REDUCTION_REQUIRED = True
_PREFIX_CAUSALITY_REQUIRED = True
_TRAINING_RELATION_EXECUTED = False
_EVALUATION_PNL_INSPECTED = False
_EVALUATION_EXECUTION_AUTHORIZED = False
_PRODUCTION_ELIGIBLE = False
_FINAL_TEST_AUTHORIZED = False
_SHARED_CASH_PROFITABILITY_ESTABLISHED = False
_LIVE_TRADING_AUTHORIZED = False

_EXPECTED_FIELDS: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": _ISSUE_NUMBER,
    "source_issue": _SOURCE_ISSUE,
    "source_publisher_run_id": _SOURCE_PUBLISHER_RUN_ID,
    "source_artifact_id": _SOURCE_ARTIFACT_ID,
    "source_artifact_api_digest": _SOURCE_ARTIFACT_API_DIGEST,
    "source_report_content_digest": _SOURCE_REPORT_CONTENT_DIGEST,
    "source_fresh_run_id": _SOURCE_FRESH_RUN_ID,
    "source_fresh_artifact_id": _SOURCE_FRESH_ARTIFACT_ID,
    "source_fresh_artifact_api_digest": _SOURCE_FRESH_ARTIFACT_API_DIGEST,
    "source_status": _SOURCE_STATUS,
    "symbols": _SYMBOLS,
    "source_quote_volume_field": _SOURCE_QUOTE_VOLUME_FIELD,
    "source_taker_buy_quote_volume_field": _SOURCE_TAKER_BUY_QUOTE_VOLUME_FIELD,
    "feature_name": _FEATURE_NAME,
    "feature_kind": _FEATURE_KIND,
    "feature_lookback_bars": _FEATURE_LOOKBACK_BARS,
    "feature_formula": _FEATURE_FORMULA,
    "feature_minimum_quote_volume": _FEATURE_MINIMUM_QUOTE_VOLUME,
    "feature_lower_bound": _FEATURE_LOWER_BOUND,
    "feature_upper_bound": _FEATURE_UPPER_BOUND,
    "fit_start": _FIT_START,
    "fit_cutoff": _FIT_CUTOFF,
    "label_execution_offset_bars": _LABEL_EXECUTION_OFFSET_BARS,
    "label_endpoint_offset_bars": _LABEL_ENDPOINT_OFFSET_BARS,
    "label_horizon_bars": _LABEL_HORIZON_BARS,
    "label_formula": _LABEL_FORMULA,
    "minimum_eligible_observations_per_symbol": (
        _MINIMUM_ELIGIBLE_OBSERVATIONS_PER_SYMBOL
    ),
    "feature_window_start_offset_bars": _FEATURE_WINDOW_START_OFFSET_BARS,
    "feature_window_stop_offset_bars_inclusive": (
        _FEATURE_WINDOW_STOP_OFFSET_BARS_INCLUSIVE
    ),
    "require_all_feature_rows_present": _REQUIRE_ALL_FEATURE_ROWS_PRESENT,
    "require_all_feature_rows_information_available": (
        _REQUIRE_ALL_FEATURE_ROWS_INFORMATION_AVAILABLE
    ),
    "require_all_feature_rows_active": _REQUIRE_ALL_FEATURE_ROWS_ACTIVE,
    "require_quote_volume_finite_nonnegative": (
        _REQUIRE_QUOTE_VOLUME_FINITE_NONNEGATIVE
    ),
    "require_taker_volume_finite_nonnegative": (
        _REQUIRE_TAKER_VOLUME_FINITE_NONNEGATIVE
    ),
    "require_taker_not_above_quote": _REQUIRE_TAKER_NOT_ABOVE_QUOTE,
    "zero_quote_denominator_action": _ZERO_QUOTE_DENOMINATOR_ACTION,
    "future_feature_rows_forbidden": _FUTURE_FEATURE_ROWS_FORBIDDEN,
    "require_label_end_strictly_before_fit_cutoff": (
        _REQUIRE_LABEL_END_STRICTLY_BEFORE_FIT_CUTOFF
    ),
    "require_execution_and_label_rows_tradable": (
        _REQUIRE_EXECUTION_AND_LABEL_ROWS_TRADABLE
    ),
    "require_execution_and_label_rows_active": (
        _REQUIRE_EXECUTION_AND_LABEL_ROWS_ACTIVE
    ),
    "calibration_method": _CALIBRATION_METHOD,
    "calibration_formula": _CALIBRATION_FORMULA,
    "expected_effect_direction": _EXPECTED_EFFECT_DIRECTION,
    "required_positive_symbol_slopes": _REQUIRED_POSITIVE_SYMBOL_SLOPES,
    "valid_status": _VALID_STATUS,
    "reject_status": _REJECT_STATUS,
    "invalid_coverage_status": _INVALID_COVERAGE_STATUS,
    "no_sign_flip_fallback": _NO_SIGN_FLIP_FALLBACK,
    "no_magnitude_threshold_after_results": _NO_MAGNITUDE_THRESHOLD_AFTER_RESULTS,
    "calibration_slope_used_as_strategy_coefficient": (
        _CALIBRATION_SLOPE_USED_AS_STRATEGY_COEFFICIENT
    ),
    "raw_field_optional_for_legacy_sources": _RAW_FIELD_OPTIONAL_FOR_LEGACY_SOURCES,
    "feature_requires_raw_taker_field": _FEATURE_REQUIRES_RAW_TAKER_FIELD,
    "missing_raw_taker_field_action": _MISSING_RAW_TAKER_FIELD_ACTION,
    "legacy_dataset_behavior_unchanged_when_feature_omitted": (
        _LEGACY_DATASET_BEHAVIOR_UNCHANGED_WHEN_FEATURE_OMITTED
    ),
    "existing_feature_bytes_unchanged_when_feature_omitted": (
        _EXISTING_FEATURE_BYTES_UNCHANGED_WHEN_FEATURE_OMITTED
    ),
    "portable_fixed_order_reduction_required": (
        _PORTABLE_FIXED_ORDER_REDUCTION_REQUIRED
    ),
    "prefix_causality_required": _PREFIX_CAUSALITY_REQUIRED,
    "training_relation_executed": _TRAINING_RELATION_EXECUTED,
    "evaluation_pnl_inspected": _EVALUATION_PNL_INSPECTED,
    "evaluation_execution_authorized": _EVALUATION_EXECUTION_AUTHORIZED,
    "production_eligible": _PRODUCTION_ELIGIBLE,
    "final_test_authorized": _FINAL_TEST_AUTHORIZED,
    "shared_cash_profitability_established": _SHARED_CASH_PROFITABILITY_ESTABLISHED,
    "live_trading_authorized": _LIVE_TRADING_AUTHORIZED,
}


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class SignedTakerFlowProtocol:
    """Immutable result-blind preregistration for Issue #558."""

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
    feature_minimum_quote_volume: float
    feature_lower_bound: float
    feature_upper_bound: float
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
    require_quote_volume_finite_nonnegative: bool
    require_taker_volume_finite_nonnegative: bool
    require_taker_not_above_quote: bool
    zero_quote_denominator_action: str
    future_feature_rows_forbidden: bool
    require_label_end_strictly_before_fit_cutoff: bool
    require_execution_and_label_rows_tradable: bool
    require_execution_and_label_rows_active: bool
    calibration_method: str
    calibration_formula: str
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
        for field_name, expected in _EXPECTED_FIELDS.items():
            actual = getattr(self, field_name)
            if type(actual) is not type(expected):
                raise ValueError(
                    "signed taker-flow fields differ from preregistered contract"
                )
            if isinstance(expected, datetime):
                if not isinstance(actual, datetime):
                    raise ValueError(
                        "signed taker-flow fields differ from preregistered contract"
                    )
                if actual.tzinfo is None or actual.utcoffset() is None:
                    raise ValueError(f"{field_name} must be timezone-aware")
                actual = actual.astimezone(UTC)
            if isinstance(expected, float) and not math.isfinite(actual):
                raise ValueError(f"{field_name} must be finite")
            if actual != expected:
                raise ValueError(
                    "signed taker-flow fields differ from preregistered contract"
                )

    def _payload_without_digest(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field_name in _EXPECTED_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                payload[field_name] = _iso_utc(value)
            elif isinstance(value, tuple):
                payload[field_name] = list(value)
            else:
                payload[field_name] = value
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self._payload_without_digest())

    def to_payload(self) -> dict[str, object]:
        payload = self._payload_without_digest()
        payload["protocol_digest"] = self.digest
        return payload


def canonical_signed_taker_flow_protocol() -> SignedTakerFlowProtocol:
    """Return the single frozen Issue #558 preregistration."""

    return SignedTakerFlowProtocol(**cast(dict[str, object], _EXPECTED_FIELDS.copy()))


def load_signed_taker_flow_protocol(path: str | Path) -> SignedTakerFlowProtocol:
    """Load only the exact sealed canonical preregistration payload."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("signed taker-flow protocol payload must be a JSON object")
    payload = cast(dict[str, object], raw)
    canonical = canonical_signed_taker_flow_protocol()
    expected = canonical.to_payload()
    if set(payload) != set(expected):
        raise ValueError("signed taker-flow protocol payload keys differ from canonical")

    minimum = payload["feature_minimum_quote_volume"]
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise ValueError("feature_minimum_quote_volume must be finite")
    if not math.isfinite(float(minimum)):
        raise ValueError("feature_minimum_quote_volume must be finite")

    if payload != expected:
        raise ValueError("signed taker-flow payload differs from preregistered contract")
    return canonical


__all__ = [
    "SignedTakerFlowProtocol",
    "canonical_signed_taker_flow_protocol",
    "load_signed_taker_flow_protocol",
]
