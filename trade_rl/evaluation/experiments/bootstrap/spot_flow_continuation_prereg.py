"""Result-blind preregistration for the Issue 584 Spot-flow continuation diagnostic."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "spot_flow_continuation_prereg_v1"
_ISSUE_NUMBER = 584
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_SPOT_DATES = ("2021-01-15", "2021-07-15", "2022-01-15", "2022-07-15")
_DECISION_MINUTES = tuple(range(15, 24 * 60 + 1, 15))

_SPOT_SOURCE_ISSUE = 578
_SPOT_SOURCE_STATUS = "PASS_SPOT_AGGTRADES_SOURCE"
_SPOT_SOURCE_PROTOCOL_HEAD = "9932775127c79a85d81fb304850c3349460e4a2a"
_SPOT_SOURCE_PROTOCOL_DIGEST = (
    "808f999915815259762067aaff381a88589a7c28ceb0383d8aa0626d5c67e8d4"
)
_SPOT_SOURCE_PROTOCOL_SEAL_RUN_ID = 34928427543
_SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_ID = 10381040623
_SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_API_DIGEST = (
    "84b3637ab61dcc6ba9942a2b41907db326bc5fd48643e437cc4505ff17f9f62f"
)
_SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_ID = 10381305016
_SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_API_DIGEST = (
    "d15223aa90a1feaeb8cd85baad9d374422f55d4cfb7d262e52ecc9e7978440a0"
)
_SPOT_SOURCE_VALIDATOR_HEAD = "a8234444cb4306c37d80c47832200f8303f8928c"
_SPOT_SOURCE_VALIDATOR_VERIFICATION_RUN_ID = 34930082702
_SPOT_SOURCE_EXECUTION_RUN_ID = 34930477567
_SPOT_SOURCE_PUBLISHER_ARTIFACT_ID = 10381815598
_SPOT_SOURCE_PUBLISHER_ARTIFACT_API_DIGEST = (
    "faad983c85f496f6c6f88fa6adacdd0f53af785d35e3c15778ace17378d7a0a7"
)
_SPOT_SOURCE_FRESH_ARTIFACT_ID = 10381247817
_SPOT_SOURCE_FRESH_ARTIFACT_API_DIGEST = (
    "0dc8faba79881ce36489bbf83f2526b7aae6fd6441539e46e0bec92f660b020f"
)
_SPOT_SOURCE_RESULT_SHA256 = (
    "dfa2a5350ffdd416c1bc06fb48539cf882a009910929b9873fd7942ef53e6030"
)
_SPOT_SOURCE_RESULT_CONTENT_DIGEST = (
    "06207c17f997ebd6c99db553cee07881a10e8b54d16af062a47e0fcb0a2cb51a"
)
_SPOT_SOURCE_USABLE_ROWS = 13_598_332
_SPOT_SOURCE_MALFORMED_ROWS = 0
_SPOT_SOURCE_INVALID_SENTINEL_ROWS = 0

_FEATURE_NAME = "15m__spot_aggressive_notional_imbalance"
_SIGNAL_WINDOW_MINUTES = 15
_WEIGHTING = "quote_notional_price_times_quantity"
_INTERVAL_SEMANTICS = "left_closed_right_open"
_FEATURE_FORMULA = "math.fsum(sign_j*price_j*quantity_j)/math.fsum(price_j*quantity_j)"
_BUYER_TOKEN = "False"
_SELLER_TOKEN = "True"
_BUYER_SIGN = 1
_SELLER_SIGN = -1

_TARGET_MARKET = "binance_usdm_perpetual"
_TARGET_TIMEFRAME = "15m"
_TARGET_SOURCE_FAMILY = "binance_vision_contract_klines"
_TARGET_TRANSPORT_MODE = "VISION"
_EXECUTION_OFFSET = 1
_ENDPOINT_OFFSET = 17
_HORIZON_BARS = 16
_HORIZON_MINUTES = 240
_LABEL_FORMULA = "log(open[t+17]/open[t+1])"
_EXPECTED_DIRECTION = "CONTINUATION"
_MIN_OBSERVATIONS = 360
_REQUIRED_POSITIVE = 4
_VALID_STATUS = "VALID_SPOT_FLOW_CONTINUATION"
_REJECT_STATUS = "REJECT_SPOT_FLOW_HYPOTHESIS"
_INVALID_STATUS = "INVALID_SPOT_FLOW_COVERAGE"
_FIXED_REDUCTION = "math.fsum_chronological"


def _canonical_values() -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "issue_number": _ISSUE_NUMBER,
        "symbols": _SYMBOLS,
        "spot_dates": _SPOT_DATES,
        "spot_archive_count": 20,
        "spot_source_issue": _SPOT_SOURCE_ISSUE,
        "spot_source_status": _SPOT_SOURCE_STATUS,
        "spot_source_protocol_head": _SPOT_SOURCE_PROTOCOL_HEAD,
        "spot_source_protocol_digest": _SPOT_SOURCE_PROTOCOL_DIGEST,
        "spot_source_protocol_seal_run_id": _SPOT_SOURCE_PROTOCOL_SEAL_RUN_ID,
        "spot_source_protocol_seal_artifact_id": _SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_ID,
        "spot_source_protocol_seal_artifact_api_digest": (
            _SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_API_DIGEST
        ),
        "spot_source_protocol_fresh_artifact_id": _SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_ID,
        "spot_source_protocol_fresh_artifact_api_digest": (
            _SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_API_DIGEST
        ),
        "spot_source_validator_head": _SPOT_SOURCE_VALIDATOR_HEAD,
        "spot_source_validator_verification_run_id": (
            _SPOT_SOURCE_VALIDATOR_VERIFICATION_RUN_ID
        ),
        "spot_source_execution_run_id": _SPOT_SOURCE_EXECUTION_RUN_ID,
        "spot_source_publisher_artifact_id": _SPOT_SOURCE_PUBLISHER_ARTIFACT_ID,
        "spot_source_publisher_artifact_api_digest": (
            _SPOT_SOURCE_PUBLISHER_ARTIFACT_API_DIGEST
        ),
        "spot_source_fresh_artifact_id": _SPOT_SOURCE_FRESH_ARTIFACT_ID,
        "spot_source_fresh_artifact_api_digest": _SPOT_SOURCE_FRESH_ARTIFACT_API_DIGEST,
        "spot_source_result_sha256": _SPOT_SOURCE_RESULT_SHA256,
        "spot_source_result_content_digest": _SPOT_SOURCE_RESULT_CONTENT_DIGEST,
        "spot_source_usable_rows": _SPOT_SOURCE_USABLE_ROWS,
        "spot_source_malformed_rows": _SPOT_SOURCE_MALFORMED_ROWS,
        "spot_source_invalid_sentinel_rows": _SPOT_SOURCE_INVALID_SENTINEL_ROWS,
        "feature_name": _FEATURE_NAME,
        "signal_window_minutes": _SIGNAL_WINDOW_MINUTES,
        "candidate_intervals_per_day": 96,
        "candidate_decisions_per_symbol": 384,
        "decision_minutes_after_day_start": _DECISION_MINUTES,
        "buyer_aggressor_token": _BUYER_TOKEN,
        "seller_aggressor_token": _SELLER_TOKEN,
        "buyer_aggressor_sign": _BUYER_SIGN,
        "seller_aggressor_sign": _SELLER_SIGN,
        "weighting": _WEIGHTING,
        "interval_semantics": _INTERVAL_SEMANTICS,
        "feature_formula": _FEATURE_FORMULA,
        "empty_interval_is_unavailable": True,
        "sentinel_rows_are_feature_events": False,
        "use_best_price_match": False,
        "provider_order_required": True,
        "event_time_strictly_before_decision": True,
        "archive_publication_time_used_as_market_event_time": False,
        "clipping_allowed": False,
        "winsorization_allowed": False,
        "zscore_allowed": False,
        "volume_scaling_allowed": False,
        "volatility_scaling_allowed": False,
        "funding_combination_allowed": False,
        "basis_combination_allowed": False,
        "price_return_input_allowed": False,
        "cross_sectional_normalization_allowed": False,
        "rank_transform_allowed": False,
        "learned_coefficient_allowed": False,
        "feature_threshold_allowed": False,
        "target_market": _TARGET_MARKET,
        "target_timeframe": _TARGET_TIMEFRAME,
        "target_source_family": _TARGET_SOURCE_FAMILY,
        "target_transport_mode": _TARGET_TRANSPORT_MODE,
        "target_source_fallback_allowed": False,
        "target_source_replacement_allowed": False,
        "target_source_checksum_required": True,
        "target_source_manifest_binding_required": True,
        "target_source_structural_preflight_required": True,
        "execution_open_offset_bars": _EXECUTION_OFFSET,
        "endpoint_open_offset_bars": _ENDPOINT_OFFSET,
        "horizon_bars": _HORIZON_BARS,
        "horizon_minutes": _HORIZON_MINUTES,
        "label_formula": _LABEL_FORMULA,
        "require_target_rows_present": True,
        "require_target_rows_contiguous": True,
        "require_target_rows_information_available": True,
        "require_target_rows_active": True,
        "require_target_rows_tradable": True,
        "require_target_open_finite_positive": True,
        "target_2023_or_later_allowed": False,
        "expected_direction": _EXPECTED_DIRECTION,
        "minimum_eligible_observations_per_symbol": _MIN_OBSERVATIONS,
        "required_positive_symbol_slopes": _REQUIRED_POSITIVE,
        "valid_status": _VALID_STATUS,
        "reject_status": _REJECT_STATUS,
        "invalid_coverage_status": _INVALID_STATUS,
        "no_intercept": True,
        "fixed_reduction": _FIXED_REDUCTION,
        "weighted_regression_allowed": False,
        "robust_regression_allowed": False,
        "require_numerator_finite": True,
        "require_denominator_finite_positive": True,
        "require_beta_finite": True,
        "no_magnitude_threshold_after_results": True,
        "training_relation_executed": False,
        "alternate_sign_allowed": False,
        "alternate_window_allowed": False,
        "alternate_horizon_allowed": False,
        "alternate_weighting_allowed": False,
        "symbol_subset_allowed": False,
        "additional_spot_dates_allowed": False,
        "economic_values_inspected": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "final_test_authorized": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class SpotFlowContinuationProtocol:
    """Immutable Issue 584 protocol. Every field is a preregistered constant."""

    schema_version: str = _SCHEMA_VERSION
    issue_number: int = _ISSUE_NUMBER
    symbols: tuple[str, ...] = _SYMBOLS
    spot_dates: tuple[str, ...] = _SPOT_DATES
    spot_archive_count: int = 20
    spot_source_issue: int = _SPOT_SOURCE_ISSUE
    spot_source_status: str = _SPOT_SOURCE_STATUS
    spot_source_protocol_head: str = _SPOT_SOURCE_PROTOCOL_HEAD
    spot_source_protocol_digest: str = _SPOT_SOURCE_PROTOCOL_DIGEST
    spot_source_protocol_seal_run_id: int = _SPOT_SOURCE_PROTOCOL_SEAL_RUN_ID
    spot_source_protocol_seal_artifact_id: int = _SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_ID
    spot_source_protocol_seal_artifact_api_digest: str = (
        _SPOT_SOURCE_PROTOCOL_SEAL_ARTIFACT_API_DIGEST
    )
    spot_source_protocol_fresh_artifact_id: int = (
        _SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_ID
    )
    spot_source_protocol_fresh_artifact_api_digest: str = (
        _SPOT_SOURCE_PROTOCOL_FRESH_ARTIFACT_API_DIGEST
    )
    spot_source_validator_head: str = _SPOT_SOURCE_VALIDATOR_HEAD
    spot_source_validator_verification_run_id: int = (
        _SPOT_SOURCE_VALIDATOR_VERIFICATION_RUN_ID
    )
    spot_source_execution_run_id: int = _SPOT_SOURCE_EXECUTION_RUN_ID
    spot_source_publisher_artifact_id: int = _SPOT_SOURCE_PUBLISHER_ARTIFACT_ID
    spot_source_publisher_artifact_api_digest: str = (
        _SPOT_SOURCE_PUBLISHER_ARTIFACT_API_DIGEST
    )
    spot_source_fresh_artifact_id: int = _SPOT_SOURCE_FRESH_ARTIFACT_ID
    spot_source_fresh_artifact_api_digest: str = _SPOT_SOURCE_FRESH_ARTIFACT_API_DIGEST
    spot_source_result_sha256: str = _SPOT_SOURCE_RESULT_SHA256
    spot_source_result_content_digest: str = _SPOT_SOURCE_RESULT_CONTENT_DIGEST
    spot_source_usable_rows: int = _SPOT_SOURCE_USABLE_ROWS
    spot_source_malformed_rows: int = _SPOT_SOURCE_MALFORMED_ROWS
    spot_source_invalid_sentinel_rows: int = _SPOT_SOURCE_INVALID_SENTINEL_ROWS
    feature_name: str = _FEATURE_NAME
    signal_window_minutes: int = _SIGNAL_WINDOW_MINUTES
    candidate_intervals_per_day: int = 96
    candidate_decisions_per_symbol: int = 384
    decision_minutes_after_day_start: tuple[int, ...] = _DECISION_MINUTES
    buyer_aggressor_token: str = _BUYER_TOKEN
    seller_aggressor_token: str = _SELLER_TOKEN
    buyer_aggressor_sign: int = _BUYER_SIGN
    seller_aggressor_sign: int = _SELLER_SIGN
    weighting: str = _WEIGHTING
    interval_semantics: str = _INTERVAL_SEMANTICS
    feature_formula: str = _FEATURE_FORMULA
    empty_interval_is_unavailable: bool = True
    sentinel_rows_are_feature_events: bool = False
    use_best_price_match: bool = False
    provider_order_required: bool = True
    event_time_strictly_before_decision: bool = True
    archive_publication_time_used_as_market_event_time: bool = False
    clipping_allowed: bool = False
    winsorization_allowed: bool = False
    zscore_allowed: bool = False
    volume_scaling_allowed: bool = False
    volatility_scaling_allowed: bool = False
    funding_combination_allowed: bool = False
    basis_combination_allowed: bool = False
    price_return_input_allowed: bool = False
    cross_sectional_normalization_allowed: bool = False
    rank_transform_allowed: bool = False
    learned_coefficient_allowed: bool = False
    feature_threshold_allowed: bool = False
    target_market: str = _TARGET_MARKET
    target_timeframe: str = _TARGET_TIMEFRAME
    target_source_family: str = _TARGET_SOURCE_FAMILY
    target_transport_mode: str = _TARGET_TRANSPORT_MODE
    target_source_fallback_allowed: bool = False
    target_source_replacement_allowed: bool = False
    target_source_checksum_required: bool = True
    target_source_manifest_binding_required: bool = True
    target_source_structural_preflight_required: bool = True
    execution_open_offset_bars: int = _EXECUTION_OFFSET
    endpoint_open_offset_bars: int = _ENDPOINT_OFFSET
    horizon_bars: int = _HORIZON_BARS
    horizon_minutes: int = _HORIZON_MINUTES
    label_formula: str = _LABEL_FORMULA
    require_target_rows_present: bool = True
    require_target_rows_contiguous: bool = True
    require_target_rows_information_available: bool = True
    require_target_rows_active: bool = True
    require_target_rows_tradable: bool = True
    require_target_open_finite_positive: bool = True
    target_2023_or_later_allowed: bool = False
    expected_direction: str = _EXPECTED_DIRECTION
    minimum_eligible_observations_per_symbol: int = _MIN_OBSERVATIONS
    required_positive_symbol_slopes: int = _REQUIRED_POSITIVE
    valid_status: str = _VALID_STATUS
    reject_status: str = _REJECT_STATUS
    invalid_coverage_status: str = _INVALID_STATUS
    no_intercept: bool = True
    fixed_reduction: str = _FIXED_REDUCTION
    weighted_regression_allowed: bool = False
    robust_regression_allowed: bool = False
    require_numerator_finite: bool = True
    require_denominator_finite_positive: bool = True
    require_beta_finite: bool = True
    no_magnitude_threshold_after_results: bool = True
    training_relation_executed: bool = False
    alternate_sign_allowed: bool = False
    alternate_window_allowed: bool = False
    alternate_horizon_allowed: bool = False
    alternate_weighting_allowed: bool = False
    symbol_subset_allowed: bool = False
    additional_spot_dates_allowed: bool = False
    economic_values_inspected: bool = False
    target_relation_computed: bool = False
    evaluation_pnl_inspected: bool = False
    evaluation_execution_authorized: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name, expected in _canonical_values().items():
            observed = getattr(self, field_name)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"{field_name} is not canonical")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field_name, value in _canonical_values().items():
            observed = getattr(self, field_name)
            if isinstance(observed, tuple):
                payload[field_name] = list(observed)
            else:
                payload[field_name] = observed
        return payload

    @property
    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SpotFlowContinuationProtocol:
        canonical = canonical_spot_flow_continuation_protocol()
        expected_fields = set(canonical.to_payload())
        if set(payload) != expected_fields:
            raise ValueError("protocol payload fields are not canonical")
        kwargs = dict(payload)
        for field_name in ("symbols", "spot_dates", "decision_minutes_after_day_start"):
            value = kwargs[field_name]
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must use canonical JSON array form")
            kwargs[field_name] = tuple(value)
        return cls(**kwargs)


def canonical_spot_flow_continuation_protocol() -> SpotFlowContinuationProtocol:
    """Return the sole canonical Issue 584 preregistration."""

    return SpotFlowContinuationProtocol()


def load_spot_flow_continuation_protocol_bytes(
    payload: bytes,
) -> SpotFlowContinuationProtocol:
    """Load only exact canonical JSON bytes; formatting drift fails closed."""

    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol payload is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("protocol payload must be a JSON object")
    protocol = SpotFlowContinuationProtocol.from_payload(decoded)
    if payload != protocol.canonical_json_bytes:
        raise ValueError("protocol bytes are not canonical")
    return protocol
