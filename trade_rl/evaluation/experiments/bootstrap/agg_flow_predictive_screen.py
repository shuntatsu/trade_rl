"""Sealed pre-result protocol for the aggTrades predictive-flow screen."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.agg_flow_capacity import (
    canonical_m2_aggtrades_flow_capacity_protocol,
)

_SCHEMA_VERSION = "aggtrades_predictive_flow_screen_protocol_v1"
_CANONICAL_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
_CANONICAL_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_SOURCE_ROSTER_PROTOCOL_DIGEST = (
    "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
)
_SOURCE_ROSTER_SEAL_ARTIFACT_ID = 10319386570
_SOURCE_ROSTER_SEAL_ARTIFACT_DIGEST = (
    "dbc6ef286abb9e4c8530089328fb64b2cfaa5e98715a82740439470a45942e57"
)
_PROVIDER_HEAD_SHA = "dd51da97845f4d8fe69c34b1c4e4859358911daf"
_PROVIDER_PARSER_BLOB_SHA = "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
_MARKET = "usds-m"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_DISCOVERY_START = datetime(2021, 1, 1, tzinfo=UTC)
_DISCOVERY_STOP_EXCLUSIVE = datetime(2023, 1, 1, tzinfo=UTC)
_EVALUATION_START = datetime(2023, 1, 1, tzinfo=UTC)
_SAMPLE_MONTH_DAYS = (1,)
_CHECKSUM_REQUIRED = True
_CHECKSUM_SUFFIX = ".CHECKSUM"
_ARCHIVE_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{date}.zip"
)
_BUYER_TAKER_WHEN_BUYER_IS_MAKER = False
_SELLER_TAKER_WHEN_BUYER_IS_MAKER = True
_EVENT_TIME_FIELD = "transact_time"
_TRADE_NOTIONAL_FORMULA = "price_times_quantity"
_FINITE_POSITIVE_TRADE_INPUTS_REQUIRED = True
_POSITIVE_TOTAL_TAKER_NOTIONAL_REQUIRED = True
_EVENT_TIMESTAMP_BEFORE_EVALUATION_REQUIRED = True
_ARCHIVE_PUBLICATION_TIME_AS_FEATURE_ALLOWED = False
_PREDICTOR_FORMULA = "buy_minus_sell_over_buy_plus_sell"
_PREDICTOR_HOUR_ALIGNMENT = "completed_utc_hour"
_PREDICTOR_AVAILABLE_AT = "end_of_completed_utc_hour"
_PREDICTOR_TRANSFORM = "identity"
_WINSORIZATION_ALLOWED = False
_FITTED_NORMALIZATION_ALLOWED = False
_VOLUME_THRESHOLD_ALLOWED = False
_PREDICTOR_CLIPPING_RULE = "mathematical_bounds_only"
_EXECUTION_ALIGNMENT = "next_bar_open_t_plus_1"
_LABEL_FORMULA = "log_open_t_plus_2_over_open_t_plus_1"
_LABEL_HORIZON_BARS = 1
_LABEL_OPEN_FINITE_POSITIVE_REQUIRED = True
_ACTIVE_TRADABLE_AT_DECISION_EXECUTION_REQUIRED = True
_CONTIGUOUS_LABEL_INTERVAL_REQUIRED = True
_LABEL_ENDPOINTS_BEFORE_EVALUATION_REQUIRED = True
_SAME_HOUR_RETURN_ALLOWED = False
_POST_2022_OBSERVATIONS_ALLOWED = False
_FIT_RULE = "ordinary_least_squares_with_intercept"
_REDUCTION_RULE = "chronological_math_fsum"
_REPORT_PEARSON_CORRELATION = True
_FORMAL_DECISION_METRIC = "beta_sign"
_STRICT_POSITIVE_BETA = True
_CALENDAR_YEAR_SPLITS = (2021, 2022)
_MIN_VALID_DAYS_PER_SYMBOL = 20
_MIN_VALID_DAYS_PER_YEAR = 10
_MIN_VALID_OBSERVATIONS_PER_SYMBOL = 480
_MIN_VALID_OBSERVATIONS_PER_YEAR = 220
_MIN_POSITIVE_FULL_SAMPLE_SYMBOLS = 4
_MIN_POSITIVE_YEAR_SYMBOLS = 3
_PASS_STATUS = "PASS_FLOW_SCREEN"
_NO_SIGNAL_STATUS = "NO_STABLE_FLOW_SIGNAL"
_INVALID_STATUS = "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
_REPLACEMENT_DATES_ALLOWED = False
_ALTERNATE_HORIZON_SEARCH_ALLOWED = False
_POSTHOC_SIGN_FLIP_ALLOWED = False
_DISJOINT_VALIDATION_REQUIRED_AFTER_PASS = True
_STRATEGY_PNL_ALLOWED = False
_NUMERIC_CAPACITY_RESULT_ALLOWED = False

_EXPECTED_FIELDS: dict[str, object] = {
    "canonical_dataset_id": _CANONICAL_DATASET_ID,
    "canonical_dataset_artifact_digest": _CANONICAL_DATASET_ARTIFACT_DIGEST,
    "source_roster_protocol_digest": _SOURCE_ROSTER_PROTOCOL_DIGEST,
    "source_roster_seal_artifact_id": _SOURCE_ROSTER_SEAL_ARTIFACT_ID,
    "source_roster_seal_artifact_digest": _SOURCE_ROSTER_SEAL_ARTIFACT_DIGEST,
    "provider_head_sha": _PROVIDER_HEAD_SHA,
    "provider_parser_blob_sha": _PROVIDER_PARSER_BLOB_SHA,
    "market": _MARKET,
    "symbols": _SYMBOLS,
    "discovery_start": _DISCOVERY_START,
    "discovery_stop_exclusive": _DISCOVERY_STOP_EXCLUSIVE,
    "evaluation_start": _EVALUATION_START,
    "sample_month_days": _SAMPLE_MONTH_DAYS,
    "checksum_required": _CHECKSUM_REQUIRED,
    "checksum_suffix": _CHECKSUM_SUFFIX,
    "archive_url_template": _ARCHIVE_URL_TEMPLATE,
    "buyer_taker_when_buyer_is_maker": _BUYER_TAKER_WHEN_BUYER_IS_MAKER,
    "seller_taker_when_buyer_is_maker": _SELLER_TAKER_WHEN_BUYER_IS_MAKER,
    "event_time_field": _EVENT_TIME_FIELD,
    "trade_notional_formula": _TRADE_NOTIONAL_FORMULA,
    "finite_positive_trade_inputs_required": _FINITE_POSITIVE_TRADE_INPUTS_REQUIRED,
    "positive_total_taker_notional_required": _POSITIVE_TOTAL_TAKER_NOTIONAL_REQUIRED,
    "event_timestamp_before_evaluation_required": _EVENT_TIMESTAMP_BEFORE_EVALUATION_REQUIRED,
    "archive_publication_time_as_feature_allowed": _ARCHIVE_PUBLICATION_TIME_AS_FEATURE_ALLOWED,
    "predictor_formula": _PREDICTOR_FORMULA,
    "predictor_hour_alignment": _PREDICTOR_HOUR_ALIGNMENT,
    "predictor_available_at": _PREDICTOR_AVAILABLE_AT,
    "predictor_transform": _PREDICTOR_TRANSFORM,
    "winsorization_allowed": _WINSORIZATION_ALLOWED,
    "fitted_normalization_allowed": _FITTED_NORMALIZATION_ALLOWED,
    "volume_threshold_allowed": _VOLUME_THRESHOLD_ALLOWED,
    "predictor_clipping_rule": _PREDICTOR_CLIPPING_RULE,
    "execution_alignment": _EXECUTION_ALIGNMENT,
    "label_formula": _LABEL_FORMULA,
    "label_horizon_bars": _LABEL_HORIZON_BARS,
    "label_open_finite_positive_required": _LABEL_OPEN_FINITE_POSITIVE_REQUIRED,
    "active_tradable_at_decision_execution_required": _ACTIVE_TRADABLE_AT_DECISION_EXECUTION_REQUIRED,
    "contiguous_label_interval_required": _CONTIGUOUS_LABEL_INTERVAL_REQUIRED,
    "label_endpoints_before_evaluation_required": _LABEL_ENDPOINTS_BEFORE_EVALUATION_REQUIRED,
    "same_hour_return_allowed": _SAME_HOUR_RETURN_ALLOWED,
    "post_2022_observations_allowed": _POST_2022_OBSERVATIONS_ALLOWED,
    "fit_rule": _FIT_RULE,
    "reduction_rule": _REDUCTION_RULE,
    "report_pearson_correlation": _REPORT_PEARSON_CORRELATION,
    "formal_decision_metric": _FORMAL_DECISION_METRIC,
    "strict_positive_beta": _STRICT_POSITIVE_BETA,
    "calendar_year_splits": _CALENDAR_YEAR_SPLITS,
    "min_valid_days_per_symbol": _MIN_VALID_DAYS_PER_SYMBOL,
    "min_valid_days_per_year": _MIN_VALID_DAYS_PER_YEAR,
    "min_valid_observations_per_symbol": _MIN_VALID_OBSERVATIONS_PER_SYMBOL,
    "min_valid_observations_per_year": _MIN_VALID_OBSERVATIONS_PER_YEAR,
    "min_positive_full_sample_symbols": _MIN_POSITIVE_FULL_SAMPLE_SYMBOLS,
    "min_positive_year_symbols": _MIN_POSITIVE_YEAR_SYMBOLS,
    "pass_status": _PASS_STATUS,
    "no_signal_status": _NO_SIGNAL_STATUS,
    "invalid_status": _INVALID_STATUS,
    "replacement_dates_allowed": _REPLACEMENT_DATES_ALLOWED,
    "alternate_horizon_search_allowed": _ALTERNATE_HORIZON_SEARCH_ALLOWED,
    "posthoc_sign_flip_allowed": _POSTHOC_SIGN_FLIP_ALLOWED,
    "disjoint_validation_required_after_pass": _DISJOINT_VALIDATION_REQUIRED_AFTER_PASS,
    "strategy_pnl_allowed": _STRATEGY_PNL_ALLOWED,
    "numeric_capacity_result_allowed": _NUMERIC_CAPACITY_RESULT_ALLOWED,
    "schema_version": _SCHEMA_VERSION,
}


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


@dataclass(frozen=True, slots=True)
class AggTradesPredictiveFlowScreenProtocol:
    """Immutable preregistration for the pre-2023 predictive-flow screen."""

    canonical_dataset_id: str
    canonical_dataset_artifact_digest: str
    source_roster_protocol_digest: str
    source_roster_seal_artifact_id: int
    source_roster_seal_artifact_digest: str
    provider_head_sha: str
    provider_parser_blob_sha: str
    market: str
    symbols: tuple[str, ...]
    discovery_start: datetime
    discovery_stop_exclusive: datetime
    evaluation_start: datetime
    sample_month_days: tuple[int, ...]
    checksum_required: bool
    checksum_suffix: str
    archive_url_template: str
    buyer_taker_when_buyer_is_maker: bool
    seller_taker_when_buyer_is_maker: bool
    event_time_field: str
    trade_notional_formula: str
    finite_positive_trade_inputs_required: bool
    positive_total_taker_notional_required: bool
    event_timestamp_before_evaluation_required: bool
    archive_publication_time_as_feature_allowed: bool
    predictor_formula: str
    predictor_hour_alignment: str
    predictor_available_at: str
    predictor_transform: str
    winsorization_allowed: bool
    fitted_normalization_allowed: bool
    volume_threshold_allowed: bool
    predictor_clipping_rule: str
    execution_alignment: str
    label_formula: str
    label_horizon_bars: int
    label_open_finite_positive_required: bool
    active_tradable_at_decision_execution_required: bool
    contiguous_label_interval_required: bool
    label_endpoints_before_evaluation_required: bool
    same_hour_return_allowed: bool
    post_2022_observations_allowed: bool
    fit_rule: str
    reduction_rule: str
    report_pearson_correlation: bool
    formal_decision_metric: str
    strict_positive_beta: bool
    calendar_year_splits: tuple[int, ...]
    min_valid_days_per_symbol: int
    min_valid_days_per_year: int
    min_valid_observations_per_symbol: int
    min_valid_observations_per_year: int
    min_positive_full_sample_symbols: int
    min_positive_year_symbols: int
    pass_status: str
    no_signal_status: str
    invalid_status: str
    replacement_dates_allowed: bool
    alternate_horizon_search_allowed: bool
    posthoc_sign_flip_allowed: bool
    disjoint_validation_required_after_pass: bool
    strategy_pnl_allowed: bool
    numeric_capacity_result_allowed: bool
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, expected in _EXPECTED_FIELDS.items():
            actual = getattr(self, field_name)
            if isinstance(expected, datetime):
                if not isinstance(actual, datetime):
                    raise ValueError(
                        "fields must match the preregistered predictive-flow contract"
                    )
                if actual.tzinfo is None or actual.utcoffset() is None:
                    raise ValueError(f"{field_name} must be timezone-aware")
                actual = actual.astimezone(UTC)
            if actual != expected:
                raise ValueError(
                    "fields must match the preregistered predictive-flow contract"
                )

    @property
    def planned_days(self) -> tuple[datetime, ...]:
        days: list[datetime] = []
        current = self.discovery_start
        while current < self.discovery_stop_exclusive:
            for day in self.sample_month_days:
                candidate = current.replace(day=day)
                if self.discovery_start <= candidate < self.discovery_stop_exclusive:
                    days.append(candidate)
            current = _next_month(current)
        return tuple(days)

    @property
    def planned_urls(self) -> tuple[str, ...]:
        return tuple(
            self.archive_url_template.format(symbol=symbol, date=f"{day:%Y-%m-%d}")
            for symbol in self.symbols
            for day in self.planned_days
        )

    def _payload_without_digest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_dataset_id": self.canonical_dataset_id,
            "canonical_dataset_artifact_digest": self.canonical_dataset_artifact_digest,
            "source_roster_protocol_digest": self.source_roster_protocol_digest,
            "source_roster_seal_artifact_id": self.source_roster_seal_artifact_id,
            "source_roster_seal_artifact_digest": self.source_roster_seal_artifact_digest,
            "provider_head_sha": self.provider_head_sha,
            "provider_parser_blob_sha": self.provider_parser_blob_sha,
            "market": self.market,
            "symbols": list(self.symbols),
            "discovery_start": _iso_utc(self.discovery_start),
            "discovery_stop_exclusive": _iso_utc(self.discovery_stop_exclusive),
            "evaluation_start": _iso_utc(self.evaluation_start),
            "sample_month_days": list(self.sample_month_days),
            "planned_urls": list(self.planned_urls),
            "checksum_required": self.checksum_required,
            "checksum_suffix": self.checksum_suffix,
            "archive_url_template": self.archive_url_template,
            "buyer_taker_when_buyer_is_maker": self.buyer_taker_when_buyer_is_maker,
            "seller_taker_when_buyer_is_maker": self.seller_taker_when_buyer_is_maker,
            "event_time_field": self.event_time_field,
            "trade_notional_formula": self.trade_notional_formula,
            "finite_positive_trade_inputs_required": self.finite_positive_trade_inputs_required,
            "positive_total_taker_notional_required": self.positive_total_taker_notional_required,
            "event_timestamp_before_evaluation_required": self.event_timestamp_before_evaluation_required,
            "archive_publication_time_as_feature_allowed": self.archive_publication_time_as_feature_allowed,
            "predictor_formula": self.predictor_formula,
            "predictor_hour_alignment": self.predictor_hour_alignment,
            "predictor_available_at": self.predictor_available_at,
            "predictor_transform": self.predictor_transform,
            "winsorization_allowed": self.winsorization_allowed,
            "fitted_normalization_allowed": self.fitted_normalization_allowed,
            "volume_threshold_allowed": self.volume_threshold_allowed,
            "predictor_clipping_rule": self.predictor_clipping_rule,
            "execution_alignment": self.execution_alignment,
            "label_formula": self.label_formula,
            "label_horizon_bars": self.label_horizon_bars,
            "label_open_finite_positive_required": self.label_open_finite_positive_required,
            "active_tradable_at_decision_execution_required": self.active_tradable_at_decision_execution_required,
            "contiguous_label_interval_required": self.contiguous_label_interval_required,
            "label_endpoints_before_evaluation_required": self.label_endpoints_before_evaluation_required,
            "same_hour_return_allowed": self.same_hour_return_allowed,
            "post_2022_observations_allowed": self.post_2022_observations_allowed,
            "fit_rule": self.fit_rule,
            "reduction_rule": self.reduction_rule,
            "report_pearson_correlation": self.report_pearson_correlation,
            "formal_decision_metric": self.formal_decision_metric,
            "strict_positive_beta": self.strict_positive_beta,
            "calendar_year_splits": list(self.calendar_year_splits),
            "min_valid_days_per_symbol": self.min_valid_days_per_symbol,
            "min_valid_days_per_year": self.min_valid_days_per_year,
            "min_valid_observations_per_symbol": self.min_valid_observations_per_symbol,
            "min_valid_observations_per_year": self.min_valid_observations_per_year,
            "min_positive_full_sample_symbols": self.min_positive_full_sample_symbols,
            "min_positive_year_symbols": self.min_positive_year_symbols,
            "pass_status": self.pass_status,
            "no_signal_status": self.no_signal_status,
            "invalid_status": self.invalid_status,
            "replacement_dates_allowed": self.replacement_dates_allowed,
            "alternate_horizon_search_allowed": self.alternate_horizon_search_allowed,
            "posthoc_sign_flip_allowed": self.posthoc_sign_flip_allowed,
            "disjoint_validation_required_after_pass": self.disjoint_validation_required_after_pass,
            "strategy_pnl_allowed": self.strategy_pnl_allowed,
            "numeric_capacity_result_allowed": self.numeric_capacity_result_allowed,
        }

    @property
    def digest(self) -> str:
        return content_digest(self._payload_without_digest())

    def to_payload(self) -> dict[str, object]:
        payload = self._payload_without_digest()
        payload["protocol_digest"] = self.digest
        return payload


def canonical_m2_aggtrades_predictive_flow_screen_protocol() -> (
    AggTradesPredictiveFlowScreenProtocol
):
    """Return the single preregistered predictive-flow protocol."""

    roster = canonical_m2_aggtrades_flow_capacity_protocol()
    if roster.digest != _SOURCE_ROSTER_PROTOCOL_DIGEST:
        raise RuntimeError("aggTrades source-roster protocol digest drifted")

    protocol = AggTradesPredictiveFlowScreenProtocol(
        canonical_dataset_id=roster.canonical_dataset_id,
        canonical_dataset_artifact_digest=roster.canonical_dataset_artifact_digest,
        source_roster_protocol_digest=roster.digest,
        source_roster_seal_artifact_id=_SOURCE_ROSTER_SEAL_ARTIFACT_ID,
        source_roster_seal_artifact_digest=_SOURCE_ROSTER_SEAL_ARTIFACT_DIGEST,
        provider_head_sha=roster.provider_head_sha,
        provider_parser_blob_sha=roster.provider_parser_blob_sha,
        market=roster.market,
        symbols=roster.symbols,
        discovery_start=roster.calibration_start,
        discovery_stop_exclusive=roster.calibration_stop_exclusive,
        evaluation_start=roster.evaluation_start,
        sample_month_days=roster.sample_month_days,
        checksum_required=roster.checksum_required,
        checksum_suffix=roster.checksum_suffix,
        archive_url_template=roster.archive_url_template,
        buyer_taker_when_buyer_is_maker=roster.buyer_taker_when_buyer_is_maker,
        seller_taker_when_buyer_is_maker=roster.seller_taker_when_buyer_is_maker,
        event_time_field=_EVENT_TIME_FIELD,
        trade_notional_formula=roster.trade_notional_formula,
        finite_positive_trade_inputs_required=_FINITE_POSITIVE_TRADE_INPUTS_REQUIRED,
        positive_total_taker_notional_required=_POSITIVE_TOTAL_TAKER_NOTIONAL_REQUIRED,
        event_timestamp_before_evaluation_required=_EVENT_TIMESTAMP_BEFORE_EVALUATION_REQUIRED,
        archive_publication_time_as_feature_allowed=_ARCHIVE_PUBLICATION_TIME_AS_FEATURE_ALLOWED,
        predictor_formula=_PREDICTOR_FORMULA,
        predictor_hour_alignment=_PREDICTOR_HOUR_ALIGNMENT,
        predictor_available_at=_PREDICTOR_AVAILABLE_AT,
        predictor_transform=_PREDICTOR_TRANSFORM,
        winsorization_allowed=_WINSORIZATION_ALLOWED,
        fitted_normalization_allowed=_FITTED_NORMALIZATION_ALLOWED,
        volume_threshold_allowed=_VOLUME_THRESHOLD_ALLOWED,
        predictor_clipping_rule=_PREDICTOR_CLIPPING_RULE,
        execution_alignment=_EXECUTION_ALIGNMENT,
        label_formula=_LABEL_FORMULA,
        label_horizon_bars=_LABEL_HORIZON_BARS,
        label_open_finite_positive_required=_LABEL_OPEN_FINITE_POSITIVE_REQUIRED,
        active_tradable_at_decision_execution_required=_ACTIVE_TRADABLE_AT_DECISION_EXECUTION_REQUIRED,
        contiguous_label_interval_required=_CONTIGUOUS_LABEL_INTERVAL_REQUIRED,
        label_endpoints_before_evaluation_required=_LABEL_ENDPOINTS_BEFORE_EVALUATION_REQUIRED,
        same_hour_return_allowed=_SAME_HOUR_RETURN_ALLOWED,
        post_2022_observations_allowed=_POST_2022_OBSERVATIONS_ALLOWED,
        fit_rule=_FIT_RULE,
        reduction_rule=_REDUCTION_RULE,
        report_pearson_correlation=_REPORT_PEARSON_CORRELATION,
        formal_decision_metric=_FORMAL_DECISION_METRIC,
        strict_positive_beta=_STRICT_POSITIVE_BETA,
        calendar_year_splits=_CALENDAR_YEAR_SPLITS,
        min_valid_days_per_symbol=_MIN_VALID_DAYS_PER_SYMBOL,
        min_valid_days_per_year=_MIN_VALID_DAYS_PER_YEAR,
        min_valid_observations_per_symbol=_MIN_VALID_OBSERVATIONS_PER_SYMBOL,
        min_valid_observations_per_year=_MIN_VALID_OBSERVATIONS_PER_YEAR,
        min_positive_full_sample_symbols=_MIN_POSITIVE_FULL_SAMPLE_SYMBOLS,
        min_positive_year_symbols=_MIN_POSITIVE_YEAR_SYMBOLS,
        pass_status=_PASS_STATUS,
        no_signal_status=_NO_SIGNAL_STATUS,
        invalid_status=_INVALID_STATUS,
        replacement_dates_allowed=_REPLACEMENT_DATES_ALLOWED,
        alternate_horizon_search_allowed=_ALTERNATE_HORIZON_SEARCH_ALLOWED,
        posthoc_sign_flip_allowed=_POSTHOC_SIGN_FLIP_ALLOWED,
        disjoint_validation_required_after_pass=_DISJOINT_VALIDATION_REQUIRED_AFTER_PASS,
        strategy_pnl_allowed=_STRATEGY_PNL_ALLOWED,
        numeric_capacity_result_allowed=_NUMERIC_CAPACITY_RESULT_ALLOWED,
    )
    if protocol.planned_days != roster.planned_days:
        raise RuntimeError("predictive-flow planned days drifted from sealed source roster")
    if protocol.planned_urls != roster.planned_urls:
        raise RuntimeError("predictive-flow planned URLs drifted from sealed source roster")
    return protocol


def load_aggtrades_predictive_flow_screen_protocol(
    path: str | Path,
) -> AggTradesPredictiveFlowScreenProtocol:
    """Load only the sealed canonical protocol payload."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("predictive-flow protocol payload must be a JSON object")
    payload = cast(dict[str, object], raw)
    canonical = canonical_m2_aggtrades_predictive_flow_screen_protocol()
    if payload != canonical.to_payload():
        raise ValueError(
            "predictive-flow protocol must match the sealed canonical payload"
        )
    return canonical


__all__ = [
    "AggTradesPredictiveFlowScreenProtocol",
    "canonical_m2_aggtrades_predictive_flow_screen_protocol",
    "load_aggtrades_predictive_flow_screen_protocol",
]
