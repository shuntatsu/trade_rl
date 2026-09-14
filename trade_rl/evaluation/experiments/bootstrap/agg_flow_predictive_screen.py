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
_TRADE_NOTIONAL_FORMULA = "price_times_quantity"
_PREDICTOR_FORMULA = "buy_minus_sell_over_buy_plus_sell"
_PREDICTOR_HOUR_ALIGNMENT = "completed_utc_hour"
_PREDICTOR_AVAILABLE_AT = "end_of_completed_utc_hour"
_EXECUTION_ALIGNMENT = "next_bar_open_t_plus_1"
_LABEL_FORMULA = "log_open_t_plus_2_over_open_t_plus_1"
_LABEL_HORIZON_BARS = 1
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
    trade_notional_formula: str
    predictor_formula: str
    predictor_hour_alignment: str
    predictor_available_at: str
    execution_alignment: str
    label_formula: str
    label_horizon_bars: int
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
        bool_fields = (
            self.checksum_required,
            self.buyer_taker_when_buyer_is_maker,
            self.seller_taker_when_buyer_is_maker,
            self.same_hour_return_allowed,
            self.post_2022_observations_allowed,
            self.report_pearson_correlation,
            self.strict_positive_beta,
            self.replacement_dates_allowed,
            self.alternate_horizon_search_allowed,
            self.posthoc_sign_flip_allowed,
            self.disjoint_validation_required_after_pass,
            self.strategy_pnl_allowed,
            self.numeric_capacity_result_allowed,
        )
        if any(type(value) is not bool for value in bool_fields):
            raise ValueError("predictive-flow boolean fields must be booleans")
        int_fields = (
            self.source_roster_seal_artifact_id,
            self.label_horizon_bars,
            self.min_valid_days_per_symbol,
            self.min_valid_days_per_year,
            self.min_valid_observations_per_symbol,
            self.min_valid_observations_per_year,
            self.min_positive_full_sample_symbols,
            self.min_positive_year_symbols,
        )
        if any(type(value) is not int for value in int_fields):
            raise ValueError("predictive-flow integer fields must be integers")
        for value, field in (
            (self.discovery_start, "discovery_start"),
            (self.discovery_stop_exclusive, "discovery_stop_exclusive"),
            (self.evaluation_start, "evaluation_start"),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")

        actual = (
            self.canonical_dataset_id,
            self.canonical_dataset_artifact_digest,
            self.source_roster_protocol_digest,
            self.source_roster_seal_artifact_id,
            self.source_roster_seal_artifact_digest,
            self.provider_head_sha,
            self.provider_parser_blob_sha,
            self.market,
            self.symbols,
            self.discovery_start.astimezone(UTC),
            self.discovery_stop_exclusive.astimezone(UTC),
            self.evaluation_start.astimezone(UTC),
            self.sample_month_days,
            self.checksum_required,
            self.checksum_suffix,
            self.archive_url_template,
            self.buyer_taker_when_buyer_is_maker,
            self.seller_taker_when_buyer_is_maker,
            self.trade_notional_formula,
            self.predictor_formula,
            self.predictor_hour_alignment,
            self.predictor_available_at,
            self.execution_alignment,
            self.label_formula,
            self.label_horizon_bars,
            self.same_hour_return_allowed,
            self.post_2022_observations_allowed,
            self.fit_rule,
            self.reduction_rule,
            self.report_pearson_correlation,
            self.formal_decision_metric,
            self.strict_positive_beta,
            self.calendar_year_splits,
            self.min_valid_days_per_symbol,
            self.min_valid_days_per_year,
            self.min_valid_observations_per_symbol,
            self.min_valid_observations_per_year,
            self.min_positive_full_sample_symbols,
            self.min_positive_year_symbols,
            self.pass_status,
            self.no_signal_status,
            self.invalid_status,
            self.replacement_dates_allowed,
            self.alternate_horizon_search_allowed,
            self.posthoc_sign_flip_allowed,
            self.disjoint_validation_required_after_pass,
            self.strategy_pnl_allowed,
            self.numeric_capacity_result_allowed,
            self.schema_version,
        )
        preregistered = (
            _CANONICAL_DATASET_ID,
            _CANONICAL_DATASET_ARTIFACT_DIGEST,
            _SOURCE_ROSTER_PROTOCOL_DIGEST,
            _SOURCE_ROSTER_SEAL_ARTIFACT_ID,
            _SOURCE_ROSTER_SEAL_ARTIFACT_DIGEST,
            _PROVIDER_HEAD_SHA,
            _PROVIDER_PARSER_BLOB_SHA,
            _MARKET,
            _SYMBOLS,
            _DISCOVERY_START,
            _DISCOVERY_STOP_EXCLUSIVE,
            _EVALUATION_START,
            _SAMPLE_MONTH_DAYS,
            _CHECKSUM_REQUIRED,
            _CHECKSUM_SUFFIX,
            _ARCHIVE_URL_TEMPLATE,
            _BUYER_TAKER_WHEN_BUYER_IS_MAKER,
            _SELLER_TAKER_WHEN_BUYER_IS_MAKER,
            _TRADE_NOTIONAL_FORMULA,
            _PREDICTOR_FORMULA,
            _PREDICTOR_HOUR_ALIGNMENT,
            _PREDICTOR_AVAILABLE_AT,
            _EXECUTION_ALIGNMENT,
            _LABEL_FORMULA,
            _LABEL_HORIZON_BARS,
            _SAME_HOUR_RETURN_ALLOWED,
            _POST_2022_OBSERVATIONS_ALLOWED,
            _FIT_RULE,
            _REDUCTION_RULE,
            _REPORT_PEARSON_CORRELATION,
            _FORMAL_DECISION_METRIC,
            _STRICT_POSITIVE_BETA,
            _CALENDAR_YEAR_SPLITS,
            _MIN_VALID_DAYS_PER_SYMBOL,
            _MIN_VALID_DAYS_PER_YEAR,
            _MIN_VALID_OBSERVATIONS_PER_SYMBOL,
            _MIN_VALID_OBSERVATIONS_PER_YEAR,
            _MIN_POSITIVE_FULL_SAMPLE_SYMBOLS,
            _MIN_POSITIVE_YEAR_SYMBOLS,
            _PASS_STATUS,
            _NO_SIGNAL_STATUS,
            _INVALID_STATUS,
            _REPLACEMENT_DATES_ALLOWED,
            _ALTERNATE_HORIZON_SEARCH_ALLOWED,
            _POSTHOC_SIGN_FLIP_ALLOWED,
            _DISJOINT_VALIDATION_REQUIRED_AFTER_PASS,
            _STRATEGY_PNL_ALLOWED,
            _NUMERIC_CAPACITY_RESULT_ALLOWED,
            _SCHEMA_VERSION,
        )
        if actual != preregistered:
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
            "trade_notional_formula": self.trade_notional_formula,
            "predictor_formula": self.predictor_formula,
            "predictor_hour_alignment": self.predictor_hour_alignment,
            "predictor_available_at": self.predictor_available_at,
            "execution_alignment": self.execution_alignment,
            "label_formula": self.label_formula,
            "label_horizon_bars": self.label_horizon_bars,
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
            "disjoint_validation_required_after_pass": (
                self.disjoint_validation_required_after_pass
            ),
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
        trade_notional_formula=roster.trade_notional_formula,
        predictor_formula=_PREDICTOR_FORMULA,
        predictor_hour_alignment=_PREDICTOR_HOUR_ALIGNMENT,
        predictor_available_at=_PREDICTOR_AVAILABLE_AT,
        execution_alignment=_EXECUTION_ALIGNMENT,
        label_formula=_LABEL_FORMULA,
        label_horizon_bars=_LABEL_HORIZON_BARS,
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
        disjoint_validation_required_after_pass=(
            _DISJOINT_VALIDATION_REQUIRED_AFTER_PASS
        ),
        strategy_pnl_allowed=_STRATEGY_PNL_ALLOWED,
        numeric_capacity_result_allowed=_NUMERIC_CAPACITY_RESULT_ALLOWED,
    )
    if protocol.planned_days != roster.planned_days:
        raise RuntimeError(
            "predictive-flow planned days drifted from sealed source roster"
        )
    if protocol.planned_urls != roster.planned_urls:
        raise RuntimeError(
            "predictive-flow planned URLs drifted from sealed source roster"
        )
    return protocol


def load_aggtrades_predictive_flow_screen_protocol(
    path: str | Path,
) -> AggTradesPredictiveFlowScreenProtocol:
    """Load only the byte-semantically sealed canonical protocol payload."""

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
