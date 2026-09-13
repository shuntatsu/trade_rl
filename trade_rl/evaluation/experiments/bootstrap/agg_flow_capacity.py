"""Sealed pre-result protocol for aggTrades flow-capacity calibration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "aggtrades_flow_capacity_protocol_v1"
_CANONICAL_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
_CANONICAL_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_CANONICAL_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
_PROVIDER_HEAD_SHA = "dd51da97845f4d8fe69c34b1c4e4859358911daf"
_PROVIDER_PARSER_BLOB_SHA = "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
_SOURCE_VALIDATION_RUN_ID = 34758933034
_SOURCE_VALIDATION_ARTIFACT_ID = 10318765553
_SOURCE_VALIDATION_ARTIFACT_DIGEST = (
    "3abd6592169ace3f308059fd524199d6fcf76c38862d0caec5e58192f6c7f7a3"
)
_SOURCE_VALIDATION_MANIFEST_SHA256 = (
    "9ebffe78c29dbe336e5ed1323b686f41fae56ef83c50104fae3b8e965eb01f0a"
)
_CAUSAL_CAPACITY_HEAD_SHA = "db8193c81dd0ca334a15ffc1895d75884238f8a9"
_PROCESSING_BAR_VOLUME_CAPACITY = False
_MARKET = "usds-m"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_CALIBRATION_START = datetime(2021, 1, 1, tzinfo=UTC)
_CALIBRATION_STOP_EXCLUSIVE = datetime(2023, 1, 1, tzinfo=UTC)
_EVALUATION_START = datetime(2023, 1, 1, tzinfo=UTC)
_SAMPLE_MONTH_DAYS = (1,)
_BURST_WINDOW_MILLISECONDS = 5_000
_LOWER_TAIL_QUANTILE = 0.10
_UTILIZATION_FRACTION = 0.25
_HARD_CEILING = 0.05
_MIN_VALID_DAYS_PER_SYMBOL = 20
_MIN_VALID_DAYS_PER_YEAR = 10
_MIN_VALID_HOURS_PER_SYMBOL = 480
_CHECKSUM_REQUIRED = True
_CHECKSUM_SUFFIX = ".CHECKSUM"
_BUYER_TAKER_WHEN_BUYER_IS_MAKER = False
_SELLER_TAKER_WHEN_BUYER_IS_MAKER = True
_ARCHIVE_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{date}.zip"
)
_TRADE_NOTIONAL_FORMULA = "price_times_quantity"
_HOUR_ALIGNMENT = "utc_hour"
_BURST_BIN_ALIGNMENT = "utc_epoch_floor_5000ms"
_FRACTION_FORMULA = "min_peak_buy_sell_5s_over_hour_total"
_ORDER_STATISTIC_RULE = "ceil_qn_minus_one"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_dataset_id",
        "canonical_dataset_artifact_digest",
        "canonical_study_digest",
        "provider_head_sha",
        "provider_parser_blob_sha",
        "source_validation_run_id",
        "source_validation_artifact_id",
        "source_validation_artifact_digest",
        "source_validation_manifest_sha256",
        "causal_capacity_head_sha",
        "processing_bar_volume_capacity",
        "market",
        "symbols",
        "calibration_start",
        "calibration_stop_exclusive",
        "evaluation_start",
        "sample_month_days",
        "burst_window_milliseconds",
        "lower_tail_quantile",
        "utilization_fraction",
        "hard_ceiling",
        "min_valid_days_per_symbol",
        "min_valid_days_per_year",
        "min_valid_hours_per_symbol",
        "checksum_required",
        "checksum_suffix",
        "buyer_taker_when_buyer_is_maker",
        "seller_taker_when_buyer_is_maker",
        "archive_url_template",
        "trade_notional_formula",
        "hour_alignment",
        "burst_bin_alignment",
        "fraction_formula",
        "order_statistic_rule",
    }
)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


def _int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_datetime(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _aware_utc(parsed, field=field)


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    values = tuple(_text(item, field=field) for item in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique non-empty values")
    return values


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    values = tuple(_int(item, field=field) for item in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique integer values")
    return values


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


@dataclass(frozen=True, slots=True)
class AggTradesFlowCapacityProtocol:
    """Immutable preregistration for pre-2023 aggTrades capacity stress."""

    canonical_dataset_id: str
    canonical_dataset_artifact_digest: str
    canonical_study_digest: str
    provider_head_sha: str
    provider_parser_blob_sha: str
    source_validation_run_id: int
    source_validation_artifact_id: int
    source_validation_artifact_digest: str
    source_validation_manifest_sha256: str
    causal_capacity_head_sha: str
    processing_bar_volume_capacity: bool
    market: str
    symbols: tuple[str, ...]
    calibration_start: datetime
    calibration_stop_exclusive: datetime
    evaluation_start: datetime
    sample_month_days: tuple[int, ...]
    burst_window_milliseconds: int
    lower_tail_quantile: float
    utilization_fraction: float
    hard_ceiling: float
    min_valid_days_per_symbol: int
    min_valid_days_per_year: int
    min_valid_hours_per_symbol: int
    checksum_required: bool
    checksum_suffix: str
    buyer_taker_when_buyer_is_maker: bool
    seller_taker_when_buyer_is_maker: bool
    archive_url_template: str
    trade_notional_formula: str
    hour_alignment: str
    burst_bin_alignment: str
    fraction_formula: str
    order_statistic_rule: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        dataset_id = _sha256(self.canonical_dataset_id, field="canonical_dataset_id")
        dataset_artifact = _sha256(
            self.canonical_dataset_artifact_digest,
            field="canonical_dataset_artifact_digest",
        )
        study_digest = _sha256(
            self.canonical_study_digest,
            field="canonical_study_digest",
        )
        provider_head = _git_sha(self.provider_head_sha, field="provider_head_sha")
        provider_blob = _git_sha(
            self.provider_parser_blob_sha,
            field="provider_parser_blob_sha",
        )
        source_validation_run_id = _int(
            self.source_validation_run_id,
            field="source_validation_run_id",
        )
        source_validation_artifact_id = _int(
            self.source_validation_artifact_id,
            field="source_validation_artifact_id",
        )
        source_validation_artifact_digest = _sha256(
            self.source_validation_artifact_digest,
            field="source_validation_artifact_digest",
        )
        source_validation_manifest_sha256 = _sha256(
            self.source_validation_manifest_sha256,
            field="source_validation_manifest_sha256",
        )
        causal_capacity_head = _git_sha(
            self.causal_capacity_head_sha,
            field="causal_capacity_head_sha",
        )
        processing_bar_volume_capacity = _bool(
            self.processing_bar_volume_capacity,
            field="processing_bar_volume_capacity",
        )
        market = _text(self.market, field="market")
        symbols = tuple(self.symbols)
        calibration_start = _aware_utc(
            self.calibration_start,
            field="calibration_start",
        )
        calibration_stop = _aware_utc(
            self.calibration_stop_exclusive,
            field="calibration_stop_exclusive",
        )
        evaluation_start = _aware_utc(
            self.evaluation_start,
            field="evaluation_start",
        )
        sample_days = tuple(self.sample_month_days)
        burst_ms = _int(
            self.burst_window_milliseconds,
            field="burst_window_milliseconds",
        )
        quantile = _float(self.lower_tail_quantile, field="lower_tail_quantile")
        utilization = _float(self.utilization_fraction, field="utilization_fraction")
        ceiling = _float(self.hard_ceiling, field="hard_ceiling")
        min_days = _int(
            self.min_valid_days_per_symbol,
            field="min_valid_days_per_symbol",
        )
        min_days_year = _int(
            self.min_valid_days_per_year,
            field="min_valid_days_per_year",
        )
        min_hours = _int(
            self.min_valid_hours_per_symbol,
            field="min_valid_hours_per_symbol",
        )
        checksum_required = _bool(self.checksum_required, field="checksum_required")
        checksum_suffix = _text(self.checksum_suffix, field="checksum_suffix")
        buyer_taker = _bool(
            self.buyer_taker_when_buyer_is_maker,
            field="buyer_taker_when_buyer_is_maker",
        )
        seller_taker = _bool(
            self.seller_taker_when_buyer_is_maker,
            field="seller_taker_when_buyer_is_maker",
        )
        url_template = _text(self.archive_url_template, field="archive_url_template")
        trade_formula = _text(
            self.trade_notional_formula,
            field="trade_notional_formula",
        )
        hour_alignment = _text(self.hour_alignment, field="hour_alignment")
        bin_alignment = _text(self.burst_bin_alignment, field="burst_bin_alignment")
        fraction_formula = _text(self.fraction_formula, field="fraction_formula")
        order_rule = _text(self.order_statistic_rule, field="order_statistic_rule")
        schema_version = _text(self.schema_version, field="schema_version")

        if not calibration_start < calibration_stop:
            raise ValueError(
                "calibration_start must be before calibration_stop_exclusive"
            )
        if calibration_stop > evaluation_start:
            raise ValueError(
                "calibration_stop_exclusive cannot exceed evaluation_start"
            )
        for identity_field, identity_value in (
            ("source_validation_run_id", source_validation_run_id),
            ("source_validation_artifact_id", source_validation_artifact_id),
        ):
            if identity_value <= 0:
                raise ValueError(f"{identity_field} must be positive")
        if any(day < 1 or day > 28 for day in sample_days):
            raise ValueError("sample_month_days must be within 1..28")
        if burst_ms <= 0:
            raise ValueError("burst_window_milliseconds must be positive")
        for field, value in (
            ("lower_tail_quantile", quantile),
            ("utilization_fraction", utilization),
            ("hard_ceiling", ceiling),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field} must be within (0, 1]")
        for field, value in (
            ("min_valid_days_per_symbol", min_days),
            ("min_valid_days_per_year", min_days_year),
            ("min_valid_hours_per_symbol", min_hours),
        ):
            if value <= 0:
                raise ValueError(f"{field} must be positive")

        actual = (
            dataset_id,
            dataset_artifact,
            study_digest,
            provider_head,
            provider_blob,
            source_validation_run_id,
            source_validation_artifact_id,
            source_validation_artifact_digest,
            source_validation_manifest_sha256,
            causal_capacity_head,
            processing_bar_volume_capacity,
            market,
            symbols,
            calibration_start,
            calibration_stop,
            evaluation_start,
            sample_days,
            burst_ms,
            quantile,
            utilization,
            ceiling,
            min_days,
            min_days_year,
            min_hours,
            checksum_required,
            checksum_suffix,
            buyer_taker,
            seller_taker,
            url_template,
            trade_formula,
            hour_alignment,
            bin_alignment,
            fraction_formula,
            order_rule,
            schema_version,
        )
        preregistered = (
            _CANONICAL_DATASET_ID,
            _CANONICAL_DATASET_ARTIFACT_DIGEST,
            _CANONICAL_STUDY_DIGEST,
            _PROVIDER_HEAD_SHA,
            _PROVIDER_PARSER_BLOB_SHA,
            _SOURCE_VALIDATION_RUN_ID,
            _SOURCE_VALIDATION_ARTIFACT_ID,
            _SOURCE_VALIDATION_ARTIFACT_DIGEST,
            _SOURCE_VALIDATION_MANIFEST_SHA256,
            _CAUSAL_CAPACITY_HEAD_SHA,
            _PROCESSING_BAR_VOLUME_CAPACITY,
            _MARKET,
            _SYMBOLS,
            _CALIBRATION_START,
            _CALIBRATION_STOP_EXCLUSIVE,
            _EVALUATION_START,
            _SAMPLE_MONTH_DAYS,
            _BURST_WINDOW_MILLISECONDS,
            _LOWER_TAIL_QUANTILE,
            _UTILIZATION_FRACTION,
            _HARD_CEILING,
            _MIN_VALID_DAYS_PER_SYMBOL,
            _MIN_VALID_DAYS_PER_YEAR,
            _MIN_VALID_HOURS_PER_SYMBOL,
            _CHECKSUM_REQUIRED,
            _CHECKSUM_SUFFIX,
            _BUYER_TAKER_WHEN_BUYER_IS_MAKER,
            _SELLER_TAKER_WHEN_BUYER_IS_MAKER,
            _ARCHIVE_URL_TEMPLATE,
            _TRADE_NOTIONAL_FORMULA,
            _HOUR_ALIGNMENT,
            _BURST_BIN_ALIGNMENT,
            _FRACTION_FORMULA,
            _ORDER_STATISTIC_RULE,
            _SCHEMA_VERSION,
        )
        if actual != preregistered:
            raise ValueError(
                "aggTrades flow-capacity fields differ from preregistered protocol"
            )

        object.__setattr__(self, "canonical_dataset_id", dataset_id)
        object.__setattr__(
            self,
            "canonical_dataset_artifact_digest",
            dataset_artifact,
        )
        object.__setattr__(self, "canonical_study_digest", study_digest)
        object.__setattr__(self, "provider_head_sha", provider_head)
        object.__setattr__(self, "provider_parser_blob_sha", provider_blob)
        object.__setattr__(self, "source_validation_run_id", source_validation_run_id)
        object.__setattr__(
            self, "source_validation_artifact_id", source_validation_artifact_id
        )
        object.__setattr__(
            self,
            "source_validation_artifact_digest",
            source_validation_artifact_digest,
        )
        object.__setattr__(
            self,
            "source_validation_manifest_sha256",
            source_validation_manifest_sha256,
        )
        object.__setattr__(self, "causal_capacity_head_sha", causal_capacity_head)
        object.__setattr__(
            self,
            "processing_bar_volume_capacity",
            processing_bar_volume_capacity,
        )
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "calibration_start", calibration_start)
        object.__setattr__(self, "calibration_stop_exclusive", calibration_stop)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "sample_month_days", sample_days)
        object.__setattr__(self, "burst_window_milliseconds", burst_ms)
        object.__setattr__(self, "lower_tail_quantile", quantile)
        object.__setattr__(self, "utilization_fraction", utilization)
        object.__setattr__(self, "hard_ceiling", ceiling)
        object.__setattr__(self, "min_valid_days_per_symbol", min_days)
        object.__setattr__(self, "min_valid_days_per_year", min_days_year)
        object.__setattr__(self, "min_valid_hours_per_symbol", min_hours)
        object.__setattr__(self, "checksum_required", checksum_required)
        object.__setattr__(self, "checksum_suffix", checksum_suffix)
        object.__setattr__(
            self,
            "buyer_taker_when_buyer_is_maker",
            buyer_taker,
        )
        object.__setattr__(
            self,
            "seller_taker_when_buyer_is_maker",
            seller_taker,
        )
        object.__setattr__(self, "archive_url_template", url_template)
        object.__setattr__(self, "trade_notional_formula", trade_formula)
        object.__setattr__(self, "hour_alignment", hour_alignment)
        object.__setattr__(self, "burst_bin_alignment", bin_alignment)
        object.__setattr__(self, "fraction_formula", fraction_formula)
        object.__setattr__(self, "order_statistic_rule", order_rule)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def planned_days(self) -> tuple[datetime, ...]:
        days: list[datetime] = []
        cursor = self.calibration_start.replace(day=1)
        while cursor < self.calibration_stop_exclusive:
            for day in self.sample_month_days:
                candidate = cursor.replace(day=day)
                if (
                    self.calibration_start
                    <= candidate
                    < self.calibration_stop_exclusive
                ):
                    days.append(candidate)
            cursor = _next_month(cursor)
        return tuple(days)

    @property
    def planned_urls(self) -> tuple[str, ...]:
        return tuple(
            self.archive_url_template.format(
                symbol=symbol,
                date=day.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
            for day in self.planned_days
        )

    def lower_tail_rank(self, sample_count: int) -> int:
        count = _int(sample_count, field="sample_count")
        if count <= 0:
            raise ValueError("sample_count must be positive")
        return max(0, math.ceil(self.lower_tail_quantile * count) - 1)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_dataset_id": self.canonical_dataset_id,
            "canonical_dataset_artifact_digest": (
                self.canonical_dataset_artifact_digest
            ),
            "canonical_study_digest": self.canonical_study_digest,
            "provider_head_sha": self.provider_head_sha,
            "provider_parser_blob_sha": self.provider_parser_blob_sha,
            "source_validation_run_id": self.source_validation_run_id,
            "source_validation_artifact_id": self.source_validation_artifact_id,
            "source_validation_artifact_digest": (
                self.source_validation_artifact_digest
            ),
            "source_validation_manifest_sha256": (
                self.source_validation_manifest_sha256
            ),
            "causal_capacity_head_sha": self.causal_capacity_head_sha,
            "processing_bar_volume_capacity": (self.processing_bar_volume_capacity),
            "market": self.market,
            "symbols": list(self.symbols),
            "calibration_start": _iso_utc(self.calibration_start),
            "calibration_stop_exclusive": _iso_utc(self.calibration_stop_exclusive),
            "evaluation_start": _iso_utc(self.evaluation_start),
            "sample_month_days": list(self.sample_month_days),
            "burst_window_milliseconds": self.burst_window_milliseconds,
            "lower_tail_quantile": self.lower_tail_quantile,
            "utilization_fraction": self.utilization_fraction,
            "hard_ceiling": self.hard_ceiling,
            "min_valid_days_per_symbol": self.min_valid_days_per_symbol,
            "min_valid_days_per_year": self.min_valid_days_per_year,
            "min_valid_hours_per_symbol": self.min_valid_hours_per_symbol,
            "checksum_required": self.checksum_required,
            "checksum_suffix": self.checksum_suffix,
            "buyer_taker_when_buyer_is_maker": (self.buyer_taker_when_buyer_is_maker),
            "seller_taker_when_buyer_is_maker": (self.seller_taker_when_buyer_is_maker),
            "archive_url_template": self.archive_url_template,
            "trade_notional_formula": self.trade_notional_formula,
            "hour_alignment": self.hour_alignment,
            "burst_bin_alignment": self.burst_bin_alignment,
            "fraction_formula": self.fraction_formula,
            "order_statistic_rule": self.order_statistic_rule,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_m2_aggtrades_flow_capacity_protocol() -> AggTradesFlowCapacityProtocol:
    """Return the frozen canonical M2 aggTrades capacity preregistration."""

    return AggTradesFlowCapacityProtocol(
        canonical_dataset_id=_CANONICAL_DATASET_ID,
        canonical_dataset_artifact_digest=_CANONICAL_DATASET_ARTIFACT_DIGEST,
        canonical_study_digest=_CANONICAL_STUDY_DIGEST,
        provider_head_sha=_PROVIDER_HEAD_SHA,
        provider_parser_blob_sha=_PROVIDER_PARSER_BLOB_SHA,
        source_validation_run_id=_SOURCE_VALIDATION_RUN_ID,
        source_validation_artifact_id=_SOURCE_VALIDATION_ARTIFACT_ID,
        source_validation_artifact_digest=_SOURCE_VALIDATION_ARTIFACT_DIGEST,
        source_validation_manifest_sha256=_SOURCE_VALIDATION_MANIFEST_SHA256,
        causal_capacity_head_sha=_CAUSAL_CAPACITY_HEAD_SHA,
        processing_bar_volume_capacity=_PROCESSING_BAR_VOLUME_CAPACITY,
        market=_MARKET,
        symbols=_SYMBOLS,
        calibration_start=_CALIBRATION_START,
        calibration_stop_exclusive=_CALIBRATION_STOP_EXCLUSIVE,
        evaluation_start=_EVALUATION_START,
        sample_month_days=_SAMPLE_MONTH_DAYS,
        burst_window_milliseconds=_BURST_WINDOW_MILLISECONDS,
        lower_tail_quantile=_LOWER_TAIL_QUANTILE,
        utilization_fraction=_UTILIZATION_FRACTION,
        hard_ceiling=_HARD_CEILING,
        min_valid_days_per_symbol=_MIN_VALID_DAYS_PER_SYMBOL,
        min_valid_days_per_year=_MIN_VALID_DAYS_PER_YEAR,
        min_valid_hours_per_symbol=_MIN_VALID_HOURS_PER_SYMBOL,
        checksum_required=_CHECKSUM_REQUIRED,
        checksum_suffix=_CHECKSUM_SUFFIX,
        buyer_taker_when_buyer_is_maker=_BUYER_TAKER_WHEN_BUYER_IS_MAKER,
        seller_taker_when_buyer_is_maker=_SELLER_TAKER_WHEN_BUYER_IS_MAKER,
        archive_url_template=_ARCHIVE_URL_TEMPLATE,
        trade_notional_formula=_TRADE_NOTIONAL_FORMULA,
        hour_alignment=_HOUR_ALIGNMENT,
        burst_bin_alignment=_BURST_BIN_ALIGNMENT,
        fraction_formula=_FRACTION_FORMULA,
        order_statistic_rule=_ORDER_STATISTIC_RULE,
    )


def load_aggtrades_flow_capacity_protocol(
    path: str | Path,
) -> AggTradesFlowCapacityProtocol:
    """Load one strict JSON preregistration without accepting unknown fields."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load aggTrades flow-capacity protocol: {source}"
        ) from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("aggTrades flow-capacity protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    keys = set(payload)
    missing = sorted(_PAYLOAD_FIELDS - keys)
    unknown = sorted(keys - _PAYLOAD_FIELDS)
    if missing or unknown:
        raise ValueError(
            "aggTrades flow-capacity protocol keys differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )

    return AggTradesFlowCapacityProtocol(
        schema_version=_text(payload["schema_version"], field="schema_version"),
        canonical_dataset_id=_sha256(
            payload["canonical_dataset_id"],
            field="canonical_dataset_id",
        ),
        canonical_dataset_artifact_digest=_sha256(
            payload["canonical_dataset_artifact_digest"],
            field="canonical_dataset_artifact_digest",
        ),
        canonical_study_digest=_sha256(
            payload["canonical_study_digest"],
            field="canonical_study_digest",
        ),
        provider_head_sha=_git_sha(
            payload["provider_head_sha"],
            field="provider_head_sha",
        ),
        provider_parser_blob_sha=_git_sha(
            payload["provider_parser_blob_sha"],
            field="provider_parser_blob_sha",
        ),
        source_validation_run_id=_int(
            payload["source_validation_run_id"],
            field="source_validation_run_id",
        ),
        source_validation_artifact_id=_int(
            payload["source_validation_artifact_id"],
            field="source_validation_artifact_id",
        ),
        source_validation_artifact_digest=_sha256(
            payload["source_validation_artifact_digest"],
            field="source_validation_artifact_digest",
        ),
        source_validation_manifest_sha256=_sha256(
            payload["source_validation_manifest_sha256"],
            field="source_validation_manifest_sha256",
        ),
        causal_capacity_head_sha=_git_sha(
            payload["causal_capacity_head_sha"],
            field="causal_capacity_head_sha",
        ),
        processing_bar_volume_capacity=_bool(
            payload["processing_bar_volume_capacity"],
            field="processing_bar_volume_capacity",
        ),
        market=_text(payload["market"], field="market"),
        symbols=_string_tuple(payload["symbols"], field="symbols"),
        calibration_start=_parse_datetime(
            payload["calibration_start"],
            field="calibration_start",
        ),
        calibration_stop_exclusive=_parse_datetime(
            payload["calibration_stop_exclusive"],
            field="calibration_stop_exclusive",
        ),
        evaluation_start=_parse_datetime(
            payload["evaluation_start"],
            field="evaluation_start",
        ),
        sample_month_days=_int_tuple(
            payload["sample_month_days"],
            field="sample_month_days",
        ),
        burst_window_milliseconds=_int(
            payload["burst_window_milliseconds"],
            field="burst_window_milliseconds",
        ),
        lower_tail_quantile=_float(
            payload["lower_tail_quantile"],
            field="lower_tail_quantile",
        ),
        utilization_fraction=_float(
            payload["utilization_fraction"],
            field="utilization_fraction",
        ),
        hard_ceiling=_float(payload["hard_ceiling"], field="hard_ceiling"),
        min_valid_days_per_symbol=_int(
            payload["min_valid_days_per_symbol"],
            field="min_valid_days_per_symbol",
        ),
        min_valid_days_per_year=_int(
            payload["min_valid_days_per_year"],
            field="min_valid_days_per_year",
        ),
        min_valid_hours_per_symbol=_int(
            payload["min_valid_hours_per_symbol"],
            field="min_valid_hours_per_symbol",
        ),
        checksum_required=_bool(
            payload["checksum_required"],
            field="checksum_required",
        ),
        checksum_suffix=_text(
            payload["checksum_suffix"],
            field="checksum_suffix",
        ),
        buyer_taker_when_buyer_is_maker=_bool(
            payload["buyer_taker_when_buyer_is_maker"],
            field="buyer_taker_when_buyer_is_maker",
        ),
        seller_taker_when_buyer_is_maker=_bool(
            payload["seller_taker_when_buyer_is_maker"],
            field="seller_taker_when_buyer_is_maker",
        ),
        archive_url_template=_text(
            payload["archive_url_template"],
            field="archive_url_template",
        ),
        trade_notional_formula=_text(
            payload["trade_notional_formula"],
            field="trade_notional_formula",
        ),
        hour_alignment=_text(payload["hour_alignment"], field="hour_alignment"),
        burst_bin_alignment=_text(
            payload["burst_bin_alignment"],
            field="burst_bin_alignment",
        ),
        fraction_formula=_text(
            payload["fraction_formula"],
            field="fraction_formula",
        ),
        order_statistic_rule=_text(
            payload["order_statistic_rule"],
            field="order_statistic_rule",
        ),
    )


__all__ = [
    "AggTradesFlowCapacityProtocol",
    "canonical_m2_aggtrades_flow_capacity_protocol",
    "load_aggtrades_flow_capacity_protocol",
]
