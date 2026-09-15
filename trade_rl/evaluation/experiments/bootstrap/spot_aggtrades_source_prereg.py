"""Result-blind source protocol for the Issue 578 Binance Spot aggTrades probe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "spot_aggtrades_source_prereg_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_DATES = ("2021-01-15", "2021-07-15", "2022-01-15", "2022-07-15")
_MARKET = "spot"
_SOURCE_FAMILY = "aggTrades"
_URL_TEMPLATE = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{date}.zip"
)
_CHECKSUM_SUFFIX = ".CHECKSUM"
_EXPECTED_HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)
_FIELD_COUNT = 8
_SENTINEL = {
    "price": "0",
    "quantity": "0",
    "first_trade_id": "-1",
    "last_trade_id": "-1",
}
_BOOLEAN_TOKENS = ("False", "True")
_ALLOWED_REPORT_FIELDS = (
    "symbol",
    "date",
    "url",
    "checksum_url",
    "archive_available",
    "raw_zip_size_bytes",
    "raw_zip_sha256",
    "checksum_available",
    "checksum_text",
    "checksum_digest",
    "checksum_verified",
    "member_name",
    "header_present",
    "normalized_schema",
    "total_row_count",
    "usable_row_count",
    "provider_invalid_sentinel_count",
    "malformed_row_count",
    "first_usable_aggregate_id",
    "last_usable_aggregate_id",
    "first_usable_event_timestamp_ms",
    "last_usable_event_timestamp_ms",
    "usable_ids_strictly_increasing_unique",
    "usable_timestamps_nondecreasing",
    "timestamps_inside_requested_utc_date",
    "schema_valid",
)
_PASS_STATUS = "PASS_SPOT_AGGTRADES_SOURCE"
_PARTIAL_STATUS = "PARTIAL_SPOT_AGGTRADES_SOURCE"
_INCOMPATIBLE_STATUS = "INCOMPATIBLE_SPOT_AGGTRADES_SOURCE"


def _require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _require_str_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class SpotAggTradesSourceProtocol:
    """Immutable structural-only authority for the frozen 20-archive probe."""

    schema_version: str = _SCHEMA_VERSION
    symbols: tuple[str, ...] = _SYMBOLS
    dates: tuple[str, ...] = _DATES
    planned_archive_count: int = len(_SYMBOLS) * len(_DATES)
    market: str = _MARKET
    source_family: str = _SOURCE_FAMILY
    url_template: str = _URL_TEMPLATE
    checksum_suffix: str = _CHECKSUM_SUFFIX
    expected_header: tuple[str, ...] = _EXPECTED_HEADER
    field_count: int = _FIELD_COUNT
    strict_boolean_tokens: tuple[str, ...] = _BOOLEAN_TOKENS
    allowed_archive_report_fields: tuple[str, ...] = _ALLOWED_REPORT_FIELDS
    pass_status: str = _PASS_STATUS
    partial_status: str = _PARTIAL_STATUS
    incompatible_status: str = _INCOMPATIBLE_STATUS
    replacement_sources_allowed: bool = False
    economic_values_allowed_in_report: bool = False
    target_relation_allowed: bool = False
    evaluation_pnl_inspected: bool = False
    feature_hypothesis_selected: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    archive_publication_time_is_market_event_time: bool = False
    post_2025_microsecond_semantics_allowed: bool = False
    fresh_reconstruction_required: bool = True
    canonical_report_byte_equality_required: bool = True

    def __post_init__(self) -> None:
        canonical_values: tuple[tuple[str, object], ...] = (
            ("schema_version", _SCHEMA_VERSION),
            ("symbols", _SYMBOLS),
            ("dates", _DATES),
            ("planned_archive_count", len(_SYMBOLS) * len(_DATES)),
            ("market", _MARKET),
            ("source_family", _SOURCE_FAMILY),
            ("url_template", _URL_TEMPLATE),
            ("checksum_suffix", _CHECKSUM_SUFFIX),
            ("expected_header", _EXPECTED_HEADER),
            ("field_count", _FIELD_COUNT),
            ("strict_boolean_tokens", _BOOLEAN_TOKENS),
            ("allowed_archive_report_fields", _ALLOWED_REPORT_FIELDS),
            ("pass_status", _PASS_STATUS),
            ("partial_status", _PARTIAL_STATUS),
            ("incompatible_status", _INCOMPATIBLE_STATUS),
        )
        for field_name, expected in canonical_values:
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} is not canonical")

        expected_flags = {
            "replacement_sources_allowed": False,
            "economic_values_allowed_in_report": False,
            "target_relation_allowed": False,
            "evaluation_pnl_inspected": False,
            "feature_hypothesis_selected": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "archive_publication_time_is_market_event_time": False,
            "post_2025_microsecond_semantics_allowed": False,
            "fresh_reconstruction_required": True,
            "canonical_report_byte_equality_required": True,
        }
        for field_name, expected in expected_flags.items():
            value = _require_bool(getattr(self, field_name), field=field_name)
            if value is not expected:
                raise ValueError(f"{field_name} is not canonical")

    @property
    def provider_invalid_sentinel(self) -> dict[str, str]:
        return dict(_SENTINEL)

    @property
    def planned_urls(self) -> tuple[str, ...]:
        return tuple(
            self.url_template.format(symbol=symbol, date=date)
            for symbol in self.symbols
            for date in self.dates
        )

    @property
    def planned_checksum_urls(self) -> tuple[str, ...]:
        return tuple(url + self.checksum_suffix for url in self.planned_urls)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "planned_archive_count": self.planned_archive_count,
            "market": self.market,
            "source_family": self.source_family,
            "url_template": self.url_template,
            "checksum_suffix": self.checksum_suffix,
            "expected_header": list(self.expected_header),
            "field_count": self.field_count,
            "provider_invalid_sentinel": self.provider_invalid_sentinel,
            "strict_boolean_tokens": list(self.strict_boolean_tokens),
            "allowed_archive_report_fields": list(self.allowed_archive_report_fields),
            "pass_status": self.pass_status,
            "partial_status": self.partial_status,
            "incompatible_status": self.incompatible_status,
            "replacement_sources_allowed": self.replacement_sources_allowed,
            "economic_values_allowed_in_report": self.economic_values_allowed_in_report,
            "target_relation_allowed": self.target_relation_allowed,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
            "feature_hypothesis_selected": self.feature_hypothesis_selected,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
            "archive_publication_time_is_market_event_time": (
                self.archive_publication_time_is_market_event_time
            ),
            "post_2025_microsecond_semantics_allowed": (
                self.post_2025_microsecond_semantics_allowed
            ),
            "fresh_reconstruction_required": self.fresh_reconstruction_required,
            "canonical_report_byte_equality_required": (
                self.canonical_report_byte_equality_required
            ),
        }

    @property
    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SpotAggTradesSourceProtocol:
        canonical = canonical_spot_aggtrades_source_protocol()
        expected_fields = set(canonical.to_payload())
        if set(payload) != expected_fields:
            raise ValueError("protocol payload fields are not canonical")

        sentinel = payload["provider_invalid_sentinel"]
        if not isinstance(sentinel, Mapping) or dict(sentinel) != _SENTINEL:
            raise ValueError("provider_invalid_sentinel is not canonical")

        kwargs = dict(payload)
        kwargs.pop("provider_invalid_sentinel")
        for field_name in (
            "symbols",
            "dates",
            "expected_header",
            "strict_boolean_tokens",
            "allowed_archive_report_fields",
        ):
            kwargs[field_name] = _require_str_tuple(
                kwargs[field_name], field=field_name
            )
        return cls(**kwargs)


def canonical_spot_aggtrades_source_protocol() -> SpotAggTradesSourceProtocol:
    """Return the sole canonical Issue 578 source protocol."""

    return SpotAggTradesSourceProtocol()


def load_spot_aggtrades_source_protocol_bytes(
    payload: bytes,
) -> SpotAggTradesSourceProtocol:
    """Load only exact canonical protocol bytes; formatting drift fails closed."""

    import json

    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol payload is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("protocol payload must be a JSON object")
    protocol = SpotAggTradesSourceProtocol.from_payload(decoded)
    if payload != protocol.canonical_json_bytes:
        raise ValueError("protocol bytes are not canonical")
    return protocol
