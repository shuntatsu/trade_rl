"""Execution of the sealed pre-2023 aggTrades flow-capacity calibration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Iterable

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.agg_flow_capacity import (
    AggTradesFlowCapacityProtocol,
    canonical_m2_aggtrades_flow_capacity_protocol,
)
from trade_rl.integrations.binance.agg_trades import BinanceAggTradesSeries

_RESULT_SCHEMA_VERSION: Final = "aggtrades_flow_capacity_calibration_result_v1"
_PROTOCOL_DIGEST: Final = (
    "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
)
_PROTOCOL_SEAL_ARTIFACT_ID: Final = 10319386570
_PROTOCOL_SEAL_ARTIFACT_DIGEST: Final = (
    "dbc6ef286abb9e4c8530089328fb64b2cfaa5e98715a82740439470a45942e57"
)
_PROTOCOL_SEALED_SOURCE_SHA: Final = "f09784d456199227a49790beb3af5baf2f92da02"
_PROTOCOL_SEALED_TREE_SHA: Final = "7caa248f3b1f5de8c93993a5d74b63a73ee93705"
_PROTOCOL_JSON_SHA256: Final = (
    "e0b21c51ad2ad8b866580f599b7d0eee7de138b5a4c88c0a3d05e74fedd4f67b"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HOUR_MILLISECONDS = 60 * 60 * 1_000
_TOLERANCE = 1e-12


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_day(value: datetime, *, field: str) -> datetime:
    resolved = _aware_utc(value, field=field)
    if resolved != resolved.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError(f"{field} must be UTC midnight")
    return resolved


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: int | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_text(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not value.strip():
        raise ValueError(f"{field} must be non-empty when present")
    return value


def _datetime_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, tz=UTC)


def _canonical_protocol(protocol: AggTradesFlowCapacityProtocol) -> None:
    canonical = canonical_m2_aggtrades_flow_capacity_protocol()
    if protocol != canonical or protocol.digest != _PROTOCOL_DIGEST:
        raise ValueError("calibration protocol differs from the sealed preregistration")


@dataclass(frozen=True, slots=True)
class AggTradesArchiveCalibration:
    """One planned archive's provenance and derived valid hourly fractions."""

    symbol: str
    day: datetime
    url: str
    accepted: bool
    rejection_reason: str | None
    raw_payload_sha256: str | None
    raw_payload_size_bytes: int | None
    checksum_text: str | None
    checksum_verified: bool
    header_present: bool | None
    row_count: int | None
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    hourly_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        url = self.url.strip()
        day = _utc_day(self.day, field="day")
        reason = _optional_text(self.rejection_reason, field="rejection_reason")
        raw_digest = _sha256(self.raw_payload_sha256, field="raw_payload_sha256")
        raw_size = _positive_int(
            self.raw_payload_size_bytes,
            field="raw_payload_size_bytes",
        )
        checksum_text = _optional_text(self.checksum_text, field="checksum_text")
        row_count = _positive_int(self.row_count, field="row_count")
        first_timestamp = (
            None
            if self.first_timestamp is None
            else _aware_utc(self.first_timestamp, field="first_timestamp")
        )
        last_timestamp = (
            None
            if self.last_timestamp is None
            else _aware_utc(self.last_timestamp, field="last_timestamp")
        )
        fractions = tuple(float(value) for value in self.hourly_fractions)

        if not symbol:
            raise ValueError("symbol must be non-empty")
        if not url:
            raise ValueError("url must be non-empty")
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not isinstance(self.checksum_verified, bool):
            raise ValueError("checksum_verified must be a boolean")
        if self.header_present is not None and not isinstance(self.header_present, bool):
            raise ValueError("header_present must be a boolean when present")
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0 + _TOLERANCE
            for value in fractions
        ):
            raise ValueError("hourly_fractions must be finite values within (0, 1]")

        if self.accepted:
            if reason is not None:
                raise ValueError("accepted archive cannot have rejection_reason")
            if (
                raw_digest is None
                or raw_size is None
                or checksum_text is None
                or not self.checksum_verified
                or self.header_present is None
                or row_count is None
                or first_timestamp is None
                or last_timestamp is None
            ):
                raise ValueError("accepted archive requires complete provenance")
            if first_timestamp > last_timestamp:
                raise ValueError("first_timestamp cannot exceed last_timestamp")
            next_day = day + timedelta(days=1)
            if not day <= first_timestamp < next_day:
                raise ValueError("first_timestamp must remain on the requested UTC day")
            if not day <= last_timestamp < next_day:
                raise ValueError("last_timestamp must remain on the requested UTC day")
        else:
            if reason is None:
                raise ValueError("rejected archive requires rejection_reason")
            if fractions:
                raise ValueError("rejected archive cannot contain hourly_fractions")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "raw_payload_sha256", raw_digest)
        object.__setattr__(self, "raw_payload_size_bytes", raw_size)
        object.__setattr__(self, "checksum_text", checksum_text)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "first_timestamp", first_timestamp)
        object.__setattr__(self, "last_timestamp", last_timestamp)
        object.__setattr__(self, "hourly_fractions", fractions)

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "date": self.day.date().isoformat(),
            "url": self.url,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "raw_payload_sha256": self.raw_payload_sha256,
            "raw_payload_size_bytes": self.raw_payload_size_bytes,
            "checksum_text": self.checksum_text,
            "checksum_verified": self.checksum_verified,
            "header_present": self.header_present,
            "row_count": self.row_count,
            "first_timestamp": (
                None
                if self.first_timestamp is None
                else _iso_utc(self.first_timestamp)
            ),
            "last_timestamp": (
                None if self.last_timestamp is None else _iso_utc(self.last_timestamp)
            ),
            "hourly_fractions": list(self.hourly_fractions),
        }


@dataclass(frozen=True, slots=True)
class AggTradesSymbolCalibrationSummary:
    """Registered coverage and lower-tail result for one symbol."""

    symbol: str
    accepted_days: int
    accepted_days_2021: int
    accepted_days_2022: int
    valid_hours: int
    rank: int | None
    q10: float | None
    symbol_cap: float | None
    failures: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "accepted_days": self.accepted_days,
            "accepted_days_2021": self.accepted_days_2021,
            "accepted_days_2022": self.accepted_days_2022,
            "valid_hours": self.valid_hours,
            "rank": self.rank,
            "q10": self.q10,
            "symbol_cap": self.symbol_cap,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class AggTradesFlowCalibrationResult:
    """Content-addressed result of exactly one sealed calibration execution."""

    status: str
    protocol_digest: str
    protocol_seal_artifact_id: int
    protocol_seal_artifact_digest: str
    protocol_sealed_source_sha: str
    protocol_sealed_tree_sha: str
    protocol_json_sha256: str
    provider_head_sha: str
    provider_parser_blob_sha: str
    planned_urls: tuple[str, ...]
    archives: tuple[AggTradesArchiveCalibration, ...]
    symbol_summaries: tuple[AggTradesSymbolCalibrationSummary, ...]
    failures: tuple[str, ...]
    schema_version: str = _RESULT_SCHEMA_VERSION

    def to_content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "protocol_digest": self.protocol_digest,
            "protocol_seal_artifact_id": self.protocol_seal_artifact_id,
            "protocol_seal_artifact_digest": self.protocol_seal_artifact_digest,
            "protocol_sealed_source_sha": self.protocol_sealed_source_sha,
            "protocol_sealed_tree_sha": self.protocol_sealed_tree_sha,
            "protocol_json_sha256": self.protocol_json_sha256,
            "provider_head_sha": self.provider_head_sha,
            "provider_parser_blob_sha": self.provider_parser_blob_sha,
            "planned_urls": list(self.planned_urls),
            "archives": [archive.to_payload() for archive in self.archives],
            "symbol_summaries": [
                summary.to_payload() for summary in self.symbol_summaries
            ],
            "failures": list(self.failures),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_content_payload())

    def to_payload(self) -> dict[str, object]:
        payload = self.to_content_payload()
        payload["result_digest"] = self.digest
        return payload


def _expected_archive_keys(
    protocol: AggTradesFlowCapacityProtocol,
) -> tuple[tuple[str, datetime, str], ...]:
    return tuple(
        (
            symbol,
            day,
            protocol.archive_url_template.format(
                symbol=symbol,
                date=day.strftime("%Y-%m-%d"),
            ),
        )
        for symbol in protocol.symbols
        for day in protocol.planned_days
    )


def _checksum_digest(checksum_text: str) -> str:
    fields = checksum_text.split()
    if not fields or _SHA256_RE.fullmatch(fields[0].lower()) is None:
        raise ValueError("checksum_text must begin with a SHA-256 digest")
    return fields[0].lower()


def evaluate_aggtrades_archive(
    protocol: AggTradesFlowCapacityProtocol,
    *,
    symbol: str,
    day: datetime,
    url: str,
    series: BinanceAggTradesSeries,
    checksum_text: str,
    checksum_verified: bool,
) -> AggTradesArchiveCalibration:
    """Derive valid hourly fractions from one accepted planned archive."""

    _canonical_protocol(protocol)
    resolved_day = _utc_day(day, field="day")
    expected_url = protocol.archive_url_template.format(
        symbol=symbol,
        date=resolved_day.strftime("%Y-%m-%d"),
    )
    expected_key = (symbol, resolved_day, expected_url)
    if expected_key not in set(_expected_archive_keys(protocol)) or url != expected_url:
        raise ValueError("archive is outside the sealed planned roster")
    if series.source_uri != url:
        raise ValueError("aggTrades source URI does not match the planned URL")
    if not checksum_verified:
        raise ValueError("official archive checksum must be verified")
    if _checksum_digest(checksum_text) != series.raw_payload_sha256:
        raise ValueError("official archive checksum differs from raw payload SHA-256")

    timestamps_ms = series.timestamps.astype("datetime64[ms]").astype(np.int64)
    round_trip = timestamps_ms.astype("datetime64[ms]").astype("datetime64[ns]")
    if not np.array_equal(round_trip, series.timestamps):
        raise ValueError("aggTrades timestamps must preserve millisecond precision")
    start_ms = int(resolved_day.timestamp() * 1_000)
    stop_ms = int((resolved_day + timedelta(days=1)).timestamp() * 1_000)
    evaluation_ms = int(protocol.evaluation_start.timestamp() * 1_000)
    if np.any(timestamps_ms < start_ms) or np.any(timestamps_ms >= stop_ms):
        raise ValueError("aggTrades timestamp escapes the requested UTC day")
    if np.any(timestamps_ms >= evaluation_ms):
        raise ValueError("aggTrades timestamp crosses the evaluation boundary")

    notionals = series.prices * series.quantities
    if not np.all(np.isfinite(notionals)) or np.any(notionals <= 0.0):
        raise ValueError("aggTrades trade notionals must be finite and positive")

    hour_keys = timestamps_ms // _HOUR_MILLISECONDS
    fractions: list[float] = []
    for hour_key in np.unique(hour_keys):
        mask = hour_keys == hour_key
        hour_notional = notionals[mask]
        total_notional = float(np.sum(hour_notional, dtype=np.float64))
        if not math.isfinite(total_notional) or total_notional <= 0.0:
            continue

        hour_timestamps = timestamps_ms[mask]
        bins = (hour_timestamps % _HOUR_MILLISECONDS) // protocol.burst_window_milliseconds
        buyer_maker = series.buyer_is_maker[mask]
        buy_weights = hour_notional[~buyer_maker]
        buy_bins = bins[~buyer_maker]
        sell_weights = hour_notional[buyer_maker]
        sell_bins = bins[buyer_maker]
        if buy_weights.size == 0 or sell_weights.size == 0:
            continue

        bin_count = _HOUR_MILLISECONDS // protocol.burst_window_milliseconds
        buy_flow = np.bincount(
            buy_bins,
            weights=buy_weights,
            minlength=bin_count,
        )
        sell_flow = np.bincount(
            sell_bins,
            weights=sell_weights,
            minlength=bin_count,
        )
        peak_buy = float(np.max(buy_flow))
        peak_sell = float(np.max(sell_flow))
        if (
            not math.isfinite(peak_buy)
            or not math.isfinite(peak_sell)
            or peak_buy <= 0.0
            or peak_sell <= 0.0
        ):
            continue
        fraction = min(peak_buy, peak_sell) / total_notional
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0 + _TOLERANCE:
            raise ValueError("derived worst-side hourly fraction is invalid")
        fractions.append(fraction)

    first_timestamp = _datetime_from_ms(int(timestamps_ms[0]))
    last_timestamp = _datetime_from_ms(int(timestamps_ms[-1]))
    return AggTradesArchiveCalibration(
        symbol=symbol,
        day=resolved_day,
        url=url,
        accepted=True,
        rejection_reason=None,
        raw_payload_sha256=series.raw_payload_sha256,
        raw_payload_size_bytes=series.raw_payload_size_bytes,
        checksum_text=checksum_text,
        checksum_verified=True,
        header_present=series.header_present,
        row_count=int(series.aggregate_trade_ids.size),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        hourly_fractions=tuple(fractions),
    )


def rejected_aggtrades_archive(
    protocol: AggTradesFlowCapacityProtocol,
    *,
    symbol: str,
    day: datetime,
    url: str,
    reason: str,
    raw_payload_sha256: str | None = None,
    raw_payload_size_bytes: int | None = None,
    checksum_text: str | None = None,
    checksum_verified: bool = False,
) -> AggTradesArchiveCalibration:
    """Record one planned archive rejection without substituting another day."""

    _canonical_protocol(protocol)
    resolved_day = _utc_day(day, field="day")
    expected_url = protocol.archive_url_template.format(
        symbol=symbol,
        date=resolved_day.strftime("%Y-%m-%d"),
    )
    if (symbol, resolved_day, expected_url) not in set(_expected_archive_keys(protocol)):
        raise ValueError("archive is outside the sealed planned roster")
    if url != expected_url:
        raise ValueError("rejected archive URL differs from the sealed planned URL")
    return AggTradesArchiveCalibration(
        symbol=symbol,
        day=resolved_day,
        url=url,
        accepted=False,
        rejection_reason=reason,
        raw_payload_sha256=raw_payload_sha256,
        raw_payload_size_bytes=raw_payload_size_bytes,
        checksum_text=checksum_text,
        checksum_verified=checksum_verified,
        header_present=None,
        row_count=None,
        first_timestamp=None,
        last_timestamp=None,
        hourly_fractions=(),
    )


def _symbol_summary(
    protocol: AggTradesFlowCapacityProtocol,
    symbol: str,
    archives: tuple[AggTradesArchiveCalibration, ...],
) -> AggTradesSymbolCalibrationSummary:
    accepted = tuple(archive for archive in archives if archive.accepted)
    accepted_2021 = sum(archive.day.year == 2021 for archive in accepted)
    accepted_2022 = sum(archive.day.year == 2022 for archive in accepted)
    fractions = tuple(
        value for archive in accepted for value in archive.hourly_fractions
    )
    failures: list[str] = []
    if len(accepted) < protocol.min_valid_days_per_symbol:
        failures.append(
            f"{symbol}: accepted_days={len(accepted)} "
            f"< {protocol.min_valid_days_per_symbol}"
        )
    if accepted_2021 < protocol.min_valid_days_per_year:
        failures.append(
            f"{symbol}: accepted_days_2021={accepted_2021} "
            f"< {protocol.min_valid_days_per_year}"
        )
    if accepted_2022 < protocol.min_valid_days_per_year:
        failures.append(
            f"{symbol}: accepted_days_2022={accepted_2022} "
            f"< {protocol.min_valid_days_per_year}"
        )
    if len(fractions) < protocol.min_valid_hours_per_symbol:
        failures.append(
            f"{symbol}: valid_hours={len(fractions)} "
            f"< {protocol.min_valid_hours_per_symbol}"
        )

    rank: int | None = None
    q10: float | None = None
    symbol_cap: float | None = None
    if not failures:
        ordered = tuple(sorted(fractions))
        rank = protocol.lower_tail_rank(len(ordered))
        q10 = ordered[rank]
        symbol_cap = min(protocol.hard_ceiling, protocol.utilization_fraction * q10)
        if (
            not math.isfinite(q10)
            or q10 <= 0.0
            or not math.isfinite(symbol_cap)
            or symbol_cap <= 0.0
        ):
            failures.append(f"{symbol}: q10 or symbol_cap is not finite and positive")
            rank = None
            q10 = None
            symbol_cap = None

    return AggTradesSymbolCalibrationSummary(
        symbol=symbol,
        accepted_days=len(accepted),
        accepted_days_2021=accepted_2021,
        accepted_days_2022=accepted_2022,
        valid_hours=len(fractions),
        rank=rank,
        q10=q10,
        symbol_cap=symbol_cap,
        failures=tuple(failures),
    )


def build_aggtrades_flow_calibration_result(
    protocol: AggTradesFlowCapacityProtocol,
    archives: Iterable[AggTradesArchiveCalibration],
) -> AggTradesFlowCalibrationResult:
    """Build the exact registered PASS/INVALID calibration result."""

    _canonical_protocol(protocol)
    records = tuple(archives)
    expected = _expected_archive_keys(protocol)
    actual = tuple((record.symbol, record.day, record.url) for record in records)
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate archive record in planned roster")
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("archive records differ from the sealed planned roster")
    by_key = {key: record for key, record in zip(actual, records, strict=True)}
    canonical_records = tuple(by_key[key] for key in expected)

    summaries = tuple(
        _symbol_summary(
            protocol,
            symbol,
            tuple(record for record in canonical_records if record.symbol == symbol),
        )
        for symbol in protocol.symbols
    )
    failures = tuple(
        failure for summary in summaries for failure in summary.failures
    )
    status = "PASS" if not failures else "INVALID"
    return AggTradesFlowCalibrationResult(
        status=status,
        protocol_digest=protocol.digest,
        protocol_seal_artifact_id=_PROTOCOL_SEAL_ARTIFACT_ID,
        protocol_seal_artifact_digest=_PROTOCOL_SEAL_ARTIFACT_DIGEST,
        protocol_sealed_source_sha=_PROTOCOL_SEALED_SOURCE_SHA,
        protocol_sealed_tree_sha=_PROTOCOL_SEALED_TREE_SHA,
        protocol_json_sha256=_PROTOCOL_JSON_SHA256,
        provider_head_sha=protocol.provider_head_sha,
        provider_parser_blob_sha=protocol.provider_parser_blob_sha,
        planned_urls=protocol.planned_urls,
        archives=canonical_records,
        symbol_summaries=summaries,
        failures=failures,
    )


__all__ = [
    "AggTradesArchiveCalibration",
    "AggTradesFlowCalibrationResult",
    "AggTradesSymbolCalibrationSummary",
    "build_aggtrades_flow_calibration_result",
    "evaluate_aggtrades_archive",
    "rejected_aggtrades_archive",
]
