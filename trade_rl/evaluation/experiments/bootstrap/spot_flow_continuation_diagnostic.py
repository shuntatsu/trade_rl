"""Deterministic training-only core for the sealed Issue 584 Spot-flow diagnostic."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    SpotFlowContinuationProtocol,
    canonical_spot_flow_continuation_protocol,
)

_SCHEMA_VERSION = "spot_flow_continuation_diagnostic_v1"
_PREREG_HEAD = "41a403bda4252f15d8c059166dd6716110a4bc37"
_PREREG_FULL_VERIFY_RUN_ID = 34952378700
_PREREG_SEAL_RUN_ID = 34952843190
_PREREG_SEAL_ARTIFACT_ID = 10390380265
_PREREG_SEAL_ARTIFACT_API_DIGEST = (
    "bd80fe0ae93ad95c5106eacc9f9b99c85164da65f5fc7ae8d290f45a03e84646"
)
_PREREG_FRESH_ARTIFACT_ID = 10390415743
_PREREG_FRESH_ARTIFACT_API_DIGEST = (
    "014cc3fc01b34a40086efe776717dfff15ffed1c5874afcc5e0aa907a9fd7115"
)
_PREREG_PROTOCOL_DIGEST = (
    "9824daffb6fbd77ab8994082cb058d8be981cc331a54434123c34b8890e94b76"
)
_TARGET_PREFLIGHT_STATUS = "PASS_USDM_15M_TARGET_SOURCE"
_TARGET_PREFLIGHT_RUN_ID = 34953161630
_TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID = 10389544902
_TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_API_DIGEST = (
    "2a1f1d7d700fc662862ebab5b41ff692d154dbfef793fb505ce7e1ea5647e2c2"
)
_TARGET_PREFLIGHT_FRESH_ARTIFACT_ID = 10389694422
_TARGET_PREFLIGHT_FRESH_ARTIFACT_API_DIGEST = (
    "8b5cbd19abb9517bf4885f3b9432b29024069f0005b8d18e89b82b2471765e3d"
)
_TARGET_PREFLIGHT_REPORT_SHA256 = (
    "721c785ddac113f7d6be2b5af3e1aa10df34ae9a16b9656e02a64c874f4526ea"
)
_TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST = (
    "bd32c2e8fb9b4b80680fdfb0b26ca4b7fd6f01104597cba8c81990faa4f312d1"
)
_TARGET_MANIFEST_DIGEST = (
    "69a4105e8839d38972a034b49c9a3d019d3bc330981fce7cb860276e9ca0b25b"
)
_QUARTER_HOUR_MS = 15 * 60 * 1_000
_FOUR_HOURS_MS = 4 * 60 * 60 * 1_000
_SPOT_HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)
_KLINE_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _require_hex(value: object, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    if value.lower() != value or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_finite_float(
    value: object, *, field: str, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved) or (positive and resolved <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return resolved


def _require_optional_finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field=field)


def _parse_int_token(value: str, *, field: str) -> int:
    if not value or value.strip() != value or value.startswith("+"):
        raise ValueError(f"{field} must be an integer token")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer token") from error
    if str(parsed) != value and not (value == "-0" and parsed == 0):
        raise ValueError(f"{field} must be an integer token")
    return parsed


def _parse_float_token(value: str, *, field: str) -> float:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be finite")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _day_bounds_ms(date: str) -> tuple[int, int]:
    try:
        start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("expected_date must use YYYY-MM-DD") from error
    if start.strftime("%Y-%m-%d") != date:
        raise ValueError("expected_date must use YYYY-MM-DD")
    return int(start.timestamp() * 1_000), int(
        (start + timedelta(days=1)).timestamp() * 1_000
    )


def _decode_csv(payload: bytes) -> list[list[str]]:
    if not isinstance(payload, bytes):
        raise TypeError("CSV payload must be bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV payload is not UTF-8") from error
    return list(csv.reader(io.StringIO(text)))


@dataclass(frozen=True, slots=True)
class SpotAggTrade:
    event_time_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool

    def __post_init__(self) -> None:
        _require_int(self.event_time_ms, field="event_time_ms")
        _require_finite_float(self.price, field="price", positive=True)
        _require_finite_float(self.quantity, field="quantity", positive=True)
        _require_bool(self.is_buyer_maker, field="is_buyer_maker")


@dataclass(frozen=True, slots=True)
class TargetBar:
    open_time_ms: int
    open_price: float
    information_available: bool = True
    active: bool = True
    tradable: bool = True

    def __post_init__(self) -> None:
        timestamp = _require_int(self.open_time_ms, field="open_time_ms")
        if timestamp % _QUARTER_HOUR_MS != 0:
            raise ValueError("target open timestamp must lie on the native 15m grid")
        _require_finite_float(self.open_price, field="open_price", positive=True)
        _require_bool(self.information_available, field="information_available")
        _require_bool(self.active, field="active")
        _require_bool(self.tradable, field="tradable")


def _parse_spot_row(
    row: list[str],
    *,
    day_start: int,
    day_end: int,
    previous_id: int | None,
    previous_timestamp: int | None,
) -> tuple[SpotAggTrade | None, int | None, int | None]:
    if len(row) != 8:
        raise ValueError("Spot aggTrades row is malformed: expected eight fields")
    (
        aggregate_raw,
        price_raw,
        quantity_raw,
        first_raw,
        last_raw,
        timestamp_raw,
        maker_raw,
        best_raw,
    ) = row
    aggregate_id = _parse_int_token(aggregate_raw, field="aggregate trade id")
    timestamp = _parse_int_token(timestamp_raw, field="event timestamp")
    if aggregate_id < 0:
        raise ValueError("aggregate trade id must be non-negative")
    if timestamp < day_start or timestamp >= day_end:
        raise ValueError("event timestamp is outside expected_date")
    if maker_raw not in {"False", "True"} or best_raw not in {"False", "True"}:
        raise ValueError("Spot aggTrades maker/best-match boolean token is invalid")

    exact_sentinel = (
        price_raw == "0"
        and quantity_raw == "0"
        and first_raw == "-1"
        and last_raw == "-1"
    )
    sentinel_component = (
        price_raw == "0" or quantity_raw == "0" or first_raw == "-1" or last_raw == "-1"
    )
    if exact_sentinel:
        return None, previous_id, previous_timestamp
    if sentinel_component:
        raise ValueError("partial provider sentinel is malformed")

    first_id = _parse_int_token(first_raw, field="first trade id")
    last_id = _parse_int_token(last_raw, field="last trade id")
    if first_id < 0 or last_id < 0 or first_id > last_id:
        raise ValueError("Spot aggTrades trade-id range is malformed")
    price = _parse_float_token(price_raw, field="price")
    quantity = _parse_float_token(quantity_raw, field="quantity")
    if price <= 0.0 or quantity <= 0.0:
        raise ValueError("Spot aggTrades price and quantity must be positive")
    if previous_id is not None and aggregate_id <= previous_id:
        raise ValueError("usable aggregate trade ids must be strictly increasing")
    if previous_timestamp is not None and timestamp < previous_timestamp:
        raise ValueError("usable event timestamps must be nondecreasing")
    return (
        SpotAggTrade(
            event_time_ms=timestamp,
            price=price,
            quantity=quantity,
            is_buyer_maker=maker_raw == "True",
        ),
        aggregate_id,
        timestamp,
    )


def parse_spot_aggtrades_csv(
    payload: bytes,
    *,
    expected_date: str,
) -> tuple[SpotAggTrade, ...]:
    """Parse only structurally valid Issue 578 Spot aggTrades rows."""

    rows = _decode_csv(payload)
    if rows and tuple(rows[0]) == _SPOT_HEADER:
        rows = rows[1:]
    day_start, day_end = _day_bounds_ms(expected_date)
    result: list[SpotAggTrade] = []
    previous_id: int | None = None
    previous_timestamp: int | None = None
    for row in rows:
        trade, previous_id, previous_timestamp = _parse_spot_row(
            row,
            day_start=day_start,
            day_end=day_end,
            previous_id=previous_id,
            previous_timestamp=previous_timestamp,
        )
        if trade is not None:
            result.append(trade)
    return tuple(result)


def _reduce_imbalance(
    signed_notionals: Sequence[float], total_notionals: Sequence[float]
) -> float | None:
    if not total_notionals:
        return None
    try:
        numerator = math.fsum(signed_notionals)
        denominator = math.fsum(total_notionals)
    except OverflowError as error:
        raise ValueError("Spot interval reduction is non-finite") from error
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError("Spot interval reduction is non-finite")
    result = numerator / denominator
    if not math.isfinite(result) or result < -1.0 - 1e-12 or result > 1.0 + 1e-12:
        raise ValueError("Spot aggressive-flow imbalance lies outside [-1,1]")
    return result


def aggregate_spot_interval(
    trades: Sequence[SpotAggTrade],
    *,
    decision_time_ms: int,
) -> float | None:
    """Compute the frozen [t-15m,t) quote-notional aggressive-flow imbalance."""

    decision = _require_int(decision_time_ms, field="decision_time_ms")
    if decision % _QUARTER_HOUR_MS != 0:
        raise ValueError("decision_time_ms must lie on the UTC quarter-hour grid")
    start = decision - _QUARTER_HOUR_MS
    if start < 0:
        raise ValueError("decision_time_ms does not have a complete 15m lookback")

    previous_timestamp: int | None = None
    signed_notionals: list[float] = []
    total_notionals: list[float] = []
    for trade in trades:
        if not isinstance(trade, SpotAggTrade):
            raise TypeError("trades must contain SpotAggTrade values")
        if previous_timestamp is not None and trade.event_time_ms < previous_timestamp:
            raise ValueError("Spot trades must remain in provider timestamp order")
        previous_timestamp = trade.event_time_ms
        if not start <= trade.event_time_ms < decision:
            continue
        notional = trade.price * trade.quantity
        if not math.isfinite(notional) or notional <= 0.0:
            raise ValueError("Spot trade quote notional must be finite and positive")
        sign = -1.0 if trade.is_buyer_maker else 1.0
        signed_notionals.append(sign * notional)
        total_notionals.append(notional)
    return _reduce_imbalance(signed_notionals, total_notionals)


def aggregate_spot_day_csv(
    payload: bytes,
    *,
    expected_date: str,
) -> tuple[float | None, ...]:
    """Stream one frozen Spot day into the exact 96 quarter-hour signals."""

    if not isinstance(payload, bytes):
        raise TypeError("CSV payload must be bytes")
    day_start, day_end = _day_bounds_ms(expected_date)
    signed_bins: list[list[float]] = [[] for _ in range(96)]
    total_bins: list[list[float]] = [[] for _ in range(96)]
    previous_id: int | None = None
    previous_timestamp: int | None = None

    stream = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline="")
    try:
        reader = csv.reader(stream)
        first = next(reader, None)
        rows = reader if first is not None and tuple(first) == _SPOT_HEADER else None

        def consume(row: list[str]) -> None:
            nonlocal previous_id, previous_timestamp
            trade, previous_id, previous_timestamp = _parse_spot_row(
                row,
                day_start=day_start,
                day_end=day_end,
                previous_id=previous_id,
                previous_timestamp=previous_timestamp,
            )
            if trade is None:
                return
            index = (trade.event_time_ms - day_start) // _QUARTER_HOUR_MS
            if index < 0 or index >= 96:
                raise ValueError("Spot trade lies outside the frozen daily bins")
            notional = trade.price * trade.quantity
            if not math.isfinite(notional) or notional <= 0.0:
                raise ValueError(
                    "Spot trade quote notional must be finite and positive"
                )
            signed_bins[index].append(
                (-1.0 if trade.is_buyer_maker else 1.0) * notional
            )
            total_bins[index].append(notional)

        if first is not None and rows is None:
            consume(first)
        if rows is None:
            rows = reader
        for row in rows:
            consume(row)
    except UnicodeDecodeError as error:
        raise ValueError("CSV payload is not UTF-8") from error
    finally:
        stream.close()

    return tuple(
        _reduce_imbalance(signed_bins[index], total_bins[index]) for index in range(96)
    )


def parse_usdm_15m_klines_csv(
    payload: bytes,
    *,
    expected_date: str,
) -> tuple[TargetBar, ...]:
    """Parse one official USD-M 15m contract-kline CSV without source fallback."""

    rows = _decode_csv(payload)
    if rows and tuple(rows[0]) == _KLINE_HEADER:
        rows = rows[1:]
    day_start, day_end = _day_bounds_ms(expected_date)
    result: list[TargetBar] = []
    previous_open_time: int | None = None
    for row in rows:
        if len(row) != 12:
            raise ValueError("USD-M kline row is malformed: expected twelve fields")
        open_time = _parse_int_token(row[0], field="open timestamp")
        close_time = _parse_int_token(row[6], field="close timestamp")
        if open_time < day_start or open_time >= day_end:
            raise ValueError("target open timestamp is outside expected_date")
        if open_time % _QUARTER_HOUR_MS != 0:
            raise ValueError("target open timestamp is off the native 15m grid")
        if close_time != open_time + _QUARTER_HOUR_MS - 1:
            raise ValueError("target close timestamp violates native 15m semantics")
        if previous_open_time is not None and open_time <= previous_open_time:
            raise ValueError("target open timestamps must be strictly increasing")
        previous_open_time = open_time

        open_price = _parse_float_token(row[1], field="open price")
        high = _parse_float_token(row[2], field="high price")
        low = _parse_float_token(row[3], field="low price")
        close = _parse_float_token(row[4], field="close price")
        volume = _parse_float_token(row[5], field="volume")
        quote_volume = _parse_float_token(row[7], field="quote volume")
        trade_count = _parse_int_token(row[8], field="trade count")
        taker_buy_volume = _parse_float_token(row[9], field="taker buy volume")
        taker_buy_quote_volume = _parse_float_token(
            row[10], field="taker buy quote volume"
        )
        _parse_float_token(row[11], field="ignore")
        if min(open_price, high, low, close) <= 0.0:
            raise ValueError("target OHLC prices must be finite and positive")
        if min(volume, quote_volume, taker_buy_volume, taker_buy_quote_volume) < 0.0:
            raise ValueError("target volume fields must be finite and non-negative")
        if trade_count < 0:
            raise ValueError("target trade count must be non-negative")
        result.append(TargetBar(open_time_ms=open_time, open_price=open_price))
    return tuple(result)


def build_four_hour_label(
    bars: Sequence[TargetBar],
    *,
    decision_time_ms: int,
) -> float | None:
    """Return log(open[t+17]/open[t+1]) using the maintained 15m clock."""

    decision = _require_int(decision_time_ms, field="decision_time_ms")
    if decision % _QUARTER_HOUR_MS != 0:
        raise ValueError("decision_time_ms must lie on the native 15m grid")
    by_time: dict[int, TargetBar] = {}
    previous: int | None = None
    for bar in bars:
        if not isinstance(bar, TargetBar):
            raise TypeError("bars must contain TargetBar values")
        if previous is not None and bar.open_time_ms <= previous:
            raise ValueError("target bars must be strictly increasing")
        previous = bar.open_time_ms
        if bar.open_time_ms in by_time:
            raise ValueError("target bars must be unique")
        by_time[bar.open_time_ms] = bar

    window: list[TargetBar] = []
    for index in range(17):
        timestamp = decision + index * _QUARTER_HOUR_MS
        candidate = by_time.get(timestamp)
        if candidate is None:
            return None
        if not (
            candidate.information_available and candidate.active and candidate.tradable
        ):
            return None
        window.append(candidate)
    execution = window[0].open_price
    endpoint = window[-1].open_price
    ratio = endpoint / execution
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None
    label = math.log(ratio)
    return label if math.isfinite(label) else None


def _expected_symbol_semantics(
    *,
    symbol: str,
    eligible_observations: int,
    numerator: float | None,
    denominator: float | None,
    beta: float | None,
    minimum_observations: int,
) -> tuple[tuple[str, ...], float | None]:
    failures: list[str] = []
    if eligible_observations < minimum_observations:
        failures.append(f"{symbol}:eligible_observations<{minimum_observations}")
    if numerator is None:
        failures.append(f"{symbol}:numerator_not_finite")
    if denominator is None or denominator <= 0.0:
        failures.append(f"{symbol}:denominator_not_finite_positive")

    expected_beta: float | None = None
    if not failures:
        assert numerator is not None and denominator is not None
        candidate = numerator / denominator
        if math.isfinite(candidate):
            expected_beta = candidate
        else:
            failures.append(f"{symbol}:beta_not_finite")

    if failures:
        if beta is not None:
            raise ValueError("beta must be null when symbol semantic failures exist")
    elif beta is None or beta != expected_beta:
        raise ValueError(
            "beta does not equal numerator / denominator semantic authority"
        )
    return tuple(failures), expected_beta


@dataclass(frozen=True, slots=True)
class SpotFlowSymbolCalibration:
    symbol: str
    eligible_observations: int
    numerator: float | None
    denominator: float | None
    beta: float | None
    positive_slope: bool
    failures: tuple[str, ...]
    minimum_observations: int = 360

    def __post_init__(self) -> None:
        _require_string(self.symbol, field="symbol")
        eligible = _require_int(
            self.eligible_observations, field="eligible_observations"
        )
        minimum = _require_int(
            self.minimum_observations, field="minimum_observations", minimum=1
        )
        numerator = _require_optional_finite(self.numerator, field="numerator")
        denominator = _require_optional_finite(self.denominator, field="denominator")
        beta = _require_optional_finite(self.beta, field="beta")
        _require_bool(self.positive_slope, field="positive_slope")
        if denominator is not None and denominator < 0.0:
            raise ValueError("denominator must be non-negative when present")
        if any(not isinstance(item, str) or not item for item in self.failures):
            raise ValueError("symbol failures must be non-empty strings")

        expected_failures, expected_beta = _expected_symbol_semantics(
            symbol=self.symbol,
            eligible_observations=eligible,
            numerator=numerator,
            denominator=denominator,
            beta=beta,
            minimum_observations=minimum,
        )
        if self.failures != expected_failures:
            raise ValueError("symbol failure list does not match semantic authority")
        if self.positive_slope is not (
            expected_beta is not None and expected_beta > 0.0
        ):
            raise ValueError("positive_slope does not match semantic beta sign")

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "eligible_observations": self.eligible_observations,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "beta": self.beta,
            "positive_slope": self.positive_slope,
            "failures": list(self.failures),
            "minimum_observations": self.minimum_observations,
        }


def _safe_fsum(values: Sequence[float]) -> float | None:
    try:
        result = math.fsum(values)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def calibrate_spot_flow_symbol(
    symbol: str,
    signals: Sequence[float],
    labels: Sequence[float],
    *,
    minimum_observations: int = 360,
) -> SpotFlowSymbolCalibration:
    """Fit the frozen no-intercept beta for one symbol in supplied order."""

    _require_string(symbol, field="symbol")
    minimum = _require_int(
        minimum_observations, field="minimum_observations", minimum=1
    )
    if len(signals) != len(labels):
        raise ValueError("signals and labels must have equal length")
    resolved_signals: list[float] = []
    resolved_labels: list[float] = []
    for index, (signal, label) in enumerate(zip(signals, labels, strict=True)):
        resolved_signals.append(
            _require_finite_float(signal, field=f"signals[{index}]")
        )
        resolved_labels.append(_require_finite_float(label, field=f"labels[{index}]"))
    eligible = len(resolved_signals)
    numerator = _safe_fsum(
        [
            signal * label
            for signal, label in zip(resolved_signals, resolved_labels, strict=True)
        ]
    )
    denominator = _safe_fsum([signal * signal for signal in resolved_signals])
    failures: list[str] = []
    if eligible < minimum:
        failures.append(f"{symbol}:eligible_observations<{minimum}")
    if numerator is None:
        failures.append(f"{symbol}:numerator_not_finite")
    if denominator is None or denominator <= 0.0:
        failures.append(f"{symbol}:denominator_not_finite_positive")
    beta: float | None = None
    if not failures:
        assert numerator is not None and denominator is not None
        candidate = numerator / denominator
        if math.isfinite(candidate):
            beta = candidate
        else:
            failures.append(f"{symbol}:beta_not_finite")
    return SpotFlowSymbolCalibration(
        symbol=symbol,
        eligible_observations=eligible,
        numerator=numerator,
        denominator=denominator,
        beta=beta,
        positive_slope=beta is not None and beta > 0.0,
        failures=tuple(failures),
        minimum_observations=minimum,
    )


@dataclass(frozen=True, slots=True)
class SpotFlowDiagnosticResult:
    protocol_digest: str
    implementation_head: str
    source_manifest_digest: str
    target_manifest_digest: str
    target_preflight_run_id: int
    target_preflight_artifact_id: int
    target_preflight_artifact_api_digest: str
    execution_run_id: int
    symbols: tuple[str, ...]
    symbol_results: tuple[SpotFlowSymbolCalibration, ...]
    positive_slope_count: int
    status: str
    failures: tuple[str, ...]
    target_preflight_status: str = _TARGET_PREFLIGHT_STATUS
    target_preflight_fresh_artifact_id: int = _TARGET_PREFLIGHT_FRESH_ARTIFACT_ID
    target_preflight_fresh_artifact_api_digest: str = (
        _TARGET_PREFLIGHT_FRESH_ARTIFACT_API_DIGEST
    )
    target_preflight_report_sha256: str = _TARGET_PREFLIGHT_REPORT_SHA256
    target_preflight_report_content_digest: str = (
        _TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST
    )
    prereg_head: str = _PREREG_HEAD
    prereg_full_verify_run_id: int = _PREREG_FULL_VERIFY_RUN_ID
    prereg_seal_run_id: int = _PREREG_SEAL_RUN_ID
    prereg_seal_artifact_id: int = _PREREG_SEAL_ARTIFACT_ID
    prereg_seal_artifact_api_digest: str = _PREREG_SEAL_ARTIFACT_API_DIGEST
    prereg_fresh_artifact_id: int = _PREREG_FRESH_ARTIFACT_ID
    prereg_fresh_artifact_api_digest: str = _PREREG_FRESH_ARTIFACT_API_DIGEST
    training_relation_executed: bool = True
    evaluation_pnl_inspected: bool = False
    evaluation_execution_authorized: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        protocol = canonical_spot_flow_continuation_protocol()
        if protocol.digest != _PREREG_PROTOCOL_DIGEST:
            raise ValueError(
                "runtime preregistration digest differs from sealed authority"
            )
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("diagnostic schema_version is not canonical")
        if self.protocol_digest != _PREREG_PROTOCOL_DIGEST:
            raise ValueError("protocol_digest is not canonical authority")
        _require_hex(self.protocol_digest, length=64, field="protocol_digest")
        _require_hex(self.implementation_head, length=40, field="implementation_head")
        _require_hex(
            self.source_manifest_digest, length=64, field="source_manifest_digest"
        )
        _require_hex(
            self.target_manifest_digest, length=64, field="target_manifest_digest"
        )
        _require_hex(
            self.target_preflight_artifact_api_digest,
            length=64,
            field="target_preflight_artifact_api_digest",
        )
        _require_int(
            self.target_preflight_run_id,
            field="target_preflight_run_id",
            minimum=1,
        )
        _require_int(
            self.target_preflight_artifact_id,
            field="target_preflight_artifact_id",
            minimum=1,
        )
        _require_int(self.execution_run_id, field="execution_run_id", minimum=1)
        if self.symbols != protocol.symbols:
            raise ValueError("diagnostic symbol roster is not canonical")
        if tuple(item.symbol for item in self.symbol_results) != self.symbols:
            raise ValueError("symbol result roster does not match canonical symbols")
        if any(
            item.minimum_observations
            != protocol.minimum_eligible_observations_per_symbol
            for item in self.symbol_results
        ):
            raise ValueError("symbol minimum coverage differs from preregistration")
        _require_int(self.positive_slope_count, field="positive_slope_count")
        observed_positive = sum(item.positive_slope for item in self.symbol_results)
        if self.positive_slope_count != observed_positive:
            raise ValueError("positive_slope_count does not match symbol results")
        expected_failures = tuple(
            failure for item in self.symbol_results for failure in item.failures
        )
        if self.failures != expected_failures:
            raise ValueError("diagnostic failures do not match symbol failures")
        if expected_failures:
            expected_status = protocol.invalid_coverage_status
        elif self.positive_slope_count >= protocol.required_positive_symbol_slopes:
            expected_status = protocol.valid_status
        else:
            expected_status = protocol.reject_status
        if self.status != expected_status:
            raise ValueError("diagnostic status does not match frozen decision rule")

        fixed_authority: dict[str, object] = {
            "target_manifest_digest": _TARGET_MANIFEST_DIGEST,
            "target_preflight_run_id": _TARGET_PREFLIGHT_RUN_ID,
            "target_preflight_artifact_id": _TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID,
            "target_preflight_artifact_api_digest": (
                _TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_API_DIGEST
            ),
            "target_preflight_status": _TARGET_PREFLIGHT_STATUS,
            "target_preflight_fresh_artifact_id": _TARGET_PREFLIGHT_FRESH_ARTIFACT_ID,
            "target_preflight_fresh_artifact_api_digest": (
                _TARGET_PREFLIGHT_FRESH_ARTIFACT_API_DIGEST
            ),
            "target_preflight_report_sha256": _TARGET_PREFLIGHT_REPORT_SHA256,
            "target_preflight_report_content_digest": (
                _TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST
            ),
            "prereg_head": _PREREG_HEAD,
            "prereg_full_verify_run_id": _PREREG_FULL_VERIFY_RUN_ID,
            "prereg_seal_run_id": _PREREG_SEAL_RUN_ID,
            "prereg_seal_artifact_id": _PREREG_SEAL_ARTIFACT_ID,
            "prereg_seal_artifact_api_digest": _PREREG_SEAL_ARTIFACT_API_DIGEST,
            "prereg_fresh_artifact_id": _PREREG_FRESH_ARTIFACT_ID,
            "prereg_fresh_artifact_api_digest": _PREREG_FRESH_ARTIFACT_API_DIGEST,
        }
        for field_name, expected in fixed_authority.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} is not canonical authority")

        for field_name in (
            "training_relation_executed",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
        ):
            _require_bool(getattr(self, field_name), field=field_name)
        if self.training_relation_executed is not True:
            raise ValueError("training_relation_executed must be true for a result")
        if any(
            (
                self.evaluation_pnl_inspected,
                self.evaluation_execution_authorized,
                self.final_test_authorized,
                self.shared_cash_profitability_established,
                self.production_eligible,
                self.live_trading_authorized,
            )
        ):
            raise ValueError("training-only diagnostic crossed a forbidden boundary")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "prereg_head": self.prereg_head,
            "prereg_full_verify_run_id": self.prereg_full_verify_run_id,
            "prereg_seal_run_id": self.prereg_seal_run_id,
            "prereg_seal_artifact_id": self.prereg_seal_artifact_id,
            "prereg_seal_artifact_api_digest": self.prereg_seal_artifact_api_digest,
            "prereg_fresh_artifact_id": self.prereg_fresh_artifact_id,
            "prereg_fresh_artifact_api_digest": self.prereg_fresh_artifact_api_digest,
            "implementation_head": self.implementation_head,
            "source_manifest_digest": self.source_manifest_digest,
            "target_manifest_digest": self.target_manifest_digest,
            "target_preflight_status": self.target_preflight_status,
            "target_preflight_run_id": self.target_preflight_run_id,
            "target_preflight_publisher_artifact_id": (
                self.target_preflight_artifact_id
            ),
            "target_preflight_publisher_artifact_api_digest": (
                self.target_preflight_artifact_api_digest
            ),
            "target_preflight_fresh_artifact_id": (
                self.target_preflight_fresh_artifact_id
            ),
            "target_preflight_fresh_artifact_api_digest": (
                self.target_preflight_fresh_artifact_api_digest
            ),
            "target_preflight_report_sha256": self.target_preflight_report_sha256,
            "target_preflight_report_content_digest": (
                self.target_preflight_report_content_digest
            ),
            "execution_run_id": self.execution_run_id,
            "symbols": list(self.symbols),
            "symbol_results": [item.to_payload() for item in self.symbol_results],
            "positive_slope_count": self.positive_slope_count,
            "status": self.status,
            "failures": list(self.failures),
            "training_relation_executed": self.training_relation_executed,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
            "evaluation_execution_authorized": self.evaluation_execution_authorized,
            "final_test_authorized": self.final_test_authorized,
            "shared_cash_profitability_established": (
                self.shared_cash_profitability_established
            ),
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
        }

    @property
    def target_preflight_publisher_artifact_id(self) -> int:
        return self.target_preflight_artifact_id

    @property
    def target_preflight_publisher_artifact_api_digest(self) -> str:
        return self.target_preflight_artifact_api_digest

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())

    def to_artifact_payload(self) -> dict[str, object]:
        return {**self.to_payload(), "content_digest": self.digest}


def build_spot_flow_diagnostic_result(
    observations: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    protocol: SpotFlowContinuationProtocol,
    *,
    implementation_head: str,
    source_manifest_digest: str,
    target_manifest_digest: str,
    target_preflight_run_id: int,
    target_preflight_artifact_id: int,
    target_preflight_artifact_api_digest: str,
    execution_run_id: int,
) -> SpotFlowDiagnosticResult:
    """Calibrate all five frozen symbols and apply only the preregistered gate."""

    canonical = canonical_spot_flow_continuation_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError("protocol differs from sealed Spot-flow preregistration")
    if tuple(observations) != protocol.symbols:
        raise ValueError("observation mapping must use the exact ordered symbol roster")
    symbol_results = tuple(
        calibrate_spot_flow_symbol(
            symbol,
            observations[symbol][0],
            observations[symbol][1],
            minimum_observations=protocol.minimum_eligible_observations_per_symbol,
        )
        for symbol in protocol.symbols
    )
    failures = tuple(failure for item in symbol_results for failure in item.failures)
    positive_count = sum(item.positive_slope for item in symbol_results)
    if failures:
        status = protocol.invalid_coverage_status
    elif positive_count >= protocol.required_positive_symbol_slopes:
        status = protocol.valid_status
    else:
        status = protocol.reject_status
    return SpotFlowDiagnosticResult(
        protocol_digest=protocol.digest,
        implementation_head=implementation_head,
        source_manifest_digest=source_manifest_digest,
        target_manifest_digest=target_manifest_digest,
        target_preflight_run_id=target_preflight_run_id,
        target_preflight_artifact_id=target_preflight_artifact_id,
        target_preflight_artifact_api_digest=target_preflight_artifact_api_digest,
        execution_run_id=execution_run_id,
        symbols=protocol.symbols,
        symbol_results=symbol_results,
        positive_slope_count=int(positive_count),
        status=status,
        failures=failures,
    )


def canonical_spot_flow_result_bytes(result: SpotFlowDiagnosticResult) -> bytes:
    if not isinstance(result, SpotFlowDiagnosticResult):
        raise TypeError("result must be SpotFlowDiagnosticResult")
    return canonical_json_bytes(result.to_artifact_payload())


def _symbol_result_from_payload(raw: object) -> SpotFlowSymbolCalibration:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("symbol result is malformed")
    expected = {
        "symbol",
        "eligible_observations",
        "numerator",
        "denominator",
        "beta",
        "positive_slope",
        "failures",
        "minimum_observations",
    }
    if set(raw) != expected:
        raise ValueError("symbol result has unknown or missing fields")
    failures_raw = raw["failures"]
    if not isinstance(failures_raw, list) or any(
        not isinstance(item, str) or not item for item in failures_raw
    ):
        raise ValueError("symbol failures are malformed")
    return SpotFlowSymbolCalibration(
        symbol=_require_string(raw["symbol"], field="symbol"),
        eligible_observations=_require_int(
            raw["eligible_observations"], field="eligible_observations"
        ),
        numerator=_require_optional_finite(raw["numerator"], field="numerator"),
        denominator=_require_optional_finite(raw["denominator"], field="denominator"),
        beta=_require_optional_finite(raw["beta"], field="beta"),
        positive_slope=_require_bool(raw["positive_slope"], field="positive_slope"),
        failures=tuple(failures_raw),
        minimum_observations=_require_int(
            raw["minimum_observations"],
            field="minimum_observations",
            minimum=1,
        ),
    )


def load_spot_flow_result_bytes(payload: bytes) -> SpotFlowDiagnosticResult:
    """Load only canonical, semantically valid Issue 584 result bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("result payload must be bytes")
    try:
        decoded: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Spot-flow result is malformed") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("Spot-flow result must be a JSON object")
    try:
        canonical = canonical_json_bytes(decoded)
    except (TypeError, ValueError) as error:
        raise ValueError("Spot-flow result is malformed") from error
    if payload != canonical:
        raise ValueError("Spot-flow result must use canonical JSON bytes")

    expected = {
        "schema_version",
        "protocol_digest",
        "prereg_head",
        "prereg_full_verify_run_id",
        "prereg_seal_run_id",
        "prereg_seal_artifact_id",
        "prereg_seal_artifact_api_digest",
        "prereg_fresh_artifact_id",
        "prereg_fresh_artifact_api_digest",
        "implementation_head",
        "source_manifest_digest",
        "target_manifest_digest",
        "target_preflight_status",
        "target_preflight_run_id",
        "target_preflight_publisher_artifact_id",
        "target_preflight_publisher_artifact_api_digest",
        "target_preflight_fresh_artifact_id",
        "target_preflight_fresh_artifact_api_digest",
        "target_preflight_report_sha256",
        "target_preflight_report_content_digest",
        "execution_run_id",
        "symbols",
        "symbol_results",
        "positive_slope_count",
        "status",
        "failures",
        "training_relation_executed",
        "evaluation_pnl_inspected",
        "evaluation_execution_authorized",
        "final_test_authorized",
        "shared_cash_profitability_established",
        "production_eligible",
        "live_trading_authorized",
        "content_digest",
    }
    if set(decoded) != expected:
        raise ValueError("Spot-flow result fields are not canonical")
    symbols_raw = decoded["symbols"]
    symbol_results_raw = decoded["symbol_results"]
    failures_raw = decoded["failures"]
    if not isinstance(symbols_raw, list) or any(
        not isinstance(item, str) or not item for item in symbols_raw
    ):
        raise ValueError("symbols are malformed")
    if not isinstance(symbol_results_raw, list):
        raise ValueError("symbol_results are malformed")
    if not isinstance(failures_raw, list) or any(
        not isinstance(item, str) or not item for item in failures_raw
    ):
        raise ValueError("failures are malformed")

    result = SpotFlowDiagnosticResult(
        schema_version=_require_string(
            decoded["schema_version"], field="schema_version"
        ),
        protocol_digest=_require_hex(
            decoded["protocol_digest"], length=64, field="protocol_digest"
        ),
        prereg_head=_require_hex(
            decoded["prereg_head"], length=40, field="prereg_head"
        ),
        prereg_full_verify_run_id=_require_int(
            decoded["prereg_full_verify_run_id"],
            field="prereg_full_verify_run_id",
            minimum=1,
        ),
        prereg_seal_run_id=_require_int(
            decoded["prereg_seal_run_id"], field="prereg_seal_run_id", minimum=1
        ),
        prereg_seal_artifact_id=_require_int(
            decoded["prereg_seal_artifact_id"],
            field="prereg_seal_artifact_id",
            minimum=1,
        ),
        prereg_seal_artifact_api_digest=_require_hex(
            decoded["prereg_seal_artifact_api_digest"],
            length=64,
            field="prereg_seal_artifact_api_digest",
        ),
        prereg_fresh_artifact_id=_require_int(
            decoded["prereg_fresh_artifact_id"],
            field="prereg_fresh_artifact_id",
            minimum=1,
        ),
        prereg_fresh_artifact_api_digest=_require_hex(
            decoded["prereg_fresh_artifact_api_digest"],
            length=64,
            field="prereg_fresh_artifact_api_digest",
        ),
        implementation_head=_require_hex(
            decoded["implementation_head"], length=40, field="implementation_head"
        ),
        source_manifest_digest=_require_hex(
            decoded["source_manifest_digest"],
            length=64,
            field="source_manifest_digest",
        ),
        target_manifest_digest=_require_hex(
            decoded["target_manifest_digest"],
            length=64,
            field="target_manifest_digest",
        ),
        target_preflight_status=_require_string(
            decoded["target_preflight_status"], field="target_preflight_status"
        ),
        target_preflight_run_id=_require_int(
            decoded["target_preflight_run_id"],
            field="target_preflight_run_id",
            minimum=1,
        ),
        target_preflight_artifact_id=_require_int(
            decoded["target_preflight_publisher_artifact_id"],
            field="target_preflight_publisher_artifact_id",
            minimum=1,
        ),
        target_preflight_artifact_api_digest=_require_hex(
            decoded["target_preflight_publisher_artifact_api_digest"],
            length=64,
            field="target_preflight_publisher_artifact_api_digest",
        ),
        target_preflight_fresh_artifact_id=_require_int(
            decoded["target_preflight_fresh_artifact_id"],
            field="target_preflight_fresh_artifact_id",
            minimum=1,
        ),
        target_preflight_fresh_artifact_api_digest=_require_hex(
            decoded["target_preflight_fresh_artifact_api_digest"],
            length=64,
            field="target_preflight_fresh_artifact_api_digest",
        ),
        target_preflight_report_sha256=_require_hex(
            decoded["target_preflight_report_sha256"],
            length=64,
            field="target_preflight_report_sha256",
        ),
        target_preflight_report_content_digest=_require_hex(
            decoded["target_preflight_report_content_digest"],
            length=64,
            field="target_preflight_report_content_digest",
        ),
        execution_run_id=_require_int(
            decoded["execution_run_id"], field="execution_run_id", minimum=1
        ),
        symbols=tuple(symbols_raw),
        symbol_results=tuple(
            _symbol_result_from_payload(item) for item in symbol_results_raw
        ),
        positive_slope_count=_require_int(
            decoded["positive_slope_count"], field="positive_slope_count"
        ),
        status=_require_string(decoded["status"], field="status"),
        failures=tuple(failures_raw),
        training_relation_executed=_require_bool(
            decoded["training_relation_executed"], field="training_relation_executed"
        ),
        evaluation_pnl_inspected=_require_bool(
            decoded["evaluation_pnl_inspected"], field="evaluation_pnl_inspected"
        ),
        evaluation_execution_authorized=_require_bool(
            decoded["evaluation_execution_authorized"],
            field="evaluation_execution_authorized",
        ),
        final_test_authorized=_require_bool(
            decoded["final_test_authorized"], field="final_test_authorized"
        ),
        shared_cash_profitability_established=_require_bool(
            decoded["shared_cash_profitability_established"],
            field="shared_cash_profitability_established",
        ),
        production_eligible=_require_bool(
            decoded["production_eligible"], field="production_eligible"
        ),
        live_trading_authorized=_require_bool(
            decoded["live_trading_authorized"], field="live_trading_authorized"
        ),
    )
    content = _require_hex(decoded["content_digest"], length=64, field="content_digest")
    if content != result.digest:
        raise ValueError(
            "Spot-flow result content digest does not match canonical payload"
        )
    return result
