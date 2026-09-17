"""Fresh, write-once public market evidence for prospective carry research."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import _aware_utc

_SPOT = "https://data-api.binance.vision/api/v3"
_FUTURES = "https://fapi.binance.com/fapi/v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _number(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("market numeric field has an invalid type")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("market numeric field must be finite and correctly signed")
    return number


def _epoch(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("source timestamp must be positive integer milliseconds")
    return value


def _fresh(value: object, received_ms: int) -> int:
    timestamp = _epoch(value)
    if not -1000 <= received_ms - timestamp <= 5000:
        raise ValueError("source timestamp is stale or in the future")
    return timestamp


def _book(value: object, *, futures: bool, received_ms: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("depth response must be an object")
    _epoch(value.get("lastUpdateId"))
    if futures:
        event = _fresh(value.get("E"), received_ms)
        transaction = _fresh(value.get("T"), received_ms)
        if event < transaction:
            raise ValueError("depth event precedes its transaction")
    prices = {}
    for side in ("bids", "asks"):
        levels = value.get(side)
        if not isinstance(levels, list) or not levels:
            raise ValueError("depth sides must contain price and quantity levels")
        numbers = []
        for level in levels:
            if not isinstance(level, list) or len(level) != 2:
                raise ValueError("depth level must contain price and quantity")
            numbers.append(_number(level[0], positive=True))
            _number(level[1], positive=True)
        pairs = zip(numbers, numbers[1:])
        if any(a <= b if side == "bids" else a >= b for a, b in pairs):
            raise ValueError("depth levels must be strictly ordered")
        prices[side] = numbers[0]
    if prices["bids"] >= prices["asks"]:
        raise ValueError("depth book is crossed or locked")
    return value


def _mark(value: object, *, symbol: str, received_ms: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("symbol") != symbol:
        raise ValueError("mark quote symbol mismatch")
    timestamp = _fresh(value.get("time"), received_ms)
    _number(value.get("markPrice"), positive=True)
    _number(value.get("indexPrice"), positive=True)
    _number(value.get("lastFundingRate"))
    if _epoch(value.get("nextFundingTime")) <= timestamp:
        raise ValueError("next funding time must follow the mark quote")
    return value


def _settlements(
    value: object, *, symbol: str, start_ms: int, received_ms: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("settled funding history must be a nonempty list")
    previous = 0
    rows = []
    for row in value:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise ValueError("settled funding symbol mismatch")
        timestamp = _epoch(row.get("fundingTime"))
        if not start_ms <= timestamp <= received_ms or timestamp <= previous:
            raise ValueError(
                "settled funding clock is duplicated, unordered or invalid"
            )
        _number(row.get("fundingRate"))
        _number(row.get("markPrice"), positive=True)
        previous = timestamp
        rows.append(row)
    if received_ms - previous > 9 * 3_600_000:
        raise ValueError("settled funding history is stale")
    return rows


def _write(path: Path, value: object) -> None:
    encoded = canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)


def capture_forward_snapshot(
    root: str | Path,
    *,
    transport: BinancePublicTransport | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Capture ten public GET responses, or persist failure without eligibility.

    This records market evidence only. It neither places orders nor simulates
    positions. Spot REST depth has receipt timing but no exchange event clock.
    """
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    client = transport or BinancePublicTransport(timeout_seconds=4, max_attempts=1)
    responses: list[dict[str, object]] = []
    try:
        started = _aware_utc(clock(), field="capture_start")
        capture_tick = monotonic()
        if not math.isfinite(capture_tick):
            raise ValueError("capture clock must be finite")
        last_receipt = started
        quote_times: list[int] = []
        start_ms = int(started.timestamp() * 1000)

        def request(label: str, url: str) -> tuple[object, int]:
            nonlocal last_receipt
            requested = _aware_utc(clock(), field="request_start")
            before = monotonic()
            raw = client._request_bytes(url)
            after = monotonic()
            elapsed = after - before
            span = after - capture_tick
            received = _aware_utc(clock(), field="response_receipt")
            with (destination / f"{label}.raw").open("xb") as stream:
                stream.write(raw)
            record = {
                "label": label,
                "url": url,
                "requested_at": requested.isoformat(),
                "received_at": received.isoformat(),
                "elapsed_seconds": elapsed,
                "capture_elapsed_seconds": span,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            _write(destination / f"{label}.json", record)
            responses.append(record)
            if received < requested or requested < last_receipt:
                raise ValueError("local clock moved backwards")
            if not math.isfinite(elapsed) or not 0 <= elapsed <= 5:
                raise ValueError("request duration exceeded freshness limit")
            if not 0 <= span <= 10:
                raise ValueError("monotonic capture span exceeds ten seconds")
            if abs((received - requested).total_seconds() - elapsed) > 1:
                raise ValueError("local clock diverges from monotonic duration")
            if abs((received - started).total_seconds() - span) > 1:
                raise ValueError("capture clock diverges from monotonic span")
            if (received - started).total_seconds() > 10:
                raise ValueError("cross-request capture span exceeds ten seconds")
            last_receipt = received
            return json.loads(raw), int(received.timestamp() * 1000)

        for label, base in (("spot_time", _SPOT), ("futures_time", _FUTURES)):
            payload, received_ms = request(label, f"{base}/time")
            if not isinstance(payload, dict):
                raise ValueError("venue clock response must be an object")
            _fresh(payload.get("serverTime"), received_ms)
        market = {}
        funding_start = start_ms - 24 * 3_600_000
        for symbol in _SYMBOLS:
            spot, received_ms = request(
                f"{symbol}_spot_depth", f"{_SPOT}/depth?symbol={symbol}&limit=20"
            )
            spot = _book(spot, futures=False, received_ms=received_ms)
            quote_times.append(received_ms)
            perp, received_ms = request(
                f"{symbol}_perp_depth", f"{_FUTURES}/depth?symbol={symbol}&limit=20"
            )
            perp = _book(perp, futures=True, received_ms=received_ms)
            quote_times.extend((_epoch(perp["E"]), _epoch(perp["T"])))
            mark, received_ms = request(
                f"{symbol}_mark", f"{_FUTURES}/premiumIndex?symbol={symbol}"
            )
            mark = _mark(mark, symbol=symbol, received_ms=received_ms)
            quote_times.append(_epoch(mark["time"]))
            history, received_ms = request(
                f"{symbol}_funding",
                f"{_FUTURES}/fundingRate?symbol={symbol}&startTime={funding_start}&limit=100",
            )
            history = _settlements(
                history, symbol=symbol, start_ms=funding_start, received_ms=received_ms
            )
            market[symbol] = {
                "spot_depth": spot,
                "perpetual_depth": perp,
                "spot_depth_time_basis": "local_receipt_only",
                "mark_quote": mark,
                "settled_funding": history,
            }
        for timestamp in quote_times:
            _fresh(timestamp, received_ms)
        snapshot = {
            "schema": "binance_forward_market_snapshot_v1",
            "eligible": True,
            "production_eligible": False,
            "started_at": started.isoformat(),
            "completed_at": responses[-1]["received_at"],
            "capture_elapsed_seconds": responses[-1]["capture_elapsed_seconds"],
            "responses": responses,
            "market": market,
        }
        _write(destination / "snapshot.json", snapshot)
        return snapshot
    except Exception as error:
        _write(
            destination / "failure.json",
            {
                "schema": "binance_forward_capture_failure_v1",
                "eligible": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_responses": responses,
            },
        )
        raise
