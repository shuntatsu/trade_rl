"""Write-once current public exchange rules for prospective paper decisions."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.integrations.binance.forward import (
    _FUTURES,
    _SPOT,
    _SYMBOLS,
    _number,
    _write,
)
from trade_rl.integrations.binance.forward_evidence import _manifest, _responses, _time
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import _aware_utc

_ROSTER = [
    ("spot_rules", f"{_SPOT}/exchangeInfo?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22%5D"),
    ("perpetual_rules", f"{_FUTURES}/exchangeInfo"),
]
_SCHEMA = "binance_forward_rules_v1"


def _rule_number(value: object, *, positive: bool = False) -> float:
    if not isinstance(value, str):
        raise ValueError("rule decimal precision requires original string fields")
    number = _number(value, positive=positive)
    if Fraction(str(value)) != Fraction(str(number)):
        raise ValueError("rule precision exceeds the supported decimal representation")
    return number


def _nonnegative(value: object) -> float:
    number = _rule_number(value)
    if number < 0:
        raise ValueError("rule must be non-negative")
    return number


def _flag(item: dict[str, Any], key: str) -> bool:
    value = item.get(key)
    if not isinstance(value, bool):
        raise ValueError("market notional applicability must be boolean")
    return value


def _notional(
    filters: dict[str, dict[str, Any]], venue: str
) -> tuple[float, float | None, int | None]:
    lower, upper, minutes = [], [], []
    for key in ("MIN_NOTIONAL", "NOTIONAL"):
        if key not in filters:
            continue
        item = filters[key]
        if venue == "perpetual":
            if key != "MIN_NOTIONAL":
                raise ValueError("unsupported perpetual notional filter")
            lower.append(_nonnegative(item.get("notional")))
            continue
        average = item.get("avgPriceMins")
        if type(average) is not int or average < 0:
            raise ValueError(
                "notional averaging minutes must be a non-negative integer"
            )
        minimum = _nonnegative(item.get("minNotional"))
        applies = _flag(
            item, "applyToMarket" if key == "MIN_NOTIONAL" else "applyMinToMarket"
        )
        if applies:
            lower.append(minimum)
            minutes.append(average)
        if key == "NOTIONAL":
            maximum = _rule_number(item.get("maxNotional"), positive=True)
            if maximum < minimum:
                raise ValueError("notional range is inverted")
            if _flag(item, "applyMaxToMarket"):
                upper.append(maximum)
                minutes.append(average)
    if not (set(filters) & {"MIN_NOTIONAL", "NOTIONAL"}):
        raise ValueError("missing notional rule")
    if len(set(minutes)) > 1:
        raise ValueError("unsupported mixed notional averaging windows")
    minimum, upper_bound = max(lower, default=0.0), min(upper, default=None)
    if upper_bound is not None and upper_bound < minimum:
        raise ValueError("notional filter intersection is empty")
    return minimum, upper_bound, minutes[0] if minutes else None


def _symbol_rule(item: dict[str, Any], symbol: str, venue: str) -> dict[str, Any]:
    if (
        item.get("status") != "TRADING"
        or item.get("baseAsset") != symbol[:-4]
        or item.get("quoteAsset") != "USDT"
    ):
        raise ValueError("symbol is not a supported trading USDT pair")
    orders = item.get("orderTypes")
    if not isinstance(orders, list) or "MARKET" not in orders:
        raise ValueError("symbol lacks market-order support")
    if venue == "spot" and item.get("isSpotTradingAllowed") is not True:
        raise ValueError("spot trading is not enabled")
    if venue == "perpetual" and (
        item.get("contractType") != "PERPETUAL" or item.get("marginAsset") != "USDT"
    ):
        raise ValueError("only USDT-margined perpetual contracts are supported")
    raw_filters = item.get("filters")
    if not isinstance(raw_filters, list):
        raise ValueError("missing exchange filters")
    filters = {}
    for rule in raw_filters:
        if not isinstance(rule, dict) or not isinstance(rule.get("filterType"), str):
            raise ValueError("malformed exchange filter")
        key = rule["filterType"]
        if key in filters:
            raise ValueError("duplicate exchange filter")
        filters[key] = rule
    if not {"PRICE_FILTER", "LOT_SIZE", "MARKET_LOT_SIZE"} <= set(filters):
        raise ValueError("missing price or quantity filters")
    risk = filters.get("POSITION_RISK_CONTROL")
    if risk is not None and risk.get("positionControlSide") != "NONE":
        raise ValueError("unsupported active position control")
    price_filter = filters["PRICE_FILTER"]
    tick = _rule_number(price_filter.get("tickSize"), positive=True)
    low_price = _nonnegative(price_filter.get("minPrice"))
    high_price = _rule_number(price_filter.get("maxPrice"), positive=True)
    if high_price < low_price:
        raise ValueError("price range is inverted")
    steps, lows, highs = [], [], []
    for key in ("LOT_SIZE", "MARKET_LOT_SIZE"):
        rule = filters[key]
        step = _nonnegative(rule.get("stepSize"))
        if key == "LOT_SIZE" and not step:
            raise ValueError("base lot size cannot be disabled")
        if step:
            steps.append(Fraction(str(step)))
        lows.append(_nonnegative(rule.get("minQty")))
        highs.append(_rule_number(rule.get("maxQty"), positive=True))
    denominator = math.lcm(*(step.denominator for step in steps))
    quantum = Fraction(
        math.lcm(*(int(step * denominator) for step in steps)), denominator
    )
    low, high = max(lows), min(highs)
    first_lots = max(1, math.ceil(Fraction(str(low)) / quantum))
    if low > high or first_lots * quantum > Fraction(str(high)):
        raise ValueError("quantity filter intersection has no admissible lot")
    if Fraction(str(float(quantum))) != quantum:
        raise ValueError(
            "joint lot precision exceeds the supported decimal representation"
        )
    minimum, maximum, average = _notional(filters, venue)
    return dict(
        lot_size=float(quantum),
        minimum_quantity=low,
        maximum_quantity=high,
        minimum_notional=minimum,
        maximum_notional=maximum,
        tick_size=tick,
        minimum_price=low_price,
        maximum_price=high_price,
        notional_average_minutes=average,
        quantity_steps=[float(step) for step in steps],
    )


def _derive(responses: dict[str, tuple[Any, datetime]]) -> dict[str, Any]:
    result = {}
    for venue in ("spot", "perpetual"):
        payload, _ = responses[f"{venue}_rules"]
        if not isinstance(payload, dict) or not isinstance(
            payload.get("symbols"), list
        ):
            raise ValueError("exchange information must contain symbols")
        selected = {}
        for item in payload["symbols"]:
            if not isinstance(item, dict):
                raise ValueError("malformed symbol metadata")
            name = item.get("symbol")
            if name in _SYMBOLS:
                if name in selected:
                    raise ValueError("duplicate requested symbol")
                selected[name] = _symbol_rule(item, name, venue)
        if set(selected) != set(_SYMBOLS):
            raise ValueError("missing requested symbol")
        result[venue] = selected
    return result


def capture_forward_rules(
    root: str | Path,
    *,
    transport: BinancePublicTransport | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Capture current unauthenticated metadata; never grants order permission."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    client = transport or BinancePublicTransport(timeout_seconds=4, max_attempts=1)
    records = []
    try:
        started = _aware_utc(clock(), field="rules capture start")
        anchor = monotonic()
        for label, url in _ROSTER:
            requested, before = _aware_utc(clock(), field="rule request"), monotonic()
            raw = client._request_bytes(url)
            after, received = monotonic(), _aware_utc(clock(), field="rule receipt")
            with (destination / f"{label}.raw").open("xb") as stream:
                stream.write(raw)
            record = dict(
                label=label,
                url=url,
                requested_at=requested.isoformat(),
                received_at=received.isoformat(),
                elapsed_seconds=after - before,
                capture_elapsed_seconds=after - anchor,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
            )
            _write(destination / f"{label}.json", record)
            records.append(record)
        manifest = dict(
            schema=_SCHEMA,
            eligible=True,
            production_eligible=False,
            started_at=started.isoformat(),
            completed_at=records[-1]["received_at"],
            capture_elapsed_seconds=records[-1]["capture_elapsed_seconds"],
            responses=records,
        )
        manifest["rules"] = _derive(_responses(destination, manifest, _ROSTER))
        _write(destination / "rules.json", manifest)
        return manifest
    except Exception as error:
        _write(
            destination / "failure.json",
            dict(
                schema="binance_forward_rules_failure_v1",
                eligible=False,
                error_type=type(error).__name__,
                error=str(error),
                completed_responses=records,
            ),
        )
        raise


def read_forward_rules(
    root: str | Path,
    *,
    expected_sha256: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Revalidate raw rules; as_of additionally requires age within one hour."""
    destination = Path(root)
    manifest = _manifest(destination, "rules.json", _SCHEMA, expected_sha256)
    derived = _derive(_responses(destination, manifest, _ROSTER))
    if canonical_json_bytes(derived) != canonical_json_bytes(manifest.get("rules")):
        raise ValueError("rule summary differs from raw metadata")
    if as_of is not None:
        age = (
            _aware_utc(as_of, field="as_of") - _time(manifest.get("completed_at"))
        ).total_seconds()
        if not 0 <= age <= 3600:
            raise ValueError("rule capture is future or expired at consumption")
    return manifest
