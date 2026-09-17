"""Read-only raw-evidence verification for prospective public snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.integrations.binance.forward import (
    _FUTURES,
    _SPOT,
    _SYMBOLS,
    _book,
    _epoch,
    _fresh,
    _mark,
    _number,
    _settlements,
)
from trade_rl.integrations.binance.types import _aware_utc


def _duplicate_free(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field in forward evidence")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _decode(raw: bytes) -> Any:
    return json.loads(
        raw, object_pairs_hook=_duplicate_free, parse_constant=_invalid_constant
    )


def _bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("forward evidence file is missing or not regular")
    return path.read_bytes()


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("forward evidence timestamp must be text")
    return _aware_utc(datetime.fromisoformat(value), field="evidence timestamp")


def _manifest(root: Path, name: str, schema: str, digest: str | None) -> dict[str, Any]:
    if (root / "failure.json").exists():
        raise ValueError("failed capture cannot be eligible evidence")
    raw = _bytes(root / name)
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("forward manifest digest mismatch")
    value = _decode(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("eligible") is not True
        or value.get("production_eligible") is not False
    ):
        raise ValueError("invalid forward evidence manifest")
    return value


def _responses(
    root: Path,
    manifest: dict[str, Any],
    roster: list[tuple[str, str]],
) -> dict[str, tuple[Any, datetime]]:
    records = manifest.get("responses")
    if not isinstance(records, list) or len(records) != len(roster):
        raise ValueError("forward response roster mismatch")
    started = _time(manifest.get("started_at"))
    previous = started
    previous_span = 0.0
    result = {}
    for record, (label, url) in zip(records, roster, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("label") != label
            or record.get("url") != url
        ):
            raise ValueError("forward response label or official URL mismatch")
        sidecar = _decode(_bytes(root / f"{label}.json"))
        if canonical_json_bytes(sidecar) != canonical_json_bytes(record):
            raise ValueError("response sidecar differs from manifest")
        raw = _bytes(root / f"{label}.raw")
        size = record.get("size_bytes")
        if (
            type(size) is not int
            or size != len(raw)
            or record.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("raw response size or digest mismatch")
        requested, received = (
            _time(record.get("requested_at")),
            _time(record.get("received_at")),
        )
        elapsed = _number(record.get("elapsed_seconds"))
        span = _number(record.get("capture_elapsed_seconds"))
        if not previous <= requested <= received:
            raise ValueError("response clock moved backwards")
        if (
            not 0 <= elapsed <= 5
            or not 0 <= span <= 10
            or span + 1e-9 < previous_span + elapsed
        ):
            raise ValueError("response monotonic duration or capture span is invalid")
        if (
            abs((received - requested).total_seconds() - elapsed) > 1
            or abs((received - started).total_seconds() - span) > 1
            or (received - started).total_seconds() > 10
        ):
            raise ValueError("response wall and monotonic clocks disagree")
        result[label] = (_decode(raw), received)
        previous, previous_span = received, span
    if (
        _time(manifest.get("completed_at")) != previous
        or manifest.get("capture_elapsed_seconds") != previous_span
    ):
        raise ValueError("capture completion differs from its final response")
    return result


def read_forward_snapshot(
    root: str | Path,
    *,
    expected_sha256: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Reconstruct captured evidence; optionally require freshness at consumption.

    Omitting as_of permits historical audit only. An expected digest binds the
    manifest to a caller's journal, rather than merely verifying internal hashes.
    """
    destination = Path(root)
    snapshot = _manifest(
        destination,
        "snapshot.json",
        "binance_forward_market_snapshot_v1",
        expected_sha256,
    )
    started = _time(snapshot.get("started_at"))
    funding_start = int(started.timestamp() * 1000) - 24 * 3_600_000
    roster = [("spot_time", f"{_SPOT}/time"), ("futures_time", f"{_FUTURES}/time")]
    for symbol in _SYMBOLS:
        roster.extend(
            [
                (f"{symbol}_spot_depth", f"{_SPOT}/depth?symbol={symbol}&limit=20"),
                (f"{symbol}_perp_depth", f"{_FUTURES}/depth?symbol={symbol}&limit=20"),
                (f"{symbol}_mark", f"{_FUTURES}/premiumIndex?symbol={symbol}"),
                (
                    f"{symbol}_funding",
                    f"{_FUTURES}/fundingRate?symbol={symbol}&startTime={funding_start}&limit=100",
                ),
            ]
        )
    responses = _responses(destination, snapshot, roster)
    quote_times = []
    for label in ("spot_time", "futures_time"):
        payload, received = responses[label]
        if not isinstance(payload, dict):
            raise ValueError("venue clock response must be an object")
        _fresh(payload.get("serverTime"), int(received.timestamp() * 1000))
    market = {}
    for symbol in _SYMBOLS:
        spot, received = responses[f"{symbol}_spot_depth"]
        spot = _book(spot, futures=False, received_ms=int(received.timestamp() * 1000))
        quote_times.append(int(received.timestamp() * 1000))
        perp, received = responses[f"{symbol}_perp_depth"]
        perp = _book(perp, futures=True, received_ms=int(received.timestamp() * 1000))
        quote_times.extend((_epoch(perp["E"]), _epoch(perp["T"])))
        mark, received = responses[f"{symbol}_mark"]
        mark = _mark(mark, symbol=symbol, received_ms=int(received.timestamp() * 1000))
        quote_times.append(_epoch(mark["time"]))
        history, received = responses[f"{symbol}_funding"]
        history = _settlements(
            history,
            symbol=symbol,
            start_ms=funding_start,
            received_ms=int(received.timestamp() * 1000),
        )
        market[symbol] = dict(
            spot_depth=spot,
            perpetual_depth=perp,
            spot_depth_time_basis="local_receipt_only",
            mark_quote=mark,
            settled_funding=history,
        )
    completed = _time(snapshot.get("completed_at"))
    for timestamp in quote_times:
        _fresh(timestamp, int(completed.timestamp() * 1000))
    if canonical_json_bytes(market) != canonical_json_bytes(snapshot.get("market")):
        raise ValueError("market summary differs from decoded raw responses")
    if as_of is not None:
        consumed = _aware_utc(as_of, field="as_of")
        if consumed < completed:
            raise ValueError("snapshot has not been received at consumption time")
        for timestamp in quote_times:
            _fresh(timestamp, int(consumed.timestamp() * 1000))
    return snapshot
