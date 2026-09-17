"""Source-derived USD-M one-way MARKET profiles; no account/order API access."""

from __future__ import annotations

import hashlib
from pathlib import Path

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import (
    _FACTORY_CAPABILITY,
    MarketOrderProfile,
    MarketOrderRule,
)
from trade_rl.integrations.binance.forward_evidence import _bytes, _decode, _time
from trade_rl.integrations.binance.forward_rules import _symbol_rule
from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot

_SOURCE = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def build_usdm_market_order_profile(
    dataset: MarketDataset,
    snapshot: BinanceExchangeInfoSnapshot,
    *,
    selected_symbols: tuple[str, ...],
    account_mode: str,
    reduce_only_exits: bool,
) -> MarketOrderProfile:
    """Validate raw bytes and derive selected rules; ignore snapshot summaries."""
    if (
        type(selected_symbols) is not tuple
        or not selected_symbols
        or any(not isinstance(s, str) for s in selected_symbols)
        or len(set(selected_symbols)) != len(selected_symbols)
        or not set(selected_symbols) <= set(dataset.symbols)
    ):
        raise ValueError("selected symbols must be unique members of the dataset")
    if snapshot.source_uri != _SOURCE:
        raise ValueError("unsupported source URI")
    if hashlib.sha256(snapshot.raw_payload).hexdigest() != snapshot.raw_payload_sha256:
        raise ValueError("source digest mismatch")
    payload = _decode(snapshot.raw_payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("source requires symbol rows")
    rows = {}
    for row in payload["symbols"]:
        if not isinstance(row, dict) or not isinstance(row.get("symbol"), str):
            raise ValueError("malformed source symbol")
        name = row["symbol"]
        if name in rows:
            raise ValueError("duplicate source symbol")
        rows[name] = row
    if not set(selected_symbols) <= rows.keys():
        raise ValueError("missing selected source symbols")
    rules = []
    for index, name in enumerate(dataset.symbols):
        if name not in selected_symbols:
            continue
        values = _symbol_rule(rows[name], name, "perpetual")
        rules.append(
            MarketOrderRule(
                symbol_index=index,
                symbol=name,
                lot_size=values["lot_size"],
                minimum_quantity=values["minimum_quantity"],
                maximum_quantity=values["maximum_quantity"],
                minimum_notional=values["minimum_notional"],
            )
        )
    profile = MarketOrderProfile(
        dataset_id=dataset.dataset_id,
        dataset_symbols=dataset.symbols,
        rules=tuple(rules),
        source_uri=snapshot.source_uri,
        source_sha256=snapshot.raw_payload_sha256,
        retrieved_at=snapshot.retrieved_at,
        account_mode=account_mode,
        reduce_only_exits=reduce_only_exits,
        _factory=_FACTORY_CAPABILITY,
    )
    profile.validate_dataset(dataset)
    return profile


def publish_usdm_market_order_profile(
    root: str | Path,
    profile: MarketOrderProfile,
    snapshot: BinanceExchangeInfoSnapshot,
    dataset: MarketDataset,
) -> str:
    """Publish once after rederivation; an incomplete publication fails closed."""
    expected = build_usdm_market_order_profile(
        dataset,
        snapshot,
        selected_symbols=tuple(r.symbol for r in profile.rules),
        account_mode=profile.account_mode,
        reduce_only_exits=profile.reduce_only_exits,
    )
    if expected != profile:
        raise ValueError("profile differs from source-derived rules")
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "exchange-info.raw.json").open("xb") as stream:
        stream.write(snapshot.raw_payload)
    with (destination / "profile.json").open("xb") as stream:
        stream.write(canonical_json_bytes(profile.canonical_payload()))
    return profile.digest


def load_usdm_market_order_profile(
    root: str | Path, dataset: MarketDataset, *, expected_digest: str
) -> MarketOrderProfile:
    """Verify external digest and independently rederive every serialized field."""
    require_sha256(expected_digest, field="expected_digest")
    source = Path(root)
    if source.is_symlink():
        raise ValueError("profile root must not be a symlink")
    raw = _bytes(source / "profile.json")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("profile digest mismatch")
    payload = _decode(raw)
    try:
        if payload["dataset_id"] != dataset.dataset_id:
            raise ValueError("profile dataset binding mismatch")
        snapshot = BinanceExchangeInfoSnapshot(
            payload={},
            raw_payload=_bytes(source / "exchange-info.raw.json"),
            source_uri=payload["source_uri"],
            retrieved_at=_time(payload["retrieved_at"]),
            raw_payload_sha256=payload["source_sha256"],
        )
        profile = build_usdm_market_order_profile(
            dataset,
            snapshot,
            selected_symbols=tuple(r["symbol"] for r in payload["rules"]),
            account_mode=payload["account_mode"],
            reduce_only_exits=payload["reduce_only_exits"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("malformed source-derived profile") from error
    if canonical_json_bytes(profile.canonical_payload()) != raw:
        raise ValueError("profile differs from source-derived rules")
    return profile
