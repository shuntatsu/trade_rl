from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.offline_signing import public_key_bytes, sign_payload

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
START = datetime(2024, 12, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)
ISSUED = datetime(2026, 7, 17, tzinfo=UTC)
TRUSTED_NOW = ISSUED + timedelta(hours=1)
KEY_ID = "binance-history-key"
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
PUBLIC_KEY = PublicVerificationKey(
    key_id=KEY_ID,
    public_key=public_key_bytes(PRIVATE_KEY),
    purpose="binance-rule-history",
    valid_from=START - timedelta(days=1),
    valid_until=TRUSTED_NOW + timedelta(days=365),
)
TRUSTED_KEYS = {KEY_ID: PUBLIC_KEY}


def signed_rule_history_document(
    *,
    market: str = "usds-m",
    symbol_order: tuple[str, ...] = SYMBOLS,
    coverage_start: datetime = START,
    coverage_end: datetime = END,
    issued_at: datetime = ISSUED,
    source_uri: str = "operator://signed-binance-rules",
    tick_size: float = 0.1,
    lot_size: float = 0.001,
    minimum_notional: float = 5.0,
    rule_effective_at: datetime = START,
    extra_rule_effective_at: datetime | None = None,
    final_tick_size: float | None = None,
    payload_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_final_tick = tick_size if final_tick_size is None else final_tick_size
    payload: dict[str, object] = {
        "schema_version": "binance_instrument_rule_history_v4",
        "policy_version": "binance_metadata_modes_v2",
        "market": market,
        "symbol_order": list(symbol_order),
        "coverage": {
            "start_time": coverage_start.isoformat(),
            "end_time": coverage_end.isoformat(),
        },
        "issued_at": issued_at.isoformat(),
        "source_uri": source_uri,
        "symbols": {
            symbol: {
                "listed_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
                "tick_size": resolved_final_tick,
                "lot_size": lot_size,
                "minimum_notional": minimum_notional,
                "execution_rules": [
                    {
                        "effective_at": effective_at.isoformat(),
                        "tick_size": tick_size,
                        "lot_size": lot_size,
                        "minimum_notional": minimum_notional,
                    }
                    for effective_at in (
                        (rule_effective_at,)
                        if extra_rule_effective_at is None
                        else (rule_effective_at, extra_rule_effective_at)
                    )
                ],
            }
            for symbol in symbol_order
        },
    }
    if payload_overrides:
        payload.update(payload_overrides)
    envelope = sign_payload(
        payload,
        key_id=KEY_ID,
        purpose="binance-rule-history",
        private_key=PRIVATE_KEY,
        signed_at=issued_at,
    )
    document = {"payload": payload, "envelope": envelope.to_mapping()}
    return json.loads(canonical_json_bytes(document))
