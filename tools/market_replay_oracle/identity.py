"""Independent v7 byte protocol; production identity helpers are not an oracle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from numpy.typing import NDArray

ARRAY_FIELDS = (
    "timestamps",
    "available_at",
    "information_available",
    "features",
    "feature_available",
    "feature_staleness",
    "feature_staleness_hours",
    "feature_missing_reason",
    "global_features",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "funding_price_rate",
    "funding_event_count",
    "tradable",
    "symbol_active",
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
    "borrow_available",
    "borrow_rate",
    "funding_due",
    "buy_allowed",
    "sell_allowed",
    "mark_price",
    "index_price",
    "dividend",
    "split_factor",
    "delisting_recovery",
    "cash_rate",
    "contract_multipliers",
)

CONTRACT_FIELDS = (
    "calendar_kind",
    "feature_config_digest",
    "feature_names",
    "global_feature_names",
    "nominal_bar_hours",
    "normalization_digest",
    "periods_per_year",
    "symbols",
    "volume_units",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        canonical(value)  # Also rejects overflowed JSON numbers such as 1e999.
        return value
    except (RecursionError, UnicodeError) as error:
        raise ValueError("invalid or excessively nested JSON") from error


def content_id(identity: dict[str, Any], arrays: Mapping[str, NDArray[Any]]) -> str:
    digest = hashlib.sha256(canonical(identity))
    for name in ARRAY_FIELDS:
        source = arrays[name]
        kind = source.dtype.kind
        dtype = "|u1" if kind == "b" else "<f8" if kind == "f" else "<i8"
        normalized = source.astype(dtype, order="C")
        descriptor = canonical(
            {"name": name, "dtype": normalized.dtype.str, "shape": list(source.shape)}
        )
        payload = normalized.tobytes(order="C")
        for block in (descriptor, payload):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()
