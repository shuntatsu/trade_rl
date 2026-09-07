"""Strict U0 config and source-catalog inputs for Universal Trade RL."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLUniverseConfig,
)

UNIVERSAL_TRADE_RL_SOURCE_CATALOG_SCHEMA: Final = "universal_trade_rl_source_catalog_v1"
_UNIVERSE_ROOT_KEYS: Final = (
    "schema_version",
    "train_symbols",
    "development_symbols",
    "admission_symbols",
    "excluded_symbols",
)
_EXCLUSION_KEYS: Final = ("symbol", "reason")
_CATALOG_ROOT_KEYS: Final = ("schema_version", "symbols")
_SOURCE_KEYS: Final = (
    "symbol",
    "dataset_digest",
    "first_timestamp_ns",
    "last_timestamp_ns",
    "row_count",
)
_SYMBOL_RE: Final = re.compile(r"^[A-Z0-9]{2,32}$")


def _load_json(path: str | Path, *, field: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{field} JSON contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} JSON is invalid") from error


def _exact_mapping(
    value: object, *, keys: tuple[str, ...], field: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result = {str(key): item for key, item in value.items()}
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(
            f"{field} must use exact keys; "
            f"expected={list(keys)}, observed={list(result)}"
        )
    return result


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _symbol(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValueError(f"{field} must use canonical uppercase market symbol syntax")
    return value


@dataclass(frozen=True, slots=True)
class UniversalTradeRLSymbolSource:
    """Exact source-data identity for one available symbol."""

    symbol: str
    dataset_digest: str
    first_timestamp_ns: int
    last_timestamp_ns: int
    row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol, field="source symbol"))
        require_sha256(self.dataset_digest, field="source dataset digest")
        first = _integer(self.first_timestamp_ns, field="first_timestamp_ns")
        last = _integer(self.last_timestamp_ns, field="last_timestamp_ns")
        rows = _integer(self.row_count, field="row_count")
        if first < 0:
            raise ValueError("first_timestamp_ns must be non-negative")
        if last <= first:
            raise ValueError("last_timestamp_ns must be later than first_timestamp_ns")
        if rows <= 0:
            raise ValueError("row_count must be positive")
        object.__setattr__(self, "first_timestamp_ns", first)
        object.__setattr__(self, "last_timestamp_ns", last)
        object.__setattr__(self, "row_count", rows)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_digest": self.dataset_digest,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "row_count": self.row_count,
            "symbol": self.symbol,
        }


def _source_records(value: object) -> tuple[UniversalTradeRLSymbolSource, ...]:
    raw = _array(value, field="source catalog symbols")
    if not raw:
        raise ValueError("source catalog symbols must be non-empty")
    records: list[UniversalTradeRLSymbolSource] = []
    for index, item in enumerate(raw):
        record = _exact_mapping(
            item,
            keys=_SOURCE_KEYS,
            field=f"source catalog symbol[{index}]",
        )
        dataset_digest = record["dataset_digest"]
        if not isinstance(dataset_digest, str):
            raise ValueError("source dataset digest must be a string")
        records.append(
            UniversalTradeRLSymbolSource(
                symbol=_symbol(record["symbol"], field="source symbol"),
                dataset_digest=dataset_digest,
                first_timestamp_ns=_integer(
                    record["first_timestamp_ns"], field="first_timestamp_ns"
                ),
                last_timestamp_ns=_integer(
                    record["last_timestamp_ns"], field="last_timestamp_ns"
                ),
                row_count=_integer(record["row_count"], field="row_count"),
            )
        )
    result = tuple(records)
    symbols = tuple(item.symbol for item in result)
    if len(set(symbols)) != len(symbols):
        raise ValueError("source catalog symbols must be unique")
    if symbols != tuple(sorted(symbols)):
        raise ValueError("source catalog symbols must be sorted")
    return result


def load_universal_trade_rl_universe_config(
    path: str | Path,
) -> UniversalTradeRLUniverseConfig:
    """Load one exact-key immutable universe-role config."""

    root = _exact_mapping(
        _load_json(path, field="Universal Trade RL universe config"),
        keys=_UNIVERSE_ROOT_KEYS,
        field="Universal Trade RL universe config",
    )
    if root["schema_version"] != "universal_trade_rl_universe_config_v1":
        raise ValueError("unsupported Universal Trade RL universe config schema")

    exclusions_raw = _array(root["excluded_symbols"], field="excluded_symbols")
    exclusions: list[UniversalTradeRLSymbolExclusion] = []
    for index, item in enumerate(exclusions_raw):
        exclusion = _exact_mapping(
            item,
            keys=_EXCLUSION_KEYS,
            field=f"excluded_symbols[{index}]",
        )
        reason = exclusion["reason"]
        if not isinstance(reason, str):
            raise ValueError("excluded symbol reason must be a string")
        exclusions.append(
            UniversalTradeRLSymbolExclusion(
                symbol=_symbol(exclusion["symbol"], field="excluded symbol"),
                reason=reason,
            )
        )

    role_values: dict[str, tuple[str, ...]] = {}
    for key in ("train_symbols", "development_symbols", "admission_symbols"):
        values = _array(root[key], field=key)
        role_values[key] = tuple(_symbol(item, field=key) for item in values)

    return UniversalTradeRLUniverseConfig(
        train_symbols=role_values["train_symbols"],
        development_symbols=role_values["development_symbols"],
        admission_symbols=role_values["admission_symbols"],
        exclusions=tuple(exclusions),
    )


def load_universal_trade_rl_source_catalog(
    path: str | Path,
) -> tuple[UniversalTradeRLSymbolSource, ...]:
    """Load exact, sorted source identities without assigning research roles."""

    root = _exact_mapping(
        _load_json(path, field="Universal Trade RL source catalog"),
        keys=_CATALOG_ROOT_KEYS,
        field="Universal Trade RL source catalog",
    )
    if root["schema_version"] != UNIVERSAL_TRADE_RL_SOURCE_CATALOG_SCHEMA:
        raise ValueError("unsupported Universal Trade RL source catalog schema")
    return _source_records(root["symbols"])


def universal_trade_rl_source_catalog_digest(
    records: tuple[UniversalTradeRLSymbolSource, ...],
) -> str:
    """Digest the complete sorted source identity surface."""

    if not isinstance(records, tuple) or not records:
        raise ValueError("source catalog records must be a non-empty tuple")
    if any(not isinstance(item, UniversalTradeRLSymbolSource) for item in records):
        raise TypeError("source catalog records must contain source identities")
    symbols = tuple(item.symbol for item in records)
    if len(set(symbols)) != len(symbols):
        raise ValueError("source catalog symbols must be unique")
    if symbols != tuple(sorted(symbols)):
        raise ValueError("source catalog symbols must be sorted")
    return content_digest(
        {
            "schema_version": UNIVERSAL_TRADE_RL_SOURCE_CATALOG_SCHEMA,
            "symbols": tuple(item.to_payload() for item in records),
        }
    )


__all__ = [
    "UNIVERSAL_TRADE_RL_SOURCE_CATALOG_SCHEMA",
    "UniversalTradeRLSymbolSource",
    "load_universal_trade_rl_source_catalog",
    "load_universal_trade_rl_universe_config",
    "universal_trade_rl_source_catalog_digest",
]
