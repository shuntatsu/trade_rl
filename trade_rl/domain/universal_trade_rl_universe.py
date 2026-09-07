"""Immutable symbol-role contracts for Universal Trade RL research."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.domain.common import domain_content_digest, require_non_empty

_UNIVERSE_CONFIG_SCHEMA: Final = "universal_trade_rl_universe_config_v1"
_SYMBOL_RE: Final = re.compile(r"^[A-Z0-9]{2,32}$")


def _canonical_symbol(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValueError(f"{field} must use canonical uppercase market symbol syntax")
    return value


def _canonical_group(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple")
    symbols = tuple(_canonical_symbol(item, field=field) for item in value)
    if not symbols:
        raise ValueError(f"{field} must be non-empty")
    if symbols != tuple(sorted(set(symbols))):
        raise ValueError(f"{field} must be sorted and unique")
    return symbols


class UniversalTradeRLSymbolRole(str, Enum):
    """Immutable role of one symbol in universal-policy research."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    ADMISSION = "admission"


@dataclass(frozen=True, slots=True)
class UniversalTradeRLSymbolExclusion:
    """Explicit reason one available symbol is not assigned a research role."""

    symbol: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _canonical_symbol(self.symbol, field="excluded symbol"),
        )
        if not isinstance(self.reason, str):
            raise TypeError("exclusion reason must be a string")
        object.__setattr__(
            self,
            "reason",
            require_non_empty(self.reason, field="exclusion reason"),
        )

    def to_payload(self) -> dict[str, str]:
        return {"reason": self.reason, "symbol": self.symbol}


@dataclass(frozen=True, slots=True)
class UniversalTradeRLUniverseConfig:
    """Artifact-compatible immutable Train/Development/Admission partition."""

    train_symbols: tuple[str, ...]
    development_symbols: tuple[str, ...]
    admission_symbols: tuple[str, ...]
    exclusions: tuple[UniversalTradeRLSymbolExclusion, ...] = ()
    schema_version: str = _UNIVERSE_CONFIG_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        train = _canonical_group(self.train_symbols, field="train_symbols")
        development = _canonical_group(
            self.development_symbols,
            field="development_symbols",
        )
        admission = _canonical_group(
            self.admission_symbols,
            field="admission_symbols",
        )
        role_sets = (set(train), set(development), set(admission))
        if any(
            left & right
            for index, left in enumerate(role_sets)
            for right in role_sets[index + 1 :]
        ):
            raise ValueError("universe role groups must be pairwise disjoint")
        if "BTCUSDT" not in train:
            raise ValueError("BTCUSDT market proxy must remain in Train")

        if not isinstance(self.exclusions, tuple):
            raise TypeError("exclusions must be a tuple")
        if any(
            not isinstance(item, UniversalTradeRLSymbolExclusion)
            for item in self.exclusions
        ):
            raise TypeError("exclusions must contain symbol exclusions")
        exclusions = tuple(sorted(self.exclusions, key=lambda item: item.symbol))
        exclusion_symbols = tuple(item.symbol for item in exclusions)
        if len(set(exclusion_symbols)) != len(exclusion_symbols):
            raise ValueError("excluded symbols must be unique")
        assigned = set().union(*role_sets)
        if assigned & set(exclusion_symbols):
            raise ValueError("a symbol cannot be assigned and excluded")
        if self.schema_version != _UNIVERSE_CONFIG_SCHEMA:
            raise ValueError("unsupported Universal Trade RL universe config schema")

        object.__setattr__(self, "train_symbols", train)
        object.__setattr__(self, "development_symbols", development)
        object.__setattr__(self, "admission_symbols", admission)
        object.__setattr__(self, "exclusions", exclusions)
        expected = domain_content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("Universal Trade RL universe config digest mismatch")
        object.__setattr__(self, "digest", expected)

    def role_for(self, symbol: str) -> UniversalTradeRLSymbolRole | None:
        resolved = _canonical_symbol(symbol, field="symbol")
        if resolved in self.train_symbols:
            return UniversalTradeRLSymbolRole.TRAIN
        if resolved in self.development_symbols:
            return UniversalTradeRLSymbolRole.DEVELOPMENT
        if resolved in self.admission_symbols:
            return UniversalTradeRLSymbolRole.ADMISSION
        return None

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_symbols": self.admission_symbols,
            "development_symbols": self.development_symbols,
            "excluded_symbols": tuple(item.to_payload() for item in self.exclusions),
            "schema_version": self.schema_version,
            "train_symbols": self.train_symbols,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "UniversalTradeRLSymbolExclusion",
    "UniversalTradeRLSymbolRole",
    "UniversalTradeRLUniverseConfig",
]
