"""Explicit execution-economics assumptions for immutable market datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

_SCHEMA_VERSION = "execution_economics_v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "max_participation_rate",
        "borrow_available",
        "borrow_rate",
    }
)


def _rate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionEconomicsConfig:
    """One explicit, deterministic dataset-level execution-economics profile."""

    fee_rate: float
    maker_fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    max_participation_rate: float
    borrow_available: bool
    borrow_rate: float
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("execution economics schema_version differs from contract")
        fee_rate = _rate(self.fee_rate, field="fee_rate")
        maker_fee_rate = _rate(self.maker_fee_rate, field="maker_fee_rate")
        taker_fee_rate = _rate(self.taker_fee_rate, field="taker_fee_rate")
        spread_rate = _rate(self.spread_rate, field="spread_rate")
        participation = _rate(
            self.max_participation_rate,
            field="max_participation_rate",
        )
        borrow_rate = _rate(self.borrow_rate, field="borrow_rate")
        if participation > 1.0:
            raise ValueError("max_participation_rate must be within [0, 1]")
        if not isinstance(self.borrow_available, bool):
            raise ValueError("borrow_available must be boolean")
        if fee_rate > 0.0 and (maker_fee_rate > 0.0 or taker_fee_rate > 0.0):
            raise ValueError(
                "fee_rate cannot be combined with maker_fee_rate or taker_fee_rate"
            )
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "maker_fee_rate", maker_fee_rate)
        object.__setattr__(self, "taker_fee_rate", taker_fee_rate)
        object.__setattr__(self, "spread_rate", spread_rate)
        object.__setattr__(self, "max_participation_rate", participation)
        object.__setattr__(self, "borrow_rate", borrow_rate)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fee_rate": self.fee_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "spread_rate": self.spread_rate,
            "max_participation_rate": self.max_participation_rate,
            "borrow_available": self.borrow_available,
            "borrow_rate": self.borrow_rate,
        }


def parse_execution_economics(value: object) -> ExecutionEconomicsConfig:
    """Parse one strict execution-economics object without implicit defaults."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("execution_economics must be an object")
    raw = cast(Mapping[str, object], value)
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"execution_economics contains unknown fields: {unknown}")
    missing = sorted(_FIELDS - set(raw))
    if missing:
        raise ValueError(f"execution_economics is missing fields: {missing}")
    if raw["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("execution economics schema_version differs from contract")
    borrow_available = raw["borrow_available"]
    if not isinstance(borrow_available, bool):
        raise ValueError("borrow_available must be boolean")
    return ExecutionEconomicsConfig(
        fee_rate=_rate(raw["fee_rate"], field="fee_rate"),
        maker_fee_rate=_rate(raw["maker_fee_rate"], field="maker_fee_rate"),
        taker_fee_rate=_rate(raw["taker_fee_rate"], field="taker_fee_rate"),
        spread_rate=_rate(raw["spread_rate"], field="spread_rate"),
        max_participation_rate=_rate(
            raw["max_participation_rate"],
            field="max_participation_rate",
        ),
        borrow_available=borrow_available,
        borrow_rate=_rate(raw["borrow_rate"], field="borrow_rate"),
    )


__all__ = ["ExecutionEconomicsConfig", "parse_execution_economics"]
