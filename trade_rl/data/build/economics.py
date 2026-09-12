"""Execution-economics contract for deterministic market-dataset builds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

_SCHEMA_VERSION = "execution_economics_profile_v1"
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "max_participation_rate",
        "borrow_available",
        "borrow_rate",
    }
)


def _text(value: object, *, field: str, strip: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    resolved = value.strip() if strip else value
    if not resolved:
        raise ValueError(f"{field} must be a non-empty string")
    return resolved


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _rate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return resolved


def _participation(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_participation_rate must be in (0, 1]")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError("max_participation_rate must be in (0, 1]")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionEconomicsProfile:
    """Resolved build-time trading-cost and participation assumptions."""

    name: str
    fee_rate: float = 0.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    spread_rate: float = 0.0
    max_participation_rate: float = 1.0
    borrow_available: bool = True
    borrow_rate: float = 0.0
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("execution economics schema_version is unsupported")
        name = _text(self.name, field="execution economics name", strip=True)
        borrow_available = _boolean(
            self.borrow_available,
            field="borrow_available",
        )

        fee_rate = _rate(self.fee_rate, field="fee_rate")
        maker_fee_rate = _rate(self.maker_fee_rate, field="maker_fee_rate")
        taker_fee_rate = _rate(self.taker_fee_rate, field="taker_fee_rate")
        spread_rate = _rate(self.spread_rate, field="spread_rate")
        borrow_rate = _rate(self.borrow_rate, field="borrow_rate")
        max_participation_rate = _participation(self.max_participation_rate)

        if fee_rate > 0.0 and (maker_fee_rate > 0.0 or taker_fee_rate > 0.0):
            raise ValueError(
                "generic fee_rate cannot be combined with maker/taker fee rates"
            )
        if not borrow_available and borrow_rate != 0.0:
            raise ValueError("borrow_rate must be zero when borrow is unavailable")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "maker_fee_rate", maker_fee_rate)
        object.__setattr__(self, "taker_fee_rate", taker_fee_rate)
        object.__setattr__(self, "spread_rate", spread_rate)
        object.__setattr__(self, "max_participation_rate", max_participation_rate)
        object.__setattr__(self, "borrow_available", borrow_available)
        object.__setattr__(self, "borrow_rate", borrow_rate)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "fee_rate": self.fee_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "spread_rate": self.spread_rate,
            "max_participation_rate": self.max_participation_rate,
            "borrow_available": self.borrow_available,
            "borrow_rate": self.borrow_rate,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        field: str = "execution_economics",
    ) -> ExecutionEconomicsProfile:
        if not isinstance(payload, dict) or any(
            not isinstance(key, str) for key in payload
        ):
            raise ValueError(f"{field} must be a JSON object")
        raw = cast(Mapping[str, object], payload)
        keys = set(raw)
        unknown = sorted(keys - _PAYLOAD_FIELDS)
        if unknown:
            raise ValueError(f"{field} contains unknown fields: {unknown}")
        missing = sorted(_PAYLOAD_FIELDS - keys)
        if missing:
            raise ValueError(f"{field} is missing required fields: {missing}")
        return cls(
            schema_version=_text(
                raw["schema_version"],
                field=f"{field}.schema_version",
            ),
            name=_text(raw["name"], field=f"{field}.name", strip=True),
            fee_rate=_rate(raw["fee_rate"], field="fee_rate"),
            maker_fee_rate=_rate(raw["maker_fee_rate"], field="maker_fee_rate"),
            taker_fee_rate=_rate(raw["taker_fee_rate"], field="taker_fee_rate"),
            spread_rate=_rate(raw["spread_rate"], field="spread_rate"),
            max_participation_rate=_participation(raw["max_participation_rate"]),
            borrow_available=_boolean(
                raw["borrow_available"],
                field="borrow_available",
            ),
            borrow_rate=_rate(raw["borrow_rate"], field="borrow_rate"),
        )


__all__ = ["ExecutionEconomicsProfile"]
