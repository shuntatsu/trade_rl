"""Deterministic balanced routing for universal single-instrument episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_non_empty,
    require_sha256,
    require_unique_non_empty,
)

INSTRUMENT_ROUTE_SCHEMA: Final = "instrument_route_v1"
DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA: Final = (
    "deterministic_balanced_instrument_router_v1"
)


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class InstrumentRoute:
    """One deterministic position in a balanced symbol cycle."""

    concrete_symbol: str
    completed_episode_count: int
    routing_cycle: int
    routing_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.concrete_symbol, str):
            raise TypeError("concrete_symbol must be a string")
        symbol = require_non_empty(
            self.concrete_symbol,
            field="concrete_symbol",
        )
        _require_non_negative_int(
            self.completed_episode_count,
            field="completed_episode_count",
        )
        _require_non_negative_int(
            self.routing_cycle,
            field="routing_cycle",
        )
        _require_non_negative_int(
            self.routing_position,
            field="routing_position",
        )
        object.__setattr__(self, "concrete_symbol", symbol)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "completed_episode_count": self.completed_episode_count,
            "concrete_symbol": self.concrete_symbol,
            "routing_cycle": self.routing_cycle,
            "routing_position": self.routing_position,
            "schema_version": INSTRUMENT_ROUTE_SCHEMA,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_json_dict())


@dataclass(frozen=True, slots=True)
class DeterministicBalancedInstrumentRouter:
    """Route each symbol exactly once per deterministic environment-local cycle."""

    train_symbols: tuple[str, ...]
    partition_digest: str
    run_seed: int
    environment_index: int

    def __post_init__(self) -> None:
        try:
            raw_symbols = tuple(self.train_symbols)
        except TypeError as error:
            raise ValueError("train_symbols must be a non-empty tuple") from error
        symbols = require_unique_non_empty(
            raw_symbols,
            field="train_symbols",
        )
        partition_digest = require_sha256(
            self.partition_digest,
            field="partition_digest",
        )
        _require_non_negative_int(self.run_seed, field="run_seed")
        _require_non_negative_int(
            self.environment_index,
            field="environment_index",
        )
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "partition_digest", partition_digest)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "environment_index": self.environment_index,
            "partition_digest": self.partition_digest,
            "run_seed": self.run_seed,
            "schema_version": DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA,
            "train_symbols": list(self.train_symbols),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_json_dict())

    def cycle_symbols(self, routing_cycle: int) -> tuple[str, ...]:
        """Return one immutable permutation for the requested cycle."""

        cycle = _require_non_negative_int(
            routing_cycle,
            field="routing_cycle",
        )

        def key(symbol: str) -> tuple[str, str]:
            return (
                content_digest(
                    {
                        "environment_index": self.environment_index,
                        "partition_digest": self.partition_digest,
                        "routing_cycle": cycle,
                        "run_seed": self.run_seed,
                        "schema_version": DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA,
                        "symbol": symbol,
                    }
                ),
                symbol,
            )

        return tuple(sorted(self.train_symbols, key=key))

    def route(self, completed_episode_count: int) -> InstrumentRoute:
        """Resolve the exact symbol and cycle position for one episode count."""

        completed = _require_non_negative_int(
            completed_episode_count,
            field="completed_episode_count",
        )
        routing_cycle, routing_position = divmod(
            completed,
            len(self.train_symbols),
        )
        concrete_symbol = self.cycle_symbols(routing_cycle)[routing_position]
        return InstrumentRoute(
            concrete_symbol=concrete_symbol,
            completed_episode_count=completed,
            routing_cycle=routing_cycle,
            routing_position=routing_position,
        )


__all__ = [
    "DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA",
    "INSTRUMENT_ROUTE_SCHEMA",
    "DeterministicBalancedInstrumentRouter",
    "InstrumentRoute",
]
