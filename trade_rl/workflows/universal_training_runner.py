"""Executable runtime assembly for Universal single-instrument training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import (
    InstrumentContract,
    InstrumentExecutionRule,
    VolumeUnit,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv


def _aware_datetime(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        resolved = value
    elif isinstance(value, str):
        token = value.strip().replace("Z", "+00:00")
        if not token:
            raise ValueError(f"{field} must not be empty")
        try:
            resolved = datetime.fromisoformat(token)
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 datetime") from error
    else:
        raise TypeError(f"{field} must be a datetime or ISO-8601 string")
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return resolved.astimezone(UTC)


def _positive_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not resolved > 0.0:
        raise ValueError(f"{field} must be positive")
    return resolved


def build_universal_instrument_contracts(
    metadata_resolution: Any,
    *,
    train_symbols: Sequence[str],
) -> dict[str, InstrumentContract]:
    """Build train-only instrument contracts used by causal descriptor generation."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols) or any(not value for value in symbols):
        raise ValueError("Universal train_symbols must be non-empty and unique")
    raw_metadata = getattr(metadata_resolution, "metadata", None)
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("Universal metadata resolution must expose metadata")
    raw_histories = getattr(metadata_resolution, "execution_rule_histories", None)
    if raw_histories is not None and not isinstance(raw_histories, Mapping):
        raise TypeError("Universal execution_rule_histories must be a mapping or null")

    contracts: dict[str, InstrumentContract] = {}
    for symbol in symbols:
        raw = raw_metadata.get(symbol)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Universal metadata is missing train symbol {symbol}")
        delisted_raw = raw.get("delisted_at")
        delisted_at = (
            None
            if delisted_raw is None
            else _aware_datetime(delisted_raw, field=f"{symbol}.delisted_at")
        )
        execution_rules: tuple[InstrumentExecutionRule, ...] = ()
        if raw_histories is not None:
            history = raw_histories.get(symbol)
            if not isinstance(history, (tuple, list)) or any(
                not isinstance(item, InstrumentExecutionRule) for item in history
            ):
                raise ValueError(
                    f"Universal execution rules are missing or invalid for {symbol}"
                )
            execution_rules = tuple(history)
        contracts[symbol] = InstrumentContract(
            symbol=symbol,
            listed_at=_aware_datetime(raw.get("listed_at"), field=f"{symbol}.listed_at"),
            delisted_at=delisted_at,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            tick_size=_positive_number(raw.get("tick_size"), field=f"{symbol}.tick_size"),
            lot_size=_positive_number(raw.get("lot_size"), field=f"{symbol}.lot_size"),
            minimum_notional=_positive_number(
                raw.get("minimum_notional"),
                field=f"{symbol}.minimum_notional",
            ),
            execution_rules=execution_rules,
        )
    return contracts


@dataclass(frozen=True, slots=True)
class UniversalRoutedEnvironmentFactory:
    """Pickle-safe routed environment factory with explicit vector-worker identity."""

    train_symbols: tuple[str, ...]
    partition_digest: str
    bindings: tuple[InstrumentDatasetBinding, ...]
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any]
    instrument_context_provider: CausalInstrumentContextProvider | None
    training_contract_digest: str
    run_seed: int

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Universal routed factory train_symbols are invalid")
        if tuple(binding.concrete_symbol for binding in self.bindings) != symbols:
            raise ValueError("Universal routed factory bindings must follow train_symbols")
        if any(binding.split != "train" for binding in self.bindings):
            raise ValueError("Universal routed training factory accepts train bindings only")
        if not callable(self.concrete_environment_factory):
            raise TypeError("concrete_environment_factory must be callable")
        if isinstance(self.run_seed, bool) or not isinstance(self.run_seed, int) or self.run_seed < 0:
            raise ValueError("Universal routed factory run_seed must be non-negative")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bindings": tuple(binding.digest for binding in self.bindings),
                "partition_digest": self.partition_digest,
                "run_seed": self.run_seed,
                "schema_version": "universal_routed_environment_factory_v1",
                "training_contract_digest": self.training_contract_digest,
                "train_symbols": self.train_symbols,
            }
        )

    def _create(self, environment_index: int) -> EpisodeRoutedSingleInstrumentEnv:
        return EpisodeRoutedSingleInstrumentEnv(
            train_symbols=self.train_symbols,
            partition_digest=self.partition_digest,
            bindings=self.bindings,
            environment_factory=self.concrete_environment_factory,
            run_seed=self.run_seed,
            environment_index=environment_index,
            instrument_context_provider=self.instrument_context_provider,
            training_contract_digest=self.training_contract_digest,
        )

    def __call__(self) -> EpisodeRoutedSingleInstrumentEnv:
        return self._create(0)

    def for_environment_index(
        self,
        index: int,
    ) -> Callable[[], EpisodeRoutedSingleInstrumentEnv]:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Universal environment index must be non-negative")
        return partial(self._create, index)


__all__ = [
    "UniversalRoutedEnvironmentFactory",
    "build_universal_instrument_contracts",
]
