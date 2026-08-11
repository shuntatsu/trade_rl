"""Immutable concrete-instrument bindings for universal single-instrument runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_non_empty,
    require_sha256,
    require_unique_non_empty,
)

GENERIC_INSTRUMENT_SYMBOL: Final = "INSTRUMENT"
GENERIC_INSTRUMENT_SYMBOLS: Final = (GENERIC_INSTRUMENT_SYMBOL,)
GENERIC_TARGET_WEIGHT_ACTION_NAMES: Final = (
    f"target_weight:{GENERIC_INSTRUMENT_SYMBOL}",
)
INSTRUMENT_SPLITS: Final = frozenset({"train", "validation", "test"})
INSTRUMENT_DATASET_BINDING_SCHEMA: Final = "instrument_dataset_binding_v1"
INSTRUMENT_EPISODE_BINDING_SCHEMA: Final = "instrument_episode_binding_v1"

_DATASET_BINDING_KEYS: Final = frozenset(
    {
        "concrete_symbol",
        "execution_metadata_digest",
        "instrument_descriptor_digest",
        "schema_version",
        "source_dataset_id",
        "split",
        "symbol_dataset_digest",
    }
)
_EPISODE_BINDING_KEYS: Final = frozenset(
    {
        "completed_episode_count",
        "concrete_symbol",
        "dataset_binding_digest",
        "environment_index",
        "episode_seed",
        "episode_start",
        "episode_stop",
        "execution_metadata_digest",
        "instrument_descriptor_digest",
        "routing_cycle",
        "routing_position",
        "schema_version",
        "source_dataset_id",
        "split",
        "symbol_dataset_digest",
    }
)


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_field_closure(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> dict[str, object]:
    payload = dict(value)
    if frozenset(payload) != expected:
        raise ValueError(f"{field} field closure mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class InstrumentDatasetBinding:
    """Immutable proof of the concrete single-symbol dataset selected for one run."""

    concrete_symbol: str
    source_dataset_id: str
    symbol_dataset_digest: str
    execution_metadata_digest: str
    instrument_descriptor_digest: str
    split: str

    def __post_init__(self) -> None:
        if not isinstance(self.concrete_symbol, str):
            raise TypeError("concrete_symbol must be a string")
        concrete_symbol = require_non_empty(
            self.concrete_symbol,
            field="concrete_symbol",
        )
        source_dataset_id = require_sha256(
            self.source_dataset_id,
            field="source_dataset_id",
        )
        symbol_dataset_digest = require_sha256(
            self.symbol_dataset_digest,
            field="symbol_dataset_digest",
        )
        execution_metadata_digest = require_sha256(
            self.execution_metadata_digest,
            field="execution_metadata_digest",
        )
        instrument_descriptor_digest = require_sha256(
            self.instrument_descriptor_digest,
            field="instrument_descriptor_digest",
        )
        if not isinstance(self.split, str) or self.split not in INSTRUMENT_SPLITS:
            raise ValueError("split must be one of train, validation, or test")
        object.__setattr__(self, "concrete_symbol", concrete_symbol)
        object.__setattr__(self, "source_dataset_id", source_dataset_id)
        object.__setattr__(
            self,
            "symbol_dataset_digest",
            symbol_dataset_digest,
        )
        object.__setattr__(
            self,
            "execution_metadata_digest",
            execution_metadata_digest,
        )
        object.__setattr__(
            self,
            "instrument_descriptor_digest",
            instrument_descriptor_digest,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "concrete_symbol": self.concrete_symbol,
            "execution_metadata_digest": self.execution_metadata_digest,
            "instrument_descriptor_digest": self.instrument_descriptor_digest,
            "schema_version": INSTRUMENT_DATASET_BINDING_SCHEMA,
            "source_dataset_id": self.source_dataset_id,
            "split": self.split,
            "symbol_dataset_digest": self.symbol_dataset_digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_json_dict())

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> InstrumentDatasetBinding:
        if not isinstance(value, Mapping):
            raise TypeError("instrument dataset binding must be a mapping")
        payload = _require_field_closure(
            value,
            expected=_DATASET_BINDING_KEYS,
            field="instrument dataset binding",
        )
        if payload["schema_version"] != INSTRUMENT_DATASET_BINDING_SCHEMA:
            raise ValueError("instrument dataset binding schema mismatch")
        return cls(
            concrete_symbol=_require_string(
                payload["concrete_symbol"],
                field="concrete_symbol",
            ),
            source_dataset_id=_require_string(
                payload["source_dataset_id"],
                field="source_dataset_id",
            ),
            symbol_dataset_digest=_require_string(
                payload["symbol_dataset_digest"],
                field="symbol_dataset_digest",
            ),
            execution_metadata_digest=_require_string(
                payload["execution_metadata_digest"],
                field="execution_metadata_digest",
            ),
            instrument_descriptor_digest=_require_string(
                payload["instrument_descriptor_digest"],
                field="instrument_descriptor_digest",
            ),
            split=_require_string(payload["split"], field="split"),
        )


@dataclass(frozen=True, slots=True)
class InstrumentEpisodeBinding:
    """Concrete identity and route coordinates for one complete episode."""

    dataset_binding: InstrumentDatasetBinding
    episode_start: int
    episode_stop: int
    episode_seed: int
    environment_index: int
    completed_episode_count: int
    routing_cycle: int
    routing_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_binding, InstrumentDatasetBinding):
            raise TypeError("dataset_binding must be an InstrumentDatasetBinding")
        episode_start = _require_non_negative_int(
            self.episode_start,
            field="episode_start",
        )
        episode_stop = _require_non_negative_int(
            self.episode_stop,
            field="episode_stop",
        )
        if episode_stop <= episode_start:
            raise ValueError("episode_stop must be greater than episode_start")
        episode_seed = _require_non_negative_int(
            self.episode_seed,
            field="episode_seed",
        )
        if episode_seed > 0xFFFFFFFF:
            raise ValueError("episode_seed must fit in an unsigned 32-bit integer")
        _require_non_negative_int(
            self.environment_index,
            field="environment_index",
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

    def to_json_dict(self) -> dict[str, object]:
        binding = self.dataset_binding
        return {
            "completed_episode_count": self.completed_episode_count,
            "concrete_symbol": binding.concrete_symbol,
            "dataset_binding_digest": binding.digest,
            "environment_index": self.environment_index,
            "episode_seed": self.episode_seed,
            "episode_start": self.episode_start,
            "episode_stop": self.episode_stop,
            "execution_metadata_digest": binding.execution_metadata_digest,
            "instrument_descriptor_digest": binding.instrument_descriptor_digest,
            "routing_cycle": self.routing_cycle,
            "routing_position": self.routing_position,
            "schema_version": INSTRUMENT_EPISODE_BINDING_SCHEMA,
            "source_dataset_id": binding.source_dataset_id,
            "split": binding.split,
            "symbol_dataset_digest": binding.symbol_dataset_digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_json_dict())

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> InstrumentEpisodeBinding:
        if not isinstance(value, Mapping):
            raise TypeError("instrument episode binding must be a mapping")
        payload = _require_field_closure(
            value,
            expected=_EPISODE_BINDING_KEYS,
            field="instrument episode binding",
        )
        if payload["schema_version"] != INSTRUMENT_EPISODE_BINDING_SCHEMA:
            raise ValueError("instrument episode binding schema mismatch")
        binding = InstrumentDatasetBinding(
            concrete_symbol=_require_string(
                payload["concrete_symbol"],
                field="concrete_symbol",
            ),
            source_dataset_id=_require_string(
                payload["source_dataset_id"],
                field="source_dataset_id",
            ),
            symbol_dataset_digest=_require_string(
                payload["symbol_dataset_digest"],
                field="symbol_dataset_digest",
            ),
            execution_metadata_digest=_require_string(
                payload["execution_metadata_digest"],
                field="execution_metadata_digest",
            ),
            instrument_descriptor_digest=_require_string(
                payload["instrument_descriptor_digest"],
                field="instrument_descriptor_digest",
            ),
            split=_require_string(payload["split"], field="split"),
        )
        observed_digest = _require_string(
            payload["dataset_binding_digest"],
            field="dataset_binding_digest",
        )
        require_sha256(observed_digest, field="dataset_binding_digest")
        if observed_digest != binding.digest:
            raise ValueError("instrument dataset binding digest mismatch")
        return cls(
            dataset_binding=binding,
            episode_start=_require_non_negative_int(
                payload["episode_start"],
                field="episode_start",
            ),
            episode_stop=_require_non_negative_int(
                payload["episode_stop"],
                field="episode_stop",
            ),
            episode_seed=_require_non_negative_int(
                payload["episode_seed"],
                field="episode_seed",
            ),
            environment_index=_require_non_negative_int(
                payload["environment_index"],
                field="environment_index",
            ),
            completed_episode_count=_require_non_negative_int(
                payload["completed_episode_count"],
                field="completed_episode_count",
            ),
            routing_cycle=_require_non_negative_int(
                payload["routing_cycle"],
                field="routing_cycle",
            ),
            routing_position=_require_non_negative_int(
                payload["routing_position"],
                field="routing_position",
            ),
        )


def validate_training_instrument_bindings(
    train_symbols: Sequence[str],
    bindings: Sequence[InstrumentDatasetBinding],
) -> dict[str, InstrumentDatasetBinding]:
    """Require an exact train-only binding closure in declared symbol order."""

    declared = require_unique_non_empty(
        tuple(train_symbols),
        field="train_symbols",
    )
    resolved: dict[str, InstrumentDatasetBinding] = {}
    for binding in bindings:
        if not isinstance(binding, InstrumentDatasetBinding):
            raise TypeError("bindings must contain InstrumentDatasetBinding values")
        symbol = binding.concrete_symbol
        if symbol in resolved:
            raise ValueError(f"duplicate instrument binding for {symbol}")
        if binding.split != "train":
            raise ValueError(
                "training instrument bindings must all use the train split"
            )
        resolved[symbol] = binding
    if set(resolved) != set(declared):
        raise ValueError("training instrument binding closure mismatch")
    return {symbol: resolved[symbol] for symbol in declared}


__all__ = [
    "GENERIC_INSTRUMENT_SYMBOL",
    "GENERIC_INSTRUMENT_SYMBOLS",
    "GENERIC_TARGET_WEIGHT_ACTION_NAMES",
    "INSTRUMENT_DATASET_BINDING_SCHEMA",
    "INSTRUMENT_EPISODE_BINDING_SCHEMA",
    "INSTRUMENT_SPLITS",
    "InstrumentDatasetBinding",
    "InstrumentEpisodeBinding",
    "validate_training_instrument_bindings",
]
