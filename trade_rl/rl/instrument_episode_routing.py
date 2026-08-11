"""Deterministic train-only routing for universal single-instrument episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

GENERIC_INSTRUMENT_SYMBOL = "INSTRUMENT"
GENERIC_INSTRUMENT_SYMBOLS = (GENERIC_INSTRUMENT_SYMBOL,)
GENERIC_INSTRUMENT_ACTION_NAMES = (f"target_weight:{GENERIC_INSTRUMENT_SYMBOL}",)

INSTRUMENT_DATASET_BINDING_SCHEMA = "instrument_dataset_binding_v1"
INSTRUMENT_EPISODE_BINDING_SCHEMA = "instrument_episode_binding_v1"
DETERMINISTIC_BALANCED_INSTRUMENT_ROUTER_SCHEMA = (
    "deterministic_balanced_instrument_router_v1"
)
_DETERMINISTIC_PERMUTATION_SCHEMA = "instrument_routing_permutation_v1"


class InstrumentDatasetSplit(str, Enum):
    """Symbol-disjoint role assigned by the immutable universal partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: object, *, field: str) -> str:
    resolved = _require_string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _require_non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{field} fields mismatch: missing={missing}, extra={extra}")


_DATASET_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "concrete_symbol",
        "source_dataset_id",
        "symbol_dataset_digest",
        "execution_metadata_digest",
        "instrument_descriptor_digest",
        "partition_digest",
        "split",
        "binding_digest",
    }
)


@dataclass(frozen=True, slots=True)
class InstrumentDatasetBinding:
    """Immutable evidence binding for one concrete single-symbol dataset."""

    concrete_symbol: str
    source_dataset_id: str
    symbol_dataset_digest: str
    execution_metadata_digest: str
    instrument_descriptor_digest: str
    partition_digest: str
    split: InstrumentDatasetSplit | str

    def __post_init__(self) -> None:
        concrete_symbol = _require_string(
            self.concrete_symbol,
            field="concrete_symbol",
        )
        if concrete_symbol == GENERIC_INSTRUMENT_SYMBOL:
            raise ValueError("concrete_symbol must not use the generic INSTRUMENT slot")
        object.__setattr__(self, "concrete_symbol", concrete_symbol)
        for field_name in (
            "source_dataset_id",
            "symbol_dataset_digest",
            "execution_metadata_digest",
            "instrument_descriptor_digest",
            "partition_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_digest(getattr(self, field_name), field=field_name),
            )
        try:
            split = InstrumentDatasetSplit(self.split)
        except (TypeError, ValueError) as error:
            raise ValueError("split is not supported") from error
        object.__setattr__(self, "split", split)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": INSTRUMENT_DATASET_BINDING_SCHEMA,
            "concrete_symbol": self.concrete_symbol,
            "source_dataset_id": self.source_dataset_id,
            "symbol_dataset_digest": self.symbol_dataset_digest,
            "execution_metadata_digest": self.execution_metadata_digest,
            "instrument_descriptor_digest": self.instrument_descriptor_digest,
            "partition_digest": self.partition_digest,
            "split": InstrumentDatasetSplit(self.split).value,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_json_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "binding_digest": self.digest}

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> InstrumentDatasetBinding:
        if not isinstance(value, Mapping):
            raise ValueError("instrument dataset binding must be a mapping")
        payload = dict(value)
        _require_exact_fields(
            payload,
            expected=_DATASET_BINDING_FIELDS,
            field="instrument dataset binding",
        )
        if payload["schema_version"] != INSTRUMENT_DATASET_BINDING_SCHEMA:
            raise ValueError("instrument dataset binding schema mismatch")
        observed_digest = _require_digest(
            payload["binding_digest"],
            field="binding_digest",
        )
        binding = cls(
            concrete_symbol=payload["concrete_symbol"],  # type: ignore[arg-type]
            source_dataset_id=payload["source_dataset_id"],  # type: ignore[arg-type]
            symbol_dataset_digest=payload["symbol_dataset_digest"],  # type: ignore[arg-type]
            execution_metadata_digest=payload["execution_metadata_digest"],  # type: ignore[arg-type]
            instrument_descriptor_digest=payload["instrument_descriptor_digest"],  # type: ignore[arg-type]
            partition_digest=payload["partition_digest"],  # type: ignore[arg-type]
            split=payload["split"],  # type: ignore[arg-type]
        )
        if observed_digest != binding.digest:
            raise ValueError("instrument dataset binding digest mismatch")
        return binding


@dataclass(frozen=True, slots=True)
class InstrumentRoute:
    """One deterministic route resolved from completed-episode count."""

    binding: InstrumentDatasetBinding
    router_digest: str
    environment_index: int
    completed_episode_count: int
    routing_cycle: int
    routing_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.binding, InstrumentDatasetBinding):
            raise TypeError("binding must be InstrumentDatasetBinding")
        _require_digest(self.router_digest, field="router_digest")
        for field_name in (
            "environment_index",
            "completed_episode_count",
            "routing_cycle",
            "routing_position",
        ):
            _require_non_negative_integer(getattr(self, field_name), field=field_name)


@dataclass(frozen=True, slots=True)
class DeterministicBalancedInstrumentRouter:
    """Stateless hash-ranked permutations with one visit per symbol per cycle."""

    bindings: tuple[InstrumentDatasetBinding, ...]
    run_seed: int
    environment_index: int
    partition_digest: str

    def __init__(
        self,
        bindings: Sequence[InstrumentDatasetBinding],
        *,
        run_seed: int,
        environment_index: int,
        partition_digest: str,
    ) -> None:
        object.__setattr__(self, "bindings", tuple(bindings))
        object.__setattr__(self, "run_seed", run_seed)
        object.__setattr__(self, "environment_index", environment_index)
        object.__setattr__(self, "partition_digest", partition_digest)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.bindings:
            raise ValueError("router bindings must not be empty")
        if any(
            not isinstance(item, InstrumentDatasetBinding) for item in self.bindings
        ):
            raise TypeError("router bindings must be InstrumentDatasetBinding values")
        run_seed = _require_non_negative_integer(self.run_seed, field="run_seed")
        environment_index = _require_non_negative_integer(
            self.environment_index,
            field="environment_index",
        )
        partition_digest = _require_digest(
            self.partition_digest,
            field="partition_digest",
        )
        if any(
            binding.split is not InstrumentDatasetSplit.TRAIN
            for binding in self.bindings
        ):
            raise ValueError("training router accepts train bindings only")
        if any(
            binding.partition_digest != partition_digest for binding in self.bindings
        ):
            raise ValueError("router binding partition digest mismatch")
        symbols = tuple(binding.concrete_symbol for binding in self.bindings)
        if len(set(symbols)) != len(symbols):
            raise ValueError("router concrete symbols must be unique")
        binding_digests = tuple(binding.digest for binding in self.bindings)
        if len(set(binding_digests)) != len(binding_digests):
            raise ValueError("router binding digests must be unique")
        ordered = tuple(sorted(self.bindings, key=lambda item: item.concrete_symbol))
        object.__setattr__(self, "bindings", ordered)
        object.__setattr__(self, "run_seed", run_seed)
        object.__setattr__(self, "environment_index", environment_index)
        object.__setattr__(self, "partition_digest", partition_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": DETERMINISTIC_BALANCED_INSTRUMENT_ROUTER_SCHEMA,
            "run_seed": self.run_seed,
            "environment_index": self.environment_index,
            "partition_digest": self.partition_digest,
            "binding_digests": tuple(binding.digest for binding in self.bindings),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_json_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "router_digest": self.digest}

    def _permutation(self, routing_cycle: int) -> tuple[InstrumentDatasetBinding, ...]:
        cycle = _require_non_negative_integer(routing_cycle, field="routing_cycle")

        def score(binding: InstrumentDatasetBinding) -> tuple[str, str]:
            ranking_digest = content_digest(
                {
                    "schema_version": _DETERMINISTIC_PERMUTATION_SCHEMA,
                    "run_seed": self.run_seed,
                    "environment_index": self.environment_index,
                    "partition_digest": self.partition_digest,
                    "routing_cycle": cycle,
                    "binding_digest": binding.digest,
                }
            )
            return ranking_digest, binding.concrete_symbol

        return tuple(sorted(self.bindings, key=score))

    def route(self, completed_episode_count: int) -> InstrumentRoute:
        count = _require_non_negative_integer(
            completed_episode_count,
            field="completed_episode_count",
        )
        routing_cycle, routing_position = divmod(count, len(self.bindings))
        permutation = self._permutation(routing_cycle)
        return InstrumentRoute(
            binding=permutation[routing_position],
            router_digest=self.digest,
            environment_index=self.environment_index,
            completed_episode_count=count,
            routing_cycle=routing_cycle,
            routing_position=routing_position,
        )


_EPISODE_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "concrete_symbol",
        "source_dataset_id",
        "symbol_dataset_digest",
        "execution_metadata_digest",
        "instrument_descriptor_digest",
        "partition_digest",
        "split",
        "dataset_binding_digest",
        "router_digest",
        "environment_index",
        "completed_episode_count",
        "routing_cycle",
        "routing_position",
        "episode_start",
        "episode_stop",
        "episode_binding_digest",
    }
)


@dataclass(frozen=True, slots=True)
class InstrumentEpisodeBinding:
    """Concrete dataset identity and routing coordinates for one episode."""

    dataset_binding: InstrumentDatasetBinding
    router_digest: str
    environment_index: int
    completed_episode_count: int
    routing_cycle: int
    routing_position: int
    episode_start: int
    episode_stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_binding, InstrumentDatasetBinding):
            raise TypeError("dataset_binding must be InstrumentDatasetBinding")
        if self.dataset_binding.split is not InstrumentDatasetSplit.TRAIN:
            raise ValueError("episode binding requires a train dataset binding")
        _require_digest(self.router_digest, field="router_digest")
        for field_name in (
            "environment_index",
            "completed_episode_count",
            "routing_cycle",
            "routing_position",
            "episode_start",
            "episode_stop",
        ):
            _require_non_negative_integer(getattr(self, field_name), field=field_name)
        if self.episode_stop <= self.episode_start:
            raise ValueError("episode_stop must be greater than episode_start")

    @classmethod
    def from_route(
        cls,
        route: InstrumentRoute,
        *,
        episode_start: int,
        episode_stop: int,
    ) -> InstrumentEpisodeBinding:
        if not isinstance(route, InstrumentRoute):
            raise TypeError("route must be InstrumentRoute")
        return cls(
            dataset_binding=route.binding,
            router_digest=route.router_digest,
            environment_index=route.environment_index,
            completed_episode_count=route.completed_episode_count,
            routing_cycle=route.routing_cycle,
            routing_position=route.routing_position,
            episode_start=episode_start,
            episode_stop=episode_stop,
        )

    def digest_payload(self) -> dict[str, object]:
        binding = self.dataset_binding
        return {
            "schema_version": INSTRUMENT_EPISODE_BINDING_SCHEMA,
            "concrete_symbol": binding.concrete_symbol,
            "source_dataset_id": binding.source_dataset_id,
            "symbol_dataset_digest": binding.symbol_dataset_digest,
            "execution_metadata_digest": binding.execution_metadata_digest,
            "instrument_descriptor_digest": binding.instrument_descriptor_digest,
            "partition_digest": binding.partition_digest,
            "split": InstrumentDatasetSplit(binding.split).value,
            "dataset_binding_digest": binding.digest,
            "router_digest": self.router_digest,
            "environment_index": self.environment_index,
            "completed_episode_count": self.completed_episode_count,
            "routing_cycle": self.routing_cycle,
            "routing_position": self.routing_position,
            "episode_start": self.episode_start,
            "episode_stop": self.episode_stop,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_json_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "episode_binding_digest": self.digest}

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> InstrumentEpisodeBinding:
        if not isinstance(value, Mapping):
            raise ValueError("instrument episode binding must be a mapping")
        payload = dict(value)
        _require_exact_fields(
            payload,
            expected=_EPISODE_BINDING_FIELDS,
            field="instrument episode binding",
        )
        if payload["schema_version"] != INSTRUMENT_EPISODE_BINDING_SCHEMA:
            raise ValueError("instrument episode binding schema mismatch")
        dataset_binding = InstrumentDatasetBinding(
            concrete_symbol=payload["concrete_symbol"],  # type: ignore[arg-type]
            source_dataset_id=payload["source_dataset_id"],  # type: ignore[arg-type]
            symbol_dataset_digest=payload["symbol_dataset_digest"],  # type: ignore[arg-type]
            execution_metadata_digest=payload["execution_metadata_digest"],  # type: ignore[arg-type]
            instrument_descriptor_digest=payload["instrument_descriptor_digest"],  # type: ignore[arg-type]
            partition_digest=payload["partition_digest"],  # type: ignore[arg-type]
            split=payload["split"],  # type: ignore[arg-type]
        )
        observed_dataset_digest = _require_digest(
            payload["dataset_binding_digest"],
            field="dataset_binding_digest",
        )
        if observed_dataset_digest != dataset_binding.digest:
            raise ValueError("instrument episode dataset binding digest mismatch")
        episode = cls(
            dataset_binding=dataset_binding,
            router_digest=payload["router_digest"],  # type: ignore[arg-type]
            environment_index=payload["environment_index"],  # type: ignore[arg-type]
            completed_episode_count=payload["completed_episode_count"],  # type: ignore[arg-type]
            routing_cycle=payload["routing_cycle"],  # type: ignore[arg-type]
            routing_position=payload["routing_position"],  # type: ignore[arg-type]
            episode_start=payload["episode_start"],  # type: ignore[arg-type]
            episode_stop=payload["episode_stop"],  # type: ignore[arg-type]
        )
        observed_episode_digest = _require_digest(
            payload["episode_binding_digest"],
            field="episode_binding_digest",
        )
        if observed_episode_digest != episode.digest:
            raise ValueError("instrument episode binding digest mismatch")
        return episode


__all__ = [
    "DETERMINISTIC_BALANCED_INSTRUMENT_ROUTER_SCHEMA",
    "GENERIC_INSTRUMENT_ACTION_NAMES",
    "GENERIC_INSTRUMENT_SYMBOL",
    "GENERIC_INSTRUMENT_SYMBOLS",
    "INSTRUMENT_DATASET_BINDING_SCHEMA",
    "INSTRUMENT_EPISODE_BINDING_SCHEMA",
    "DeterministicBalancedInstrumentRouter",
    "InstrumentDatasetBinding",
    "InstrumentDatasetSplit",
    "InstrumentEpisodeBinding",
    "InstrumentRoute",
]
