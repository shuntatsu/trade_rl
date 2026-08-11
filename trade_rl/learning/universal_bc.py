from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from trade_rl.artifacts.hashing import content_digest


class CriticWarmStartPhase(str, Enum):
    CRITIC_ONLY = "critic_only"
    JOINT_FINE_TUNE = "joint_fine_tune"

    @classmethod
    def ordered(cls) -> tuple["CriticWarmStartPhase", ...]:
        return (cls.CRITIC_ONLY, cls.JOINT_FINE_TUNE)


@dataclass(frozen=True)
class SymbolBalancedBatchSampler:
    sample_indices: Mapping[str, tuple[int, ...]]
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not self.sample_indices:
            raise ValueError("sample_indices must not be empty")
        normalized: dict[str, tuple[int, ...]] = {}
        for symbol, raw_indices in self.sample_indices.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("every symbol must be non-empty")
            indices = tuple(raw_indices)
            if not indices:
                raise ValueError("every symbol must have at least one sample index")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            ):
                raise ValueError("sample indices must be non-negative integers")
            if len(set(indices)) != len(indices):
                raise ValueError("sample indices must be unique within each symbol")
            normalized[symbol] = indices
        object.__setattr__(self, "sample_indices", normalized)

    def _batch_shape(self, batch_size: int) -> tuple[tuple[str, ...], int]:
        symbols = tuple(sorted(self.sample_indices))
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be positive")
        if batch_size % len(symbols) != 0:
            raise ValueError("batch_size must be divisible by the symbol count")
        return symbols, batch_size // len(symbols)

    def batch(
        self, *, batch_size: int, batch_index: int
    ) -> tuple[tuple[str, int], ...]:
        symbols, per_symbol = self._batch_shape(batch_size)
        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or batch_index < 0
        ):
            raise ValueError("batch_index must be non-negative")
        result: list[tuple[str, int]] = []
        for symbol in symbols:
            indices = list(self.sample_indices[symbol])
            rng = random.Random(_stable_seed(self.seed, batch_index, symbol))
            rng.shuffle(indices)
            if per_symbol > len(indices):
                repeats, remainder = divmod(per_symbol, len(indices))
                selected = indices * repeats + indices[:remainder]
            else:
                selected = indices[:per_symbol]
            result.extend((symbol, index) for index in selected)
        return tuple(result)

    def epoch_batches(
        self, *, batch_size: int, epoch: int
    ) -> tuple[tuple[int, ...], ...]:
        """Return deterministic equal-symbol batches covering every sample each epoch."""

        symbols, per_symbol = self._batch_shape(batch_size)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        maximum_symbol_count = max(
            len(self.sample_indices[symbol]) for symbol in symbols
        )
        batch_count = math.ceil(maximum_symbol_count / per_symbol)
        target_per_symbol = batch_count * per_symbol
        streams: dict[str, tuple[int, ...]] = {}
        for symbol in symbols:
            stream: list[int] = []
            cycle = 0
            while len(stream) < target_per_symbol:
                cycle_indices = list(self.sample_indices[symbol])
                rng = random.Random(_stable_seed(self.seed, epoch, symbol, cycle))
                rng.shuffle(cycle_indices)
                stream.extend(cycle_indices)
                cycle += 1
            streams[symbol] = tuple(stream[:target_per_symbol])

        batches: list[tuple[int, ...]] = []
        for batch_index in range(batch_count):
            start = batch_index * per_symbol
            stop = start + per_symbol
            batch: list[int] = []
            for symbol in symbols:
                batch.extend(streams[symbol][start:stop])
            batches.append(tuple(batch))
        return tuple(batches)


def _stable_seed(seed: int, *parts: object) -> int:
    token = ":".join(str(value) for value in (seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class UniversalTeacherArtifact:
    teacher_digest: str
    train_symbols: tuple[str, ...]
    normalizer_digest: str
    feature_schema_digest: str
    artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        teacher_digest: str,
        train_symbols: Sequence[str],
        teacher_symbols: Sequence[str],
        normalizer_digest: str,
        feature_schema_digest: str,
    ) -> "UniversalTeacherArtifact":
        expected = tuple(train_symbols)
        actual = tuple(teacher_symbols)
        if actual != expected:
            raise ValueError("teacher symbols must equal train symbols")
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("train symbols must be non-empty and unique")
        artifact_digest = content_digest(
            {
                "version": "universal_teacher_artifact_v1",
                "teacher_digest": teacher_digest,
                "train_symbols": expected,
                "normalizer_digest": normalizer_digest,
                "feature_schema_digest": feature_schema_digest,
            }
        )
        return cls(
            teacher_digest=teacher_digest,
            train_symbols=expected,
            normalizer_digest=normalizer_digest,
            feature_schema_digest=feature_schema_digest,
            artifact_digest=artifact_digest,
        )


@dataclass(frozen=True)
class CriticWarmStartPlan:
    critic_only_steps: int
    joint_fine_tune_steps: int
    joint_actor_learning_rate_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.critic_only_steps <= 0 or self.joint_fine_tune_steps <= 0:
            raise ValueError("warm-start phases must have positive step budgets")
        if not 0.0 < self.joint_actor_learning_rate_scale <= 1.0:
            raise ValueError("joint actor learning-rate scale must be in (0, 1]")

    @property
    def phases(self) -> tuple[CriticWarmStartPhase, ...]:
        return CriticWarmStartPhase.ordered()
