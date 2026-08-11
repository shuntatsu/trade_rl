from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        if not self.sample_indices:
            raise ValueError("sample_indices must not be empty")
        for symbol, indices in self.sample_indices.items():
            if not symbol or not indices:
                raise ValueError("every symbol must have at least one sample index")

    def batch(
        self, *, batch_size: int, batch_index: int
    ) -> tuple[tuple[str, int], ...]:
        symbols = tuple(sorted(self.sample_indices))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size % len(symbols) != 0:
            raise ValueError("batch_size must be divisible by the symbol count")
        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        per_symbol = batch_size // len(symbols)
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


def _stable_seed(seed: int, batch_index: int, symbol: str) -> int:
    token = f"{seed}:{batch_index}:{symbol}".encode()
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
        artifact_digest = _digest(
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
