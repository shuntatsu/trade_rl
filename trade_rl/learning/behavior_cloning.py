"""Framework-neutral behavior-cloning configuration and result contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest


class ObservationBatchProvider(Protocol):
    sample_count: int

    def get(self, indices: np.ndarray) -> object: ...


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    epochs: int = 15
    learning_rate: float = 1e-3
    batch_size: int = 256
    validation_fraction: float = 0.0
    early_stopping_patience: int = 3
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("early_stopping_patience", self.early_stopping_patience),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.validation_fraction)
            or not 0.0 <= self.validation_fraction < 0.5
        ):
            raise ValueError("validation_fraction must be within [0, 0.5)")
        if (
            not math.isfinite(self.minimum_improvement)
            or self.minimum_improvement < 0.0
        ):
            raise ValueError("minimum_improvement must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BehaviorCloningResult:
    initial_mse: float
    final_mse: float
    sample_count: int
    observation_digest: str
    action_digest: str
    teacher_config_digest: str
    config: BehaviorCloningConfig
    seed: int
    validation_mse: float | None = None
    validation_sample_count: int = 0
    best_epoch: int = 0

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "action_digest": self.action_digest,
                "best_epoch": self.best_epoch,
                "config": asdict(self.config),
                "final_mse": self.final_mse,
                "initial_mse": self.initial_mse,
                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v2",
                "seed": self.seed,
                "teacher_config_digest": self.teacher_config_digest,
                "validation_mse": self.validation_mse,
                "validation_sample_count": self.validation_sample_count,
            }
        )


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "ObservationBatchProvider",
]
