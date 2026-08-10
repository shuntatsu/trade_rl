"""Framework-neutral behavior-cloning configuration and result contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.hierarchical_bc_metrics import (
    HierarchicalBehaviorCloningLosses,
    HierarchicalBehaviorCloningMetrics,
)


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
    gate_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    composed_loss_weight: float = 1.0
    max_positive_class_weight: float = 20.0
    gate_prediction_threshold: float = 0.5

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
        weights = (
            self.gate_loss_weight,
            self.target_loss_weight,
            self.composed_loss_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError(
                "hierarchical loss weights must be finite and non-negative"
            )
        if sum(weights) <= 0.0:
            raise ValueError("at least one hierarchical loss weight must be positive")
        if (
            not math.isfinite(self.max_positive_class_weight)
            or self.max_positive_class_weight < 1.0
        ):
            raise ValueError(
                "max_positive_class_weight must be finite and at least one"
            )
        if (
            not math.isfinite(self.gate_prediction_threshold)
            or not 0.0 < self.gate_prediction_threshold < 1.0
        ):
            raise ValueError("gate_prediction_threshold must be within (0, 1)")


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
    hierarchical_label_digest: str | None = None
    initial_hierarchical_losses: HierarchicalBehaviorCloningLosses | None = None
    final_hierarchical_losses: HierarchicalBehaviorCloningLosses | None = None
    validation_hierarchical_losses: HierarchicalBehaviorCloningLosses | None = None
    initial_hierarchical_metrics: HierarchicalBehaviorCloningMetrics | None = None
    final_hierarchical_metrics: HierarchicalBehaviorCloningMetrics | None = None
    validation_hierarchical_metrics: HierarchicalBehaviorCloningMetrics | None = None
    training_sample_count: int | None = None
    excluded_sample_count: int = 0
    split_digest: str | None = None

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "action_digest": self.action_digest,
                "best_epoch": self.best_epoch,
                "config": asdict(self.config),
                "final_hierarchical_losses": (
                    None
                    if self.final_hierarchical_losses is None
                    else asdict(self.final_hierarchical_losses)
                ),
                "final_hierarchical_metrics": (
                    None
                    if self.final_hierarchical_metrics is None
                    else asdict(self.final_hierarchical_metrics)
                ),
                "excluded_sample_count": self.excluded_sample_count,
                "final_mse": self.final_mse,
                "hierarchical_label_digest": self.hierarchical_label_digest,
                "initial_hierarchical_losses": (
                    None
                    if self.initial_hierarchical_losses is None
                    else asdict(self.initial_hierarchical_losses)
                ),
                "initial_hierarchical_metrics": (
                    None
                    if self.initial_hierarchical_metrics is None
                    else asdict(self.initial_hierarchical_metrics)
                ),
                "initial_mse": self.initial_mse,
                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v4",
                "seed": self.seed,
                "split_digest": self.split_digest,
                "teacher_config_digest": self.teacher_config_digest,
                "training_sample_count": self.training_sample_count,
                "validation_hierarchical_losses": (
                    None
                    if self.validation_hierarchical_losses is None
                    else asdict(self.validation_hierarchical_losses)
                ),
                "validation_hierarchical_metrics": (
                    None
                    if self.validation_hierarchical_metrics is None
                    else asdict(self.validation_hierarchical_metrics)
                ),
                "validation_mse": self.validation_mse,
                "validation_sample_count": self.validation_sample_count,
            }
        )


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "ObservationBatchProvider",
]
