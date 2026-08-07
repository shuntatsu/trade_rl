"""Stage-scoped construction of causal signal providers for walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from trade_rl.domain.common import require_sha256
from trade_rl.integrations.signal_artifacts import (
    LoadedAlphaArtifact,
    LoadedFactorArtifact,
    load_alpha_artifact,
    load_factor_artifact,
)


class SignalStage(StrEnum):
    TRAIN = "train"
    CHECKPOINT = "checkpoint"
    SELECTION = "selection"
    TEST = "test"


class SignalRun(Protocol):
    alpha_artifact: Path | None
    factor_artifact: Path | None


@dataclass(frozen=True, slots=True)
class FoldSignals:
    stage: SignalStage
    alpha: LoadedAlphaArtifact | None
    factor: LoadedFactorArtifact | None


@dataclass(frozen=True, slots=True)
class FoldSignalProviderFactory:
    dataset_id: str
    expected_symbols: tuple[str, ...] | None = None
    expected_factor_names: tuple[str, ...] | None = None
    expected_symbol_count: int | None = None

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")

    @classmethod
    def trend_only(cls, *, dataset_id: str) -> FoldSignalProviderFactory:
        return cls(dataset_id=dataset_id)

    def build(
        self,
        *,
        stage: SignalStage,
        run: SignalRun,
        evaluation_start: int | None = None,
    ) -> FoldSignals:
        if stage is not SignalStage.TRAIN and evaluation_start is None:
            raise ValueError("non-training signal stages require evaluation_start")
        strict_start = None if stage is SignalStage.TRAIN else evaluation_start
        alpha = (
            None
            if run.alpha_artifact is None
            else load_alpha_artifact(
                run.alpha_artifact,
                dataset_id=self.dataset_id,
                evaluation_start=strict_start,
                expected_symbols=self.expected_symbols,
            )
        )
        factor = (
            None
            if run.factor_artifact is None
            else load_factor_artifact(
                run.factor_artifact,
                dataset_id=self.dataset_id,
                evaluation_start=strict_start,
                expected_names=self.expected_factor_names,
                expected_symbols=self.expected_symbol_count,
            )
        )
        return FoldSignals(stage=stage, alpha=alpha, factor=factor)


__all__ = [
    "FoldSignalProviderFactory",
    "FoldSignals",
    "SignalStage",
]
