"""Application workflow for pure fold planning, execution, and OOS stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.walk_forward.folds import WalkForwardFold, build_folds
from trade_rl.evaluation.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    stitch_oos,
)


class FoldRunner(Protocol):
    """Application boundary for fold-local training and sealed OOS execution."""

    def run_fold(self, fold: WalkForwardFold) -> FoldOOSResult: ...


@dataclass(frozen=True, slots=True)
class WalkForwardWorkflowConfig:
    n_bars: int
    train_bars: int
    checkpoint_bars: int
    selection_bars: int
    test_bars: int
    purge_bars: int
    step_bars: int | None = None
    max_folds: int | None = None
    expanding_train: bool = True

    def build_folds(self) -> tuple[WalkForwardFold, ...]:
        return build_folds(
            n_bars=self.n_bars,
            train_bars=self.train_bars,
            checkpoint_bars=self.checkpoint_bars,
            selection_bars=self.selection_bars,
            test_bars=self.test_bars,
            purge_bars=self.purge_bars,
            step_bars=self.step_bars,
            max_folds=self.max_folds,
            expanding_train=self.expanding_train,
        )


@dataclass(frozen=True, slots=True)
class WalkForwardWorkflowResult:
    folds: tuple[WalkForwardFold, ...]
    fold_results: tuple[FoldOOSResult, ...]
    stitched: StitchedOOS
    metrics: PerformanceMetrics


def _validate_fold_result(
    fold: WalkForwardFold,
    result: FoldOOSResult,
) -> None:
    if result.fold_index != fold.fold_index:
        raise ValueError("fold identity mismatch between plan and runner result")
    if (result.start, result.stop) != (fold.test.start, fold.test.stop):
        raise ValueError("fold OOS range mismatch between plan and runner result")


def run_walk_forward(
    config: WalkForwardWorkflowConfig,
    *,
    runner: FoldRunner,
) -> WalkForwardWorkflowResult:
    """Run fold-local work without coupling pure evaluation to training code."""

    folds = config.build_folds()
    results: list[FoldOOSResult] = []
    for fold in folds:
        result = runner.run_fold(fold)
        _validate_fold_result(fold, result)
        results.append(result)
    stitched = stitch_oos(tuple(results))
    return WalkForwardWorkflowResult(
        folds=folds,
        fold_results=tuple(results),
        stitched=stitched,
        metrics=evaluate_performance(stitched.returns),
    )
