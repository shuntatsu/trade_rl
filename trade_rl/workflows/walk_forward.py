"""Application workflow for pure fold planning, execution, and OOS stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.fold_metrics import (
    IndependentFoldSummary,
    summarize_independent_folds,
)
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.walk_forward.folds import WalkForwardFold, build_folds
from trade_rl.evaluation.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    StitchMode,
    stitch_oos,
)
from trade_rl.workflows.fold_runner import FoldExecutionResult


class FoldRunner(Protocol):
    """Application boundary for fold-local training and sealed OOS execution."""

    def run_fold(self, fold: WalkForwardFold) -> FoldOOSResult: ...


class DetailedFoldRunner(Protocol):
    """Boundary for identity-bound selection and paired sealed OOS execution."""

    def run_fold(self, fold: WalkForwardFold) -> FoldExecutionResult: ...


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
    stitch_mode: StitchMode = StitchMode.INDEPENDENT_FOLDS

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


@dataclass(frozen=True, slots=True)
class WalkForwardExecutionResult:
    """Selected and baseline outer-OOS evidence from concrete nested folds."""

    dataset_id: str
    folds: tuple[WalkForwardFold, ...]
    fold_results: tuple[FoldExecutionResult, ...]
    selected_stitched: StitchedOOS
    baseline_stitched: StitchedOOS
    selected_metrics: PerformanceMetrics | None
    baseline_metrics: PerformanceMetrics | None
    selected_independent_summary: IndependentFoldSummary | None
    baseline_independent_summary: IndependentFoldSummary | None
    evaluation_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.evaluation_digest, field="evaluation_digest")
        if len(self.folds) != len(self.fold_results):
            raise ValueError("fold plan and detailed result counts must match")


def _validate_fold_result(
    fold: WalkForwardFold,
    result: FoldOOSResult,
) -> None:
    if result.fold_index != fold.fold_index:
        raise ValueError("fold identity mismatch between plan and runner result")
    if (result.start, result.stop) != (fold.test.start, fold.test.stop):
        raise ValueError("fold OOS range mismatch between plan and runner result")


def _validate_detailed_fold_result(
    *,
    dataset_id: str,
    fold: WalkForwardFold,
    result: FoldExecutionResult,
) -> None:
    if result.dataset_id != dataset_id:
        raise ValueError("fold dataset identity mismatch")
    if result.fold_index != fold.fold_index:
        raise ValueError("fold identity mismatch between plan and detailed result")
    _validate_fold_result(fold, result.selected_oos)
    _validate_fold_result(fold, result.baseline_oos)


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
    stitched = stitch_oos(tuple(results), mode=config.stitch_mode)
    return WalkForwardWorkflowResult(
        folds=folds,
        fold_results=tuple(results),
        stitched=stitched,
        metrics=evaluate_performance(stitched.returns),
    )


def execute_walk_forward(
    config: WalkForwardWorkflowConfig,
    *,
    dataset_id: str,
    runner: DetailedFoldRunner,
) -> WalkForwardExecutionResult:
    """Execute concrete nested folds and preserve selected/baseline OOS identity."""

    require_sha256(dataset_id, field="dataset_id")
    folds = config.build_folds()
    detailed_results: list[FoldExecutionResult] = []
    selected_results: list[FoldOOSResult] = []
    baseline_results: list[FoldOOSResult] = []
    for fold in folds:
        result = runner.run_fold(fold)
        _validate_detailed_fold_result(
            dataset_id=dataset_id,
            fold=fold,
            result=result,
        )
        detailed_results.append(result)
        selected_results.append(result.selected_oos)
        baseline_results.append(result.baseline_oos)

    selected_stitched = stitch_oos(
        tuple(selected_results),
        mode=config.stitch_mode,
    )
    baseline_stitched = stitch_oos(
        tuple(baseline_results),
        mode=config.stitch_mode,
    )
    evaluation_digest = content_digest(
        {
            "dataset_id": dataset_id,
            "folds": tuple(
                {
                    "fold_index": result.fold_index,
                    "selected_configuration": (result.selection.selected_configuration),
                    "selected_policy_digest": (result.selection.selected_policy_digest),
                    "selection_evaluation_digest": (result.selection_evaluation_digest),
                    "test_evaluation_digest": result.test_evaluation_digest,
                }
                for result in detailed_results
            ),
            "baseline_diagnostics": baseline_stitched.diagnostics.digest_payload(),
            "schema_version": "walk_forward_execution_v2",
            "selected_diagnostics": selected_stitched.diagnostics.digest_payload(),
            "stitch_mode": config.stitch_mode.value,
        }
    )
    independent = config.stitch_mode is StitchMode.INDEPENDENT_FOLDS
    return WalkForwardExecutionResult(
        dataset_id=dataset_id,
        folds=folds,
        fold_results=tuple(detailed_results),
        selected_stitched=selected_stitched,
        baseline_stitched=baseline_stitched,
        selected_metrics=(
            None
            if independent
            else evaluate_performance(
                selected_stitched.returns,
                turnover_total=selected_stitched.diagnostics.turnover_total,
                total_cost=selected_stitched.diagnostics.total_cost,
                funding_pnl=selected_stitched.diagnostics.funding_pnl,
                borrow_cost=selected_stitched.diagnostics.borrow_cost,
                n_trades=selected_stitched.diagnostics.n_trades,
                rebalance_events=selected_stitched.diagnostics.rebalance_events,
                termination_count=selected_stitched.diagnostics.termination_count,
            )
        ),
        baseline_metrics=(
            None
            if independent
            else evaluate_performance(
                baseline_stitched.returns,
                turnover_total=baseline_stitched.diagnostics.turnover_total,
                total_cost=baseline_stitched.diagnostics.total_cost,
                funding_pnl=baseline_stitched.diagnostics.funding_pnl,
                borrow_cost=baseline_stitched.diagnostics.borrow_cost,
                n_trades=baseline_stitched.diagnostics.n_trades,
                rebalance_events=baseline_stitched.diagnostics.rebalance_events,
                termination_count=baseline_stitched.diagnostics.termination_count,
            )
        ),
        selected_independent_summary=(
            summarize_independent_folds(tuple(selected_results))
            if independent
            else None
        ),
        baseline_independent_summary=(
            summarize_independent_folds(tuple(baseline_results))
            if independent
            else None
        ),
        evaluation_digest=evaluation_digest,
    )
