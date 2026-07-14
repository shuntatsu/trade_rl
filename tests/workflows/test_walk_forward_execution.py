from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from trade_rl.domain.selection import PolicyMode, SelectionDecision
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult
from trade_rl.workflows.fold_runner import FoldExecutionResult
from trade_rl.workflows.walk_forward import (
    WalkForwardWorkflowConfig,
    execute_walk_forward,
)

DATASET_ID = "a" * 64
SIGNAL_DIGEST = "b" * 64
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


@dataclass
class DetailedRunner:
    dataset_id: str = DATASET_ID
    calls: list[WalkForwardFold] = field(default_factory=list)

    def run_fold(self, fold: WalkForwardFold) -> FoldExecutionResult:
        self.calls.append(fold)
        policy_digest = f"{fold.fold_index + 1:064x}"
        selection_digest = f"{100 + fold.fold_index:064x}"
        test_digest = f"{200 + fold.fold_index:064x}"
        selected = FoldOOSResult(
            fold_index=fold.fold_index,
            start=fold.test.start,
            stop=fold.test.stop,
            returns=ReturnSeries(
                values=tuple(0.002 for _ in range(fold.test.size)),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=8_760,
            ),
        )
        baseline = FoldOOSResult(
            fold_index=fold.fold_index,
            start=fold.test.start,
            stop=fold.test.stop,
            returns=ReturnSeries(
                values=tuple(0.001 for _ in range(fold.test.size)),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=8_760,
            ),
        )
        return FoldExecutionResult(
            dataset_id=self.dataset_id,
            fold_index=fold.fold_index,
            selection=SelectionDecision(
                dataset_id=self.dataset_id,
                mode=PolicyMode.RESIDUAL_POLICY,
                selected_configuration=f"candidate_{fold.fold_index}",
                selected_policy_digest=policy_digest,
                signal_digest=SIGNAL_DIGEST,
                evaluation_digest=selection_digest,
                selected_at=NOW,
                reasons=("candidate exceeded baseline",),
            ),
            selection_evaluation_digest=selection_digest,
            test_evaluation_digest=test_digest,
            selected_oos=selected,
            baseline_oos=baseline,
        )


def workflow_config() -> WalkForwardWorkflowConfig:
    return WalkForwardWorkflowConfig(
        n_bars=220,
        train_bars=80,
        checkpoint_bars=10,
        selection_bars=10,
        test_bars=20,
        purge_bars=2,
        step_bars=20,
        max_folds=2,
    )


def test_execute_walk_forward_stitches_selected_and_baseline_evidence() -> None:
    runner = DetailedRunner()

    result = execute_walk_forward(
        workflow_config(),
        dataset_id=DATASET_ID,
        runner=runner,
    )

    assert len(runner.calls) == 2
    assert result.dataset_id == DATASET_ID
    assert result.selected_stitched.fold_indices == (0, 1)
    assert result.baseline_stitched.fold_indices == (0, 1)
    assert result.selected_metrics is None
    assert result.baseline_metrics is None
    assert result.selected_independent_summary.fold_count == 2
    assert result.baseline_independent_summary.fold_count == 2
    assert (
        result.selected_independent_summary.mean_fold_return
        > result.baseline_independent_summary.mean_fold_return
    )
    assert len(result.evaluation_digest) == 64


def test_execute_walk_forward_rejects_fold_dataset_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset identity"):
        execute_walk_forward(
            workflow_config(),
            dataset_id=DATASET_ID,
            runner=DetailedRunner(dataset_id="f" * 64),
        )
