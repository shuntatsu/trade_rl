from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult
from trade_rl.workflows.walk_forward import (
    WalkForwardWorkflowConfig,
    run_walk_forward,
)


@dataclass
class FakeRunner:
    calls: list[WalkForwardFold] = field(default_factory=list)

    def run_fold(self, fold: WalkForwardFold) -> FoldOOSResult:
        self.calls.append(fold)
        values = tuple(0.001 * (fold.fold_index + 1) for _ in range(fold.test.size))
        return FoldOOSResult(
            fold_index=fold.fold_index,
            start=fold.test.start,
            stop=fold.test.stop,
            returns=ReturnSeries(
                values=values,
                kind=ReturnKind.BASE_BAR,
                periods_per_year=8_760,
            ),
        )


def test_workflow_builds_runs_and_stitches_folds() -> None:
    runner = FakeRunner()
    config = WalkForwardWorkflowConfig(
        n_bars=260,
        train_bars=80,
        checkpoint_bars=10,
        selection_bars=10,
        test_bars=20,
        purge_bars=2,
        step_bars=20,
        max_folds=3,
    )

    result = run_walk_forward(config, runner=runner)

    assert len(runner.calls) == 3
    assert result.stitched.fold_indices == (0, 1, 2)
    assert result.metrics.n_periods == 60
    assert result.metrics.return_kind is ReturnKind.BASE_BAR
    assert result.metrics.total_return > 0.0


def test_workflow_rejects_runner_output_for_the_wrong_fold() -> None:
    class WrongRunner:
        def run_fold(self, fold: WalkForwardFold) -> FoldOOSResult:
            return FoldOOSResult(
                fold_index=fold.fold_index + 1,
                start=fold.test.start,
                stop=fold.test.stop,
                returns=ReturnSeries(
                    values=tuple(0.0 for _ in range(fold.test.size)),
                    kind=ReturnKind.BASE_BAR,
                    periods_per_year=8_760,
                ),
            )

    with pytest.raises(ValueError, match="fold identity"):
        run_walk_forward(
            WalkForwardWorkflowConfig(
                n_bars=180,
                train_bars=80,
                checkpoint_bars=10,
                selection_bars=10,
                test_bars=20,
                purge_bars=2,
                max_folds=1,
            ),
            runner=WrongRunner(),
        )
