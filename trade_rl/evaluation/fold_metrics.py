"""Statistics for independently reset outer-OOS folds."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median

from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult


@dataclass(frozen=True, slots=True)
class IndependentFoldSummary:
    fold_count: int
    mean_fold_return: float
    median_fold_return: float
    worst_fold_return: float
    best_fold_return: float
    positive_fold_rate: float
    turnover_total: float
    total_cost: float
    funding_pnl: float
    borrow_cost: float
    n_trades: int
    rebalance_events: int

    def continuous_max_drawdown(self) -> float:
        raise ValueError(
            "continuous-account drawdown is undefined for independently reset folds"
        )

    def continuous_total_return(self) -> float:
        raise ValueError(
            "continuous-account total return is undefined for independently reset folds"
        )


def summarize_independent_folds(
    results: tuple[FoldOOSResult, ...],
) -> IndependentFoldSummary:
    if not results:
        raise ValueError("at least one independent fold is required")
    fold_returns = tuple(compound_return(result.returns.values) for result in results)
    return IndependentFoldSummary(
        fold_count=len(results),
        mean_fold_return=fmean(fold_returns),
        median_fold_return=float(median(fold_returns)),
        worst_fold_return=min(fold_returns),
        best_fold_return=max(fold_returns),
        positive_fold_rate=sum(value > 0.0 for value in fold_returns)
        / len(fold_returns),
        turnover_total=sum(result.diagnostics.turnover_total for result in results),
        total_cost=sum(result.diagnostics.total_cost for result in results),
        funding_pnl=sum(result.diagnostics.funding_pnl for result in results),
        borrow_cost=sum(result.diagnostics.borrow_cost for result in results),
        n_trades=sum(result.diagnostics.n_trades for result in results),
        rebalance_events=sum(result.diagnostics.rebalance_events for result in results),
    )


__all__ = ["IndependentFoldSummary", "summarize_independent_folds"]
