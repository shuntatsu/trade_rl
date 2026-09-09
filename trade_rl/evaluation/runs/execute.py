"""In-memory execution boundary for one resolved candidate run."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import UniversalStrategyComparison
from trade_rl.evaluation.runs.candidate_suite import run_lean_candidate_suite
from trade_rl.evaluation.runs.config import ResolvedCandidateRunSpec


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """One resolved in-memory candidate-suite result before publication."""

    spec: ResolvedCandidateRunSpec
    symbols: tuple[str, ...]
    comparison: UniversalStrategyComparison


def execute_candidate_run(
    dataset: MarketDataset,
    spec: ResolvedCandidateRunSpec,
) -> CandidateRunResult:
    """Execute the existing candidate suite against one resolved run spec."""

    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("dataset id does not match resolved candidate run spec")
    comparison = run_lean_candidate_suite(
        dataset,
        spec.lean_config,
        start_index=spec.evaluation_start_index,
        stop_index=spec.evaluation_stop_index,
        gross_budget=spec.config.gross_budget,
        initial_capital=spec.config.initial_capital,
        execution_cost=None,
        risk=None,
    )
    return CandidateRunResult(
        spec=spec,
        symbols=tuple(dataset.symbols),
        comparison=comparison,
    )


__all__ = ["CandidateRunResult", "execute_candidate_run"]
