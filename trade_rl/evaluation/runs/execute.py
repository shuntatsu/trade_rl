"""In-memory execution boundary for one resolved candidate run."""

from __future__ import annotations

from dataclasses import dataclass, replace

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import UniversalStrategyComparison
from trade_rl.evaluation.runs.candidate_suite import run_lean_candidate_suite
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    LEGACY_DATASET_EXECUTION_OVERLAY,
    ResolvedCandidateRunSpec,
)
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3


def execution_cost_for_overlay(execution_overlay: str) -> ExecutionCostConfig:
    if execution_overlay == LEGACY_DATASET_EXECUTION_OVERLAY:
        return ExecutionCostConfig.zero()
    if execution_overlay == CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY:
        return replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=False,
        )
    raise ValueError(f"unsupported execution_overlay: {execution_overlay}")


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """One resolved in-memory candidate-suite result before publication."""

    spec: ResolvedCandidateRunSpec
    symbols: tuple[str, ...]
    comparison: UniversalStrategyComparison
    ppo_training_timesteps: int
    ppo_training_minimum_hold_suppressed_count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.ppo_training_timesteps, bool)
            or not isinstance(self.ppo_training_timesteps, int)
            or self.ppo_training_timesteps <= 0
        ):
            raise ValueError("ppo_training_timesteps must be a positive integer")
        if (
            isinstance(self.ppo_training_minimum_hold_suppressed_count, bool)
            or not isinstance(self.ppo_training_minimum_hold_suppressed_count, int)
            or self.ppo_training_minimum_hold_suppressed_count < 0
        ):
            raise ValueError(
                "ppo_training_minimum_hold_suppressed_count must be a non-negative integer"
            )


def execute_candidate_run(
    dataset: MarketDataset,
    spec: ResolvedCandidateRunSpec,
) -> CandidateRunResult:
    """Execute the existing candidate suite against one resolved run spec."""

    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("dataset id does not match resolved candidate run spec")
    suite_result = run_lean_candidate_suite(
        dataset,
        spec.lean_config,
        start_index=spec.evaluation_start_index,
        stop_index=spec.evaluation_stop_index,
        gross_budget=spec.config.gross_budget,
        initial_capital=spec.config.initial_capital,
        execution_cost=execution_cost_for_overlay(spec.execution_overlay),
        risk=(
            None
            if spec.config.pretrade_risk_config is None
            else PreTradeRisk(spec.config.pretrade_risk_config)
        ),
        include_ppo_shared_cash_replay=(
            spec.config.ppo_observation_schema == PPO_OBSERVATION_SCHEMA_V3
        ),
    )
    training_timesteps = suite_result.ppo_training_timesteps
    if training_timesteps is None:
        raise ValueError("candidate suite did not report PPO realized timesteps")
    return CandidateRunResult(
        spec=spec,
        symbols=tuple(dataset.symbols),
        comparison=suite_result,
        ppo_training_timesteps=training_timesteps,
        ppo_training_minimum_hold_suppressed_count=(
            suite_result.ppo_training_minimum_hold_suppressed_count or 0
        ),
    )


__all__ = ["CandidateRunResult", "execute_candidate_run"]
