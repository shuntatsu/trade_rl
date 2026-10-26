"""Explicit four-family replay under a pinned v2 execution scenario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from trade_rl._validation import require_sha256
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.runs.portfolio_family_comparison import (
    FourFamilyPortfolioReplayResult,
    PortfolioFamilyReplayOutcome,
    _run_four_family_shared_portfolio_replay_core,
)
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.strategies.forecasts.ridge_artifact import RidgeForecastArtifactBinding
from trade_rl.strategies.portfolio import PortfolioPolicy
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig

if TYPE_CHECKING:
    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
    from trade_rl.strategies.rl.portfolio_transform import (
        PortfolioScenarioObservationRuntimeBindingV2,
    )


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioScenarioReplayV2Result:
    """Actual native outcomes bound to one verified v2 runtime scenario.

    ``common_replay`` contains the four in-memory native arm results. It is not
    a v1 persisted result artifact and does not authorize publication.
    """

    common_replay: FourFamilyPortfolioReplayResult
    scenario: PortfolioRuntimeExecutionScenarioV2
    scenario_sha256: str
    scenario_inference_source_sha256: str
    runtime_binding: PortfolioScenarioObservationRuntimeBindingV2

    @property
    def training_context_sha256(self) -> str:
        return self.runtime_binding.training_context_sha256

    @property
    def transform_digest(self) -> str:
        return self.runtime_binding.transform_digest

    @property
    def checkpoint_sha256(self) -> str:
        return self.runtime_binding.checkpoint_sha256

    @property
    def base_execution_policy_digest(self) -> str:
        return self.runtime_binding.base_execution_policy_digest

    @property
    def runtime_execution_policy_digest(self) -> str:
        return self.runtime_binding.runtime_execution_policy_digest

    @property
    def training_bound_digest(self) -> str:
        return self.runtime_binding.training_bound_digest

    @property
    def runtime_bound_digest(self) -> str:
        return self.runtime_binding.runtime_bound_digest


def run_four_family_shared_portfolio_scenario_replay_v2(
    *,
    dataset_artifact_root: str | Path,
    expected_dataset_artifact_sha256: str,
    expected_symbols: tuple[str, ...],
    ridge_artifact_root: str | Path,
    expected_ridge_artifact_sha256: str,
    expected_ridge_binding: RidgeForecastArtifactBinding,
    bundle_root: str | Path,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    expected_scenario: PortfolioRuntimeExecutionScenarioV2,
    expected_scenario_sha256: str,
    expected_scenario_inference_source_sha256: str,
    bound: BoundObjectiveClock,
    start_index: int,
    stop_index: int,
    initial_capital: float,
    target_abs_weights: tuple[float, ...],
    trend_config: TrendIntentConfig,
    ridge_entry_threshold: float,
    ridge_exit_threshold: float,
    ridge_one_way_switch_cost: float | None,
    risk_config: PreTradeRiskConfig,
    market_order_profile: MarketOrderProfile | None = None,
    on_family_completed: (Callable[[PortfolioFamilyReplayOutcome], None] | None) = None,
) -> FourFamilyPortfolioScenarioReplayV2Result:
    """Run all four native arms using only the scenario's runtime cost config.

    This explicit v2 entry retains the trained checkpoint and fitted statistics.
    The runtime scenario changes supported execution settings only; it does not
    refit, reserve a durable attempt, publish an artifact, or select a winner.
    """

    if type(expected_scenario) is not PortfolioRuntimeExecutionScenarioV2:
        raise TypeError(
            "expected_scenario must be exactly PortfolioRuntimeExecutionScenarioV2"
        )
    require_sha256(expected_scenario_sha256, field="expected_scenario_sha256")
    require_sha256(
        expected_scenario_inference_source_sha256,
        field="expected_scenario_inference_source_sha256",
    )
    expected_scenario.to_bytes()
    if expected_scenario.digest != expected_scenario_sha256:
        raise ValueError("scenario digest differs from its external expected pin")

    # Check the v2 inference implementation before any artifact load. The loader
    # repeats this check at its checkpoint boundary as a second fail-closed gate.
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        capture_portfolio_scenario_inference_source_digest,
    )

    if (
        capture_portfolio_scenario_inference_source_digest()
        != expected_scenario_inference_source_sha256
    ):
        raise ValueError("current scenario inference source differs from its pin")

    from trade_rl.strategies.rl.portfolio_scenario_inference import (
        PortfolioScenarioInferencePolicy,
        load_portfolio_scenario_inference_bundle,
    )

    def load_scenario_policy(runtime_env: SharedPortfolioGymEnv) -> PortfolioPolicy:
        policy = load_portfolio_scenario_inference_bundle(
            bundle_root,
            expected_bundle_sha256=expected_bundle_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_completion_sha256=expected_completion_sha256,
            expected_training_context_sha256=expected_training_context_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_scenario=expected_scenario,
            expected_scenario_sha256=expected_scenario_sha256,
            expected_scenario_inference_source_sha256=(
                expected_scenario_inference_source_sha256
            ),
            runtime_env=runtime_env,
        )
        if type(policy) is not PortfolioScenarioInferencePolicy:
            raise TypeError(
                "scenario loader must return exactly PortfolioScenarioInferencePolicy"
            )
        binding = policy.runtime_binding
        if (
            binding.scenario_sha256 != expected_scenario_sha256
            or binding.scenario_inference_source_sha256
            != expected_scenario_inference_source_sha256
            or binding.training_context_sha256 != expected_training_context_sha256
            or binding.transform_digest != expected_scenario.transform_digest
            or binding.checkpoint_sha256 != expected_checkpoint_sha256
            or binding.base_execution_policy_digest
            != expected_scenario.base_execution_cost.execution_policy_digest
            or binding.runtime_execution_policy_digest
            != expected_scenario.runtime_execution_cost.execution_policy_digest
            or binding.runtime_bound_digest != runtime_env.bound.digest
            or binding.dataset_id != runtime_env.dataset.dataset_id
            or binding.runtime_start_index != start_index
            or binding.runtime_stop_index != stop_index
        ):
            raise ValueError(
                "loaded scenario policy binding differs from replay inputs"
            )
        return policy

    execution = _run_four_family_shared_portfolio_replay_core(
        dataset_artifact_root=dataset_artifact_root,
        expected_dataset_artifact_sha256=expected_dataset_artifact_sha256,
        expected_symbols=expected_symbols,
        ridge_artifact_root=ridge_artifact_root,
        expected_ridge_artifact_sha256=expected_ridge_artifact_sha256,
        expected_ridge_binding=expected_ridge_binding,
        bundle_root=bundle_root,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_training_context_sha256=expected_training_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        bound=bound,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=initial_capital,
        target_abs_weights=target_abs_weights,
        trend_config=trend_config,
        ridge_entry_threshold=ridge_entry_threshold,
        ridge_exit_threshold=ridge_exit_threshold,
        ridge_one_way_switch_cost=ridge_one_way_switch_cost,
        execution_cost=expected_scenario.runtime_execution_cost,
        risk_config=risk_config,
        inference_policy_loader=load_scenario_policy,
        market_order_profile=market_order_profile,
        on_family_completed=on_family_completed,
    )
    policy = execution.inference_policy
    if type(policy) is not PortfolioScenarioInferencePolicy:
        raise RuntimeError("native comparison returned an unexpected PPO policy")
    return FourFamilyPortfolioScenarioReplayV2Result(
        common_replay=execution.common_replay,
        scenario=expected_scenario,
        scenario_sha256=expected_scenario_sha256,
        scenario_inference_source_sha256=(expected_scenario_inference_source_sha256),
        runtime_binding=policy.runtime_binding,
    )


__all__ = [
    "FourFamilyPortfolioScenarioReplayV2Result",
    "run_four_family_shared_portfolio_scenario_replay_v2",
]
