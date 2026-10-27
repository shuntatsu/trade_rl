"""Paired verified-data and frozen-policy replay entrypoint."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from trade_rl._validation import require_sha256
from trade_rl.data.artifacts.publication import (
    PublishedDatasetArtifact,
    load_published_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock

if TYPE_CHECKING:
    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv


def run_published_portfolio_inference_replay(
    *,
    dataset_artifact_root: str | Path,
    expected_dataset_artifact_sha256: str,
    bundle_root: str | Path,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    bound: BoundObjectiveClock,
    start_index: int,
    stop_index: int,
    initial_capital: float,
    target_abs_weights: tuple[float, ...],
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    market_order_profile: MarketOrderProfile | None,
    capture_ledger_evidence: bool,
) -> tuple[PublishedDatasetArtifact, SharedCashPortfolioReplayResult]:
    """Load one verified Dataset, validate one policy shell, and replay once.

    Dataset and bundle identities must be supplied independently by the caller.
    The unreset Gym environment is only a compatibility validator; the
    evaluation replay creates the sole native account/session and owns terminal
    settlement. The returned artifact is the identity paired with the exact
    Dataset object passed to both validation and replay.
    """

    require_sha256(
        expected_dataset_artifact_sha256,
        field="expected_dataset_artifact_sha256",
    )
    if not isinstance(bound, BoundObjectiveClock):
        raise TypeError("bound must be a BoundObjectiveClock")
    if type(start_index) is not int or type(stop_index) is not int:
        raise TypeError("start_index and stop_index must be native integers")
    capital_input: object = initial_capital
    if isinstance(capital_input, bool) or not isinstance(capital_input, Real):
        raise TypeError("initial_capital must be a real non-boolean number")
    resolved_capital = float(capital_input)
    if not math.isfinite(resolved_capital) or resolved_capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    if resolved_capital != bound.objective.capital.total_initial_capital:
        raise ValueError("initial_capital differs from the bound objective")
    if not isinstance(execution_cost, ExecutionCostConfig):
        raise TypeError("execution_cost must be an ExecutionCostConfig")
    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("risk_config must be a PreTradeRiskConfig")
    if market_order_profile is not None and not isinstance(
        market_order_profile, MarketOrderProfile
    ):
        raise TypeError("market_order_profile must be verified or None")
    if type(capture_ledger_evidence) is not bool:
        raise TypeError("capture_ledger_evidence must be a boolean")

    artifact, dataset = load_published_market_dataset_artifact(dataset_artifact_root)
    if artifact.artifact_digest != expected_dataset_artifact_sha256:
        raise ValueError("published Dataset artifact differs from the expected digest")
    if not isinstance(dataset, MarketDataset):
        raise TypeError("verified Dataset loader returned an invalid Dataset")

    # Keep the optional Gym/SB3 inference stack out of evaluation module import.
    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        load_portfolio_inference_bundle,
    )

    validation_env: SharedPortfolioGymEnv = SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=bound,
        target_abs_weights=target_abs_weights,
        execution_cost=execution_cost,
        risk_config=risk_config,
        market_order_profile=market_order_profile,
    )
    policy = load_portfolio_inference_bundle(
        bundle_root,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_training_context_sha256=expected_training_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        runtime_env=validation_env,
    )
    replay_result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=resolved_capital,
        execution_cost=execution_cost,
        risk=PreTradeRisk(risk_config),
        market_order_profile=market_order_profile,
        settle_terminal_position=True,
        capture_ledger_evidence=capture_ledger_evidence,
    )
    return artifact, replay_result


__all__ = ["run_published_portfolio_inference_replay"]
