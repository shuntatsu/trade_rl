"""Continuous allocation OOS fold runner with explicit policy switches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.rl_allocation.continuation import (
    allocation_state_digest,
    export_allocation_continuation,
    resume_allocation_continuation,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.simulation.stateful.execution import StatefulExecutionResult

FoldAction = Callable[[np.ndarray, str], object]


@dataclass(frozen=True, slots=True)
class AllocationFoldPolicy:
    """Runtime policy declaration bound to one allocation recipe identity."""

    policy_digest: str
    recipe_digest: str
    action: FoldAction = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require_sha256(self.policy_digest, field="policy_digest")
        require_sha256(self.recipe_digest, field="recipe_digest")
        if not callable(self.action):
            raise ValueError("fold policy action must be callable")


@dataclass(frozen=True, slots=True)
class ContinuousAllocationWalkForwardResult:
    """Completed contiguous OOS chain and the policy identity used per fold."""

    folds: tuple[FoldOOSResult, ...]
    policy_digests: tuple[str, ...]
    stitched: StitchedOOS

    def __post_init__(self) -> None:
        if not self.folds or len(self.folds) != len(self.policy_digests):
            raise ValueError("continuous result requires one policy per fold")
        if self.stitched.mode is not StitchMode.CONTINUOUS_ACCOUNT:
            raise ValueError("continuous result must use continuous-account stitching")


def _preflight(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    policies: tuple[AllocationFoldPolicy, ...],
) -> None:
    if (
        type(folds) is not tuple
        or not folds
        or any(type(fold) is not WalkForwardFold for fold in folds)
    ):
        raise ValueError("continuous runner requires immutable walk-forward folds")
    if (
        type(environments) is not tuple
        or len(environments) != len(folds)
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError(
            "continuous runner requires one allocation environment per fold"
        )
    if (
        type(policies) is not tuple
        or len(policies) != len(environments)
        or any(type(policy) is not AllocationFoldPolicy for policy in policies)
    ):
        raise ValueError("continuous runner requires one declared policy per fold")

    first = environments[0]
    fold_indices = tuple(fold.fold_index for fold in folds)
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("continuous walk-forward fold indices must be unique")
    if any(right <= left for left, right in zip(fold_indices, fold_indices[1:])):
        raise ValueError(
            "continuous walk-forward fold indices must increase chronologically"
        )
    for index, (fold, env, policy) in enumerate(
        zip(folds, environments, policies, strict=True)
    ):
        if hasattr(env, "book") or not env._terminated:
            raise RuntimeError("continuous runner requires fresh uninitialized folds")
        env.validate_binding()
        if (env.start_index, env.stop_index) != (fold.test.start, fold.test.stop):
            raise ValueError(
                "allocation OOS environment differs from declared fold test range"
            )
        if policy.recipe_digest != env.recipe_digest:
            raise ValueError("fold policy recipe differs from allocation runtime")
        if env.dataset is not first.dataset:
            raise ValueError("continuous folds must share the same Dataset instance")
        if env.account_id != first.account_id:
            raise ValueError("continuous folds must share one account identity")
        if env.symbol_index != first.symbol_index:
            raise ValueError("continuous folds must share one symbol identity")
        if env.initial_capital != first.initial_capital:
            raise ValueError("continuous folds must share one capital denominator")
        if (
            env.executor.execution_policy_digest
            != first.executor.execution_policy_digest
        ):
            raise ValueError("continuous folds must share one execution policy")
        if content_digest(env.risk_config) != content_digest(first.risk_config):
            raise ValueError("continuous folds must share one risk profile")
        if index:
            previous_fold = folds[index - 1]
            if fold.test.start != previous_fold.test.stop:
                raise ValueError("continuous fold test ranges must be contiguous")
            if env.start_index != environments[index - 1].stop_index:
                raise ValueError("continuous OOS environments must be contiguous")


def _fold_diagnostics(
    env: AllocationTradingEnv,
    *,
    turnover_before: float,
    cost_before: float,
    funding_before: float,
    borrow_before: float,
    fills_before: int,
    rebalances_before: int,
) -> ExecutionDiagnostics:
    return ExecutionDiagnostics(
        turnover_total=env.book.turnover_total - turnover_before,
        total_cost=env.book.total_cost - cost_before,
        funding_pnl=env.book.funding_pnl - funding_before,
        borrow_cost=env.book.borrow_cost - borrow_before,
        n_trades=env.book.fill_count - fills_before,
        rebalance_events=env.book.rebalance_events - rebalances_before,
    )


def run_continuous_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    policies: tuple[AllocationFoldPolicy, ...],
    *,
    reset_seed: int | None = None,
) -> ContinuousAllocationWalkForwardResult:
    """Run contiguous OOS folds with one account and explicit policy switches."""
    if reset_seed is not None and (
        isinstance(reset_seed, bool)
        or not isinstance(reset_seed, int)
        or reset_seed < 0
    ):
        raise ValueError("reset_seed must be a non-negative integer or None")
    _preflight(folds, environments, policies)

    results: list[FoldOOSResult] = []
    first = environments[0]
    observation, _ = first.reset(seed=reset_seed)
    opening_digest = allocation_state_digest(first)

    for index, (fold, env, policy) in enumerate(
        zip(folds, environments, policies, strict=True)
    ):
        if index:
            previous = environments[index - 1]
            continuation = export_allocation_continuation(previous)
            observation, info = resume_allocation_continuation(env, continuation)
            opening_digest = str(info["opening_state_digest"])

        turnover_before = env.book.turnover_total
        cost_before = env.book.total_cost
        funding_before = env.book.funding_pnl
        borrow_before = env.book.borrow_cost
        fills_before = env.book.fill_count
        rebalances_before = env.book.rebalance_events
        interval_returns: list[float] = []

        while True:
            raw_action = policy.action(observation, env.recipe_digest)
            if (
                isinstance(raw_action, (bool, np.bool_))
                or not isinstance(raw_action, Integral)
                or not 0 <= int(raw_action) <= 3
            ):
                raise ValueError(
                    "fold policy action must be one integer within {0, 1, 2, 3}"
                )
            observation, _, terminated, truncated, info = env.step(int(raw_action))
            if truncated:
                raise ValueError("continuous allocation fold cannot truncate")
            execution = info.get("execution")
            if not isinstance(execution, StatefulExecutionResult):
                raise RuntimeError("allocation step did not return canonical execution")
            interval_returns.append(float(execution.interval_net_return))
            if terminated:
                if env.book.termination_reason is not None:
                    raise ValueError(
                        "economic termination prevents continuous account stitching"
                    )
                if env.index != env.stop_index:
                    raise ValueError("allocation fold terminated before its horizon")
                break

        closing_digest = allocation_state_digest(env)
        results.append(
            FoldOOSResult(
                fold_index=fold.fold_index,
                start=env.start_index,
                stop=env.stop_index,
                returns=ReturnSeries(
                    tuple(interval_returns),
                    ReturnKind.DECISION_STEP,
                    env.dataset.periods_per_year,
                ),
                diagnostics=_fold_diagnostics(
                    env,
                    turnover_before=turnover_before,
                    cost_before=cost_before,
                    funding_before=funding_before,
                    borrow_before=borrow_before,
                    fills_before=fills_before,
                    rebalances_before=rebalances_before,
                ),
                opening_state_digest=opening_digest,
                closing_state_digest=closing_digest,
            )
        )

    completed = tuple(results)
    stitched = stitch_oos(completed, mode=StitchMode.CONTINUOUS_ACCOUNT)
    return ContinuousAllocationWalkForwardResult(
        folds=completed,
        policy_digests=tuple(policy.policy_digest for policy in policies),
        stitched=stitched,
    )


__all__ = [
    "AllocationFoldPolicy",
    "ContinuousAllocationWalkForwardResult",
    "run_continuous_allocation_walk_forward",
]
