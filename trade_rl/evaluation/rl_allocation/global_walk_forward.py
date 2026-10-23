"""Opt-in frozen policy switches inside one native global OOS episode."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from trade_rl.artifacts.verified_file import file_digest
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    _fold_diagnostics,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.global_execution_context import (
    GlobalAllocationExecutionCollector,
    validate_global_collector,
)
from trade_rl.evaluation.rl_allocation.policy_admission import (
    AllocationFoldPolicyArtifact,
    _bind_loaded_policy,
)
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.strategies.rl.allocation_artifact import (
    load_allocation_policy,
    read_allocation_policy_manifest,
)


@dataclass(frozen=True, slots=True)
class GlobalAllocationWalkForwardResult:
    """Segmented global episode; deliberately distinct from local-fold evidence."""

    folds: tuple[FoldOOSResult, ...]
    policy_digests: tuple[str, ...]
    stitched: StitchedOOS

    def __post_init__(self) -> None:
        if not self.folds or len(self.folds) != len(self.policy_digests):
            raise ValueError("global result requires one policy per segment")
        if self.stitched.mode is not StitchMode.CONTINUOUS_ACCOUNT:
            raise ValueError("global result requires continuous-account stitching")


def _seed(reset_seed: int | None) -> None:
    if reset_seed is not None and (type(reset_seed) is not int or reset_seed < 0):
        raise ValueError("reset_seed must be a native non-negative integer or None")


def _roster(
    folds: tuple[WalkForwardFold, ...], policies: tuple[AllocationFoldPolicy, ...]
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            id(type(f)),
            (id(type(f.fold_index)), f.fold_index),
            (id(type(f.purge_bars)), f.purge_bars),
            tuple(
                (id(type(r)), id(type(r.start)), r.start, id(type(r.stop)), r.stop)
                for r in (
                    f.train,
                    f.checkpoint_validation,
                    f.configuration_selection,
                    f.test,
                )
            ),
            id(type(p)),
            (id(type(p.policy_digest)), p.policy_digest),
            (id(type(p.recipe_digest)), p.recipe_digest),
            id(p.action),
        )
        for f, p in zip(folds, policies, strict=True)
    )


def validate_global_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    policies: tuple[AllocationFoldPolicy, ...],
) -> None:
    """Validate the complete immutable roster before initializing an account."""
    if (
        type(folds) is not tuple
        or not folds
        or any(type(f) is not WalkForwardFold for f in folds)
    ):
        raise ValueError("global runner requires immutable native walk-forward folds")
    if type(env) is not AllocationTradingEnv:
        raise ValueError("global runner requires one native allocation environment")
    if (
        type(policies) is not tuple
        or len(policies) != len(folds)
        or any(type(p) is not AllocationFoldPolicy for p in policies)
    ):
        raise ValueError("global runner requires one declared policy per segment")
    if hasattr(env, "book") or not env._terminated:
        raise RuntimeError("global runner requires a fresh uninitialized environment")
    if type(env.start_index) is not int or type(env.stop_index) is not int:
        raise ValueError("global environment ranges must use native integers")
    env.validate_binding()
    previous_stop, previous_id = env.start_index, -1
    for fold, policy in zip(folds, policies, strict=True):
        if type(fold.fold_index) is not int or type(fold.purge_bars) is not int:
            raise ValueError("global fold scalars must use native integers")
        for interval in (
            fold.train,
            fold.checkpoint_validation,
            fold.configuration_selection,
            fold.test,
        ):
            if (
                type(interval) is not IndexRange
                or type(interval.start) is not int
                or type(interval.stop) is not int
            ):
                raise ValueError(
                    "global fold ranges must use native integer IndexRange"
                )
            interval.__post_init__()
        # Recheck dataclass structural claims as well as native scalar types.
        fold.__post_init__()
        if fold.fold_index <= previous_id or fold.test.start != previous_stop:
            raise ValueError(
                "global segments must be unique, increasing and contiguous"
            )
        if fold.test.stop > env.stop_index:
            raise ValueError("global segment exceeds the native horizon")
        if policy.recipe_digest != env.recipe_digest:
            raise ValueError("segment policy differs from the full global recipe")
        previous_stop, previous_id = fold.test.stop, fold.fold_index
    if previous_stop != env.stop_index:
        raise ValueError("global segments must cover the whole native OOS horizon")


def run_global_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    policies: tuple[AllocationFoldPolicy, ...],
    *,
    reset_seed: int | None = None,
    collector: GlobalAllocationExecutionCollector | None = None,
) -> GlobalAllocationWalkForwardResult:
    """Reset once; segment boundaries preserve the active native state/clock."""
    _seed(reset_seed)
    validate_global_allocation_walk_forward(folds, env, policies)
    validate_global_collector(env, collector)
    recipe_digest = env.recipe_digest
    policy_digests = tuple(p.policy_digest for p in policies)
    # Strong references make action identity stable; callable __eq__ is unused.
    actions = tuple(p.action for p in policies)
    roster = _roster(folds, policies)
    if collector is not None:
        collector.before_reset(folds, env, policies, reset_seed)
        env._transition_recorder = collector
    try:
        observation, _ = env.reset(seed=reset_seed)
        if collector is not None:
            collector.after_reset()
        results = []
        for fold, policy, action in zip(folds, policies, actions, strict=True):
            opening = allocation_state_digest(env)
            turnover, cost, funding, borrow = (
                env.book.turnover_total,
                env.book.total_cost,
                env.book.funding_pnl,
                env.book.borrow_cost,
            )
            fills, rebalances = env.book.fill_count, env.book.rebalance_events
            returns = []
            while env.index < fold.test.stop:
                if _roster(folds, policies) != roster:
                    raise ValueError("global declared roster changed during execution")
                if (
                    env.recipe_digest != recipe_digest
                    or policy.recipe_digest != recipe_digest
                ):
                    raise ValueError("global recipe changed during execution")
                if collector is not None:
                    collector.before_action(
                        fold.fold_index,
                        policy.policy_digest,
                        observation,
                        recipe_digest,
                    )
                try:
                    raw_action = action(observation, recipe_digest)
                except Exception as error:
                    if collector is not None:
                        collector.action_failed(error)
                    raise
                if collector is not None:
                    collector.after_action(raw_action)
                if _roster(folds, policies) != roster:
                    raise ValueError(
                        "global declared roster changed during policy invocation"
                    )
                if (
                    isinstance(raw_action, (bool, np.bool_))
                    or not isinstance(raw_action, Integral)
                    or not 0 <= int(raw_action) <= 3
                ):
                    raise ValueError(
                        "segment policy action must be an integer within {0, 1, 2, 3}"
                    )
                if env.recipe_digest != recipe_digest:
                    raise ValueError("global recipe changed during policy invocation")
                observation, _, terminated, truncated, info = env.step(int(raw_action))
                if collector is not None:
                    collector.after_step(terminated, truncated, info)
                if truncated or env.book.termination_reason is not None:
                    raise ValueError(
                        "economic termination/truncation prevents global stitching"
                    )
                if terminated != (env.index == env.stop_index):
                    raise ValueError(
                        "native terminal must occur only at global horizon end"
                    )
                execution = info.get("execution")
                if not isinstance(execution, StatefulExecutionResult):
                    raise RuntimeError(
                        "allocation step did not return canonical execution"
                    )
                returns.append(float(execution.interval_net_return))
            results.append(
                FoldOOSResult(
                    fold.fold_index,
                    fold.test.start,
                    fold.test.stop,
                    ReturnSeries(
                        tuple(returns),
                        ReturnKind.DECISION_STEP,
                        env.dataset.periods_per_year,
                    ),
                    _fold_diagnostics(
                        env,
                        turnover_before=turnover,
                        cost_before=cost,
                        funding_before=funding,
                        borrow_before=borrow,
                        fills_before=fills,
                        rebalances_before=rebalances,
                    ),
                    opening,
                    allocation_state_digest(env),
                )
            )
        completed = tuple(results)
        result = GlobalAllocationWalkForwardResult(
            completed,
            policy_digests,
            stitch_oos(completed, mode=StitchMode.CONTINUOUS_ACCOUNT),
        )
        if collector is not None:
            collector.completed(result.folds)
        return result
    except Exception as error:
        if collector is not None:
            collector.failed(error)
        raise
    finally:
        if collector is not None and env._transition_recorder is collector:
            env._transition_recorder = None


def run_artifact_bound_global_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
    *,
    reset_seed: int | None = None,
    collector: GlobalAllocationExecutionCollector | None = None,
) -> GlobalAllocationWalkForwardResult:
    """Preflight all v5 temporal claims, then load/recheck every pinned model."""
    _seed(reset_seed)
    validate_global_collector(env, collector)
    if (
        type(artifacts) is not tuple
        or not artifacts
        or any(
            type(a) is not AllocationFoldPolicyArtifact or type(a.fold_index) is not int
            for a in artifacts
        )
    ):
        raise ValueError("global admission requires immutable native policy artifacts")
    policies = tuple(
        AllocationFoldPolicy(a.expected_digest, a.expected_recipe_digest, lambda *_: 0)
        for a in artifacts
    )
    validate_global_allocation_walk_forward(folds, env, policies)
    if any(f.fold_index != a.fold_index for f, a in zip(folds, artifacts, strict=True)):
        raise ValueError("global artifact fold differs from declared segment")
    cutoff = int(
        env.dataset.timestamps[env.start_index].astype("datetime64[ns]").astype("int64")
    )
    manifests = []
    for artifact in artifacts:
        manifests.append(
            read_allocation_policy_manifest(
                artifact.bundle_root,
                expected_digest=artifact.expected_digest,
                expected_recipe_digest=artifact.expected_recipe_digest,
                training_cutoff_ns=cutoff,
            )
        )
    if collector is not None:
        for artifact, manifest in zip(artifacts, manifests, strict=True):
            if (
                file_digest(
                    artifact.bundle_root / "policy.zip",
                    field="global original allocation policy",
                )
                != manifest["policy_sha256"]
            ):
                raise ValueError(
                    "global original allocation policy archive digest mismatch"
                )
    models = []
    admitted = []
    for artifact in artifacts:
        loaded = load_allocation_policy(
            artifact.bundle_root,
            expected_digest=artifact.expected_digest,
            expected_recipe_digest=artifact.expected_recipe_digest,
            training_cutoff_ns=cutoff,
        )
        admitted.append(_bind_loaded_policy(loaded, artifact))
        models.append(loaded)
    if collector is not None:
        collector.loaded_policies(tuple(models))
    return run_global_allocation_walk_forward(
        folds, env, tuple(admitted), reset_seed=reset_seed, collector=collector
    )


__all__ = [
    "GlobalAllocationWalkForwardResult",
    "validate_global_allocation_walk_forward",
    "run_global_allocation_walk_forward",
    "run_artifact_bound_global_allocation_walk_forward",
]
