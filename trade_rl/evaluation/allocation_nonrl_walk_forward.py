"""Fixed, seedless allocator baseline on the canonical continuous account.

The result wrapper prevents accidental NONRL labeling of caller-supplied action
policies. Its in-process declarations are not proof of origin or authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
    AllocationComparisonEvidence,
    AllocationValidity,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    build_continuous_allocation_comparison_evidence,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    ContinuousAllocationWalkForwardResult,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.global_execution_context import (
    GlobalAllocationExecutionCollector,
)
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    GlobalAllocationWalkForwardResult,
    run_global_allocation_walk_forward,
)
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
)


def allocation_nonrl_rule_digest(candidate_recipe_digest: str) -> str:
    require_sha256(candidate_recipe_digest, field="candidate_recipe_digest")
    return content_digest(
        {
            "schema": "allocation_nonrl_rule_v1",
            "candidate_recipe_digest": candidate_recipe_digest,
            "action_semantics": "residual_exact_baseline_v1",
            "raw_action": 2,
        }
    )


@dataclass(frozen=True, slots=True)
class ContinuousNonRLAllocationResult:
    walk_forward: ContinuousAllocationWalkForwardResult
    candidate_recipe_digest: str
    forecast_context_digest: str
    runtime_recipe_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.walk_forward) is not ContinuousAllocationWalkForwardResult:
            raise ValueError("nonRL result requires a canonical continuous result")
        for name in ("candidate_recipe_digest", "forecast_context_digest"):
            require_sha256(getattr(self, name), field=name)
        if type(self.runtime_recipe_digests) is not tuple or len(
            self.runtime_recipe_digests
        ) != len(self.walk_forward.folds):
            raise ValueError("nonRL result requires one full runtime recipe per fold")
        for digest in self.runtime_recipe_digests:
            require_sha256(digest, field="runtime_recipe_digest")
        if self.walk_forward.policy_digests != (self.rule_digest,) * len(
            self.walk_forward.folds
        ):
            raise ValueError("nonRL result must carry the fixed allocator rule")

    @property
    def rule_digest(self) -> str:
        return allocation_nonrl_rule_digest(self.candidate_recipe_digest)


def _carriers(
    environments: tuple[AllocationTradingEnv, ...],
    *,
    required_action_mode: str = "residual",
) -> str:
    if type(required_action_mode) is not str or required_action_mode not in {
        "residual",
        "direct",
    }:
        raise ValueError("native control requires residual or direct action semantics")
    if (
        type(environments) is not tuple
        or not environments
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("nonRL runner requires native allocation environments")
    first = environments[0]
    candidate = allocation_candidate_recipe_digest(first.recipe)
    packets = {(packet.symbol, packet.as_of): packet for packet in first.stream.packets}
    for env in environments:
        env.validate_binding()
        if (
            env.execution_cost.order_type != "market"
            or env.execution_cost.order_latency_bars != 0
        ):
            raise ValueError("nonRL allocation requires MARKET with zero extra latency")
        if not env.risk_config.drawdown_start < env.risk_config.drawdown_stop <= 0.20:
            raise ValueError("nonRL allocation requires the native drawdown guardrail")
        if env.action_contract.mode != required_action_mode:
            if required_action_mode == "residual":
                raise ValueError("nonRL baseline requires residual action2 carriers")
            raise ValueError("cash control requires direct HOLD-current carriers")
        if allocation_candidate_recipe_digest(env.recipe) != candidate:
            raise ValueError("nonRL candidate recipe differs across folds")
        if (
            env.stream.dataset_id != env.dataset.dataset_id
            or env.stream.digest != first.stream.digest
        ):
            raise ValueError("nonRL forecast context differs across folds")
        times = env.dataset.timestamps[env.start_index : env.stop_index].astype(
            "datetime64[ns]"
        )
        if set(env._estimates) != {int(time.astype("int64")) for time in times}:
            raise ValueError("nonRL cost coverage differs from fold decision clocks")
        for time in times:
            packet = packets.get((env.dataset.symbols[env.symbol_index], time))
            cost = env._estimates[int(time.astype("int64"))]
            if (
                packet is None
                or type(cost) is not HorizonCostEstimates
                or packet.forecast_available_at != time
                or packet.horizon_seconds != env.expected_horizon_seconds
                or cost.symbol != packet.symbol
                or cost.decision_time != time
                or cost.available_at > time
                or cost.horizon_end != packet.horizon_end
            ):
                raise ValueError("nonRL forecast/cost coverage or clocks differ")
    return candidate


def run_continuous_nonrl_allocation(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
) -> ContinuousNonRLAllocationResult:
    candidate = _carriers(environments)
    rule = allocation_nonrl_rule_digest(candidate)
    runtime = tuple(env.recipe_digest for env in environments)
    policies = tuple(
        AllocationFoldPolicy(rule, digest, lambda _observation, _recipe: 2)
        for digest in runtime
    )
    # The common runner performs its whole chain preflight before first reset.
    # No policy RNG exists; execution retains its declared economic random_seed.
    result = run_continuous_allocation_walk_forward(folds, environments, policies)
    return ContinuousNonRLAllocationResult(
        result, candidate, environments[0].stream.digest, runtime
    )


def build_continuous_nonrl_allocation_evidence(
    contract: AllocationComparisonContract,
    *,
    scenario: str,
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousNonRLAllocationResult,
    validity_evidence_digest: str,
    validity: AllocationValidity = AllocationValidity.VALID,
) -> AllocationComparisonEvidence:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("nonRL evidence requires a comparison contract")
    if type(result) is not ContinuousNonRLAllocationResult:
        raise ValueError("nonRL evidence requires the fixed allocator result")
    # Recheck structural declarations; these do not authenticate an execution.
    result.__post_init__()
    if (
        _carriers(environments) != result.candidate_recipe_digest
        or result.candidate_recipe_digest != contract.nonrl_recipe_digest
        or result.forecast_context_digest != contract.forecast_context_digest
        or result.runtime_recipe_digests
        != tuple(env.recipe_digest for env in environments)
    ):
        raise ValueError("nonRL result differs from candidate/forecast/runtime context")
    return build_continuous_allocation_comparison_evidence(
        contract,
        candidate=AllocationCandidateKind.NONRL,
        scenario=scenario,
        seed=None,
        folds=folds,
        environments=environments,
        result=result.walk_forward,
        validity_evidence_digest=validity_evidence_digest,
        validity=validity,
    )


def _validate_global_control_roster(result: GlobalAllocationWalkForwardResult) -> None:
    if type(result) is not GlobalAllocationWalkForwardResult:
        raise ValueError("native control requires the exact global walk-forward result")
    if (
        type(result.folds) is not tuple
        or type(result.policy_digests) is not tuple
        or any(type(fold) is not FoldOOSResult for fold in result.folds)
        or type(result.stitched) is not StitchedOOS
    ):
        raise ValueError("global control requires an immutable native result roster")
    result.__post_init__()
    for digest in result.policy_digests:
        if type(digest) is not str:
            raise ValueError("global policy digest must be a native SHA-256 string")
        require_sha256(digest, field="policy_digest")


@dataclass(frozen=True, slots=True)
class GlobalNonRLAllocationResult:
    """Fixed seedless rule on one global account; not execution authenticity."""

    walk_forward: GlobalAllocationWalkForwardResult
    candidate_recipe_digest: str
    forecast_context_digest: str
    runtime_recipe_digest: str

    def __post_init__(self) -> None:
        _validate_global_control_roster(self.walk_forward)
        for name in (
            "candidate_recipe_digest",
            "forecast_context_digest",
            "runtime_recipe_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        if self.walk_forward.policy_digests != (self.rule_digest,) * len(
            self.walk_forward.folds
        ):
            raise ValueError("global nonRL result must carry the fixed allocator rule")

    @property
    def rule_digest(self) -> str:
        return allocation_nonrl_rule_digest(self.candidate_recipe_digest)


def run_global_nonrl_allocation(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    *,
    collector: GlobalAllocationExecutionCollector | None = None,
) -> GlobalNonRLAllocationResult:
    """Execute only residual action2 over one complete native horizon/account."""
    candidate = _carriers((env,))
    if type(folds) is not tuple:
        raise ValueError("global nonRL requires immutable native folds")
    policy = AllocationFoldPolicy(
        allocation_nonrl_rule_digest(candidate),
        env.recipe_digest,
        lambda _observation, _recipe: 2,
    )
    result = run_global_allocation_walk_forward(
        folds, env, (policy,) * len(folds), collector=collector
    )
    return GlobalNonRLAllocationResult(
        result, candidate, env.stream.digest, env.recipe_digest
    )


__all__ = [
    "ContinuousNonRLAllocationResult",
    "GlobalNonRLAllocationResult",
    "allocation_nonrl_rule_digest",
    "build_continuous_nonrl_allocation_evidence",
    "run_continuous_nonrl_allocation",
    "run_global_nonrl_allocation",
]
