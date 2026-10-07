"""Opt-in configured-fee-only admission on one native global allocation clock."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import file_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_business_objective_digest,
    allocation_comparison_scenario_digest,
    allocation_economic_clock_digest,
    allocation_fold_plan_digest,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.global_execution_context import (
    GlobalAllocationExecutionCollector,
    allocation_array_pin,
    allocation_execution_runtime,
    allocation_source_envelope_digest,
    validate_global_collector,
)
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    GlobalAllocationWalkForwardResult,
    run_global_allocation_walk_forward,
    validate_global_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.policy_admission import (
    AllocationFoldPolicyArtifact,
)
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.strategies.rl.allocation_artifact import (
    load_allocation_policy,
    read_allocation_policy_manifest,
)
from trade_rl.strategies.rl.allocation_fee_stress import (
    AllocationFeeStressBinding,
    AllocationFeeStressPolicyView,
    allocation_policy_state_digest,
)


def _array_pin(values: np.ndarray) -> dict[str, Any]:
    return allocation_array_pin(values)


def _source_context(env: AllocationTradingEnv) -> str:
    return allocation_source_envelope_digest(env)


def _runtime(env: AllocationTradingEnv) -> dict[str, Any]:
    return allocation_execution_runtime(env)


@dataclass(frozen=True, slots=True)
class FeeStressedGlobalAllocationResult:
    """Native global execution plus separate receipt, outside common comparison."""

    walk_forward: GlobalAllocationWalkForwardResult
    _receipt: bytes

    def __post_init__(self) -> None:
        if type(self.walk_forward) is not GlobalAllocationWalkForwardResult:
            raise ValueError("fee result requires a distinct native global result")

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt)


def _comparison(
    contract: AllocationComparisonContract,
    candidate: AllocationCandidateKind,
    scenario: str,
    folds: tuple[WalkForwardFold, ...],
    base: AllocationTradingEnv,
    stress: AllocationTradingEnv,
) -> str:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("fee declaration requires an allocation comparison contract")
    contract.__post_init__()
    if type(candidate) is not AllocationCandidateKind or candidate not in (
        AllocationCandidateKind.RESIDUAL_PPO,
        AllocationCandidateKind.DIRECT_PPO,
    ):
        raise ValueError("fee view requires an RL candidate")
    if base.action_contract.mode != (
        "residual" if candidate is AllocationCandidateKind.RESIDUAL_PPO else "direct"
    ):
        raise ValueError("fee candidate differs from actual action mode")
    expected = (
        contract.residual_recipe_digest
        if candidate is AllocationCandidateKind.RESIDUAL_PPO
        else contract.direct_recipe_digest
    )
    if (
        allocation_candidate_recipe_digest(base.recipe) != expected
        or contract.dataset_id != base.dataset.dataset_id
        or contract.forecast_context_digest != base.stream.digest
        or contract.objective_digest
        != allocation_business_objective_digest(base.bound.objective)
        or contract.clock_digest != allocation_economic_clock_digest(base.bound.clock)
        or contract.fold_plan_digest != allocation_fold_plan_digest(folds)
        or contract.initial_capital != base.initial_capital
        or contract.account_mode != base.bound.objective.capital.account_mode
        or contract.maximum_drawdown != base.risk_config.drawdown_stop
        or contract.economics_digest != base.executor.execution_policy_digest
        or contract.risk_digest != content_digest(base.risk_config)
    ):
        raise ValueError("fee comparison differs from actual base runtime")
    if type(scenario) is not str or scenario == "base":
        raise ValueError("fee stress requires a declared non-base scenario")
    matches = tuple(s for s in contract.scenarios if s.name == scenario)
    digest = allocation_comparison_scenario_digest(
        name=scenario,
        dataset_id=stress.dataset.dataset_id,
        forecast_context_digest=stress.stream.digest,
        economics_digest=stress.executor.execution_policy_digest,
        risk_digest=content_digest(stress.risk_config),
    )
    if len(matches) != 1 or matches[0].digest != digest:
        raise ValueError("fee scenario differs from actual stress runtime")
    return digest


def _bind_action(
    env: AllocationTradingEnv,
    expected: dict[str, Any],
    view: AllocationFeeStressPolicyView,
    policy_digest: str,
    fold_index: int,
    rows: list[dict[str, Any]],
) -> Any:
    dataset = env.dataset

    def invoke(observation: object, runtime_recipe_digest: str) -> int:
        before = _runtime(env)
        if env.dataset is not dataset or canonical_json_bytes(
            before
        ) != canonical_json_bytes(expected):
            raise ValueError("fee runtime or source contents changed before prediction")
        state = allocation_state_digest(env)
        if state != env.decision.baseline.context.state_digest:
            raise ValueError("fee prepared decision differs from current account state")
        values = np.asarray(observation)
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape != env.observation_space.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "fee actor input differs from native finite float32 observation"
            )
        row = {
            "fold_index": fold_index,
            "decision_index": env.index,
            "decision_time_ns": int(
                env.dataset.timestamps[env.index]
                .astype("datetime64[ns]")
                .astype("int64")
            ),
            "state_digest": state,
            "policy_digest": policy_digest,
            "runtime_recipe_digest": runtime_recipe_digest,
            "source_context_digest": before["source_context_digest"],
            "input": _array_pin(values),
            "input_hex": np.ascontiguousarray(values).tobytes().hex(),
            "cash": env.book.cash,
            "equity": env.book.portfolio_value,
        }
        action = view.action(values, runtime_recipe_digest=runtime_recipe_digest)
        if (
            canonical_json_bytes(_runtime(env)) != canonical_json_bytes(expected)
            or allocation_state_digest(env) != state
        ):
            raise ValueError("fee runtime or account changed during prediction")
        rows.append(row | {"action": action})
        return action

    return invoke


def run_fee_stressed_global_allocation(
    folds: tuple[WalkForwardFold, ...],
    *,
    base_env: AllocationTradingEnv,
    stress_env: AllocationTradingEnv,
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
    contract: AllocationComparisonContract,
    candidate: AllocationCandidateKind,
    scenario: str,
    fee_factor: float,
    reset_seed: int | None = None,
    collector: GlobalAllocationExecutionCollector | None = None,
) -> FeeStressedGlobalAllocationResult:
    """Preflight/load originals, then reset once with exact fee-only policy views."""
    if (
        type(artifacts) is not tuple
        or not artifacts
        or any(
            type(a) is not AllocationFoldPolicyArtifact or type(a.fold_index) is not int
            for a in artifacts
        )
    ):
        raise ValueError("fee admission requires immutable native policy artifacts")
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("fee declaration requires an allocation comparison contract")
    if type(reset_seed) is not int or reset_seed not in contract.rl_seeds:
        raise ValueError("fee run requires one declared native RL seed")
    validate_global_collector(stress_env, collector)
    base, stress = _runtime(base_env), _runtime(stress_env)
    if (
        base_env is stress_env
        or hasattr(base_env, "book")
        or not base_env._terminated
        or base_env.dataset is not stress_env.dataset
        or base["source_context_digest"] != stress["source_context_digest"]
    ):
        raise ValueError("fee base/stress require fresh same-source global accounts")
    scenario_digest = _comparison(
        contract, candidate, scenario, folds, base_env, stress_env
    )
    placeholders = tuple(
        AllocationFoldPolicy(a.expected_digest, stress_env.recipe_digest, lambda *_: 0)
        for a in artifacts
    )
    validate_global_allocation_walk_forward(folds, stress_env, placeholders)
    if any(
        a.fold_index != f.fold_index
        or a.expected_recipe_digest != base_env.recipe_digest
        for a, f in zip(artifacts, folds, strict=True)
    ):
        raise ValueError("fee artifact fold/base recipe differs from declaration")
    cutoff = int(
        stress_env.dataset.timestamps[stress_env.start_index]
        .astype("datetime64[ns]")
        .astype("int64")
    )
    manifests = tuple(
        read_allocation_policy_manifest(
            a.bundle_root,
            expected_digest=a.expected_digest,
            expected_recipe_digest=a.expected_recipe_digest,
            training_cutoff_ns=cutoff,
        )
        for a in artifacts
    )
    if any(manifest["training"]["seed"] != reset_seed for manifest in manifests):
        raise ValueError("fee original model seed differs from comparison seed")
    for artifact, manifest in zip(artifacts, manifests, strict=True):
        if (
            file_digest(
                artifact.bundle_root / "policy.zip",
                field="fee original allocation policy",
            )
            != manifest["policy_sha256"]
        ):
            raise ValueError("fee original allocation policy archive digest mismatch")
    declarations = []
    for a, manifest in zip(artifacts, manifests, strict=True):
        declarations.append(
            {
                "schema": "allocation_fee_stress_binding_v1",
                "base_recipe": base["recipe"],
                "stress_recipe": stress["recipe"],
                "base_cost_payload": base["cost_payload"],
                "stress_cost_payload": stress["cost_payload"],
                "fee_factor": fee_factor,
                "original_policy_digest": a.expected_digest,
                "original_policy_sha256": manifest["policy_sha256"],
                "model_state_digest": "0" * 64,
                "contract_digest": contract.digest,
                "scenario_digest": scenario_digest,
            }
        )
        # Domain transformation is checked before any optional backend load.
        AllocationFeeStressBinding(declarations[-1])
    models = tuple(
        load_allocation_policy(
            a.bundle_root,
            expected_digest=a.expected_digest,
            expected_recipe_digest=a.expected_recipe_digest,
            training_cutoff_ns=cutoff,
        )
        for a in artifacts
    )
    if any(model.manifest["training"]["seed"] != reset_seed for model in models):
        raise ValueError("fee original model seed differs from comparison seed")
    if collector is not None:
        collector.loaded_policies(models)
    states = tuple(allocation_policy_state_digest(model) for model in models)
    bindings = tuple(
        AllocationFeeStressBinding(d | {"model_state_digest": state})
        for d, state in zip(declarations, states, strict=True)
    )
    views = tuple(
        AllocationFeeStressPolicyView(model, binding)
        for model, binding in zip(models, bindings, strict=True)
    )
    rows: list[dict[str, Any]] = []
    policies = tuple(
        AllocationFoldPolicy(
            a.expected_digest,
            stress_env.recipe_digest,
            _bind_action(
                stress_env, stress, view, a.expected_digest, a.fold_index, rows
            ),
        )
        for a, view in zip(artifacts, views, strict=True)
    )
    result = run_global_allocation_walk_forward(
        folds, stress_env, policies, reset_seed=reset_seed, collector=collector
    )
    for view in views:
        view.validate_original()
    if canonical_json_bytes(_runtime(stress_env)) != canonical_json_bytes(stress):
        raise ValueError("fee runtime changed before completion")
    receipt = {
        "schema": "allocation_fee_global_execution_receipt_v1",
        "contract_digest": contract.digest,
        "scenario": scenario,
        "scenario_digest": scenario_digest,
        "candidate": candidate.value,
        "seed": reset_seed,
        "base_recipe_digest": base_env.recipe_digest,
        "runtime_recipe_digest": stress_env.recipe_digest,
        "runtime": stress,
        "binding_digests": [b.digest for b in bindings],
        "original_policy_digests": [a.expected_digest for a in artifacts],
        "original_policy_sha256": [m.manifest["policy_sha256"] for m in models],
        "model_states_before": list(states),
        "model_states_after": [allocation_policy_state_digest(m) for m in models],
        "fold_plan_digest": allocation_fold_plan_digest(folds),
        "actor_rows": rows,
        "opening_state_digest": result.folds[0].opening_state_digest,
        "closing_state_digest": result.folds[-1].closing_state_digest,
        "native_execution_summary": result.stitched.diagnostics.digest_payload(),
        "source_semantics": "declared_envelope_plus_actual_actor_rows_not_authenticated_usage",
    }
    return FeeStressedGlobalAllocationResult(result, canonical_json_bytes(receipt))
