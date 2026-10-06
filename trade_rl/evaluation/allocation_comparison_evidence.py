"""Bind actual continuous allocation OOS results to the pure comparison matrix."""

from __future__ import annotations

import math

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
    AllocationComparisonEvidence,
    AllocationComparisonScenario,
    AllocationValidity,
)
from trade_rl.evaluation.objectives import FinancialClockContract, ObjectiveContract
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    ContinuousAllocationWalkForwardResult,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def allocation_business_objective_digest(objective: ObjectiveContract) -> str:
    """Digest business semantics while excluding fold/scenario/candidate bindings."""
    if type(objective) is not ObjectiveContract:
        raise ValueError("business objective profile requires ObjectiveContract")
    payload = objective.payload()
    return content_digest(
        {
            "schema": "allocation_business_objective_profile_v1",
            "objective_id": payload["objective_id"],
            "tax_treatment": payload["tax_treatment"],
            "infrastructure_cost_treatment": payload["infrastructure_cost_treatment"],
            "external_cash_flow_convention": payload["external_cash_flow_convention"],
            "capital": payload["capital"],
            "terminal_valuation": payload["terminal_valuation"],
            "maximum_drawdown": payload["maximum_drawdown"],
        }
    )


def allocation_economic_clock_digest(clock: FinancialClockContract) -> str:
    """Digest economic timing while excluding PPO rollout/GAE training choices."""
    if type(clock) is not FinancialClockContract:
        raise ValueError("economic clock profile requires FinancialClockContract")
    return content_digest(
        {
            "schema": "allocation_economic_clock_profile_v1",
            "decision_interval_seconds": clock.decision_interval_seconds,
            "execution_interval_seconds": clock.execution_interval_seconds,
            "reward_interval_seconds": clock.reward_interval_seconds,
            "economic_horizon_seconds": clock.economic_horizon_seconds,
            "gamma": clock.gamma,
            "reward_schema": clock.reward_schema,
        }
    )


def allocation_comparison_scenario_digest(
    *,
    name: str,
    dataset_id: str,
    forecast_context_digest: str,
    economics_digest: str,
    risk_digest: str,
) -> str:
    scenario = require_non_empty(name, field="scenario")
    for field, value in (
        ("dataset_id", dataset_id),
        ("forecast_context_digest", forecast_context_digest),
        ("economics_digest", economics_digest),
        ("risk_digest", risk_digest),
    ):
        require_sha256(value, field=field)
    return content_digest(
        {
            "schema": "allocation_comparison_scenario_context_v1",
            "name": scenario,
            "dataset_id": dataset_id,
            "forecast_context_digest": forecast_context_digest,
            "economics_digest": economics_digest,
            "risk_digest": risk_digest,
        }
    )


def allocation_fold_plan_digest(folds: tuple[WalkForwardFold, ...]) -> str:
    if (
        type(folds) is not tuple
        or not folds
        or any(type(fold) is not WalkForwardFold for fold in folds)
    ):
        raise ValueError("fold plan requires a non-empty immutable fold tuple")
    indices = tuple(fold.fold_index for fold in folds)
    if len(set(indices)) != len(indices) or any(
        right <= left for left, right in zip(indices, indices[1:])
    ):
        raise ValueError("fold plan indices must be unique and increasing")
    return content_digest(
        {
            "schema": "allocation_fold_plan_v1",
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "train": [fold.train.start, fold.train.stop],
                    "checkpoint_validation": [
                        fold.checkpoint_validation.start,
                        fold.checkpoint_validation.stop,
                    ],
                    "configuration_selection": [
                        fold.configuration_selection.start,
                        fold.configuration_selection.stop,
                    ],
                    "test": [fold.test.start, fold.test.stop],
                    "purge_bars": fold.purge_bars,
                }
                for fold in folds
            ],
        }
    )


def allocation_policy_schedule_digest(
    candidate: AllocationCandidateKind,
    seed: int,
    policy_digests: tuple[str, ...],
) -> str:
    if candidate not in (
        AllocationCandidateKind.RESIDUAL_PPO,
        AllocationCandidateKind.DIRECT_PPO,
    ):
        raise ValueError("only RL candidates have a policy schedule digest")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("policy schedule seed must be a non-negative integer")
    if (
        type(policy_digests) is not tuple
        or not policy_digests
        or any(not isinstance(value, str) for value in policy_digests)
    ):
        raise ValueError("policy schedule requires immutable policy digests")
    for value in policy_digests:
        require_sha256(value, field="policy_digest")
    return content_digest(
        {
            "schema": "allocation_policy_schedule_v1",
            "candidate": candidate.value,
            "seed": seed,
            "fold_policy_digests": list(policy_digests),
        }
    )


def continuous_account_profit_and_drawdown(
    returns: ReturnSeries,
) -> tuple[float, float]:
    if type(returns) is not ReturnSeries:
        raise ValueError("continuous profit requires ReturnSeries")
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns.values:
        wealth *= 1.0 + value
        if not math.isfinite(wealth) or wealth < 0.0:
            raise ValueError("continuous account wealth became invalid")
        peak = max(peak, wealth)
        if peak > 0.0:
            maximum_drawdown = max(maximum_drawdown, 1.0 - wealth / peak)
    return wealth - 1.0, maximum_drawdown


def _candidate_recipe(
    contract: AllocationComparisonContract,
    candidate: AllocationCandidateKind,
) -> str:
    if candidate is AllocationCandidateKind.NONRL:
        return contract.nonrl_recipe_digest
    if candidate is AllocationCandidateKind.RESIDUAL_PPO:
        return contract.residual_recipe_digest
    return contract.direct_recipe_digest


def _scenario(
    contract: AllocationComparisonContract,
    name: str,
) -> AllocationComparisonScenario:
    matches = tuple(value for value in contract.scenarios if value.name == name)
    if len(matches) != 1:
        raise ValueError("comparison scenario is not declared exactly once")
    return matches[0]


def validate_continuous_allocation_comparison_context(
    contract: AllocationComparisonContract,
    *,
    expected_recipe_digest: str,
    scenario: str,
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousAllocationWalkForwardResult,
) -> None:
    require_sha256(expected_recipe_digest, field="expected_recipe_digest")
    if (
        type(environments) is not tuple
        or len(environments) != len(folds)
        or not environments
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("comparison evidence requires one allocation env per fold")
    if type(result) is not ContinuousAllocationWalkForwardResult:
        raise ValueError("comparison evidence requires continuous allocation result")
    if allocation_fold_plan_digest(folds) != contract.fold_plan_digest:
        raise ValueError("comparison fold plan differs from runtime")
    if result.stitched.mode is not StitchMode.CONTINUOUS_ACCOUNT:
        raise ValueError("comparison evidence requires continuous account stitching")
    if result.stitched.gaps:
        raise ValueError("comparison evidence cannot contain OOS gaps")
    if result.stitched.fold_indices != tuple(fold.fold_index for fold in folds):
        raise ValueError("continuous result fold identity differs from declaration")
    if result.stitched.boundaries != tuple(
        (fold.test.start, fold.test.stop) for fold in folds
    ):
        raise ValueError("continuous result boundaries differ from fold plan")
    if len(result.folds) != len(folds) or len(result.policy_digests) != len(folds):
        raise ValueError("continuous result does not cover every fold")
    rebuilt_stitched = stitch_oos(
        result.folds,
        mode=StitchMode.CONTINUOUS_ACCOUNT,
    )
    if rebuilt_stitched != result.stitched:
        raise ValueError(
            "continuous result stitched evidence differs from fold evidence"
        )
    if result.stitched.returns.kind is not ReturnKind.DECISION_STEP:
        raise ValueError("allocation comparison requires decision-step OOS returns")

    first = environments[0]
    if first.dataset.dataset_id != contract.dataset_id:
        raise ValueError("comparison Dataset differs from runtime")
    if first.stream.digest != contract.forecast_context_digest:
        raise ValueError("comparison forecast context differs from runtime")
    if (
        allocation_business_objective_digest(first.bound.objective)
        != contract.objective_digest
    ):
        raise ValueError("comparison business objective differs from runtime")
    if allocation_economic_clock_digest(first.bound.clock) != contract.clock_digest:
        raise ValueError("comparison economic clock differs from runtime")
    if first.bound.objective.capital.account_mode != contract.account_mode:
        raise ValueError("comparison account mode differs from runtime")
    if first.initial_capital != contract.initial_capital:
        raise ValueError("comparison initial capital differs from runtime")
    if first.bound.objective.maximum_drawdown != contract.maximum_drawdown:
        raise ValueError("comparison drawdown guardrail differs from runtime")

    for fold, env, fold_result in zip(folds, environments, result.folds, strict=True):
        env.validate_binding()
        if (
            env.dataset is not first.dataset
            or env.dataset.dataset_id != contract.dataset_id
        ):
            raise ValueError("comparison Dataset identity differs across folds")
        if env.stream.digest != contract.forecast_context_digest:
            raise ValueError("comparison forecast context differs across folds")
        if (
            allocation_business_objective_digest(env.bound.objective)
            != contract.objective_digest
        ):
            raise ValueError("comparison business objective differs across folds")
        if allocation_economic_clock_digest(env.bound.clock) != contract.clock_digest:
            raise ValueError("comparison economic clock differs across folds")
        if env.recipe_digest != expected_recipe_digest:
            raise ValueError("comparison candidate recipe differs from runtime")
        if (env.start_index, env.stop_index) != (fold.test.start, fold.test.stop):
            raise ValueError("comparison OOS environment differs from fold test range")
        if (fold_result.start, fold_result.stop) != (fold.test.start, fold.test.stop):
            raise ValueError("comparison result fold range differs from declaration")
        if (
            fold_result.opening_state_digest is None
            or fold_result.closing_state_digest is None
        ):
            raise ValueError("comparison result requires complete state-chain digests")
        if not env._terminated or env.index != env.stop_index:
            raise ValueError("comparison OOS environment did not complete its fold")
        if env.book.termination_reason is not None:
            raise ValueError(
                "comparison continuous result ended economically terminated"
            )

    if (
        allocation_state_digest(environments[-1])
        != result.folds[-1].closing_state_digest
    ):
        raise ValueError(
            "comparison closing state differs from canonical final account"
        )

    actual_economics = first.executor.execution_policy_digest
    actual_risk = content_digest(first.risk_config)
    expected_scenario = allocation_comparison_scenario_digest(
        name=scenario,
        dataset_id=contract.dataset_id,
        forecast_context_digest=contract.forecast_context_digest,
        economics_digest=actual_economics,
        risk_digest=actual_risk,
    )
    if _scenario(contract, scenario).digest != expected_scenario:
        raise ValueError("comparison scenario differs from runtime economics/risk")
    if scenario == "base" and (
        actual_economics != contract.economics_digest
        or actual_risk != contract.risk_digest
    ):
        raise ValueError("base comparison scenario must use contract economics/risk")


def allocation_comparison_oos_source_digest(
    contract: AllocationComparisonContract,
    *,
    scenario: str,
) -> str:
    return content_digest(
        {
            "schema": "allocation_comparison_oos_source_v1",
            "dataset_id": contract.dataset_id,
            "forecast_context_digest": contract.forecast_context_digest,
            "fold_plan_digest": contract.fold_plan_digest,
            "scenario_digest": _scenario(contract, scenario).digest,
        }
    )


def allocation_continuous_ledger_summary_digest(
    result: ContinuousAllocationWalkForwardResult,
    *,
    source_digest: str,
) -> str:
    return content_digest(
        {
            "schema": "allocation_continuous_ledger_evidence_v1",
            "oos_source_digest": source_digest,
            "returns": {
                "kind": result.stitched.returns.kind.value,
                "periods_per_year": result.stitched.returns.periods_per_year,
                "values": list(result.stitched.returns.values),
            },
            "state_chain": [
                {
                    "fold_index": value.fold_index,
                    "opening": value.opening_state_digest,
                    "closing": value.closing_state_digest,
                }
                for value in result.folds
            ],
        }
    )


def allocation_continuous_execution_summary_digest(
    result: ContinuousAllocationWalkForwardResult,
    *,
    source_digest: str,
) -> str:
    return content_digest(
        {
            "schema": "allocation_continuous_execution_evidence_v1",
            "oos_source_digest": source_digest,
            "boundaries": [list(value) for value in result.stitched.boundaries],
            "fold_diagnostics": [
                {
                    "fold_index": value.fold_index,
                    "diagnostics": value.diagnostics.digest_payload(),
                }
                for value in result.folds
            ],
            "combined_diagnostics": result.stitched.diagnostics.digest_payload(),
        }
    )


def canonical_continuous_allocation_metrics(
    contract: AllocationComparisonContract,
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousAllocationWalkForwardResult,
) -> tuple[float, float]:
    """Cross-check stitched returns against canonical final account economics."""
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("canonical metrics require allocation comparison contract")
    if (
        type(environments) is not tuple
        or not environments
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("canonical metrics require allocation environments")
    if type(result) is not ContinuousAllocationWalkForwardResult:
        raise ValueError("canonical metrics require continuous allocation result")
    return_profit, return_path_drawdown = continuous_account_profit_and_drawdown(
        result.stitched.returns
    )
    final_book = environments[-1].book
    terminal_profit = final_book.portfolio_value / contract.initial_capital - 1.0
    if not math.isclose(
        terminal_profit,
        return_profit,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "continuous account returns disagree with canonical terminal equity"
        )
    max_drawdown = final_book.max_drawdown
    if return_path_drawdown > max_drawdown + 1e-12:
        raise ValueError("return-path drawdown exceeds canonical account drawdown")
    return terminal_profit, max_drawdown


def build_continuous_allocation_comparison_evidence(
    contract: AllocationComparisonContract,
    *,
    candidate: AllocationCandidateKind,
    scenario: str,
    seed: int | None,
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousAllocationWalkForwardResult,
    validity_evidence_digest: str,
    validity: AllocationValidity = AllocationValidity.VALID,
) -> AllocationComparisonEvidence:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("comparison evidence builder requires comparison contract")
    require_sha256(validity_evidence_digest, field="validity_evidence_digest")
    if type(validity) is not AllocationValidity:
        raise ValueError("validity must be AllocationValidity")
    validate_continuous_allocation_comparison_context(
        contract,
        expected_recipe_digest=_candidate_recipe(contract, candidate),
        scenario=scenario,
        folds=folds,
        environments=environments,
        result=result,
    )

    if candidate is AllocationCandidateKind.NONRL:
        if seed is not None:
            raise ValueError("nonRL comparison evidence must not use a seed")
        policy_digest = None
    else:
        if seed is None or seed not in contract.rl_seeds:
            raise ValueError("RL comparison evidence requires one declared seed")
        policy_digest = allocation_policy_schedule_digest(
            candidate,
            seed,
            result.policy_digests,
        )

    terminal_profit, max_drawdown = canonical_continuous_allocation_metrics(
        contract,
        environments,
        result,
    )
    source_digest = allocation_comparison_oos_source_digest(
        contract,
        scenario=scenario,
    )
    opening = result.folds[0].opening_state_digest
    closing = result.folds[-1].closing_state_digest
    if opening is None or closing is None:
        raise ValueError("comparison result state chain is incomplete")

    termination_reasons = result.stitched.diagnostics.termination_reasons
    termination_reason = (
        None if not termination_reasons else ",".join(termination_reasons)
    )
    return AllocationComparisonEvidence(
        contract_digest=contract.digest,
        candidate=candidate,
        scenario=scenario,
        seed=seed,
        policy_digest=policy_digest,
        oos_source_digest=source_digest,
        ledger_digest=allocation_continuous_ledger_summary_digest(
            result, source_digest=source_digest
        ),
        execution_digest=allocation_continuous_execution_summary_digest(
            result, source_digest=source_digest
        ),
        validity_evidence_digest=validity_evidence_digest,
        recipe_digest=_candidate_recipe(contract, candidate),
        opening_state_digest=opening,
        closing_state_digest=closing,
        terminal_profit_rate=terminal_profit,
        max_drawdown=max_drawdown,
        validity=validity,
        coverage_complete=True,
        termination_reason=termination_reason,
    )


__all__ = [
    "allocation_business_objective_digest",
    "allocation_comparison_oos_source_digest",
    "allocation_comparison_scenario_digest",
    "allocation_continuous_execution_summary_digest",
    "allocation_continuous_ledger_summary_digest",
    "allocation_economic_clock_digest",
    "allocation_fold_plan_digest",
    "allocation_policy_schedule_digest",
    "build_continuous_allocation_comparison_evidence",
    "canonical_continuous_allocation_metrics",
    "continuous_account_profit_and_drawdown",
    "validate_continuous_allocation_comparison_context",
]
