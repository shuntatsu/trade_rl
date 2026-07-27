"""Deterministic constrained-policy report aggregation."""

from __future__ import annotations

from statistics import fmean

from trade_rl.domain.constraint_contracts import CONSTRAINT_COST_NAMES
from trade_rl.evaluation.constrained_policy_report._models import (
    ConstrainedPolicyEligibility,
    ConstrainedPolicyReport,
    ConstraintAggregateSummary,
    ConstraintCostSummary,
    ConstraintFoldSummary,
)
from trade_rl.evaluation.constrained_policy_report._observations import (
    ConstraintCostObservation,
    ConstraintFoldEvidence,
    ConstraintPolicyObservation,
)
from trade_rl.evaluation.constrained_policy_report._validation import (
    _BUDGET_TOLERANCE,
    _DIAGNOSTIC_FIELDS,
    _complete_max,
    _complete_mean,
    _require_non_empty,
)


def _cost_summary(
    *,
    seed_costs: tuple[ConstraintCostObservation, ...],
    ensemble_cost: ConstraintCostObservation,
) -> ConstraintCostSummary:
    observations = (*seed_costs, ensemble_cost)
    return ConstraintCostSummary(
        name=ensemble_cost.name,
        aggregation=ensemble_cost.aggregation,
        unit=ensemble_cost.unit,
        budget=ensemble_cost.budget,
        mean=ensemble_cost.observed_value,
        worst_seed=max(cost.observed_value for cost in seed_costs),
        worst_fold=ensemble_cost.observed_value,
        completed_episode_denominator=min(
            cost.completed_episode_denominator for cost in observations
        ),
        censored_episode_count=max(
            cost.censored_episode_count for cost in observations
        ),
        raw_estimate=_complete_mean(tuple(cost.raw_estimate for cost in observations)),
        ema_estimate=_complete_mean(tuple(cost.ema_estimate for cost in observations)),
        multiplier_mean=_complete_mean(
            tuple(cost.multiplier_mean for cost in observations)
        ),
        multiplier_max=_complete_max(
            tuple(cost.multiplier_max for cost in observations)
        ),
        upper_cap_fraction=_complete_mean(
            tuple(cost.upper_cap_fraction for cost in observations)
        ),
        lower_bound_fraction=_complete_mean(
            tuple(cost.lower_bound_fraction for cost in observations)
        ),
        cost_critic_explained_variance=_complete_mean(
            tuple(cost.cost_critic_explained_variance for cost in observations)
        ),
        cost_critic_loss=_complete_mean(
            tuple(cost.cost_critic_loss for cost in observations)
        ),
    )


def _scope(observation: ConstraintPolicyObservation) -> str:
    return "ensemble" if observation.seed is None else f"seed={observation.seed}"


def _scenario_reasons(
    *,
    fold_index: int,
    scenario: str,
    constrained: bool,
    seeds: tuple[ConstraintPolicyObservation, ...],
    ensemble: ConstraintPolicyObservation,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        ensemble.expected_member_policy_digests
        != ensemble.evaluated_member_policy_digests
    ):
        reasons.append(
            f"member_identity_mismatch:fold={fold_index}:scenario={scenario}"
        )
    for observation in seeds:
        if (
            observation.expected_member_policy_digests
            != observation.evaluated_member_policy_digests
        ):
            reasons.append(
                "member_identity_mismatch:"
                f"fold={fold_index}:scenario={scenario}:seed={observation.seed}"
            )
    if not constrained:
        return tuple(sorted(set(reasons)))

    for observation in (*seeds, ensemble):
        if observation.costs is None:
            reasons.append(
                "constraint_evidence_missing:"
                f"fold={fold_index}:scenario={scenario}:scope={_scope(observation)}"
            )
            continue
        for cost in observation.costs:
            if cost.observed_value > cost.budget + _BUDGET_TOLERANCE:
                reasons.append(
                    "constraint_budget_exceeded:"
                    f"fold={fold_index}:scenario={scenario}:cost={cost.name}:"
                    f"scope={_scope(observation)}"
                )
            if cost.completed_episode_denominator < cost.minimum_completed_episodes:
                reasons.append(
                    "constraint_support_below_minimum:"
                    f"fold={fold_index}:scenario={scenario}:cost={cost.name}:"
                    f"observed={cost.completed_episode_denominator}:"
                    f"required={cost.minimum_completed_episodes}"
                )
            for field_name in _DIAGNOSTIC_FIELDS:
                if getattr(cost, field_name) is None:
                    reasons.append(
                        "constraint_diagnostics_missing:"
                        f"fold={fold_index}:scenario={scenario}:cost={cost.name}:"
                        f"field={field_name}"
                    )
    return tuple(sorted(set(reasons)))


def _fold_summary(
    *,
    fold: ConstraintFoldEvidence,
    scenario: str,
    seeds: tuple[ConstraintPolicyObservation, ...],
    ensemble: ConstraintPolicyObservation,
    reasons: tuple[str, ...],
) -> ConstraintFoldSummary:
    observations = (*seeds, ensemble)
    constraints: tuple[ConstraintCostSummary, ...] | None
    if not fold.constrained:
        constraints = None
    elif ensemble.costs is None or any(seed.costs is None for seed in seeds):
        constraints = None
    else:
        seed_cost_sets = tuple(seed.costs for seed in seeds)
        constraints = tuple(
            _cost_summary(
                seed_costs=tuple(
                    cost_set[index]
                    for cost_set in seed_cost_sets
                    if cost_set is not None
                ),
                ensemble_cost=ensemble.costs[index],
            )
            for index in range(len(CONSTRAINT_COST_NAMES))
        )
    return ConstraintFoldSummary(
        fold_index=fold.fold_index,
        configuration=fold.configuration,
        constrained=fold.constrained,
        scenario=scenario,
        seed_count=len(seeds),
        ensemble_policy_digest=ensemble.policy_digest,
        expected_member_policy_digests=ensemble.expected_member_policy_digests,
        evaluated_member_policy_digests=ensemble.evaluated_member_policy_digests,
        total_return=ensemble.total_return,
        worst_seed_total_return=min(seed.total_return for seed in seeds),
        maximum_drawdown=max(item.maximum_drawdown for item in observations),
        maximum_turnover_per_day=max(item.turnover_per_day for item in observations),
        maximum_economic_cost_fraction=max(
            item.economic_cost_fraction for item in observations
        ),
        mean_raw_to_filled_distortion=float(
            fmean(item.raw_to_filled_distortion for item in observations)
        ),
        penalty_to_reward_l2_ratio=(
            _complete_mean(
                tuple(item.penalty_to_reward_l2_ratio for item in observations)
            )
            if fold.constrained
            else None
        ),
        constraints=constraints,
        eligibility=ConstrainedPolicyEligibility(reasons),
    )


def _aggregate_cost(
    summaries: tuple[ConstraintCostSummary, ...],
) -> ConstraintCostSummary:
    first = summaries[0]
    return ConstraintCostSummary(
        name=first.name,
        aggregation=first.aggregation,
        unit=first.unit,
        budget=first.budget,
        mean=float(fmean(summary.mean for summary in summaries)),
        worst_seed=max(summary.worst_seed for summary in summaries),
        worst_fold=max(summary.mean for summary in summaries),
        completed_episode_denominator=min(
            summary.completed_episode_denominator for summary in summaries
        ),
        censored_episode_count=max(
            summary.censored_episode_count for summary in summaries
        ),
        raw_estimate=_complete_mean(
            tuple(summary.raw_estimate for summary in summaries)
        ),
        ema_estimate=_complete_mean(
            tuple(summary.ema_estimate for summary in summaries)
        ),
        multiplier_mean=_complete_mean(
            tuple(summary.multiplier_mean for summary in summaries)
        ),
        multiplier_max=_complete_max(
            tuple(summary.multiplier_max for summary in summaries)
        ),
        upper_cap_fraction=_complete_mean(
            tuple(summary.upper_cap_fraction for summary in summaries)
        ),
        lower_bound_fraction=_complete_mean(
            tuple(summary.lower_bound_fraction for summary in summaries)
        ),
        cost_critic_explained_variance=_complete_mean(
            tuple(summary.cost_critic_explained_variance for summary in summaries)
        ),
        cost_critic_loss=_complete_mean(
            tuple(summary.cost_critic_loss for summary in summaries)
        ),
    )


def _aggregate_summary(
    *,
    configuration: str,
    constrained: bool,
    scenario: str,
    folds: tuple[ConstraintFoldSummary, ...],
    reasons: tuple[str, ...],
) -> ConstraintAggregateSummary:
    constraints: tuple[ConstraintCostSummary, ...] | None
    if not constrained or any(summary.constraints is None for summary in folds):
        constraints = None
    else:
        constraint_sets = tuple(summary.constraints for summary in folds)
        constraints = tuple(
            _aggregate_cost(
                tuple(
                    constraint_set[index]
                    for constraint_set in constraint_sets
                    if constraint_set is not None
                )
            )
            for index in range(len(CONSTRAINT_COST_NAMES))
        )
    return ConstraintAggregateSummary(
        configuration=configuration,
        constrained=constrained,
        scenario=scenario,
        fold_count=len(folds),
        seed_count=sum(summary.seed_count for summary in folds),
        mean_total_return=float(fmean(summary.total_return for summary in folds)),
        worst_seed_total_return=min(
            summary.worst_seed_total_return for summary in folds
        ),
        worst_fold_total_return=min(summary.total_return for summary in folds),
        maximum_drawdown=max(summary.maximum_drawdown for summary in folds),
        maximum_turnover_per_day=max(
            summary.maximum_turnover_per_day for summary in folds
        ),
        maximum_economic_cost_fraction=max(
            summary.maximum_economic_cost_fraction for summary in folds
        ),
        mean_raw_to_filled_distortion=float(
            fmean(summary.mean_raw_to_filled_distortion for summary in folds)
        ),
        penalty_to_reward_l2_ratio=(
            _complete_mean(
                tuple(summary.penalty_to_reward_l2_ratio for summary in folds)
            )
            if constrained
            else None
        ),
        constraints=constraints,
        eligibility=ConstrainedPolicyEligibility(reasons),
    )


def build_constrained_policy_report(
    folds: tuple[ConstraintFoldEvidence, ...],
    *,
    required_scenarios: tuple[str, ...] = ("nominal", "joint_2x"),
) -> ConstrainedPolicyReport:
    """Build one deterministic report without concatenating fold return series."""

    normalized_folds = tuple(sorted(tuple(folds), key=lambda item: item.fold_index))
    if not normalized_folds:
        raise ValueError("constrained policy report requires at least one fold")
    fold_indices = tuple(fold.fold_index for fold in normalized_folds)
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("constrained policy report fold indices must be unique")
    configurations = {fold.configuration for fold in normalized_folds}
    if len(configurations) != 1:
        raise ValueError("constrained policy report requires one configuration")
    constrained_values = {fold.constrained for fold in normalized_folds}
    if len(constrained_values) != 1:
        raise ValueError("constrained policy report mode must be consistent")
    configuration = normalized_folds[0].configuration
    constrained = normalized_folds[0].constrained

    required = tuple(
        _require_non_empty(scenario, field="required_scenario")
        for scenario in tuple(required_scenarios)
    )
    if not required or len(set(required)) != len(required):
        raise ValueError("required scenarios must be non-empty and unique")

    all_scenarios = {
        observation.scenario
        for fold in normalized_folds
        for observation in (*fold.seed_observations, *fold.ensemble_observations)
    }
    scenario_order = (*required, *tuple(sorted(all_scenarios - set(required))))
    scenario_rank = {scenario: index for index, scenario in enumerate(scenario_order)}

    fold_summaries: list[ConstraintFoldSummary] = []
    overall_reasons: list[str] = []
    reasons_by_scenario: dict[str, list[str]] = {
        scenario: [] for scenario in scenario_order
    }
    expected_seed_set: tuple[int, ...] | None = None

    for fold in normalized_folds:
        seed_map = {
            scenario: tuple(
                sorted(
                    (
                        observation
                        for observation in fold.seed_observations
                        if observation.scenario == scenario
                    ),
                    key=lambda item: int(item.seed or 0),
                )
            )
            for scenario in all_scenarios
        }
        ensemble_map = {
            observation.scenario: observation
            for observation in fold.ensemble_observations
        }
        for required_scenario in required:
            seeds = seed_map.get(required_scenario, ())
            ensemble = ensemble_map.get(required_scenario)
            if not seeds or ensemble is None:
                reason = (
                    "missing_required_scenario:"
                    f"fold={fold.fold_index}:scenario={required_scenario}"
                )
                overall_reasons.append(reason)
                reasons_by_scenario[required_scenario].append(reason)
                continue
            observed_seed_set = tuple(int(item.seed or 0) for item in seeds)
            if expected_seed_set is None:
                expected_seed_set = observed_seed_set
            elif observed_seed_set != expected_seed_set:
                reason = (
                    "seed_support_mismatch:"
                    f"fold={fold.fold_index}:scenario={required_scenario}"
                )
                overall_reasons.append(reason)
                reasons_by_scenario[required_scenario].append(reason)

        for scenario in scenario_order:
            seeds = seed_map.get(scenario, ())
            ensemble = ensemble_map.get(scenario)
            if not seeds or ensemble is None:
                continue
            reasons = _scenario_reasons(
                fold_index=fold.fold_index,
                scenario=scenario,
                constrained=constrained,
                seeds=seeds,
                ensemble=ensemble,
            )
            fold_summaries.append(
                _fold_summary(
                    fold=fold,
                    scenario=scenario,
                    seeds=seeds,
                    ensemble=ensemble,
                    reasons=reasons,
                )
            )
            reasons_by_scenario[scenario].extend(reasons)
            if scenario in required:
                overall_reasons.extend(reasons)

    fold_summary_tuple = tuple(
        sorted(
            fold_summaries,
            key=lambda item: (item.fold_index, scenario_rank[item.scenario]),
        )
    )
    aggregate_summaries = tuple(
        _aggregate_summary(
            configuration=configuration,
            constrained=constrained,
            scenario=scenario,
            folds=tuple(
                summary
                for summary in fold_summary_tuple
                if summary.scenario == scenario
            ),
            reasons=tuple(reasons_by_scenario[scenario]),
        )
        for scenario in scenario_order
        if any(summary.scenario == scenario for summary in fold_summary_tuple)
    )
    return ConstrainedPolicyReport(
        configuration=configuration,
        constrained=constrained,
        required_scenarios=required,
        fold_summaries=fold_summary_tuple,
        aggregate_summaries=aggregate_summaries,
        eligibility=ConstrainedPolicyEligibility(tuple(overall_reasons)),
    )
