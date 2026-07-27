"""Deterministic fail-closed summaries for constrained-policy evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian_statistics import (
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)

_SCHEMA_VERSION: Final = "constrained_policy_report_v1"
_BUDGET_TOLERANCE: Final = 1e-12
_DIAGNOSTIC_FIELDS: Final = (
    "raw_estimate",
    "ema_estimate",
    "multiplier_mean",
    "multiplier_max",
    "upper_cap_fraction",
    "lower_bound_fraction",
    "cost_critic_explained_variance",
    "cost_critic_loss",
)


def _require_non_empty(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return value.lower()


def _require_finite(
    value: float,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return resolved


def _optional_finite(
    value: float | None,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_finite(
        value,
        field=field,
        minimum=minimum,
        maximum=maximum,
    )


def _require_integer(
    value: int,
    *,
    field: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _normalized_digests(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized = tuple(
        _require_sha256(value, field=f"{field}[{index}]")
        for index, value in enumerate(tuple(values))
    )
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique digests")
    return normalized


def _complete_mean(values: tuple[float | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(fmean(float(value) for value in values if value is not None))


def _complete_max(values: tuple[float | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return max(float(value) for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class ConstraintCostObservation:
    """One canonical cost observed for one seed or deployable ensemble."""

    name: str
    observed_value: float
    budget: float
    completed_episode_denominator: int
    censored_episode_count: int
    minimum_completed_episodes: int
    raw_estimate: float | None
    ema_estimate: float | None
    multiplier_mean: float | None
    multiplier_max: float | None
    upper_cap_fraction: float | None
    lower_bound_fraction: float | None
    cost_critic_explained_variance: float | None
    cost_critic_loss: float | None

    def __post_init__(self) -> None:
        if self.name not in CONSTRAINT_COST_NAMES:
            raise ValueError(f"unknown constraint cost: {self.name}")
        object.__setattr__(
            self,
            "observed_value",
            _require_finite(
                self.observed_value,
                field="observed_value",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "budget",
            _require_finite(self.budget, field="budget", minimum=0.0),
        )
        object.__setattr__(
            self,
            "completed_episode_denominator",
            _require_integer(
                self.completed_episode_denominator,
                field="completed_episode_denominator",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "censored_episode_count",
            _require_integer(
                self.censored_episode_count,
                field="censored_episode_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_completed_episodes",
            _require_integer(
                self.minimum_completed_episodes,
                field="minimum_completed_episodes",
                minimum=1,
            ),
        )
        for field_name in ("raw_estimate", "ema_estimate"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite(
                    getattr(self, field_name),
                    field=field_name,
                    minimum=0.0,
                ),
            )
        for field_name in ("multiplier_mean", "multiplier_max"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite(
                    getattr(self, field_name),
                    field=field_name,
                    minimum=0.0,
                ),
            )
        for field_name in ("upper_cap_fraction", "lower_bound_fraction"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite(
                    getattr(self, field_name),
                    field=field_name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "cost_critic_explained_variance",
            _optional_finite(
                self.cost_critic_explained_variance,
                field="cost_critic_explained_variance",
            ),
        )
        object.__setattr__(
            self,
            "cost_critic_loss",
            _optional_finite(
                self.cost_critic_loss,
                field="cost_critic_loss",
                minimum=0.0,
            ),
        )

    @property
    def aggregation(self) -> str:
        return canonical_constraint_aggregation(self.name).value

    @property
    def unit(self) -> str:
        return canonical_constraint_unit(self.name)

    def digest_payload(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "budget": self.budget,
            "censored_episode_count": self.censored_episode_count,
            "completed_episode_denominator": self.completed_episode_denominator,
            "cost_critic_explained_variance": (
                self.cost_critic_explained_variance
            ),
            "cost_critic_loss": self.cost_critic_loss,
            "ema_estimate": self.ema_estimate,
            "lower_bound_fraction": self.lower_bound_fraction,
            "minimum_completed_episodes": self.minimum_completed_episodes,
            "multiplier_max": self.multiplier_max,
            "multiplier_mean": self.multiplier_mean,
            "name": self.name,
            "observed_value": self.observed_value,
            "raw_estimate": self.raw_estimate,
            "unit": self.unit,
            "upper_cap_fraction": self.upper_cap_fraction,
        }


@dataclass(frozen=True, slots=True)
class ConstraintPolicyObservation:
    """One seed-local or deployable-ensemble scenario observation."""

    scenario: str
    seed: int | None
    policy_digest: str
    expected_member_policy_digests: tuple[str, ...]
    evaluated_member_policy_digests: tuple[str, ...]
    costs: tuple[ConstraintCostObservation, ...] | None
    penalty_to_reward_l2_ratio: float | None
    raw_to_filled_distortion: float
    total_return: float
    maximum_drawdown: float
    turnover_per_day: float
    economic_cost_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario",
            _require_non_empty(self.scenario, field="scenario"),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                _require_integer(self.seed, field="seed", minimum=0),
            )
        object.__setattr__(
            self,
            "policy_digest",
            _require_sha256(self.policy_digest, field="policy_digest"),
        )
        expected = _normalized_digests(
            tuple(self.expected_member_policy_digests),
            field="expected_member_policy_digests",
        )
        evaluated = _normalized_digests(
            tuple(self.evaluated_member_policy_digests),
            field="evaluated_member_policy_digests",
        )
        if self.seed is not None and (
            len(expected) != 1
            or len(evaluated) != 1
            or evaluated[0] != self.policy_digest
        ):
            raise ValueError("seed observation must identify one evaluated policy")
        object.__setattr__(self, "expected_member_policy_digests", expected)
        object.__setattr__(self, "evaluated_member_policy_digests", evaluated)

        costs = None if self.costs is None else tuple(self.costs)
        if costs is not None:
            names = tuple(cost.name for cost in costs)
            if names != CONSTRAINT_COST_NAMES:
                raise ValueError("constraint costs must preserve canonical order")
            if self.penalty_to_reward_l2_ratio is None:
                raise ValueError(
                    "ordinary PPO cannot contain constraint evidence without penalty diagnostics"
                )
        elif self.penalty_to_reward_l2_ratio is not None:
            raise ValueError("ordinary PPO cannot contain constraint penalty evidence")
        object.__setattr__(self, "costs", costs)
        object.__setattr__(
            self,
            "penalty_to_reward_l2_ratio",
            _optional_finite(
                self.penalty_to_reward_l2_ratio,
                field="penalty_to_reward_l2_ratio",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "raw_to_filled_distortion",
            _require_finite(
                self.raw_to_filled_distortion,
                field="raw_to_filled_distortion",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "total_return",
            _require_finite(self.total_return, field="total_return"),
        )
        object.__setattr__(
            self,
            "maximum_drawdown",
            _require_finite(
                self.maximum_drawdown,
                field="maximum_drawdown",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "turnover_per_day",
            _require_finite(
                self.turnover_per_day,
                field="turnover_per_day",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "economic_cost_fraction",
            _require_finite(
                self.economic_cost_fraction,
                field="economic_cost_fraction",
                minimum=0.0,
            ),
        )

    @property
    def constrained(self) -> bool:
        return self.costs is not None

    def digest_payload(self) -> dict[str, object]:
        return {
            "costs": (
                None
                if self.costs is None
                else tuple(cost.digest_payload() for cost in self.costs)
            ),
            "economic_cost_fraction": self.economic_cost_fraction,
            "evaluated_member_policy_digests": (
                self.evaluated_member_policy_digests
            ),
            "expected_member_policy_digests": (
                self.expected_member_policy_digests
            ),
            "maximum_drawdown": self.maximum_drawdown,
            "penalty_to_reward_l2_ratio": self.penalty_to_reward_l2_ratio,
            "policy_digest": self.policy_digest,
            "raw_to_filled_distortion": self.raw_to_filled_distortion,
            "scenario": self.scenario,
            "seed": self.seed,
            "total_return": self.total_return,
            "turnover_per_day": self.turnover_per_day,
        }


@dataclass(frozen=True, slots=True)
class ConstraintFoldEvidence:
    """All seed and deployable evidence for one candidate in one fold."""

    fold_index: int
    configuration: str
    constrained: bool
    seed_observations: tuple[ConstraintPolicyObservation, ...]
    ensemble_observations: tuple[ConstraintPolicyObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold_index",
            _require_integer(self.fold_index, field="fold_index", minimum=0),
        )
        object.__setattr__(
            self,
            "configuration",
            _require_non_empty(self.configuration, field="configuration"),
        )
        if not isinstance(self.constrained, bool):
            raise ValueError("constrained must be a boolean")
        seeds = tuple(self.seed_observations)
        ensembles = tuple(self.ensemble_observations)
        if not seeds or not ensembles:
            raise ValueError("fold evidence requires seed and ensemble observations")
        if any(observation.seed is None for observation in seeds):
            raise ValueError("seed observations require a seed")
        if any(observation.seed is not None for observation in ensembles):
            raise ValueError("ensemble observations cannot declare a seed")
        if any(observation.constrained != self.constrained for observation in (*seeds, *ensembles)):
            if self.constrained:
                raise ValueError("constrained fold requires constraint evidence")
            raise ValueError("ordinary PPO cannot contain constraint evidence")
        seed_keys = tuple((item.scenario, item.seed) for item in seeds)
        if len(set(seed_keys)) != len(seed_keys):
            raise ValueError("seed scenario observations must be unique")
        ensemble_scenarios = tuple(item.scenario for item in ensembles)
        if len(set(ensemble_scenarios)) != len(ensemble_scenarios):
            raise ValueError("ensemble scenario observations must be unique")
        object.__setattr__(
            self,
            "seed_observations",
            tuple(sorted(seeds, key=lambda item: (item.scenario, int(item.seed or 0)))),
        )
        object.__setattr__(
            self,
            "ensemble_observations",
            tuple(sorted(ensembles, key=lambda item: item.scenario)),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "constrained": self.constrained,
            "ensemble_observations": tuple(
                observation.digest_payload()
                for observation in self.ensemble_observations
            ),
            "fold_index": self.fold_index,
            "seed_observations": tuple(
                observation.digest_payload() for observation in self.seed_observations
            ),
        }


@dataclass(frozen=True, slots=True)
class ConstrainedPolicyEligibility:
    """Stable fail-closed eligibility decision."""

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.reasons)))
        object.__setattr__(self, "reasons", normalized)

    @property
    def eligible(self) -> bool:
        return not self.reasons

    def digest_payload(self) -> dict[str, object]:
        return {"eligible": self.eligible, "reasons": self.reasons}


@dataclass(frozen=True, slots=True)
class ConstraintCostSummary:
    """Conservative fold-local or cross-fold summary for one cost."""

    name: str
    aggregation: str
    unit: str
    budget: float
    mean: float
    worst_seed: float
    worst_fold: float
    completed_episode_denominator: int
    censored_episode_count: int
    raw_estimate: float | None
    ema_estimate: float | None
    multiplier_mean: float | None
    multiplier_max: float | None
    upper_cap_fraction: float | None
    lower_bound_fraction: float | None
    cost_critic_explained_variance: float | None
    cost_critic_loss: float | None

    def digest_payload(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "budget": self.budget,
            "censored_episode_count": self.censored_episode_count,
            "completed_episode_denominator": self.completed_episode_denominator,
            "cost_critic_explained_variance": (
                self.cost_critic_explained_variance
            ),
            "cost_critic_loss": self.cost_critic_loss,
            "ema_estimate": self.ema_estimate,
            "lower_bound_fraction": self.lower_bound_fraction,
            "mean": self.mean,
            "multiplier_max": self.multiplier_max,
            "multiplier_mean": self.multiplier_mean,
            "name": self.name,
            "raw_estimate": self.raw_estimate,
            "unit": self.unit,
            "upper_cap_fraction": self.upper_cap_fraction,
            "worst_fold": self.worst_fold,
            "worst_seed": self.worst_seed,
        }


@dataclass(frozen=True, slots=True)
class ConstraintFoldSummary:
    """One fold/scenario deployable summary with seed-distribution evidence."""

    fold_index: int
    configuration: str
    constrained: bool
    scenario: str
    seed_count: int
    ensemble_policy_digest: str
    expected_member_policy_digests: tuple[str, ...]
    evaluated_member_policy_digests: tuple[str, ...]
    total_return: float
    worst_seed_total_return: float
    maximum_drawdown: float
    maximum_turnover_per_day: float
    maximum_economic_cost_fraction: float
    mean_raw_to_filled_distortion: float
    penalty_to_reward_l2_ratio: float | None
    constraints: tuple[ConstraintCostSummary, ...] | None
    eligibility: ConstrainedPolicyEligibility

    def digest_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "constrained": self.constrained,
            "constraints": (
                None
                if self.constraints is None
                else tuple(item.digest_payload() for item in self.constraints)
            ),
            "eligibility": self.eligibility.digest_payload(),
            "ensemble_policy_digest": self.ensemble_policy_digest,
            "evaluated_member_policy_digests": (
                self.evaluated_member_policy_digests
            ),
            "expected_member_policy_digests": (
                self.expected_member_policy_digests
            ),
            "fold_index": self.fold_index,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_economic_cost_fraction": (
                self.maximum_economic_cost_fraction
            ),
            "maximum_turnover_per_day": self.maximum_turnover_per_day,
            "mean_raw_to_filled_distortion": self.mean_raw_to_filled_distortion,
            "penalty_to_reward_l2_ratio": self.penalty_to_reward_l2_ratio,
            "scenario": self.scenario,
            "seed_count": self.seed_count,
            "total_return": self.total_return,
            "worst_seed_total_return": self.worst_seed_total_return,
        }


@dataclass(frozen=True, slots=True)
class ConstraintAggregateSummary:
    """Cross-fold summary for one execution scenario."""

    configuration: str
    constrained: bool
    scenario: str
    fold_count: int
    seed_count: int
    mean_total_return: float
    worst_seed_total_return: float
    worst_fold_total_return: float
    maximum_drawdown: float
    maximum_turnover_per_day: float
    maximum_economic_cost_fraction: float
    mean_raw_to_filled_distortion: float
    penalty_to_reward_l2_ratio: float | None
    constraints: tuple[ConstraintCostSummary, ...] | None
    eligibility: ConstrainedPolicyEligibility

    def digest_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "constrained": self.constrained,
            "constraints": (
                None
                if self.constraints is None
                else tuple(item.digest_payload() for item in self.constraints)
            ),
            "eligibility": self.eligibility.digest_payload(),
            "fold_count": self.fold_count,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_economic_cost_fraction": (
                self.maximum_economic_cost_fraction
            ),
            "maximum_turnover_per_day": self.maximum_turnover_per_day,
            "mean_raw_to_filled_distortion": self.mean_raw_to_filled_distortion,
            "mean_total_return": self.mean_total_return,
            "penalty_to_reward_l2_ratio": self.penalty_to_reward_l2_ratio,
            "scenario": self.scenario,
            "seed_count": self.seed_count,
            "worst_fold_total_return": self.worst_fold_total_return,
            "worst_seed_total_return": self.worst_seed_total_return,
        }


@dataclass(frozen=True, slots=True)
class ConstrainedPolicyReport:
    """Deterministic fold and aggregate summaries for one configuration."""

    configuration: str
    constrained: bool
    required_scenarios: tuple[str, ...]
    fold_summaries: tuple[ConstraintFoldSummary, ...]
    aggregate_summaries: tuple[ConstraintAggregateSummary, ...]
    eligibility: ConstrainedPolicyEligibility
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported constrained policy report schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "aggregate_summaries": tuple(
                summary.digest_payload() for summary in self.aggregate_summaries
            ),
            "configuration": self.configuration,
            "constrained": self.constrained,
            "eligibility": self.eligibility.digest_payload(),
            "fold_summaries": tuple(
                summary.digest_payload() for summary in self.fold_summaries
            ),
            "required_scenarios": self.required_scenarios,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


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
        raw_estimate=_complete_mean(
            tuple(cost.raw_estimate for cost in observations)
        ),
        ema_estimate=_complete_mean(
            tuple(cost.ema_estimate for cost in observations)
        ),
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
    if ensemble.expected_member_policy_digests != ensemble.evaluated_member_policy_digests:
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

    observations = (*seeds, ensemble)
    for observation in observations:
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
            if (
                cost.completed_episode_denominator
                < cost.minimum_completed_episodes
            ):
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
    else:
        if ensemble.costs is None or any(seed.costs is None for seed in seeds):
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
        expected_member_policy_digests=(
            ensemble.expected_member_policy_digests
        ),
        evaluated_member_policy_digests=(
            ensemble.evaluated_member_policy_digests
        ),
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
            tuple(
                summary.cost_critic_explained_variance for summary in summaries
            )
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
    if not constrained:
        constraints = None
    elif any(summary.constraints is None for summary in folds):
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
        seed_map: dict[str, tuple[ConstraintPolicyObservation, ...]] = {}
        for scenario in all_scenarios:
            seed_map[scenario] = tuple(
                sorted(
                    (
                        observation
                        for observation in fold.seed_observations
                        if observation.scenario == scenario
                    ),
                    key=lambda item: int(item.seed or 0),
                )
            )
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


__all__ = [
    "ConstrainedPolicyEligibility",
    "ConstrainedPolicyReport",
    "ConstraintAggregateSummary",
    "ConstraintCostObservation",
    "ConstraintCostSummary",
    "ConstraintFoldEvidence",
    "ConstraintFoldSummary",
    "ConstraintPolicyObservation",
    "build_constrained_policy_report",
]
