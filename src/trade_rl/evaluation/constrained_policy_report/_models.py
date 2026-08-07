"""Immutable constrained-policy report summaries."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.constrained_policy_report._validation import _SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ConstrainedPolicyEligibility:
    """Stable fail-closed eligibility decision."""

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))

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
            "cost_critic_explained_variance": self.cost_critic_explained_variance,
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
            "evaluated_member_policy_digests": self.evaluated_member_policy_digests,
            "expected_member_policy_digests": self.expected_member_policy_digests,
            "fold_index": self.fold_index,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_economic_cost_fraction": self.maximum_economic_cost_fraction,
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
            "maximum_economic_cost_fraction": self.maximum_economic_cost_fraction,
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
