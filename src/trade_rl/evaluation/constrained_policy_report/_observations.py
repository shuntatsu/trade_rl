"""Immutable constrained-policy evidence observations."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.domain.constraint_contracts import (
    CONSTRAINT_COST_NAMES,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)
from trade_rl.evaluation.constrained_policy_report._validation import (
    _normalized_digests,
    _optional_finite,
    _require_finite,
    _require_integer,
    _require_non_empty,
    _require_sha256,
)


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
            _require_finite(self.observed_value, field="observed_value", minimum=0.0),
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
            "cost_critic_explained_variance": self.cost_critic_explained_variance,
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
            if tuple(cost.name for cost in costs) != CONSTRAINT_COST_NAMES:
                raise ValueError("constraint costs must preserve canonical order")
            if self.penalty_to_reward_l2_ratio is None:
                raise ValueError(
                    "ordinary PPO cannot contain constraint evidence without "
                    "penalty diagnostics"
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
            "evaluated_member_policy_digests": self.evaluated_member_policy_digests,
            "expected_member_policy_digests": self.expected_member_policy_digests,
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
        if any(
            observation.constrained != self.constrained
            for observation in (*seeds, *ensembles)
        ):
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
