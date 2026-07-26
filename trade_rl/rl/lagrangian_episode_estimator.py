"""Time-aware completed-episode statistics for Lagrangian constraints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from trade_rl.rl.lagrangian import ConstraintEstimate, LagrangianSchema


class CompletionKind(str, Enum):
    """Explicit completion semantics for one environment transition."""

    NONE = "none"
    ECONOMIC_TERMINATION = "economic_termination"
    TIME_LIMIT_COMPLETION = "time_limit_completion"
    CENSORED_EXTERNAL_TRUNCATION = "censored_external_truncation"


def classify_completion_kind(
    *,
    terminated: bool,
    truncated: bool,
    truncation_reason: str | None,
    time_limit_reason: str = "time_limit",
) -> CompletionKind:
    """Classify Gymnasium completion metadata and fail closed on ambiguity."""

    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("completion flags must be booleans")
    if not isinstance(time_limit_reason, str) or not time_limit_reason:
        raise ValueError("time-limit completion reason must be non-empty")
    if terminated and truncated:
        raise ValueError("completion cannot be both terminated and truncated")
    if terminated:
        if truncation_reason is not None:
            raise ValueError("economic completion cannot carry a truncation reason")
        return CompletionKind.ECONOMIC_TERMINATION
    if truncated:
        if not isinstance(truncation_reason, str) or not truncation_reason:
            raise ValueError("truncated completion requires an explicit reason")
        if truncation_reason.startswith("shadow_"):
            return CompletionKind.CENSORED_EXTERNAL_TRUNCATION
        if truncation_reason == time_limit_reason:
            return CompletionKind.TIME_LIMIT_COMPLETION
        raise ValueError(f"unknown completion truncation reason: {truncation_reason}")
    if truncation_reason is not None:
        raise ValueError("non-completion transition cannot carry a truncation reason")
    return CompletionKind.NONE


@dataclass(frozen=True, slots=True)
class EpisodeEstimateBatch:
    """Completed-episode estimates emitted from one rollout ingestion."""

    estimates: dict[str, ConstraintEstimate | None]
    completed_episode_count: int
    censored_episode_count: int


class TimeAwareCompletedEpisodeCostAccumulator:
    """Aggregate explicit policy completions while excluding censored resets."""

    _STATE_VERSION = "time_aware_completed_episode_cost_accumulator_v1"
    _EVENT_TOLERANCE = 1e-12
    _AREA_COSTS = frozenset({"drawdown_excess", "margin_deficit_fraction"})
    _EVENT_COSTS = frozenset({"drawdown_stop_event", "forced_liquidation_event"})
    _DECISION_MEAN_COSTS = frozenset({"gross_exposure_request_excess"})
    _TIME_WEIGHTED_RATE_COSTS = frozenset({"daily_turnover"})
    _EPISODE_SUM_COSTS = frozenset({"execution_cost_fraction"})

    def __init__(self, *, n_envs: int, schema: LagrangianSchema) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
            raise ValueError("n_envs must be a positive integer")
        if not isinstance(schema, LagrangianSchema):
            raise TypeError("schema must be a LagrangianSchema")
        maintained = (
            self._AREA_COSTS
            | self._EVENT_COSTS
            | self._DECISION_MEAN_COSTS
            | self._TIME_WEIGHTED_RATE_COSTS
            | self._EPISODE_SUM_COSTS
        )
        if set(schema.names) != maintained:
            raise ValueError(
                "episode estimator requires the complete canonical cost schema"
            )
        self.n_envs = n_envs
        self.schema = schema
        shape = (n_envs, len(schema.names))
        self._episode_raw_sums = np.zeros(shape, dtype=np.float64)
        self._episode_time_weighted_sums = np.zeros(shape, dtype=np.float64)
        self._episode_elapsed_hours = np.zeros(n_envs, dtype=np.float64)
        self._episode_step_counts = np.zeros(n_envs, dtype=np.int64)
        self._censored_episode_count = 0

    @property
    def censored_episode_count(self) -> int:
        return self._censored_episode_count

    def _validated_rollout(
        self,
        *,
        costs: np.ndarray,
        transition_elapsed_hours: np.ndarray,
        completion_kinds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cost_array = np.asarray(costs, dtype=np.float64)
        expected_suffix = (self.n_envs, len(self.schema.names))
        if cost_array.ndim != 3 or cost_array.shape[1:] != expected_suffix:
            raise ValueError(
                "cost rollout shape must be [steps, n_envs, n_costs] "
                f"with suffix {expected_suffix}"
            )
        if cost_array.shape[0] <= 0:
            raise ValueError("cost rollout must contain at least one step")
        if not np.all(np.isfinite(cost_array)) or np.any(cost_array < 0.0):
            raise ValueError("cost rollout values must be finite and non-negative")

        transition_shape = (cost_array.shape[0], self.n_envs)
        elapsed = np.asarray(transition_elapsed_hours, dtype=np.float64)
        if elapsed.shape != transition_shape:
            raise ValueError(
                f"transition elapsed hours must have shape {transition_shape}"
            )
        if not np.all(np.isfinite(elapsed)) or np.any(elapsed <= 0.0):
            raise ValueError("transition elapsed hours must be finite and positive")

        raw_kinds = np.asarray(completion_kinds, dtype=object)
        if raw_kinds.shape != transition_shape:
            raise ValueError(f"completion kinds must have shape {transition_shape}")
        kinds = np.empty(transition_shape, dtype=object)
        for index in np.ndindex(transition_shape):
            value = raw_kinds[index]
            if not isinstance(value, CompletionKind):
                raise ValueError("completion kinds must contain CompletionKind values")
            kinds[index] = value

        for cost_index, name in enumerate(self.schema.names):
            if name in self._EVENT_COSTS and np.any(
                cost_array[:, :, cost_index] > 1.0 + self._EVENT_TOLERANCE
            ):
                raise ValueError(f"event cost {name} must be within [0, 1]")
        return cost_array, elapsed, kinds

    def _episode_contribution(
        self,
        *,
        env_index: int,
        cost_index: int,
        name: str,
    ) -> float:
        raw_sum = float(self._episode_raw_sums[env_index, cost_index])
        weighted_sum = float(self._episode_time_weighted_sums[env_index, cost_index])
        elapsed_hours = float(self._episode_elapsed_hours[env_index])
        step_count = int(self._episode_step_counts[env_index])
        if elapsed_hours <= 0.0 or step_count <= 0:
            raise RuntimeError("completed episode has no positive elapsed time")
        if name in self._AREA_COSTS:
            contribution = weighted_sum / 24.0
        elif name in self._EVENT_COSTS:
            if raw_sum > 1.0 + self._EVENT_TOLERANCE:
                raise ValueError(
                    f"event cost {name} occurred more than once in one episode"
                )
            contribution = min(raw_sum, 1.0)
        elif name in self._DECISION_MEAN_COSTS:
            contribution = raw_sum / step_count
        elif name in self._TIME_WEIGHTED_RATE_COSTS:
            contribution = weighted_sum / elapsed_hours
        elif name in self._EPISODE_SUM_COSTS:
            contribution = raw_sum
        else:
            raise RuntimeError(f"unsupported canonical episode aggregation: {name}")
        if not math.isfinite(contribution) or contribution < 0.0:
            raise RuntimeError("episode constraint aggregation became invalid")
        return contribution

    def _clear_environment(self, env_index: int) -> None:
        self._episode_raw_sums[env_index].fill(0.0)
        self._episode_time_weighted_sums[env_index].fill(0.0)
        self._episode_elapsed_hours[env_index] = 0.0
        self._episode_step_counts[env_index] = 0

    def ingest_rollout(
        self,
        *,
        costs: np.ndarray,
        transition_elapsed_hours: np.ndarray,
        completion_kinds: np.ndarray,
    ) -> EpisodeEstimateBatch:
        """Consume one rollout and emit statistics for valid policy completions."""

        cost_array, elapsed, kinds = self._validated_rollout(
            costs=costs,
            transition_elapsed_hours=transition_elapsed_hours,
            completion_kinds=completion_kinds,
        )
        raw_sums = self._episode_raw_sums.copy()
        weighted_sums = self._episode_time_weighted_sums.copy()
        elapsed_sums = self._episode_elapsed_hours.copy()
        step_counts = self._episode_step_counts.copy()
        previous_censored_count = self._censored_episode_count
        numerators = np.zeros(len(self.schema.names), dtype=np.float64)
        completed_count = 0
        censored_count = 0

        try:
            self._episode_raw_sums = raw_sums
            self._episode_time_weighted_sums = weighted_sums
            self._episode_elapsed_hours = elapsed_sums
            self._episode_step_counts = step_counts
            for step_index in range(cost_array.shape[0]):
                for env_index in range(self.n_envs):
                    hours = float(elapsed[step_index, env_index])
                    values = cost_array[step_index, env_index]
                    self._episode_raw_sums[env_index] += values
                    self._episode_time_weighted_sums[env_index] += values * hours
                    self._episode_elapsed_hours[env_index] += hours
                    self._episode_step_counts[env_index] += 1
                    for cost_index, name in enumerate(self.schema.names):
                        if (
                            name in self._EVENT_COSTS
                            and self._episode_raw_sums[env_index, cost_index]
                            > 1.0 + self._EVENT_TOLERANCE
                        ):
                            raise ValueError(
                                f"event cost {name} occurred more than once in one episode"
                            )

                    kind = kinds[step_index, env_index]
                    if kind is CompletionKind.NONE:
                        continue
                    if kind is CompletionKind.CENSORED_EXTERNAL_TRUNCATION:
                        censored_count += 1
                        self._clear_environment(env_index)
                        continue
                    if kind not in {
                        CompletionKind.ECONOMIC_TERMINATION,
                        CompletionKind.TIME_LIMIT_COMPLETION,
                    }:
                        raise RuntimeError("unsupported completion kind")
                    for cost_index, name in enumerate(self.schema.names):
                        numerators[cost_index] += self._episode_contribution(
                            env_index=env_index,
                            cost_index=cost_index,
                            name=name,
                        )
                    completed_count += 1
                    self._clear_environment(env_index)
        except Exception:
            self._episode_raw_sums = raw_sums
            self._episode_time_weighted_sums = weighted_sums
            self._episode_elapsed_hours = elapsed_sums
            self._episode_step_counts = step_counts
            self._censored_episode_count = previous_censored_count
            raise

        self._censored_episode_count = previous_censored_count + censored_count
        estimates: dict[str, ConstraintEstimate | None]
        if completed_count == 0:
            estimates = {name: None for name in self.schema.names}
        else:
            estimates = {
                name: ConstraintEstimate(
                    name=name,
                    numerator=float(numerators[index]),
                    denominator=completed_count,
                )
                for index, name in enumerate(self.schema.names)
            }
        return EpisodeEstimateBatch(
            estimates=estimates,
            completed_episode_count=completed_count,
            censored_episode_count=censored_count,
        )

    def state_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible unfinished-episode state."""

        return {
            "censored_episode_count": self._censored_episode_count,
            "cost_names": list(self.schema.names),
            "episode_elapsed_hours": self._episode_elapsed_hours.tolist(),
            "episode_raw_sums": self._episode_raw_sums.tolist(),
            "episode_step_counts": self._episode_step_counts.tolist(),
            "episode_time_weighted_sums": self._episode_time_weighted_sums.tolist(),
            "n_envs": self.n_envs,
            "schema_digest": self.schema.digest,
            "schema_version": self._STATE_VERSION,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore state only when schema identity, shapes, and values match."""

        if not isinstance(state, Mapping):
            raise TypeError("episode estimator state must be a mapping")
        if state.get("schema_version") != self._STATE_VERSION:
            raise ValueError("episode estimator state schema version mismatch")
        raw_names = state.get("cost_names")
        if (
            not isinstance(raw_names, (list, tuple))
            or tuple(raw_names) != self.schema.names
        ):
            raise ValueError("episode estimator state cost schema mismatch")
        if state.get("schema_digest") != self.schema.digest:
            raise ValueError("episode estimator state schema digest mismatch")
        if state.get("n_envs") != self.n_envs:
            raise ValueError("episode estimator state environment count mismatch")
        try:
            raw_sums = np.asarray(state["episode_raw_sums"], dtype=np.float64)
            weighted_sums = np.asarray(
                state["episode_time_weighted_sums"],
                dtype=np.float64,
            )
            elapsed = np.asarray(state["episode_elapsed_hours"], dtype=np.float64)
            raw_steps = np.asarray(state["episode_step_counts"], dtype=np.float64)
            raw_censored = state["censored_episode_count"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("episode estimator state payload is invalid") from error
        expected_cost_shape = self._episode_raw_sums.shape
        expected_env_shape = self._episode_elapsed_hours.shape
        if (
            raw_sums.shape != expected_cost_shape
            or weighted_sums.shape != expected_cost_shape
        ):
            raise ValueError("episode estimator state cost shape mismatch")
        if elapsed.shape != expected_env_shape or raw_steps.shape != expected_env_shape:
            raise ValueError("episode estimator state environment shape mismatch")
        for values, field_name in (
            (raw_sums, "raw sums"),
            (weighted_sums, "weighted sums"),
            (elapsed, "elapsed hours"),
            (raw_steps, "step counts"),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"episode estimator state {field_name} must be finite and non-negative"
                )
        step_counts = raw_steps.astype(np.int64)
        if not np.array_equal(step_counts, raw_steps):
            raise ValueError("episode estimator state step counts must be integers")
        if isinstance(raw_censored, bool) or not isinstance(raw_censored, int):
            raise ValueError("episode estimator censored count must be an integer")
        if raw_censored < 0:
            raise ValueError("episode estimator censored count must be non-negative")
        for cost_index, name in enumerate(self.schema.names):
            if name in self._EVENT_COSTS and np.any(
                raw_sums[:, cost_index] > 1.0 + self._EVENT_TOLERANCE
            ):
                raise ValueError(f"episode estimator event state is invalid for {name}")
        empty = step_counts == 0
        if np.any(elapsed[empty] != 0.0):
            raise ValueError("empty episode state must have zero elapsed time")
        if np.any(raw_sums[empty] != 0.0) or np.any(weighted_sums[empty] != 0.0):
            raise ValueError("empty episode state must have zero cost state")
        if np.any((step_counts > 0) & (elapsed <= 0.0)):
            raise ValueError("non-empty episode state requires positive elapsed time")

        self._episode_raw_sums = raw_sums.copy()
        self._episode_time_weighted_sums = weighted_sums.copy()
        self._episode_elapsed_hours = elapsed.copy()
        self._episode_step_counts = step_counts.copy()
        self._censored_episode_count = raw_censored


__all__ = [
    "CompletionKind",
    "EpisodeEstimateBatch",
    "TimeAwareCompletedEpisodeCostAccumulator",
    "classify_completion_kind",
]
