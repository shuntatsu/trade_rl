"""Chronological independent-account schedule runtime for allocation PPO."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.training_source import allocation_training_source
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
    AllocationTrainingWindow,
)


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def allocation_training_window(
    env: AllocationTradingEnv,
    *,
    role: Literal["train", "validation", "held_out"] = "train",
) -> AllocationTrainingWindow:
    """Bind one complete canonical account episode to its actual source envelope."""
    if type(env) is not AllocationTradingEnv:
        raise ValueError("training window requires an AllocationTradingEnv")
    env.validate_binding()
    source = allocation_training_source(env)
    source_scope = {key: value for key, value in source.items() if key != "dataset_id"}
    return AllocationTrainingWindow(
        role=role,
        dataset_id=source["dataset_id"],
        symbol=source["symbol"],
        start_index=source["start_index"],
        stop_index=source["stop_index"],
        decision_start_ns=_timestamp_ns(env.dataset.timestamps[env.start_index]),
        terminal_time_ns=_timestamp_ns(env.dataset.timestamps[env.stop_index]),
        source_digest=content_digest(source_scope),
    )


class AllocationTrainingScheduleEnv(gym.Env):
    """Cycle declared full episodes; only true reset advances to a new account.

    The wrapper does not reinterpret a PPO rollout boundary as an account boundary.
    Each child remains the canonical AllocationTradingEnv and therefore owns the
    actual risk, execution, accounting and reward transition.
    """

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        schedule: AllocationTrainingSchedule,
        environments: tuple[AllocationTradingEnv, ...],
    ) -> None:
        if type(schedule) is not AllocationTrainingSchedule:
            raise ValueError(
                "runtime requires an immutable allocation training schedule"
            )
        if (
            type(environments) is not tuple
            or not environments
            or any(type(env) is not AllocationTradingEnv for env in environments)
        ):
            raise ValueError(
                "runtime requires immutable allocation training environments"
            )
        if len({id(env) for env in environments}) != len(environments):
            raise ValueError("training environments must be unique account instances")

        first = environments[0]
        for env in environments[1:]:
            if env.bound.clock != first.bound.clock:
                raise ValueError(
                    "scheduled training environments must share one financial clock"
                )
            if env.recipe_digest != first.recipe_digest:
                raise ValueError(
                    "scheduled training environments must share one recipe"
                )
            if (
                env.observation_space.shape != first.observation_space.shape
                or env.observation_space.dtype != first.observation_space.dtype
                or type(env.action_space) is not type(first.action_space)
                or getattr(env.action_space, "n", None)
                != getattr(first.action_space, "n", None)
                or getattr(env.action_space, "start", None)
                != getattr(first.action_space, "start", None)
            ):
                raise ValueError("scheduled policy spaces must be identical")

        actual: dict[str, AllocationTradingEnv] = {}
        actual_windows: dict[str, AllocationTrainingWindow] = {}
        for env in environments:
            env.validate_binding()
            window = allocation_training_window(env)
            if window.window_id in actual:
                raise ValueError(
                    "training environment source identities must be unique"
                )
            actual[window.window_id] = env
            actual_windows[window.window_id] = window
        declared = {window.window_id: window for window in schedule.training_windows}
        if set(actual) != set(declared) or len(actual) != len(
            schedule.train_window_ids
        ):
            raise ValueError("runtime environments must match training windows exactly")
        if any(
            actual_windows[identity].payload() != declared[identity].payload()
            for identity in schedule.train_window_ids
        ):
            raise ValueError(
                "training runtime source lineage differs from its schedule"
            )

        self.schedule = schedule
        self._declared_windows = declared
        self._environments = actual
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        self._cursor = 0
        self._active: AllocationTradingEnv | None = None
        self._active_window_id: str | None = None
        self._awaiting_reset = True
        self._usage = {
            identity: {"reset_count": 0, "decision_count": 0}
            for identity in schedule.train_window_ids
        }

    @property
    def template_env(self) -> AllocationTradingEnv:
        """First declared train window; common policy/clock semantics only."""
        return self._environments[self.schedule.train_window_ids[0]]

    @property
    def training_environments(self) -> tuple[AllocationTradingEnv, ...]:
        """Declared train environments in immutable sampling order."""
        return tuple(
            self._environments[identity] for identity in self.schedule.train_window_ids
        )

    def validate_sources(self) -> None:
        """Rebind every actual child to the immutable declared source envelope."""
        for identity in self.schedule.train_window_ids:
            self._validate_window(identity, self._environments[identity])

    @property
    def active_window_id(self) -> str:
        if self._active_window_id is None:
            raise RuntimeError("training schedule has not been reset")
        return self._active_window_id

    @property
    def active_env(self) -> AllocationTradingEnv:
        if self._active is None:
            raise RuntimeError("training schedule has not been reset")
        return self._active

    def _validate_window(self, identity: str, env: AllocationTradingEnv) -> None:
        current = allocation_training_window(env)
        declared = self._declared_windows[identity]
        if current.window_id != identity or current.payload() != declared.payload():
            raise ValueError("training schedule source changed after declaration")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._active is not None and not self._awaiting_reset:
            raise RuntimeError(
                "training window must reach a true terminal before reset"
            )
        identity = self.schedule.train_window_ids[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.schedule.train_window_ids)
        env = self._environments[identity]
        self._validate_window(identity, env)
        observation, info = env.reset(seed=seed, options=options)
        self._active = env
        self._active_window_id = identity
        self._awaiting_reset = False
        self._usage[identity]["reset_count"] += 1
        return observation, {
            **info,
            "training_window_id": identity,
            "training_window_ordinal": self.schedule.train_window_ids.index(identity),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if (
            self._active is None
            or self._active_window_id is None
            or self._awaiting_reset
        ):
            raise RuntimeError("training schedule requires reset before step")
        observation, reward, terminated, truncated, info = self._active.step(action)
        if truncated:
            raise ValueError(
                "scheduled training cannot treat truncation as a clean reset"
            )
        identity = self._active_window_id
        self._usage[identity]["decision_count"] += 1
        if terminated:
            self._awaiting_reset = True
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                **info,
                "training_window_id": identity,
                "training_window_ordinal": self.schedule.train_window_ids.index(
                    identity
                ),
            },
        )

    def usage_payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_training_schedule_usage_v1",
            "schedule_digest": self.schedule.digest,
            "sampler": self.schedule.sampler,
            "reset_semantics": self.schedule.reset_semantics,
            "rollout_boundary_semantics": self.schedule.rollout_boundary_semantics,
            "windows": [
                {"window_id": identity, **self._usage[identity]}
                for identity in self.schedule.train_window_ids
            ],
        }

    def close(self) -> None:
        for env in self._environments.values():
            env.close()


__all__ = ["AllocationTrainingScheduleEnv", "allocation_training_window"]
