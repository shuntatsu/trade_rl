"""Independent constraint-cost storage aligned with SB3 rollout ordering."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.rl.cost_learning import CostLearningSchema
from trade_rl.rl.cost_returns import compute_cost_returns_and_advantages
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode_estimator import (
    CompletionKind,
    classify_completion_kind,
)

_FLOAT32_BYTES = int(np.dtype(np.float32).itemsize)
_FLOAT64_BYTES = int(np.dtype(np.float64).itemsize)
_BOOL_BYTES = int(np.dtype(np.bool_).itemsize)
_UINT8_BYTES = int(np.dtype(np.uint8).itemsize)
_COST_FLOAT_ARRAY_COUNT = 5
_COST_BOOL_ARRAY_COUNT = 2
_COMPLETION_KIND_TO_CODE = {
    CompletionKind.NONE: 0,
    CompletionKind.ECONOMIC_TERMINATION: 1,
    CompletionKind.TIME_LIMIT_COMPLETION: 2,
    CompletionKind.CENSORED_EXTERNAL_TRUNCATION: 3,
}
_CODE_TO_COMPLETION_KIND = np.asarray(
    [
        CompletionKind.NONE,
        CompletionKind.ECONOMIC_TERMINATION,
        CompletionKind.TIME_LIMIT_COMPLETION,
        CompletionKind.CENSORED_EXTERNAL_TRUNCATION,
    ],
    dtype=object,
)


def _positive_integer(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_matrix(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain finite values")
    return array


@dataclass(frozen=True, slots=True)
class CostRolloutBatch:
    """One flattened cost-learning minibatch in canonical head order."""

    cost_names: tuple[str, ...]
    costs: np.ndarray
    old_cost_values: np.ndarray
    cost_advantages: np.ndarray
    cost_returns: np.ndarray


class CostRolloutStorage:
    """Store independent cost transitions without modifying SB3 reward storage."""

    def __init__(
        self,
        *,
        buffer_size: int,
        n_envs: int,
        schema: CostLearningSchema,
        store_episode_metadata: bool = False,
    ) -> None:
        self.buffer_size = _positive_integer(buffer_size, field_name="buffer_size")
        self.n_envs = _positive_integer(n_envs, field_name="n_envs")
        if not isinstance(schema, CostLearningSchema):
            raise TypeError("schema must be a CostLearningSchema")
        if not isinstance(store_episode_metadata, bool):
            raise TypeError("store_episode_metadata must be a boolean")
        self.schema = schema
        self.cost_names = schema.names
        self.n_costs = len(self.cost_names)
        self.store_episode_metadata = store_episode_metadata
        self.reset()

    def reset(self) -> None:
        shape = (self.buffer_size, self.n_envs, self.n_costs)
        self.costs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.terminal_values = np.zeros(shape, dtype=np.float32)
        mask_shape = (self.buffer_size, self.n_envs)
        self.terminated = np.zeros(mask_shape, dtype=np.bool_)
        self.truncated = np.zeros(mask_shape, dtype=np.bool_)
        if self.store_episode_metadata:
            self.transition_elapsed_hours: np.ndarray | None = np.full(
                mask_shape,
                np.nan,
                dtype=np.float64,
            )
            self.completion_kind_codes: np.ndarray | None = np.zeros(
                mask_shape,
                dtype=np.uint8,
            )
        else:
            self.transition_elapsed_hours = None
            self.completion_kind_codes = None
        self.pos = 0
        self.full = False
        self.finalized = False

    @property
    def completion_kinds(self) -> np.ndarray:
        """Return decoded completion kinds for Lagrangian episode estimation."""

        if self.completion_kind_codes is None:
            raise RuntimeError("episode metadata storage is not enabled")
        return _CODE_TO_COMPLETION_KIND[self.completion_kind_codes]

    def _cost_matrix_from_infos(
        self,
        infos: Sequence[Mapping[str, object]],
    ) -> np.ndarray:
        if len(infos) != self.n_envs:
            raise ValueError("infos must contain one mapping per environment")
        rows: list[list[float]] = []
        for info in infos:
            if "constraint_costs" not in info:
                raise ValueError("info is missing constraint_costs")
            costs = info["constraint_costs"]
            if not isinstance(costs, ConstraintCostVector):
                raise ValueError("constraint_costs must be a ConstraintCostVector")
            values = costs.constraint_dict()
            rows.append([float(values[name]) for name in self.cost_names])
        matrix = np.asarray(rows, dtype=np.float32)
        expected_shape = (self.n_envs, self.n_costs)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError("constraint_costs contain invalid values")
        if np.any(matrix < 0.0):
            raise ValueError("constraint_costs must be non-negative")
        return matrix

    def _episode_metadata_from_infos(
        self,
        *,
        infos: Sequence[Mapping[str, object]],
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.store_episode_metadata:
            return None
        elapsed = np.empty(self.n_envs, dtype=np.float64)
        codes = np.empty(self.n_envs, dtype=np.uint8)
        for env_index, info in enumerate(infos):
            raw_elapsed = info.get("transition_elapsed_hours")
            if isinstance(raw_elapsed, bool) or not isinstance(
                raw_elapsed, (int, float)
            ):
                raise ValueError(
                    "transition_elapsed_hours must be a finite positive number"
                )
            elapsed_value = float(raw_elapsed)
            if not math.isfinite(elapsed_value) or elapsed_value <= 0.0:
                raise ValueError(
                    "transition_elapsed_hours must be a finite positive number"
                )
            elapsed[env_index] = elapsed_value

            raw_reason = info.get("termination_reason")
            if raw_reason is not None and not isinstance(raw_reason, str):
                raise ValueError(
                    "completion termination_reason must be a string or null"
                )
            raw_time_limit = info.get("TimeLimit.truncated", False)
            if not isinstance(raw_time_limit, bool):
                raise ValueError("completion TimeLimit.truncated must be a boolean")
            is_terminated = bool(terminated[env_index])
            is_truncated = bool(truncated[env_index])
            if raw_time_limit and not is_truncated:
                raise ValueError("completion time-limit metadata disagrees with flags")
            if is_terminated and raw_time_limit:
                raise ValueError("completion cannot terminate and be a time limit")

            if is_terminated:
                classification_reason = None
            elif is_truncated and raw_reason is None and raw_time_limit:
                classification_reason = "time_limit"
            else:
                classification_reason = raw_reason
            kind = classify_completion_kind(
                terminated=is_terminated,
                truncated=is_truncated,
                truncation_reason=classification_reason,
            )
            codes[env_index] = _COMPLETION_KIND_TO_CODE[kind]
        return elapsed, codes

    def add_from_infos(
        self,
        *,
        infos: Sequence[Mapping[str, object]],
        cost_values: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        terminal_cost_values: np.ndarray,
    ) -> None:
        if self.full:
            raise RuntimeError("cost rollout storage is already full")
        shape = (self.n_envs, self.n_costs)
        value_matrix = _finite_matrix(
            cost_values,
            shape=shape,
            field_name="cost_values",
        )
        terminal_value_matrix = _finite_matrix(
            terminal_cost_values,
            shape=shape,
            field_name="terminal_cost_values",
        )
        terminated_array = np.asarray(terminated, dtype=np.bool_)
        truncated_array = np.asarray(truncated, dtype=np.bool_)
        if terminated_array.shape != (self.n_envs,) or truncated_array.shape != (
            self.n_envs,
        ):
            raise ValueError("termination flags must have shape [environments]")
        if np.any(terminated_array & truncated_array):
            raise ValueError("a transition cannot both terminate and truncate")

        cost_matrix = self._cost_matrix_from_infos(infos)
        episode_metadata = self._episode_metadata_from_infos(
            infos=infos,
            terminated=terminated_array,
            truncated=truncated_array,
        )
        index = self.pos
        self.costs[index] = cost_matrix
        self.values[index] = value_matrix
        self.terminated[index] = terminated_array
        self.truncated[index] = truncated_array
        self.terminal_values[index] = np.where(
            truncated_array[:, None],
            terminal_value_matrix,
            0.0,
        )
        if episode_metadata is not None:
            elapsed, codes = episode_metadata
            if (
                self.transition_elapsed_hours is None
                or self.completion_kind_codes is None
            ):
                raise RuntimeError("episode metadata arrays are unavailable")
            self.transition_elapsed_hours[index] = elapsed
            self.completion_kind_codes[index] = codes
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def finalize(self, *, last_cost_values: np.ndarray) -> None:
        if not self.full:
            raise RuntimeError("cost rollout storage requires a full rollout")
        if self.finalized:
            raise RuntimeError("cost rollout storage is already finalized")
        last_values = _finite_matrix(
            last_cost_values,
            shape=(self.n_envs, self.n_costs),
            field_name="last_cost_values",
        )
        result = compute_cost_returns_and_advantages(
            costs=self.costs,
            values=self.values,
            terminated=self.terminated,
            truncated=self.truncated,
            terminal_values=self.terminal_values,
            last_values=last_values,
            gammas=np.asarray(
                [spec.gamma for spec in self.schema.specs],
                dtype=np.float64,
            ),
            gae_lambdas=np.asarray(
                [spec.gae_lambda for spec in self.schema.specs],
                dtype=np.float64,
            ),
        )
        self.advantages[...] = result.advantages.astype(np.float32, copy=False)
        self.returns[...] = result.returns.astype(np.float32, copy=False)
        self.finalized = True

    def _flatten(self, array: np.ndarray) -> np.ndarray:
        trailing_shape = array.shape[2:]
        return array.swapaxes(0, 1).reshape(
            self.buffer_size * self.n_envs,
            *trailing_shape,
        )

    def sample(self, batch_indices: np.ndarray) -> CostRolloutBatch:
        if not self.finalized:
            raise RuntimeError("cost rollout storage must be finalized before sampling")
        indices = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
        transition_count = self.buffer_size * self.n_envs
        if np.any(indices < 0) or np.any(indices >= transition_count):
            raise IndexError("cost rollout sample index is out of range")
        return CostRolloutBatch(
            cost_names=self.cost_names,
            costs=self._flatten(self.costs)[indices],
            old_cost_values=self._flatten(self.values)[indices],
            cost_advantages=self._flatten(self.advantages)[indices],
            cost_returns=self._flatten(self.returns)[indices],
        )


def estimate_cost_rollout_storage_bytes(
    buffer_size: int,
    n_envs: int,
    n_costs: int,
    *,
    store_episode_metadata: bool = False,
) -> int:
    """Return the exact NumPy payload allocated by ``CostRolloutStorage``."""

    steps = _positive_integer(buffer_size, field_name="buffer_size")
    environments = _positive_integer(n_envs, field_name="n_envs")
    costs = _positive_integer(n_costs, field_name="n_costs")
    if not isinstance(store_episode_metadata, bool):
        raise TypeError("store_episode_metadata must be a boolean")
    transitions = steps * environments
    float_bytes = transitions * costs * _COST_FLOAT_ARRAY_COUNT * _FLOAT32_BYTES
    bool_bytes = transitions * _COST_BOOL_ARRAY_COUNT * _BOOL_BYTES
    metadata_bytes = (
        transitions * (_FLOAT64_BYTES + _UINT8_BYTES) if store_episode_metadata else 0
    )
    return float_bytes + bool_bytes + metadata_bytes


__all__ = [
    "CostRolloutBatch",
    "CostRolloutStorage",
    "estimate_cost_rollout_storage_bytes",
]
