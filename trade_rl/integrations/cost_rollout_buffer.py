"""Independent constraint-cost storage aligned with SB3 rollout ordering."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.rl.cost_learning import CostLearningSchema
from trade_rl.rl.cost_returns import compute_cost_returns_and_advantages
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode import (
    EpisodeCompletionKind,
    classify_episode_completion,
)

_FLOAT32_BYTES = int(np.dtype(np.float32).itemsize)
_FLOAT64_BYTES = int(np.dtype(np.float64).itemsize)
_BOOL_BYTES = int(np.dtype(np.bool_).itemsize)
_INT8_BYTES = int(np.dtype(np.int8).itemsize)
_COST_FLOAT_ARRAY_COUNT = 5
_COST_BOOL_ARRAY_COUNT = 2


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


def _finite_positive_elapsed(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("transition_elapsed_hours must be finite and positive")
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("transition_elapsed_hours must be finite and positive")
    return elapsed


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
        require_episode_metadata: bool = False,
    ) -> None:
        self.buffer_size = _positive_integer(buffer_size, field_name="buffer_size")
        self.n_envs = _positive_integer(n_envs, field_name="n_envs")
        if not isinstance(schema, CostLearningSchema):
            raise TypeError("schema must be a CostLearningSchema")
        if not isinstance(require_episode_metadata, bool):
            raise TypeError("require_episode_metadata must be a boolean")
        self.schema = schema
        self.cost_names = schema.names
        self.n_costs = len(self.cost_names)
        self.require_episode_metadata = require_episode_metadata
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
        self.elapsed_hours = np.full(mask_shape, np.nan, dtype=np.float64)
        self.completion_kinds = np.full(
            mask_shape,
            EpisodeCompletionKind.NONE,
            dtype=np.int8,
        )
        self.pos = 0
        self.full = False
        self.finalized = False

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

    @staticmethod
    def _elapsed_from_info(info: Mapping[str, object]) -> object | None:
        explicit = info.get("transition_elapsed_hours")
        costs = info.get("constraint_costs")
        vector_elapsed = (
            costs.transition_elapsed_hours
            if isinstance(costs, ConstraintCostVector)
            else None
        )
        if explicit is not None and vector_elapsed is not None:
            explicit_value = _finite_positive_elapsed(explicit)
            if not math.isclose(
                explicit_value,
                vector_elapsed,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("transition elapsed metadata mismatch")
        return explicit if explicit is not None else vector_elapsed

    def _episode_metadata_from_infos(
        self,
        *,
        infos: Sequence[Mapping[str, object]],
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        elapsed_hours = np.full(self.n_envs, np.nan, dtype=np.float64)
        completion_kinds = np.full(
            self.n_envs,
            EpisodeCompletionKind.NONE,
            dtype=np.int8,
        )
        for index, info in enumerate(infos):
            raw_elapsed = self._elapsed_from_info(info)
            if raw_elapsed is None:
                if self.require_episode_metadata:
                    raise ValueError("info is missing transition_elapsed_hours")
                continue
            elapsed = _finite_positive_elapsed(raw_elapsed)
            kind = classify_episode_completion(
                terminated=bool(terminated[index]),
                truncated=bool(truncated[index]),
                time_limit_truncated=bool(info.get("TimeLimit.truncated", False)),
                termination_reason=info.get("termination_reason"),
            )
            elapsed_hours[index] = elapsed
            completion_kinds[index] = int(kind)
        return elapsed_hours, completion_kinds

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
        elapsed_hours, completion_kinds = self._episode_metadata_from_infos(
            infos=infos,
            terminated=terminated_array,
            truncated=truncated_array,
        )

        index = self.pos
        self.costs[index] = self._cost_matrix_from_infos(infos)
        self.values[index] = value_matrix
        self.terminated[index] = terminated_array
        self.truncated[index] = truncated_array
        self.elapsed_hours[index] = elapsed_hours
        self.completion_kinds[index] = completion_kinds
        self.terminal_values[index] = np.where(
            truncated_array[:, None],
            terminal_value_matrix,
            0.0,
        )
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
) -> int:
    """Return the exact NumPy payload allocated by ``CostRolloutStorage``."""

    steps = _positive_integer(buffer_size, field_name="buffer_size")
    environments = _positive_integer(n_envs, field_name="n_envs")
    costs = _positive_integer(n_costs, field_name="n_costs")
    transitions = steps * environments
    float_bytes = transitions * costs * _COST_FLOAT_ARRAY_COUNT * _FLOAT32_BYTES
    bool_bytes = transitions * _COST_BOOL_ARRAY_COUNT * _BOOL_BYTES
    metadata_bytes = transitions * (_FLOAT64_BYTES + _INT8_BYTES)
    return float_bytes + bool_bytes + metadata_bytes


__all__ = [
    "CostRolloutBatch",
    "CostRolloutStorage",
    "estimate_cost_rollout_storage_bytes",
]
