"""Shared fail-closed deterministic aggregation for SB3 policy ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _validated_action_size(action_size: int | None) -> int | None:
    if action_size is None:
        return None
    if isinstance(action_size, bool) or not isinstance(action_size, int):
        raise TypeError("action_size must be an integer or None")
    if action_size <= 0:
        raise ValueError("action_size must be positive")
    return action_size


def _validated_member_action(
    raw: object,
    *,
    action_size: int | None,
    member_index: int,
    context: str,
) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)
    if action_size is not None and action.shape != (action_size,):
        raise ValueError(f"{context} member {member_index} action shape mismatch")
    if not np.isfinite(action).all():
        raise ValueError(f"{context} member {member_index} action must be finite")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"{context} member {member_index} action violates bounds")
    return action


def predict_deterministic_mean_action(
    models: Sequence[Any],
    observation: object,
    *,
    action_size: int | None = None,
    context: str = "SB3 ensemble",
) -> np.ndarray:
    """Predict every member deterministically and return their validated mean."""

    if not isinstance(context, str) or not context:
        raise ValueError("ensemble context must be a non-empty string")
    if not models:
        raise ValueError(f"{context} must contain at least one member")
    resolved_action_size = _validated_action_size(action_size)
    actions: list[np.ndarray] = []
    for member_index, model in enumerate(models):
        try:
            raw, _ = model.predict(observation, deterministic=True)
        except Exception as error:
            raise ValueError(
                f"{context} member {member_index} prediction failed"
            ) from error
        actions.append(
            _validated_member_action(
                raw,
                action_size=resolved_action_size,
                member_index=member_index,
                context=context,
            )
        )
    if resolved_action_size is None:
        shapes = {action.shape for action in actions}
        if len(shapes) != 1:
            raise ValueError(f"{context} member action shapes disagree")
    averaged = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(averaged).all():
        raise ValueError(f"{context} mean action must be finite")
    return np.asarray(averaged, dtype=np.float32)


__all__ = ["predict_deterministic_mean_action"]
