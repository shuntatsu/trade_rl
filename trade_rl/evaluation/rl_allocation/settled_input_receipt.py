"""Observed settled model inputs; no learner import or replay-derived evidence."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest

if TYPE_CHECKING:
    from trade_rl.evaluation.rl_allocation.settled_env import (
        SettledAllocationTradingEnv,
    )


def _tensor_array(value: Any) -> np.ndarray:
    """Copy a tensor already supplied/returned by the actual model."""
    return value.detach().cpu().numpy().copy()


def _array_record(
    value: np.ndarray, dtype: str, shape: tuple[int, ...]
) -> dict[str, Any]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(dtype)
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError("settled observed array dtype/shape/finite values differ")
    return {
        "dtype": dtype,
        "shape": list(shape),
        "order": "C",
        "hex": value.tobytes(order="C").hex(),
    }


class SettledInputRecorder:
    """Immutable input observations; counts are derived from captured events."""

    def __init__(self, env: SettledAllocationTradingEnv) -> None:
        shape = env.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError("settled capture requires a vector observation")
        self.width = shape[0]
        self.schema_digest = content_digest(env.recipe["observation"])
        self._events: list[bytes] = []
        self._resets: list[bytes] = []

    @property
    def events(self) -> tuple[bytes, ...]:
        return tuple(self._events)

    def record(self, role: str, values: np.ndarray, identity: dict[str, Any]) -> int:
        if role not in (
            "actor",
            "live_boundary_critic",
            "prepared_reset_boundary_critic",
            "terminal_sentinel",
        ):
            raise ValueError("settled input role must denote an actual observation")
        shape = (self.width,) if role == "terminal_sentinel" else (1, self.width)
        tensor = _array_record(values, "<f4", shape)
        if role == "terminal_sentinel" and np.any(values):
            raise ValueError("true terminal observation must be a zero sentinel")
        ordinal = len(self._events)
        self._events.append(
            canonical_json_bytes(
                {
                    "kind": "input",
                    "ordinal": ordinal,
                    "role": role,
                    "identity": identity,
                    "tensor": tensor,
                    "model_consumed": role != "terminal_sentinel",
                }
            )
        )
        return ordinal

    def record_reset(
        self, values: np.ndarray, identity: dict[str, Any], *, prepared: bool
    ) -> None:
        self._resets.append(
            canonical_json_bytes(
                {
                    "identity": identity,
                    "prepared": prepared,
                    "tensor": _array_record(values, "<f4", (self.width,)),
                }
            )
        )

    def payload(self) -> dict[str, Any]:
        events = [json.loads(raw) for raw in self._events]
        roles = (
            "actor",
            "live_boundary_critic",
            "prepared_reset_boundary_critic",
            "terminal_sentinel",
        )
        counts, digests = {}, {}
        for role in roles:
            digest, count = sha256(), 0
            for event in events:
                if event["role"] != role:
                    continue
                tensor = event["tensor"]
                raw = bytes.fromhex(tensor["hex"])
                header = canonical_json_bytes(
                    {key: value for key, value in event.items() if key != "tensor"}
                    | {"tensor": {k: v for k, v in tensor.items() if k != "hex"}}
                )
                digest.update(
                    len(header).to_bytes(8, "big")
                    + header
                    + len(raw).to_bytes(8, "big")
                    + raw
                )
                count += 1
            counts[role], digests[role] = count, digest.hexdigest()
        actor_episodes = {
            event["identity"]["episode_id"]
            for event in events
            if event["role"] == "actor"
        }
        unused = [
            json.loads(raw)
            for raw in self._resets
            if json.loads(raw)["prepared"]
            and json.loads(raw)["identity"]["episode_id"] not in actor_episodes
        ]
        return {
            "schema": "allocation_settled_input_observations_v1",
            "observation_schema_digest": self.schema_digest,
            "width": self.width,
            "dtype": "little_endian_float32",
            "events": events,
            "role_counts": counts,
            "role_digests": digests,
            # This role is derived from real reset/forward ordering, never added
            # as a fictional model call during finalization.
            "prepared_reset_without_actor": unused,
            "prepared_reset_without_actor_count": len(unused),
        }


__all__ = ["SettledInputRecorder"]
