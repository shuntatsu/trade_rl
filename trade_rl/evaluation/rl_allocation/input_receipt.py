"""Hash actual actor tensors and collector-boundary inputs, not emitted rows."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv


class AllocationInputRecorder:
    def __init__(self, env: AllocationTradingEnv) -> None:
        self.env = env
        shape = env.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError("v2 collector requires a vector observation space")
        self.width = shape[0]
        self.schema_digest = content_digest(env.recipe["observation"])
        self.digests = {
            phase: sha256()
            for phase in ("actor", "rollout_boundary", "terminal_sentinel")
        }
        self.counts = dict.fromkeys(self.digests, 0)

    def _record(self, phase: str, values: np.ndarray, **facts: object) -> None:
        shape = (self.width,) if phase == "terminal_sentinel" else (1, self.width)
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape != shape
            or not np.isfinite(values).all()
        ):
            raise ValueError("actual v2 collector input differs from its float32 shape")
        raw = np.ascontiguousarray(values, dtype="<f4").tobytes()
        header = canonical_json_bytes({"phase": phase, **facts})
        self.digests[phase].update(
            len(header).to_bytes(8, "big") + header + len(raw).to_bytes(8, "big") + raw
        )
        self.counts[phase] += 1

    def callback(
        self, base: type[Any], observe: Callable[[dict[str, Any], dict[str, Any]], bool]
    ) -> Any:
        def step(callback: Any) -> bool:
            values = callback.locals
            result = observe(values, callback.globals)
            execution = values["infos"][0]["execution"]
            self._record(
                "actor",
                values["obs_tensor"].detach().cpu().numpy(),
                index=execution.next_index - 1,
                episode_start=bool(values["self"]._last_episode_starts[0]),
            )
            if bool(values["dones"][0]):
                info = values["infos"][0]
                sentinel = info["terminal_observation"]
                if info["TimeLimit.truncated"] or np.any(sentinel):
                    raise ValueError(
                        "v2 requires true termination with a zero sentinel"
                    )
                self._record("terminal_sentinel", sentinel, index=execution.next_index)
            return result

        def boundary(callback: Any) -> None:
            values = callback.locals
            self._record(
                "rollout_boundary",
                values["new_obs"],
                index=self.env.index,
                last_transition_index=values["infos"][0]["execution"].next_index,
                done=bool(values["dones"][0]),
            )

        return type(
            "AllocationInputCallback",
            (base,),
            {"_on_step": step, "_on_rollout_end": boundary},
        )()

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "allocation_ppo_observation_consumption_v2",
            "observation_schema_digest": self.schema_digest,
            "width": self.width,
            "dtype": "little_endian_float32",
            "actor_count": self.counts["actor"],
            "rollout_boundary_count": self.counts["rollout_boundary"],
            "terminal_count": self.counts["terminal_sentinel"],
            "actor_digest": self.digests["actor"].hexdigest(),
            "rollout_boundary_digest": self.digests["rollout_boundary"].hexdigest(),
            "terminal_digest": self.digests["terminal_sentinel"].hexdigest(),
        }
        if self.env.feature_preprocessing is not None:
            payload.update(
                schema="allocation_ppo_observation_consumption_v3",
                preprocessing_digest=self.env.feature_preprocessing.digest,
            )
        return payload
