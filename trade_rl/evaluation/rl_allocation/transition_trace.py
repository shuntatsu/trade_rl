"""Freeze actual actor output and verify chronological native buffer admission."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_decision import AllocationActionExecutionResult
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.transition_arrays import (
    array_hex,
    buffer_payload,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    freeze_allocation_execution,
)
from trade_rl.evaluation.rl_allocation.transition_validation import (
    validate_transition_events,
)
from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy


class AllocationTransitionRecorder:
    """Single-fit observer; capture is not successful learning or authorization."""

    def __init__(self, env: AllocationTradingEnv) -> None:
        if env.observation_schema is None:
            raise ValueError("transition trace requires observation v2 or v3")
        self.env = env
        shape = env.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError("transition trace requires a vector observation")
        self.width = shape[0]
        self.n_steps = env.bound.clock.rollout_steps
        self._events: list[bytes] = []
        self._transitions: list[dict[str, Any]] = []
        self._rollouts = 0
        self._episode = -1
        self._started = False
        self._pending: bytes | None = None
        self._manifest: bytes | None = None

    @property
    def events(self) -> tuple[bytes, ...]:
        return tuple(self._events)

    @property
    def complete(self) -> bool:
        return self._manifest is not None

    @property
    def training_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise ValueError("transition trace has no successful final fit")
        return json.loads(self._manifest)

    def attach(self) -> None:
        if self.env._transition_recorder is not None:
            raise ValueError("transition recorder slot is occupied")
        if self._started:
            raise ValueError("transition recorder must be fresh for one fit")
        self._started = True
        self.env._transition_recorder = self

    def validate_fit(self, env: AllocationTradingEnv) -> None:
        if self.env is not env or self._started or env._transition_recorder is not None:
            raise ValueError("transition trace requires its fresh unoccupied account")

    def detach(self) -> None:
        if self.env._transition_recorder is self:
            self.env._transition_recorder = None

    def freeze_execution(self, result: AllocationActionExecutionResult) -> bytes:
        if self.env._transition_recorder is not self or self._pending is not None:
            raise ValueError("transition execution occurred outside its collector")
        self._pending = freeze_allocation_execution(self.env, result)
        return self._pending

    def callback(self, base: type[Any]) -> Any:
        def step(callback: Any) -> bool:
            values = callback.locals
            if len(values["infos"]) != 1 or self._pending is None:
                raise ValueError("transition capture requires one frozen account")
            info = values["infos"][0]
            if info["transition_trace"] != self._pending:
                raise ValueError("transition facts differ from pre-overwrite capture")
            facts = json.loads(self._pending)
            self._pending = None
            episode_start = bool(values["self"]._last_episode_starts[0])
            if episode_start:
                self._episode += 1
            actor = {
                "observation": array_hex(
                    values["obs_tensor"].detach().cpu().numpy(), "<f4", (1, self.width)
                ),
                "action": array_hex(values["actions"], "<i8", (1,)),
                "action_code": int(values["actions"][0]),
                "value": array_hex(
                    values["values"].detach().cpu().numpy(), "<f4", (1, 1)
                ),
                "log_prob": array_hex(
                    values["log_probs"].detach().cpu().numpy(), "<f4", (1,)
                ),
                "reward": array_hex(values["rewards"], "<f4", (1,)),
            }
            done = bool(values["dones"][0])
            terminal = None
            if done:
                if info["TimeLimit.truncated"] or np.any(info["terminal_observation"]):
                    raise ValueError("transition trace requires true zero termination")
                terminal = array_hex(info["terminal_observation"], "<f4", (self.width,))
            sequence = len(self._transitions)
            row = {
                "kind": "transition",
                "sequence": sequence,
                "rollout": sequence // self.n_steps,
                "local": sequence % self.n_steps,
                "episode": self._episode,
                "episode_start": episode_start,
                "done": done,
                "actor": actor,
                "next_observation": array_hex(
                    values["new_obs"], "<f4", (1, self.width)
                ),
                "terminal": terminal,
                "facts": facts,
            }
            self._transitions.append(row)
            self._events.append(canonical_json_bytes(row))
            return True

        def end(callback: Any) -> None:
            values = callback.locals
            buffer = values["self"].rollout_buffer
            rows = self._transitions[self._rollouts * self.n_steps :]
            if (
                len(rows) != self.n_steps
                or not buffer.full
                or buffer.pos != self.n_steps
                or buffer.generator_ready
            ):
                raise ValueError(
                    "transition buffer admission is incomplete or reordered"
                )
            expected = buffer_payload(rows, self.width)
            for name, raw in expected.items():
                actual = getattr(buffer, name)
                shape: tuple[int, ...] = (self.n_steps, 1)
                if name in ("observations", "actions"):
                    shape += (self.width if name == "observations" else 1,)
                if (
                    actual.dtype != np.dtype("<f4")
                    or actual.shape != shape
                    or actual.tobytes().hex() != raw
                ):
                    raise ValueError(
                        f"transition buffer {name} differs from actual actor rows"
                    )
            event = {
                "kind": "rollout",
                "rollout": self._rollouts,
                "first": rows[0]["sequence"],
                "last": rows[-1]["sequence"],
                "buffer_digest": content_digest(expected),
                "new_observation": array_hex(values["new_obs"], "<f4", (1, self.width)),
                "done": bool(values["dones"][0]),
                "index": self.env.index,
                "last_transition_index": values["infos"][0]["execution"].next_index,
            }
            self._events.append(canonical_json_bytes(event))
            self._rollouts += 1

        return type(
            "AllocationTransitionCallback",
            (base,),
            {"_on_step": step, "_on_rollout_end": end},
        )()

    def finish(self, policy: AllocationPPOPolicy) -> None:
        manifest = policy.manifest
        training = manifest["training"]
        if (
            self.env._transition_recorder is not None
            or self._pending is not None
            or manifest["schema"]
            not in (
                "allocation_ppo_inference_bundle_v3",
                "allocation_ppo_inference_bundle_v4",
            )
            or len(self._transitions) != training["actual_timesteps"]
            or self._rollouts * self.n_steps != len(self._transitions)
            or manifest["recipe_digest"] != self.env.recipe_digest
        ):
            raise ValueError(
                "transition trace cannot complete without admitted successful fit"
            )
        validate_transition_events(
            [json.loads(event) for event in self._events], manifest
        )
        self._manifest = canonical_json_bytes(manifest)
