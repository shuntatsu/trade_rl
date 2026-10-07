"""Opt-in actual scheduled actor/buffer observer; never another learner or ledger."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_decision import AllocationActionExecutionResult
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
    validate_scheduled_transition_events,
)
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
)
from trade_rl.evaluation.rl_allocation.transition_arrays import (
    array_hex,
    buffer_payload,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    freeze_allocation_execution,
)
from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def parameter_digests(policy: Any) -> dict[str, str]:
    """Actual named actor/critic bytes, not a parameter-improvement assertion."""
    groups: dict[str, dict[str, Any]] = {"actor": {}, "critic": {}}
    for name, parameter in policy.named_parameters():
        value = parameter.detach().cpu().numpy()
        if value.dtype != np.dtype("<f4") or not np.isfinite(value).all():
            raise ValueError("scheduled diagnostics require finite float32 parameters")
        group = (
            "critic"
            if name.startswith(("mlp_extractor.value_net.", "value_net."))
            else "actor"
        )
        groups[group][name] = {
            "shape": list(value.shape),
            "dtype": "<f4",
            "bytes": value.tobytes().hex(),
        }
    if any(not values for values in groups.values()):
        raise ValueError(
            "scheduled diagnostics require actual actor and critic tensors"
        )
    return {name: content_digest(values) for name, values in groups.items()}


class _ChildObserver:
    def __init__(
        self, owner: ScheduledAllocationTransitionRecorder, child: AllocationTradingEnv
    ) -> None:
        self.owner, self.child = owner, child

    def freeze_execution(self, result: AllocationActionExecutionResult) -> bytes:
        owner = self.owner
        if (
            owner.env.active_env is not self.child
            or self.child._transition_recorder is not self
            or owner._pending is not None
        ):
            raise ValueError(
                "scheduled execution occurred outside its old child collector"
            )
        raw = freeze_allocation_execution(self.child, result)
        owner._pending = (raw, owner._identity(self.child, owner._episode))
        return raw


class ScheduledAllocationTransitionRecorder:
    """Single finite fit; capture consistency is not learnability or authenticity."""

    def __init__(
        self, env: AllocationTrainingScheduleEnv, *, learner_diagnostics: str = "none"
    ) -> None:
        if (
            type(env) is not AllocationTrainingScheduleEnv
            or env.template_env.observation_schema is None
        ):
            raise ValueError(
                "scheduled trace requires the exact explicit-observation schedule"
            )
        if type(learner_diagnostics) is not str or learner_diagnostics not in (
            "none",
            "native_ppo_update_v1",
        ):
            raise ValueError("scheduled learner diagnostics mode is undeclared")
        shape = env.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError("scheduled trace requires vector observations")
        self.env, self.width = env, shape[0]
        self.n_steps = env.template_env.bound.clock.rollout_steps
        self._learner_diagnostics = learner_diagnostics
        self._declared_mode = learner_diagnostics
        self._events: list[bytes] = []
        self._rows: list[dict[str, Any]] = []
        self._adapters = tuple(
            _ChildObserver(self, child) for child in env.training_environments
        )
        self._pending: tuple[bytes, dict[str, Any]] | None = None
        self._bootstrap: dict[str, str] | None = None
        self._manifest: bytes | None = None
        self._started = self._attached = False
        self._episode = self._rollouts = 0
        self._model: Any = None
        self._original_predict: Any = None
        self._predict_wrapper: Any = None
        self._had_predict_attribute = False
        self._pre_parameters: dict[str, str] | None = None
        self._final_parameters: dict[str, str] | None = None

    @property
    def learner_diagnostics(self) -> str:
        if self._learner_diagnostics != self._declared_mode:
            raise ValueError("scheduled diagnostics mode changed after declaration")
        return self._declared_mode

    @property
    def events(self) -> tuple[bytes, ...]:
        return tuple(self._events)

    @property
    def complete(self) -> bool:
        return self._manifest is not None

    @property
    def training_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise ValueError("scheduled trace has no admitted successful final fit")
        return json.loads(self._manifest)

    def validate_fit(
        self,
        env: AllocationTrainingScheduleEnv,
        protocol: AllocationPPOTrainingProtocol,
    ) -> None:
        if (
            env is not self.env
            or self._started
            or protocol.n_steps != self.n_steps
            or any(a.child._transition_recorder is not None for a in self._adapters)
        ):
            raise ValueError(
                "scheduled trace requires its fresh unoccupied accounts and clock"
            )
        if any(
            row["reset_count"] or row["decision_count"]
            for row in cast(list[dict[str, Any]], env.usage_payload()["windows"])
        ):
            raise ValueError("scheduled trace requires an unused schedule")

    def _identity(self, child: AllocationTradingEnv, episode: int) -> dict[str, Any]:
        ordinal = next(
            i
            for i, value in enumerate(self.env.training_environments)
            if value is child
        )
        window = self.env.schedule.training_windows[ordinal]
        return {
            "window_id": window.window_id,
            "dataset_id": child.dataset.dataset_id,
            "source_scope_digest": window.source_digest,
            "window_ordinal": ordinal,
            "episode": episode,
            "index": child.index,
            "time_ns": int(
                child.dataset.timestamps[child.index]
                .astype("datetime64[ns]")
                .astype(np.int64)
            ),
        }

    def attach(self, model: Any) -> None:
        if self._started or any(
            a.child._transition_recorder is not None for a in self._adapters
        ):
            raise ValueError("scheduled trace requires fresh unoccupied observer slots")
        self._model = model
        self._original_predict = model.policy.predict_values
        self._had_predict_attribute = "predict_values" in model.policy.__dict__

        def predict(observation: Any) -> Any:
            if self._bootstrap is not None:
                raise ValueError("scheduled rollout bootstrap was repeated")
            tensor = array_hex(
                observation.detach().cpu().numpy(), "<f4", (1, self.width)
            )
            value = self._original_predict(observation)
            self._bootstrap = {
                "observation": tensor,
                "value": array_hex(value.detach().cpu().numpy(), "<f4", (1, 1)),
            }
            return value

        self._predict_wrapper = predict
        self._started = self._attached = True
        try:
            model.policy.predict_values = predict
            for adapter in self._adapters:
                adapter.child._transition_recorder = adapter
        except BaseException:
            self.detach()
            raise

    def detach(self) -> None:
        for adapter in self._adapters:
            if adapter.child._transition_recorder is adapter:
                adapter.child._transition_recorder = None
        if self._attached:
            policy = self._model.policy
            replaced = policy.predict_values is not self._predict_wrapper
            if self._had_predict_attribute:
                policy.predict_values = self._original_predict
            else:
                del policy.predict_values
            self._attached = False
            if replaced:
                raise ValueError("scheduled bootstrap observer was replaced")

    def _append(self, event: dict[str, Any]) -> None:
        self._events.append(canonical_json_bytes(event))

    def callback(self, base: type[Any]) -> Any:
        def start(callback: Any) -> None:
            model = callback.locals["self"]
            identity = self._identity(self.env.active_env, 0)
            info = model.env.reset_infos[0]
            if (
                info["training_window_id"] != identity["window_id"]
                or info["training_window_ordinal"] != 0
            ):
                raise ValueError(
                    "scheduled initial reset differs from declared first child"
                )
            self._append(
                {
                    "kind": "start",
                    "identity": identity,
                    "observation": array_hex(model._last_obs, "<f4", (1, self.width)),
                }
            )

        def step(callback: Any) -> bool:
            values = callback.locals
            if (
                len(values["infos"]) != 1
                or len(values["dones"]) != 1
                or self._pending is None
            ):
                raise ValueError(
                    "scheduled actor capture requires one frozen old account"
                )
            raw, old = self._pending
            info = values["infos"][0]
            if (
                info["transition_trace"] != raw
                or info["training_window_id"] != old["window_id"]
                or info["training_window_ordinal"] != old["window_ordinal"]
            ):
                raise ValueError(
                    "scheduled old child info differs from pre-overwrite capture"
                )
            self._pending = None
            done = bool(values["dones"][0])
            terminal = None
            if done:
                if info["TimeLimit.truncated"] or np.any(info["terminal_observation"]):
                    raise ValueError(
                        "scheduled trace requires true zero terminal sentinel"
                    )
                terminal = array_hex(info["terminal_observation"], "<f4", (self.width,))
                self._episode += 1
            current = self._identity(self.env.active_env, self._episode)
            if done:
                reset_info = values["self"].env.reset_infos[0]
                if (
                    reset_info["training_window_id"] != current["window_id"]
                    or reset_info["training_window_ordinal"]
                    != current["window_ordinal"]
                ):
                    raise ValueError(
                        "scheduled autoreset differs from prepared next child"
                    )
            sequence = len(self._rows)
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
            row = {
                "kind": "transition",
                "sequence": sequence,
                "rollout": sequence // self.n_steps,
                "local": sequence % self.n_steps,
                "old": old,
                "episode_start": bool(values["self"]._last_episode_starts[0]),
                "done": done,
                "actor": actor,
                "next": current,
                "next_role": "prepared_reset" if done else "live_successor",
                "next_observation": array_hex(
                    values["new_obs"], "<f4", (1, self.width)
                ),
                "terminal": terminal,
                "facts": json.loads(raw),
            }
            self._rows.append(row)
            self._append(row)
            return True

        def end(callback: Any) -> None:
            values = callback.locals
            buffer = values["self"].rollout_buffer
            rows = self._rows[self._rollouts * self.n_steps :]
            if (
                len(rows) != self.n_steps
                or not buffer.full
                or buffer.pos != self.n_steps
                or buffer.generator_ready
                or self._bootstrap is None
            ):
                raise ValueError(
                    "scheduled buffer/bootstrap admission is incomplete or reordered"
                )
            expected = buffer_payload(rows, self.width)
            for name, raw in expected.items():
                shape: tuple[int, ...] = (self.n_steps, 1)
                if name in ("observations", "actions"):
                    shape += (self.width if name == "observations" else 1,)
                if array_hex(getattr(buffer, name), "<f4", shape) != raw:
                    raise ValueError(
                        f"scheduled buffer {name} differs from actual actor rows"
                    )
            current = self._identity(self.env.active_env, self._episode)
            event = {
                "kind": "rollout",
                "rollout": self._rollouts,
                "first": rows[0]["sequence"],
                "last": rows[-1]["sequence"],
                "buffer_digest": content_digest(expected),
                "last_old": rows[-1]["old"],
                "current": current,
                "done": rows[-1]["done"],
                "bootstrap": self._bootstrap,
                "advantages": array_hex(buffer.advantages, "<f4", (self.n_steps, 1)),
                "returns": array_hex(buffer.returns, "<f4", (self.n_steps, 1)),
            }
            if self.learner_diagnostics != "none":
                self._pre_parameters = parameter_digests(self._model.policy)
            self._append(event)
            self._bootstrap = None
            self._rollouts += 1

        return type(
            "ScheduledAllocationTransitionCallback",
            (base,),
            {"_on_step": step, "_on_training_start": start, "_on_rollout_end": end},
        )()

    def observe_update(self, rollout: int, steps: int, epochs: int) -> None:
        if self.learner_diagnostics == "none":
            return
        if rollout != self._rollouts or self._pre_parameters is None:
            raise ValueError(
                "scheduled update is not bound to its prior admitted rollout"
            )
        logged = self._model.logger.name_to_value
        if (
            type(logged.get("train/n_updates")) is not int
            or logged["train/n_updates"] != self._model._n_updates
        ):
            raise ValueError("scheduled diagnostics logger update is missing or stale")
        names = (
            "value_loss",
            "policy_gradient_loss",
            "entropy_loss",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        )
        diagnostics: dict[str, float | None] = {}
        for name in names:
            if f"train/{name}" not in logged or isinstance(
                logged[f"train/{name}"], (bool, np.bool_)
            ):
                raise ValueError("scheduled native diagnostic is missing or boolean")
            value = float(logged[f"train/{name}"])
            if name == "explained_variance" and np.isnan(value):
                diagnostics[name] = None
            elif not np.isfinite(value):
                raise ValueError("scheduled native diagnostic is nonfinite")
            else:
                diagnostics[name] = value
        boundary = json.loads(self._events[-1])
        if boundary["kind"] != "rollout":
            raise ValueError("scheduled update does not immediately follow its rollout")
        self._append(
            {
                "kind": "update",
                "rollout": rollout - 1,
                "first": boundary["first"],
                "last": boundary["last"],
                "buffer_digest": boundary["buffer_digest"],
                "optimizer_steps": steps,
                "epochs": epochs,
                "n_updates": int(self._model._n_updates),
                "diagnostics": diagnostics,
                "explained_variance_reason": "zero_target_variance"
                if diagnostics["explained_variance"] is None
                else None,
                "parameters_before": self._pre_parameters,
                "parameters_after": parameter_digests(self._model.policy),
            }
        )
        self._pre_parameters = None

    def finish(self, policy: AllocationPPOPolicy) -> None:
        manifest = policy.manifest
        if (
            not self._started
            or self._attached
            or self._pending is not None
            or self._bootstrap is not None
            or any(a.child._transition_recorder is not None for a in self._adapters)
            or manifest["recipe_digest"] != self.env.template_env.recipe_digest
        ):
            raise ValueError(
                "scheduled trace cannot complete without detached successful fit"
            )
        validate_scheduled_transition_events(
            [json.loads(event) for event in self._events],
            manifest,
            learner_diagnostics=self.learner_diagnostics,
        )
        final_parameters = parameter_digests(self._model.policy)
        if self.learner_diagnostics != "none":
            final_event = json.loads(self._events[-1])
            if final_event["parameters_after"] != final_parameters:
                raise ValueError(
                    "scheduled parameters changed after final native update"
                )
        self._final_parameters = final_parameters
        self._manifest = canonical_json_bytes(manifest)

    def validate_current_parameters(self) -> None:
        """Detect accidental post-fit mutation before sidecar/saved-model linkage."""
        if (
            self._final_parameters is None
            or parameter_digests(self._model.policy) != self._final_parameters
        ):
            raise ValueError(
                "scheduled final live parameters changed after successful fit"
            )
