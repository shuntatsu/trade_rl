"""Once-call observers of the actual settled collector, model and native buffer.

Capture is diagnostic, not source authentication or bundle admission. The fit
validator owns lower receipt derivation and publication.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.settled_input_receipt import (
    SettledInputRecorder,
    _array_record,
    _tensor_array,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _json,
    native_allocation_account_facts,
)

if TYPE_CHECKING:
    from trade_rl.evaluation.rl_allocation.settled_env import (
        SettledAllocationTradingEnv,
    )

_BUFFER_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "log_probs",
    "advantages",
    "returns",
)


class _InstanceLease:
    """Restore only the wrapper this lease installed; never overwrite a stranger."""

    def __init__(self, owner: Any, name: str, factory: Callable[..., Any]) -> None:
        self.owner, self.name = owner, name
        self.present = name in owner.__dict__
        self.prior_attribute = owner.__dict__.get(name)
        self.original = getattr(owner, name)
        self.wrapper = factory(self.original)
        setattr(owner, name, self.wrapper)

    def restore(self) -> bool:
        if self.owner.__dict__.get(self.name) is not self.wrapper:
            return False
        if self.present:
            setattr(self.owner, self.name, self.prior_attribute)
        else:
            delattr(self.owner, self.name)
        return True


def _best_effort_note(error: BaseException, message: str) -> None:
    """Foreign exception bookkeeping must never replace the original failure."""
    try:
        BaseException.add_note(error, message)
    except BaseException:
        pass


class SettledTransitionRecorder:
    """Fresh single-fit recorder; events and tensor arrays are detached bytes."""

    def __init__(
        self,
        env: SettledAllocationTradingEnv,
        *,
        inputs: SettledInputRecorder | None = None,
    ) -> None:
        self.env, self.inputs = env, inputs or SettledInputRecorder(env)
        self.width = self.inputs.width
        self.n_steps = env.bound.clock.rollout_steps
        self._events: list[bytes] = []
        self._leases: list[_InstanceLease] = []
        self._claimed: list[Any] = []
        self._started = self._attached = self._collecting = False
        self._ended = False
        self._manifest: bytes | None = None
        self._forward: dict[str, Any] | None = None
        self._step_return: dict[str, Any] | None = None
        self._stage: dict[str, Any] | None = None
        self._critic: dict[str, Any] | None = None
        self._critic_result: Any = None
        self._identity: dict[str, Any] | None = None
        self._observation: dict[str, Any] | None = None
        self._awaiting_reset = False
        self._append_count = self._rollout_count = self._update_count = 0
        self._reset_count = 0
        self._declared: bytes | None = None
        self._last_buffer: dict[str, Any] | None = None
        self._call_frontier: dict[str, Any] = {}

    @property
    def events(self) -> tuple[bytes, ...]:
        return tuple(self._events)

    @property
    def complete(self) -> bool:
        return self._manifest is not None

    @property
    def training_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise ValueError("settled capture has no admitted successful publication")
        return json.loads(self._manifest)

    def _emit(self, kind: str, **facts: Any) -> dict[str, Any]:
        raw = canonical_json_bytes(
            {
                "kind": kind,
                "ordinal": len(self._events),
                **facts,
            }
        )
        self._events.append(raw)
        return json.loads(raw)

    def _parameters(self) -> dict[str, Any]:
        result = {}
        for name, tensor in sorted(self.model.policy.state_dict().items()):
            value = _tensor_array(tensor)
            result[name] = _array_record(value, value.dtype.str, value.shape)
        return result

    def _actual_identity(self) -> dict[str, Any]:
        index = self.env.index
        return {
            "episode_id": self.env._episode_sequence,
            "actor_step": self.env.actor_step,
            "actor_index": index,
            "time_ns": int(
                self.env.dataset.timestamps[index]
                .astype("datetime64[ns]")
                .astype(np.int64)
            ),
            "account_id": self.env.account_id,
            "recipe_digest": self.env.recipe_digest,
            "source_envelope_digest": self.env._source_digest,
            "reset_ordinal": self._reset_count - 1,
        }

    def record_failure(self, stage: str, error: BaseException) -> None:
        """Keep diagnostics while callers rethrow the same original exception."""
        try:
            frontier = getattr(error, "settled_execution_receipt", None)
            available_buffer = {}
            if hasattr(self, "buffer"):
                for name in _BUFFER_FIELDS:
                    value = getattr(self.buffer, name)
                    available_buffer[name] = {
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "order": "C",
                        "hex": value.tobytes(order="C").hex(),
                    }
            account = None
            if hasattr(self.env, "book") and hasattr(self.env, "order_book"):
                account = native_allocation_account_facts(
                    self.env.book, self.env.order_book
                )
            self._emit(
                "failure",
                stage=stage,
                error_type=type(error).__name__,
                error_message=str(error),
                frontier_hex=frontier.hex() if isinstance(frontier, bytes) else None,
                pending_forward=self._forward,
                pending_step=self._step_return,
                pending_stage=self._stage,
                successful_append_count=self._append_count,
                actual_model_timesteps=getattr(
                    getattr(self, "model", None), "num_timesteps", None
                ),
                actual_model_n_updates=getattr(
                    getattr(self, "model", None), "_n_updates", None
                ),
                buffer_pos=getattr(getattr(self, "buffer", None), "pos", None),
                available_buffer=available_buffer,
                native_frontier=account,
                available_call_frontier=self._call_frontier,
                actual_frontier_index=getattr(self.env, "index", None),
            )
            setattr(
                error, "settled_capture_trace", canonical_json_bytes(self.payload())
            )
        except BaseException as diagnostic_error:
            _best_effort_note(
                error,
                "Settled diagnostic capture failed: " + type(diagnostic_error).__name__,
            )

    def validate_fit(self, env: SettledAllocationTradingEnv) -> None:
        if env is not self.env or self._started:
            raise ValueError("settled recorder must be fresh for its exact environment")

    def validate_model_frontier(self, model: Any) -> None:
        """Join the final observed update to the actual model being published."""
        if model is not self.model or not self._ended or self._attached:
            raise ValueError(
                "settled model frontier requires completed clean observation"
            )
        updates = [
            json.loads(raw)
            for raw in self._events
            if json.loads(raw)["kind"] == "update"
        ]
        if not updates:
            raise ValueError("settled model frontier has no actual returned update")
        last = updates[-1]
        if (
            canonical_json_bytes(last["parameters_after"])
            != canonical_json_bytes(self._parameters())
            or model.num_timesteps != last["actual_timesteps"]
            or model._n_updates != last["n_updates"]
        ):
            raise ValueError(
                "actual settled model changed after its observed final update"
            )

    def attach(self, model: Any) -> None:
        self.validate_fit(self.env)
        if model.num_timesteps != 0 or model._n_updates != 0:
            raise ValueError("settled recorder requires a fresh actual model")
        self.model, self.buffer = model, model.rollout_buffer
        if self.buffer.buffer_size != self.n_steps or self.buffer.n_envs != 1:
            raise ValueError(
                "settled recorder requires the declared single-account buffer"
            )
        self._declared = canonical_json_bytes(
            {
                "recipe": self.env.recipe,
                "recipe_digest": self.env.recipe_digest,
                "objective": self.env.bound.objective.payload(),
                "settled_clock": self.env.bound.clock.payload(),
                "source_envelope": self.env._source_envelope,
            }
        )
        owners = (self.env, model, model.policy, self.buffer)
        if any(
            getattr(owner, "_settled_capture_owner", None) is not None
            for owner in owners
        ):
            raise ValueError("settled observer owner is already occupied")
        self._started = True
        try:
            for owner in owners:
                setattr(owner, "_settled_capture_owner", self)
                self._claimed.append(owner)
            for owner, name, factory in (
                (self.env, "reset", self._reset_wrapper),
                (self.env, "step", self._step_wrapper),
                (model.policy, "forward", self._forward_wrapper),
                (model.policy, "predict_values", self._critic_wrapper),
                (self.buffer, "add", self._add_wrapper),
                (self.buffer, "compute_returns_and_advantage", self._compute_wrapper),
            ):
                self._leases.append(_InstanceLease(owner, name, factory))
            self._attached = True
        except BaseException as error:
            self.record_failure("attach", error)
            self.detach(primary_error=error)
            raise

    def detach(self, *, primary_error: BaseException | None = None) -> None:
        foreign = []
        for lease in reversed(self._leases):
            try:
                restored = lease.restore()
            except BaseException as error:
                foreign.append(lease.name + ": " + type(error).__name__)
                continue
            if not restored:
                foreign.append(lease.name)
        self._leases.clear()
        for owner in self._claimed:
            try:
                if getattr(owner, "_settled_capture_owner", None) is self:
                    delattr(owner, "_settled_capture_owner")
                else:
                    foreign.append("owner_slot")
            except BaseException as error:
                foreign.append("owner_slot: " + type(error).__name__)
        self._claimed.clear()
        self._attached = False
        if foreign:
            ownership_error = ValueError(
                "settled observer ownership changed: " + ", ".join(foreign)
            )
            self.record_failure("detach", ownership_error)
            if primary_error is not None:
                _best_effort_note(primary_error, str(ownership_error))
                self.record_failure("detach", primary_error)
            else:
                raise ownership_error

    def _delegate(
        self, stage: str, original: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        try:
            return original(*args, **kwargs)
        except BaseException as error:
            self.record_failure(stage + ".original", error)
            raise

    def _observe(self, stage: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except BaseException as error:
            self.record_failure(stage + ".observation", error)
            raise

    def _reset_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def reset(*args: Any, **kwargs: Any) -> Any:
            self._call_frontier = {
                "stage": "env.reset",
                "awaiting_reset": self._awaiting_reset,
            }
            result = self._delegate("env.reset", original, *args, **kwargs)

            def capture() -> None:
                observation, info = result
                self._call_frontier["returned_observation"] = _array_record(
                    observation, "<f4", (self.width,)
                )
                prepared = self._awaiting_reset
                self._reset_count += 1
                identity = self._actual_identity()
                if identity["episode_id"] != info["episode_id"]:
                    raise ValueError(
                        "actual reset return and environment identity differ"
                    )
                self._identity = identity
                self._observation = _array_record(observation, "<f4", (self.width,))
                account = native_allocation_account_facts(
                    self.env.book, self.env.order_book
                )
                self.inputs.record_reset(
                    observation.copy(), identity, prepared=prepared
                )
                self._emit(
                    "reset",
                    reset_ordinal=self._reset_count - 1,
                    prepared=prepared,
                    identity=identity,
                    observation=self._observation,
                    native_account=account,
                    observation_witness={
                        "decision": _json(self.env.decision),
                        "snapshot": self.env._encoded_snapshot.payload(),
                    },
                )
                self._awaiting_reset = False

            self._observe("env.reset", capture)
            return result

        return reset

    def _forward_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def forward(observation: Any, *args: Any, **kwargs: Any) -> Any:
            supplied = _array_record(_tensor_array(observation), "<f4", (1, self.width))
            identity = self._identity
            parameters = content_digest(self._parameters())
            self._call_frontier = {
                "stage": "policy.forward",
                "identity": identity,
                "input": supplied,
                "parameters_digest": parameters,
            }
            result = self._delegate(
                "policy.forward", original, observation, *args, **kwargs
            )

            def capture() -> None:
                if (
                    not self._collecting
                    or self._forward is not None
                    or identity is None
                ):
                    raise ValueError(
                        "actor model call occurred outside its unique collector row"
                    )
                if (
                    self._observation is None
                    or supplied["hex"] != self._observation["hex"]
                ):
                    raise ValueError(
                        "actor tensor differs from actual reset/live observation"
                    )
                actions, values, log_probs = result
                actor = {
                    "observation": supplied,
                    "action": _array_record(_tensor_array(actions), "<i8", (1,)),
                    "value": _array_record(_tensor_array(values), "<f4", (1, 1)),
                    "log_prob": _array_record(_tensor_array(log_probs), "<f4", (1,)),
                }
                self._call_frontier["returned_actor"] = actor
                input_ordinal = self.inputs.record(
                    "actor", _tensor_array(observation), identity
                )
                self._forward = self._emit(
                    "actor_forward",
                    identity=identity,
                    actor=actor,
                    parameters_digest=parameters,
                    input_ordinal=input_ordinal,
                )

            self._observe("policy.forward", capture)
            return result

        return forward

    def _step_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def step(*args: Any, **kwargs: Any) -> Any:
            supplied_action = np.asarray(args[0] if args else kwargs["action"]).copy()
            self._call_frontier = {
                "stage": "env.step",
                "supplied_action": {
                    "dtype": supplied_action.dtype.str,
                    "shape": list(supplied_action.shape),
                    "order": "C",
                    "hex": supplied_action.tobytes(order="C").hex(),
                },
            }
            result = self._delegate("env.step", original, *args, **kwargs)

            def capture() -> None:
                observation, reward, terminated, truncated, info = result
                raw = info["settled_transition"]
                if isinstance(raw, bytes):
                    self._call_frontier["settled_transition_hex"] = raw.hex()
                if (
                    type(raw) is not bytes
                    or canonical_json_bytes(json.loads(raw)) != raw
                ):
                    raise ValueError(
                        "actual settled transition must be detached canonical bytes"
                    )
                facts = json.loads(raw)
                if self._step_return is not None or self._forward is None:
                    raise ValueError("native return lacks its unique actual actor call")
                identity = self._forward["identity"]
                action_record = _array_record(
                    supplied_action, "<i8", supplied_action.shape
                )
                if (
                    supplied_action.size != 1
                    or action_record["hex"] != self._forward["actor"]["action"]["hex"]
                ):
                    raise ValueError(
                        "actual native action differs from sampled model action"
                    )
                if (facts["episode_id"], facts["actor_step"], facts["actor_index"]) != (
                    identity["episode_id"],
                    identity["actor_step"],
                    identity["actor_index"],
                ):
                    raise ValueError("native transition and actor identity differ")
                if (
                    facts["reward"] != reward
                    or facts["terminated"] is not terminated
                    or truncated
                ):
                    raise ValueError("native settled return reward/termination differs")
                self._step_return = self._emit(
                    "env_transition",
                    identity=identity,
                    settled_transition_hex=raw.hex(),
                    supplied_action=action_record,
                    returned_observation=_array_record(
                        observation, "<f4", (self.width,)
                    ),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    observation_witness=None
                    if terminated
                    else {
                        "decision": _json(self.env.decision),
                        "snapshot": self.env._encoded_snapshot.payload(),
                    },
                )
                self._observation = self._step_return["returned_observation"]
                if terminated:
                    if np.any(observation):
                        raise ValueError("true terminal native observation is not zero")
                    terminal_identity = identity | {
                        "physical_frontier_index": facts["settlement_witness"][
                            "frontier_index"
                        ],
                        "actor_steps_processed": facts["settlement_witness"][
                            "actor_steps_processed"
                        ],
                    }
                    self.inputs.record(
                        "terminal_sentinel", observation.copy(), terminal_identity
                    )
                    self._identity = None
                    self._awaiting_reset = True
                else:
                    self._identity = self._actual_identity()

            self._observe("env.step", capture)
            return result

        return step

    def _critic_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def predict_values(observation: Any, *args: Any, **kwargs: Any) -> Any:
            supplied = _array_record(_tensor_array(observation), "<f4", (1, self.width))
            parameters = content_digest(self._parameters())
            self._call_frontier = {
                "stage": "policy.predict_values",
                "identity": self._identity,
                "input": supplied,
                "parameters_digest": parameters,
            }
            result = self._delegate(
                "policy.predict_values", original, observation, *args, **kwargs
            )

            def capture() -> None:
                self._call_frontier["result"] = _array_record(
                    _tensor_array(result), "<f4", (1, 1)
                )
                if (
                    not self._collecting
                    or self._stage is not None
                    or self._append_count != (self._rollout_count + 1) * self.n_steps
                    or self._critic is not None
                    or self._identity is None
                    or self._observation is None
                    or supplied["hex"] != self._observation["hex"]
                ):
                    raise ValueError(
                        "critic call is not the actual unique full-rollout endpoint"
                    )
                last = self._last_appended_stage
                role = (
                    "prepared_reset_boundary_critic"
                    if last["done"]
                    else "live_boundary_critic"
                )
                input_ordinal = self.inputs.record(
                    role, _tensor_array(observation), self._identity
                )
                self._critic = self._emit(
                    "critic",
                    identity=self._identity,
                    role=role,
                    input=supplied,
                    result=_array_record(_tensor_array(result), "<f4", (1, 1)),
                    parameters_digest=parameters,
                    input_ordinal=input_ordinal,
                )
                self._critic_result = result

            self._observe("policy.predict_values", capture)
            return result

        return predict_values

    def _add_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def add(
            obs: Any,
            action: Any,
            reward: Any,
            episode_start: Any,
            value: Any,
            log_prob: Any,
        ) -> Any:
            pos = self.buffer.pos
            arguments = {
                "observations": _array_record(obs.copy(), "<f4", (1, self.width)),
                "actions": _array_record(action.copy(), "<i8", (1, 1)),
                "rewards": _array_record(reward.copy(), "<f4", (1,)),
                "episode_starts": _array_record(episode_start.copy(), "|b1", (1,)),
                "values": _array_record(_tensor_array(value), "<f4", (1, 1)),
                "log_probs": _array_record(_tensor_array(log_prob), "<f4", (1,)),
            }
            self._call_frontier = {
                "stage": "buffer.add",
                "pos_before": pos,
                "arguments": arguments,
            }
            result = self._delegate(
                "buffer.add",
                original,
                obs,
                action,
                reward,
                episode_start,
                value,
                log_prob,
            )

            def capture() -> None:
                stage = self._stage
                if stage is None or pos != stage["local"] or self.buffer.pos != pos + 1:
                    raise ValueError(
                        "returned append position/staged collector row differs"
                    )
                actor = stage["actor"]
                comparisons = (
                    ("observations", "observation"),
                    ("actions", "action"),
                    ("rewards", "reward"),
                    ("values", "value"),
                    ("log_probs", "log_prob"),
                )
                for argument, key in comparisons:
                    if arguments[argument]["hex"] != actor[key]["hex"]:
                        raise ValueError(
                            "actual buffer.add argument differs from model/collector"
                        )
                flag = np.frombuffer(
                    bytes.fromhex(arguments["episode_starts"]["hex"]), dtype=np.bool_
                )[0]
                if bool(flag) is not stage["episode_start"]:
                    raise ValueError(
                        "actual append episode mask differs from collector"
                    )
                stored = {}
                for name in _BUFFER_FIELDS[:6]:
                    shape = (
                        (1, self.width)
                        if name == "observations"
                        else (1, 1)
                        if name == "actions"
                        else (1,)
                    )
                    stored[name] = _array_record(
                        getattr(self.buffer, name)[pos].copy(), "<f4", shape
                    )
                    source = arguments[name]
                    values = np.frombuffer(
                        bytes.fromhex(source["hex"]), dtype=np.dtype(source["dtype"])
                    )
                    expected = values.astype("<f4").reshape(shape)
                    if stored[name]["hex"] != expected.tobytes(order="C").hex():
                        raise ValueError(
                            "actual stored buffer row differs from returned add arguments"
                        )
                self._emit(
                    "append",
                    sequence=stage["sequence"],
                    rollout=stage["rollout"],
                    local=stage["local"],
                    pos_before=pos,
                    pos_after=self.buffer.pos,
                    stage_ordinal=stage["ordinal"],
                    arguments=arguments,
                    stored_row=stored,
                    actual_timesteps=self.model.num_timesteps,
                )
                self._append_count += 1
                self._last_appended_stage = stage
                self._stage = self._forward = self._step_return = None

            self._observe("buffer.add", capture)
            return result

        return add

    def _buffer_snapshot(self) -> dict[str, Any]:
        if (
            self.buffer.pos != self.n_steps
            or not self.buffer.full
            or self.buffer.generator_ready
        ):
            raise ValueError("native buffer must remain chronological before get")
        result = {}
        for name in _BUFFER_FIELDS:
            shape = (
                (self.n_steps, 1, self.width)
                if name == "observations"
                else ((self.n_steps, 1, 1) if name == "actions" else (self.n_steps, 1))
            )
            result[name] = _array_record(
                getattr(self.buffer, name).copy(), "<f4", shape
            )
        return result

    def _compute_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def compute(last_values: Any, dones: np.ndarray) -> Any:
            actual_values = _array_record(_tensor_array(last_values), "<f4", (1, 1))
            actual_dones = _array_record(dones.copy(), "|b1", (1,))
            same_result = last_values is self._critic_result
            self._call_frontier = {
                "stage": "buffer.compute",
                "last_values": actual_values,
                "dones": actual_dones,
                "same_critic_result_object": same_result,
            }
            result = self._delegate(
                "buffer.compute", original, last_values=last_values, dones=dones
            )

            def capture() -> None:
                if (
                    self._critic is None
                    or not same_result
                    or actual_values != self._critic["result"]
                    or self._append_count != (self._rollout_count + 1) * self.n_steps
                    or self._stage is not None
                    or bool(dones[0]) is not self._last_appended_stage["done"]
                    or self._last_buffer is not None
                ):
                    raise ValueError(
                        "returned GAE has missing/reordered actual endpoint capture"
                    )
                self._last_buffer = self._buffer_snapshot()
                self._emit(
                    "rollout",
                    rollout=self._rollout_count,
                    first=self._rollout_count * self.n_steps,
                    last=self._append_count - 1,
                    critic_ordinal=self._critic["ordinal"],
                    last_values=actual_values,
                    dones=actual_dones,
                    same_critic_result_object=same_result,
                    chronological_buffer=self._last_buffer,
                    parameters_before=self._parameters(),
                    actual_timesteps=self.model.num_timesteps,
                )
                self._rollout_count += 1

            self._observe("buffer.compute", capture)
            return result

        return compute

    def _stage_step(self, local_values: dict[str, Any]) -> None:
        if local_values["self"] is not self.model or self._stage is not None:
            raise ValueError(
                "collector callback is not the unique attached native model"
            )
        if (
            len(local_values["infos"]) != 1
            or self._forward is None
            or self._step_return is None
        ):
            raise ValueError(
                "collector stage lacks actual actor/native pre-autoreset capture"
            )
        info = local_values["infos"][0]
        if info.get("TimeLimit.truncated", False):
            raise ValueError(
                "settled collector must not timeout-bootstrap terminal sentinels"
            )
        if (
            info["settled_transition"].hex()
            != self._step_return["settled_transition_hex"]
        ):
            raise ValueError("collector native bytes differ from pre-autoreset return")
        actor = {
            "observation": _array_record(
                _tensor_array(local_values["obs_tensor"]), "<f4", (1, self.width)
            ),
            "action": _array_record(local_values["actions"].copy(), "<i8", (1,)),
            "value": _array_record(
                _tensor_array(local_values["values"]), "<f4", (1, 1)
            ),
            "log_prob": _array_record(
                _tensor_array(local_values["log_probs"]), "<f4", (1,)
            ),
            "reward": _array_record(local_values["rewards"].copy(), "<f4", (1,)),
        }
        for name in ("observation", "action", "value", "log_prob"):
            if actor[name] != self._forward["actor"][name]:
                raise ValueError("collector tensors differ from actual model forward")
        expected_reward = np.asarray([self._step_return["reward"]], dtype="<f4")
        if actor["reward"]["hex"] != expected_reward.tobytes().hex():
            raise ValueError("collector reward differs from signed native transition")
        done = bool(local_values["dones"][0])
        if done is not self._step_return["terminated"]:
            raise ValueError("collector done differs from native termination")
        terminal = None
        if done:
            terminal = _array_record(
                info["terminal_observation"].copy(), "<f4", (self.width,)
            )
            if (
                terminal != self._step_return["returned_observation"]
                or self._awaiting_reset
            ):
                raise ValueError(
                    "terminal sentinel/prepared reset capture is incomplete"
                )
        new_observation = _array_record(
            local_values["new_obs"].copy(), "<f4", (1, self.width)
        )
        if (
            self._observation is None
            or new_observation["hex"] != self._observation["hex"]
        ):
            raise ValueError(
                "collector successor differs from native live/prepared reset"
            )
        self._stage = self._emit(
            "stage",
            sequence=self._append_count,
            rollout=self._rollout_count,
            local=self._append_count - self._rollout_count * self.n_steps,
            actor_forward_ordinal=self._forward["ordinal"],
            env_transition_ordinal=self._step_return["ordinal"],
            identity=self._forward["identity"],
            episode_start=bool(self.model._last_episode_starts[0]),
            done=done,
            actor=actor,
            terminal=terminal,
            next_observation=new_observation,
            next_identity=self._identity,
            next_role="prepared_reset" if done else "live_successor",
        )

    def observe_update(self, rollout: int, optimizer_calls: int, epochs: int) -> None:
        """Called only by the existing actual returned Adam/epoch recorder flush."""
        if rollout != self._update_count + 1 or rollout != self._rollout_count:
            raise ValueError("actual optimizer flush cannot join its preceding rollout")
        if self._collecting or self._last_buffer is None:
            raise ValueError(
                "actual optimizer flush preceded chronological rollout end"
            )
        diagnostics = {}
        for name, value in sorted(self.model.logger.name_to_value.items()):
            if not name.startswith("train/"):
                continue
            value = value.item() if isinstance(value, np.generic) else value
            if isinstance(value, float) and not np.isfinite(value):
                returns = self._last_buffer["returns"]
                actual_returns = np.frombuffer(
                    bytes.fromhex(returns["hex"]), dtype="<f4"
                )
                if name != "train/explained_variance" or np.var(actual_returns) != 0:
                    raise ValueError(
                        "native train diagnostics contain unexplained nonfinite values"
                    )
                diagnostics[name] = {"value": None, "reason": "zero_target_variance"}
            else:
                diagnostics[name] = value
        self._emit(
            "update",
            rollout=rollout - 1,
            actual_timesteps=self.model.num_timesteps,
            optimizer_calls=optimizer_calls,
            epochs=epochs,
            n_updates=self.model._n_updates,
            diagnostics=diagnostics,
            parameters_after=self._parameters(),
        )
        self._update_count += 1

    def callback(self, base: type[Any]) -> Any:
        def training_start(callback: Any) -> None:
            def capture() -> None:
                if self._reset_count != 1 or self._identity is None:
                    raise ValueError(
                        "learn setup did not return its captured initial reset"
                    )
                last = _array_record(
                    self.model._last_obs.copy(), "<f4", (1, self.width)
                )
                if self._observation is None or last["hex"] != self._observation["hex"]:
                    raise ValueError(
                        "model initial input differs from actual reset return"
                    )
                if (
                    self.model.env.reset_infos[0]["episode_id"]
                    != self._identity["episode_id"]
                ):
                    raise ValueError("VecEnv reset identity differs from native reset")
                self._emit("training_start", actual_timesteps=self.model.num_timesteps)

            self._observe("callback.training_start", capture)

        def rollout_start(callback: Any) -> None:
            def capture() -> None:
                if (
                    self._update_count != self._rollout_count
                    or self.buffer.pos != 0
                    or self.buffer.full
                    or self.buffer.generator_ready
                    or self._stage is not None
                    or self._forward is not None
                ):
                    raise ValueError(
                        "rollout start preceded final prior update/clean buffer reset"
                    )
                self._collecting = True
                self._critic = None
                self._critic_result = None
                self._last_buffer = None
                self._emit(
                    "rollout_start",
                    rollout=self._rollout_count,
                    actual_timesteps=self.model.num_timesteps,
                )

            self._observe("callback.rollout_start", capture)

        def step(callback: Any) -> bool:
            self._observe("callback.step", lambda: self._stage_step(callback.locals))
            return True

        def rollout_end(callback: Any) -> None:
            def capture() -> None:
                if (
                    self._last_buffer is None
                    or self._buffer_snapshot() != self._last_buffer
                ):
                    raise ValueError("buffer changed after actual GAE and before get")
                self._collecting = False
                self._emit(
                    "rollout_end",
                    rollout=self._rollout_count - 1,
                    actual_timesteps=self.model.num_timesteps,
                )

            self._observe("callback.rollout_end", capture)

        def training_end(callback: Any) -> None:
            def capture() -> None:
                if (
                    self._collecting
                    or self._stage is not None
                    or self._forward is not None
                    or self._update_count != self._rollout_count
                    or self._append_count != self.model.num_timesteps
                ):
                    raise ValueError(
                        "training end lacks final actual update/complete append capture"
                    )
                self._ended = True
                self._emit("training_end", actual_timesteps=self.model.num_timesteps)

            self._observe("callback.training_end", capture)

        return type(
            "SettledTransitionCallback",
            (base,),
            {
                "_on_training_start": training_start,
                "_on_rollout_start": rollout_start,
                "_on_step": step,
                "_on_rollout_end": rollout_end,
                "_on_training_end": training_end,
            },
        )()

    def payload(self) -> dict[str, Any]:
        events = [json.loads(raw) for raw in self._events]
        return {
            "schema": "allocation_settled_observed_capture_v1",
            "declared": json.loads(self._declared)
            if self._declared is not None
            else None,
            "events": events,
            "inputs": self.inputs.payload(),
            "successful_append_count": sum(
                event["kind"] == "append" for event in events
            ),
            "returned_rollout_count": sum(
                event["kind"] == "rollout" for event in events
            ),
            "returned_update_count": sum(event["kind"] == "update" for event in events),
            "failure_count": sum(event["kind"] == "failure" for event in events),
        }

    def mark_complete(self, validated_manifest: dict[str, Any]) -> None:
        if (
            self._attached
            or not self._ended
            or self._manifest is not None
            or self._append_count <= 0
            or self._rollout_count <= 0
            or self._update_count != self._rollout_count
            or self._append_count != self.model.num_timesteps
            or any(json.loads(raw)["kind"] == "failure" for raw in self._events)
            or validated_manifest.get("schema") != "allocation_ppo_inference_bundle_v6"
        ):
            raise ValueError(
                "capture cannot complete before validation, cleanup and v6 publication"
            )
        # Root has already validated native source/trace/GAE/lower receipt/model.
        # No core event or input role is added here; bound core digests stay stable.
        self._manifest = canonical_json_bytes(validated_manifest)


__all__ = ["SettledTransitionRecorder"]
