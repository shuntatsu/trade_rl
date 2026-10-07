"""Fixed explicit constructors and read-only completed optimizer-call telemetry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.strategies.rl.allocation_model import (
    validate_allocation_backend_versions,
    validate_allocation_protocol_model,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def validate_protocol_clock(
    env: AllocationTradingEnv, protocol: AllocationPPOTrainingProtocol
) -> None:
    if not isinstance(protocol, AllocationPPOTrainingProtocol):
        raise ValueError("training_protocol must be an explicit immutable declaration")
    if env.observation_schema is None:
        raise ValueError("explicit training protocol requires observation v2")
    clock = env.bound.clock
    if (protocol.n_steps, protocol.gamma, protocol.gae_lambda) != (
        clock.rollout_steps,
        clock.gamma,
        clock.gae_lambda,
    ):
        raise ValueError("explicit protocol differs from the financial clock")


def construct_protocol_ppo(
    env: AllocationTradingEnv, protocol: AllocationPPOTrainingProtocol, *, seed: int
) -> Any:
    validate_allocation_backend_versions()
    torch = importlib.import_module("torch")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("explicit protocol requires ambient float32 construction")
    sb3 = importlib.import_module("stable_baselines3")
    buffers = importlib.import_module("stable_baselines3.common.buffers")
    layers = importlib.import_module("stable_baselines3.common.torch_layers")
    torch.set_num_threads(1)
    payload: dict[str, Any] = protocol.payload()
    kwargs = {
        key: value
        for key, value in payload["ppo"].items()
        if key
        not in {
            "learning_rate_schedule",
            "clip_range_schedule",
            "clip_range_vf_schedule",
            "rollout_buffer_class",
        }
    }
    kwargs["rollout_buffer_class"] = buffers.RolloutBuffer
    policy_kwargs = {
        key: value
        for key, value in payload["policy"].items()
        if key not in {"class", "activation_fn", "features_extractor_class"}
    }
    optimizer_kwargs = {
        key: value
        for key, value in payload["optimizer"].items()
        if key not in {"class", "learning_rate_source"}
    }
    optimizer_kwargs["betas"] = protocol.adam_betas
    policy_kwargs.update(
        activation_fn=torch.nn.Tanh,
        features_extractor_class=layers.FlattenExtractor,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
    )
    model = sb3.PPO(
        "MlpPolicy", env, seed=seed, device="cpu", policy_kwargs=policy_kwargs, **kwargs
    )
    validate_allocation_protocol_model(model, protocol)
    return model


class AllocationUpdateRecorder:
    """Count returned Adam calls, not parameter improvement or successful learning."""

    def __init__(
        self,
        model: Any,
        *,
        completed_update: Callable[[int, int, int], None] | None = None,
    ) -> None:
        if model.num_timesteps != 0 or model._n_updates != 0:
            raise ValueError("explicit protocol requires a fresh model")
        self.model = model
        self.steps = self.rollouts = self.epochs = 0
        self.digest = sha256()
        self.pending: tuple[int, int] | None = None
        self._completed_update = completed_update
        self._previous_steps = 0
        self.handle = model.policy.optimizer.register_step_post_hook(self._step)

    def _step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        del args, kwargs
        if optimizer is not self.model.policy.optimizer or self.pending is None:
            raise ValueError(
                "optimizer call occurred outside the observed rollout update"
            )
        self.steps += 1
        event = canonical_json_bytes(
            {
                "step": self.steps,
                "rollout": self.rollouts + 1,
                "timesteps": self.model.num_timesteps,
            }
        )
        self.digest.update(len(event).to_bytes(8, "big") + event)

    def _flush(self) -> None:
        if self.pending is not None:
            epoch_start, timesteps = self.pending
            if self.model.num_timesteps != timesteps:
                raise ValueError("update capture must precede the next collection")
            entered_epochs = self.model._n_updates - epoch_start
            self.epochs += entered_epochs
            self.rollouts += 1
            self.pending = None
            if self._completed_update is not None:
                self._completed_update(
                    self.rollouts, self.steps - self._previous_steps, entered_epochs
                )
            self._previous_steps = self.steps

    def callback(self, base: type[Any]) -> Any:
        def start(_: Any) -> None:
            self._flush()

        def end(_: Any) -> None:
            if self.pending is not None:
                raise ValueError("previous rollout update was not captured")
            self.pending = (self.model._n_updates, self.model.num_timesteps)

        return type(
            "AllocationUpdateCallback",
            (base,),
            {
                "_on_step": lambda _: True,
                "_on_rollout_start": start,
                "_on_rollout_end": end,
                "_on_training_end": start,
            },
        )()

    def close(self) -> None:
        self.handle.remove()

    def payload(self) -> dict[str, object]:
        self._flush()
        return {
            "schema": "allocation_ppo_optimization_receipt_v1",
            "rollout_update_count": self.rollouts,
            "successful_optimizer_step_calls": self.steps,
            "epoch_iterations": self.epochs,
            "optimizer_step_events_digest": self.digest.hexdigest(),
            "final_capture": "after_final_train_v1",
        }
