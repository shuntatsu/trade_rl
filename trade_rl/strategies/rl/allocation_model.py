"""Explicit four-action allocation PPO inference with a frozen receipt."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def validate_allocation_backend_versions() -> None:
    for distribution, expected in (("stable-baselines3", "2.3.2"), ("torch", "2.4.1")):
        try:
            actual = version(distribution).split("+", 1)[0]
        except PackageNotFoundError as error:
            raise ValueError("explicit protocol backend is unavailable") from error
        if actual != expected:
            raise ValueError("installed backend release differs from explicit protocol")


def _class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_linear(layer: Any, inputs: int, outputs: int) -> None:
    if (
        _class_name(layer) != "torch.nn.modules.linear.Linear"
        or (layer.in_features, layer.out_features) != (inputs, outputs)
        or tuple(layer.weight.shape) != (outputs, inputs)
        or layer.bias is None
        or tuple(layer.bias.shape) != (outputs,)
    ):
        raise ValueError("actual linear architecture differs from protocol")


def validate_allocation_protocol_model(
    model: Any, protocol: AllocationPPOTrainingProtocol
) -> None:
    """Read actual settings without optional imports or upper dependencies."""
    validate_allocation_backend_versions()
    p: dict[str, Any] = protocol.payload()
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "learning_rate",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "sde_sample_freq",
        "tensorboard_log",
        "verbose",
    ):
        if canonical_json_bytes(getattr(model, name, None)) != canonical_json_bytes(
            p["ppo"][name]
        ):
            raise ValueError(f"actual {name} differs from explicit protocol")
    for name in ("lr_schedule", "clip_range", "clip_range_vf"):
        expected = (
            protocol.learning_rate if name == "lr_schedule" else getattr(protocol, name)
        )
        schedule = getattr(model, name, None)
        if expected is None:
            valid = schedule is None
        else:
            valid = callable(schedule) and all(
                schedule(progress) == expected for progress in (1.0, 0.5, 0.0)
            )
        if not valid:
            raise ValueError("actual constant schedule differs from protocol")
    if (
        _class_name(model) != "stable_baselines3.ppo.ppo.PPO"
        or str(model.device) != "cpu"
        or model.n_envs != 1
        or model._stats_window_size != 100
        or model.rollout_buffer_kwargs != {}
        or _class_name(model.rollout_buffer_class) != p["ppo"]["rollout_buffer_class"]
    ):
        raise ValueError("actual PPO backend differs from protocol")
    policy = model.policy
    simple = {
        key: value
        for key, value in p["policy"].items()
        if key not in {"class", "activation_fn", "features_extractor_class"}
    }
    for name in (
        "net_arch",
        "ortho_init",
        "features_extractor_kwargs",
        "share_features_extractor",
        "normalize_images",
        "log_std_init",
    ):
        if canonical_json_bytes(getattr(policy, name, None)) != canonical_json_bytes(
            simple[name]
        ):
            raise ValueError("actual policy settings differ from protocol")
    if (
        _class_name(policy) != "stable_baselines3.common.policies.ActorCriticPolicy"
        or _class_name(policy.activation_fn) != "torch.nn.modules.activation.Tanh"
        or _class_name(policy.features_extractor_class)
        != p["policy"]["features_extractor_class"]
        or _class_name(policy.action_dist)
        != "stable_baselines3.common.distributions.CategoricalDistribution"
    ):
        raise ValueError("actual policy classes differ from protocol")
    if (
        _class_name(policy.features_extractor)
        != p["policy"]["features_extractor_class"]
        or policy.pi_features_extractor is not policy.features_extractor
        or policy.vf_features_extractor is not policy.features_extractor
        or _class_name(policy.features_extractor.flatten)
        != "torch.nn.modules.flatten.Flatten"
        or (
            policy.features_extractor.flatten.start_dim,
            policy.features_extractor.flatten.end_dim,
        )
        != (1, -1)
        or policy.features_extractor.features_dim != policy.features_dim
        or model.observation_space.shape != (policy.features_dim,)
    ):
        raise ValueError("actual shared feature extractor differs from protocol")
    for name, widths, output in (
        ("policy_net", protocol.pi_layers, policy.action_net),
        ("value_net", protocol.vf_layers, policy.value_net),
    ):
        layers = list(getattr(policy.mlp_extractor, name))
        if len(layers) != 2 * len(widths):
            raise ValueError("actual network depth differs from protocol")
        inputs = policy.features_dim
        for index, width in enumerate(widths):
            _validate_linear(layers[2 * index], inputs, width)
            if _class_name(layers[2 * index + 1]) != "torch.nn.modules.activation.Tanh":
                raise ValueError("actual network activation differs from protocol")
            inputs = width
        _validate_linear(output, inputs, 4 if name == "policy_net" else 1)
    optimizer = policy.optimizer
    settings = {
        key: value
        for key, value in p["optimizer"].items()
        if key not in {"class", "learning_rate_source"}
    } | {"lr": protocol.learning_rate}
    if (
        _class_name(optimizer) != "torch.optim.adam.Adam"
        or len(optimizer.param_groups) != 1
    ):
        raise ValueError("actual optimizer differs from protocol")
    for key, expected in settings.items():
        if any(
            canonical_json_bytes(group.get(key)) != canonical_json_bytes(expected)
            for group in (optimizer.defaults, *optimizer.param_groups)
        ):
            raise ValueError("actual Adam settings differ from protocol")
    parameters = list(policy.parameters())
    if (
        len(parameters) != len(optimizer.param_groups[0]["params"])
        or {id(value) for value in parameters}
        != {id(value) for value in optimizer.param_groups[0]["params"]}
        or any(
            str(value.dtype) != "torch.float32" or str(value.device) != "cpu"
            for value in parameters
        )
    ):
        raise ValueError("actual policy parameters differ from CPU float32 protocol")
    for name, expected in simple.items():
        if canonical_json_bytes(model.policy_kwargs.get(name)) != canonical_json_bytes(
            expected
        ):
            raise ValueError("actual declared policy kwargs differ from protocol")


def validate_allocation_model(model: Any, manifest: dict[str, Any]) -> None:
    observation, action = model.observation_space, model.action_space
    count = len(manifest["recipe"]["feature_names"]) + len(
        manifest["recipe"]["observation"]["fields"]
    )
    if manifest["recipe"]["schema"] in (
        "allocation_ppo_recipe_v2",
        "allocation_ppo_recipe_v3",
    ):
        count = len(manifest["recipe"]["observation"]["fields"])
    if (
        not isinstance(observation, gym.spaces.Box)
        or observation.shape != (count,)
        or observation.dtype != np.dtype(np.float32)
        or not np.isneginf(observation.low).all()
        or not np.isposinf(observation.high).all()
        or not isinstance(action, gym.spaces.Discrete)
        or action.n != 4
        or action.start != 0
    ):
        raise ValueError(
            "policy spaces differ from the allocation observation/action contract"
        )
    training = manifest["training"]
    ppo = training["ppo"]
    for key in (
        "gamma",
        "gae_lambda",
        "n_steps",
        "batch_size",
        "n_epochs",
        "learning_rate",
    ):
        if getattr(model, key, None) != ppo[key]:
            raise ValueError(f"actual PPO {key} differs from its receipt")
    if (
        str(model.device) != ppo["device"]
        or model.policy.net_arch != ppo["net_arch"]
        or model.num_timesteps != training["actual_timesteps"]
        or model.seed != training["seed"]
        or getattr(model, "n_envs", None) != 1
    ):
        raise ValueError("actual PPO runtime differs from its training receipt")
    if manifest["schema"] in (
        "allocation_ppo_inference_bundle_v3",
        "allocation_ppo_inference_bundle_v4",
        "allocation_ppo_inference_bundle_v5",
    ):
        validate_allocation_protocol_model(
            model, AllocationPPOTrainingProtocol.from_payload(training["protocol"])
        )
        if (
            type(model._n_updates) is not int
            or model._n_updates < 0
            or model._n_updates != training["optimization"]["epoch_iterations"]
        ):
            raise ValueError("actual epoch iterations differ from protocol receipt")


class AllocationPPOPolicy:
    """Deterministic allocation inference; no research or trading authorization."""

    def __init__(self, model: Any, manifest: dict[str, Any]) -> None:
        normalized = validate_allocation_manifest(manifest)
        validate_allocation_model(model, normalized)
        self.model = model
        self._manifest = canonical_json_bytes(normalized)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest)

    def action(self, observation: np.ndarray, *, runtime_recipe_digest: str) -> int:
        require_sha256(runtime_recipe_digest, field="runtime_recipe_digest")
        manifest = self.manifest
        if runtime_recipe_digest != manifest["recipe_digest"]:
            raise ValueError("runtime recipe differs from the frozen allocation policy")
        validate_allocation_model(self.model, manifest)
        with np.errstate(over="raise", invalid="raise"):
            try:
                original = np.asarray(observation)
                if original.dtype.kind not in "iuf":
                    raise ValueError("allocation observation must contain real numbers")
                values = original.astype(np.float32)
            except (TypeError, ValueError, FloatingPointError) as error:
                raise ValueError(
                    "allocation observation must fit finite float32 values"
                ) from error
        if (
            values.shape != self.model.observation_space.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "allocation observation differs from its finite vector contract"
            )
        raw, _ = self.model.predict(values, deterministic=True)
        codes = np.asarray(raw).reshape(-1)
        if (
            codes.size != 1
            or isinstance(codes[0], (bool, np.bool_))
            or not isinstance(codes[0], Integral)
            or not 0 <= int(codes[0]) <= 3
        ):
            raise ValueError(
                "allocation action must be one integer within {0, 1, 2, 3}"
            )
        return int(codes[0])
