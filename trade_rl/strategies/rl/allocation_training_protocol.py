"""Immutable pinned categorical PPO declarations, without a learner consumer.

Class names are closed identifiers, never imports. Declaring settings does not
construct a model, bind a financial clock, count updates or authorize research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_PPO_FIELDS = (
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "learning_rate",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
)


def _integer(value: object, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a native integer >= {minimum}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside finite float reporting") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0 else result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _array_tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _native_json(value: object) -> None:
    """Reject conversions and mutable subclasses at the public read boundary."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("protocol JSON numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _native_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("protocol JSON keys must be native strings")
            _native_json(item)
        return
    raise ValueError("protocol payload must contain only native JSON values")


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocationPPOTrainingProtocol:
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    clip_range: float
    clip_range_vf: float | None
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None
    pi_layers: tuple[int, ...]
    vf_layers: tuple[int, ...]
    adam_betas: tuple[float, float]
    adam_eps: float
    adam_weight_decay: float
    adam_amsgrad: bool

    def __post_init__(self) -> None:
        for name, minimum in (("n_steps", 2), ("batch_size", 2), ("n_epochs", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.n_steps % self.batch_size:
            raise ValueError("one-account rollout must contain complete minibatches")
        for name in (
            "gamma",
            "gae_lambda",
            "learning_rate",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "adam_eps",
            "adam_weight_decay",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.gamma != 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("allocation gamma must equal one and GAE be within [0,1]")
        for name in ("learning_rate", "clip_range", "max_grad_norm", "adam_eps"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("ent_coef", "vf_coef", "adam_weight_decay"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("clip_range_vf", "target_kl"):
            value = getattr(self, name)
            if value is not None:
                normalized = _number(value, name)
                if normalized <= 0.0:
                    raise ValueError(f"{name} must be positive or None")
                object.__setattr__(self, name, normalized)
        for name in ("normalize_advantage", "adam_amsgrad"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a native boolean")
        for name in ("pi_layers", "vf_layers"):
            widths = getattr(self, name)
            if type(widths) is not tuple or not widths:
                raise ValueError(f"{name} must be a nonempty immutable tuple")
            for width in widths:
                _integer(width, name, 1)
        if type(self.adam_betas) is not tuple or len(self.adam_betas) != 2:
            raise ValueError("adam_betas must be an immutable pair")
        betas = tuple(_number(value, "adam beta") for value in self.adam_betas)
        if any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("Adam betas must be within [0,1)")
        object.__setattr__(self, "adam_betas", betas)

    def payload(self) -> dict[str, object]:
        """Resolve all constructor identifiers/flags into detached finite JSON."""
        return {
            "schema": "allocation_ppo_training_protocol_v1",
            "backend": {
                "algorithm": "stable_baselines3.PPO",
                "stable_baselines3": "2.3.2",
                "torch": "2.4.1",
                "device": "cpu",
                "n_envs": 1,
                "num_threads": 1,
                "parameter_dtype": "float32",
            },
            "ppo": {
                **{name: getattr(self, name) for name in _PPO_FIELDS},
                "learning_rate_schedule": "constant_v1",
                "clip_range_schedule": "constant_v1",
                "clip_range_vf_schedule": "none_or_constant_v1",
                "use_sde": False,
                "sde_sample_freq": -1,
                "rollout_buffer_class": "stable_baselines3.common.buffers.RolloutBuffer",
                "rollout_buffer_kwargs": {},
                "stats_window_size": 100,
                "tensorboard_log": None,
                "verbose": 0,
                "_init_setup_model": True,
            },
            "policy": {
                "class": "MlpPolicy",
                "net_arch": {"pi": list(self.pi_layers), "vf": list(self.vf_layers)},
                "activation_fn": "torch.nn.Tanh",
                "ortho_init": True,
                "features_extractor_class": "stable_baselines3.common.torch_layers.FlattenExtractor",
                "features_extractor_kwargs": {},
                "share_features_extractor": True,
                "normalize_images": True,
                "log_std_init": 0.0,
                "full_std": True,
                "use_expln": False,
                "squash_output": False,
            },
            "optimizer": {
                "class": "torch.optim.Adam",
                "learning_rate_source": "ppo.learning_rate",
                "betas": list(self.adam_betas),
                "eps": self.adam_eps,
                "weight_decay": self.adam_weight_decay,
                "amsgrad": self.adam_amsgrad,
                "foreach": False,
                "fused": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            },
            "fit": {
                "initialization": "fresh_model_v1",
                "reset_num_timesteps": True,
                "progress_bar": False,
                "log_interval": 1,
                "tb_log_name": "PPO",
            },
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationPPOTrainingProtocol:
        """Read only the exact closed, resolved canonical declaration."""
        try:
            _native_json(value)
            data = _mapping(value, "protocol")
            ppo = _mapping(data["ppo"], "ppo")
            policy = _mapping(data["policy"], "policy")
            architecture = _mapping(policy["net_arch"], "net_arch")
            optimizer = _mapping(data["optimizer"], "optimizer")
            declaration = cls(
                **{name: ppo[name] for name in _PPO_FIELDS},
                pi_layers=_array_tuple(architecture["pi"], "pi"),
                vf_layers=_array_tuple(architecture["vf"], "vf"),
                adam_betas=_array_tuple(optimizer["betas"], "betas"),
                adam_eps=optimizer["eps"],
                adam_weight_decay=optimizer["weight_decay"],
                adam_amsgrad=optimizer["amsgrad"],
            )
            # JSON bytes distinguish True/1, False/0 and 0.0/-0.0, unlike dict
            # equality. Reconstruction closes every nested key and fixed flag.
            if canonical_json_bytes(value) != canonical_json_bytes(
                declaration.payload()
            ):
                raise ValueError("payload differs from its closed resolved declaration")
            return declaration
        except (KeyError, TypeError, OverflowError, RecursionError) as error:
            raise ValueError("invalid allocation PPO protocol payload") from error


__all__ = ["AllocationPPOTrainingProtocol"]
