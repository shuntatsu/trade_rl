"""Sequential CPU A2C fitting over the shared PPO trading environment."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.dataset_scope import validated_training_scope
from trade_rl.strategies.rl.intent import _PredictPolicy, _ThreeActionIntentStrategy
from trade_rl.strategies.rl.ppo import (
    PPOTradingEnv,
    _agent_stop_index,
    _validate_sequential_symbol_coverage,
)
from trade_rl.strategies.rl.ppo_normalization import (
    PPOFeatureNormalizer,
    fit_ppo_feature_normalizer,
)

A2C_ROLLOUT_STEPS = 5
A2C_STEP_ROUNDING = "ceil_to_complete_rollout"

_A2C_LEARNING_RATE = 7e-4
_A2C_GAMMA = 0.99
_A2C_GAE_LAMBDA = 1.0
_A2C_ENT_COEF = 0.0
_A2C_VF_COEF = 0.5
_A2C_MAX_GRAD_NORM = 0.5
_A2C_RMS_PROP_EPS = 1e-5
_A2C_STATS_WINDOW_SIZE = 100


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def effective_a2c_timesteps(requested_timesteps: int) -> int:
    """Return the complete-rollout step count SB3 A2C will execute."""

    requested = _positive_integer(requested_timesteps, name="total_timesteps")
    return (
        (requested + A2C_ROLLOUT_STEPS - 1) // A2C_ROLLOUT_STEPS
    ) * A2C_ROLLOUT_STEPS


def a2c_training_config() -> dict[str, object]:
    """Return the explicit, JSON-safe A2C configuration recorded with fits."""

    return {
        "policy": "MlpPolicy",
        "learning_rate": _A2C_LEARNING_RATE,
        "n_steps": A2C_ROLLOUT_STEPS,
        "gamma": _A2C_GAMMA,
        "gae_lambda": _A2C_GAE_LAMBDA,
        "ent_coef": _A2C_ENT_COEF,
        "vf_coef": _A2C_VF_COEF,
        "max_grad_norm": _A2C_MAX_GRAD_NORM,
        "rms_prop_eps": _A2C_RMS_PROP_EPS,
        "use_rms_prop": True,
        "normalize_advantage": False,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "stats_window_size": _A2C_STATS_WINDOW_SIZE,
        "tensorboard_log": None,
        "verbose": 0,
        "device": "cpu",
        "_init_setup_model": True,
        "policy_kwargs": {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": "torch.nn.Tanh",
            "ortho_init": True,
            "log_std_init": 0.0,
            "full_std": True,
            "use_expln": False,
            "squash_output": False,
            "features_extractor_class": "stable_baselines3.common.torch_layers.FlattenExtractor",
            "features_extractor_kwargs": {},
            "share_features_extractor": True,
            "normalize_images": False,
            "optimizer_class": "torch.optim.RMSprop",
            "optimizer_kwargs": {
                "alpha": 0.99,
                "eps": 1e-5,
                "weight_decay": 0.0,
                "momentum": 0.0,
                "centered": False,
                "capturable": False,
                "foreach": None,
                "maximize": False,
                "differentiable": False,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class A2CFitMetadata:
    """Requested/effective budget and sequential symbol coverage evidence."""

    requested_timesteps: int
    effective_timesteps: int
    rollout_steps: int
    step_rounding: str
    seed: int
    fit_symbol_indices: tuple[int, ...]
    fit_symbols: tuple[str, ...]
    episode_steps: int
    required_coverage_timesteps: int
    nominal_full_episodes_per_symbol: int

    def __post_init__(self) -> None:
        requested = _positive_integer(
            self.requested_timesteps,
            name="requested_timesteps",
        )
        effective = _positive_integer(
            self.effective_timesteps,
            name="effective_timesteps",
        )
        rollout = _positive_integer(self.rollout_steps, name="rollout_steps")
        _non_negative_integer(self.seed, name="seed")
        episode_steps = _positive_integer(self.episode_steps, name="episode_steps")
        required = _positive_integer(
            self.required_coverage_timesteps,
            name="required_coverage_timesteps",
        )
        full_episodes = _positive_integer(
            self.nominal_full_episodes_per_symbol,
            name="nominal_full_episodes_per_symbol",
        )
        indices = tuple(self.fit_symbol_indices)
        symbols = tuple(self.fit_symbols)
        if (
            self.step_rounding != A2C_STEP_ROUNDING
            or effective != ((requested + rollout - 1) // rollout) * rollout
            or rollout != A2C_ROLLOUT_STEPS
            or len(indices) != len(symbols)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or len(set(indices)) != len(indices)
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or len(set(symbols)) != len(symbols)
            or required != episode_steps * len(symbols)
            or effective < required
            or full_episodes != effective // required
        ):
            raise ValueError("A2C fit metadata differs from the pinned fit contract")
        object.__setattr__(self, "fit_symbol_indices", indices)
        object.__setattr__(self, "fit_symbols", symbols)

    def to_payload(self) -> dict[str, object]:
        return {
            "requested_timesteps": self.requested_timesteps,
            "effective_timesteps": self.effective_timesteps,
            "rollout_steps": self.rollout_steps,
            "step_rounding": self.step_rounding,
            "seed": self.seed,
            "fit_symbol_indices": list(self.fit_symbol_indices),
            "fit_symbols": list(self.fit_symbols),
            "episode_steps": self.episode_steps,
            "required_coverage_timesteps": self.required_coverage_timesteps,
            "nominal_full_episodes_per_symbol": self.nominal_full_episodes_per_symbol,
            "training_config": a2c_training_config(),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> A2CFitMetadata:
        keys = {
            "requested_timesteps",
            "effective_timesteps",
            "rollout_steps",
            "step_rounding",
            "seed",
            "fit_symbol_indices",
            "fit_symbols",
            "episode_steps",
            "required_coverage_timesteps",
            "nominal_full_episodes_per_symbol",
            "training_config",
        }
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ValueError("invalid A2C fit metadata fields")
        indices = payload["fit_symbol_indices"]
        symbols = payload["fit_symbols"]
        config = payload["training_config"]
        if not isinstance(indices, list) or not isinstance(symbols, list):
            raise ValueError("A2C fit scope metadata must use JSON arrays")
        if not isinstance(config, dict):
            raise ValueError("A2C training configuration must be a JSON object")
        if canonical_json_bytes(config) != canonical_json_bytes(a2c_training_config()):
            raise ValueError("A2C training configuration differs from the pinned fit")
        return cls(
            requested_timesteps=payload["requested_timesteps"],
            effective_timesteps=payload["effective_timesteps"],
            rollout_steps=payload["rollout_steps"],
            step_rounding=payload["step_rounding"],
            seed=payload["seed"],
            fit_symbol_indices=tuple(indices),
            fit_symbols=tuple(symbols),
            episode_steps=payload["episode_steps"],
            required_coverage_timesteps=payload["required_coverage_timesteps"],
            nominal_full_episodes_per_symbol=payload[
                "nominal_full_episodes_per_symbol"
            ],
        )


class A2CIntentStrategy(_ThreeActionIntentStrategy):
    """A2C-named adapter with optional fit-budget evidence."""

    _family_name = "A2C"

    def __init__(
        self,
        policy: _PredictPolicy,
        *,
        feature_indices: tuple[int, ...],
        feature_names: tuple[str, ...] | None = None,
        feature_normalizer: PPOFeatureNormalizer | None = None,
        fit_metadata: A2CFitMetadata | None = None,
    ) -> None:
        super().__init__(
            policy,
            feature_indices=feature_indices,
            feature_names=feature_names,
            feature_normalizer=feature_normalizer,
        )
        if fit_metadata is not None and not isinstance(fit_metadata, A2CFitMetadata):
            raise ValueError("fit_metadata must be A2CFitMetadata or None")
        self._fit_metadata = fit_metadata

    @property
    def fit_metadata(self) -> A2CFitMetadata | None:
        return self._fit_metadata


def fit_a2c_strategy(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    total_timesteps: int,
    seed: int = 0,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk_config: PreTradeRiskConfig | None = None,
    normalize_features: bool = False,
    settle_terminal_position: bool = False,
) -> A2CIntentStrategy:
    """Fit one sequential CPU A2C policy using explicit rollout-level budget."""

    requested = _positive_integer(total_timesteps, name="total_timesteps")
    actual_seed = _non_negative_integer(seed, name="seed")
    if not isinstance(settle_terminal_position, bool):
        raise ValueError("settle_terminal_position must be boolean")
    if not isinstance(normalize_features, bool):
        raise ValueError("normalize_features must be boolean")

    resolved_execution_cost = execution_cost or ExecutionCostConfig.zero()
    policy_stop = _agent_stop_index(
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=resolved_execution_cost,
        settle_terminal_position=settle_terminal_position,
    )
    indices, fit_symbols = validated_training_scope(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
    )
    episode_steps = policy_stop - start_index
    _validate_sequential_symbol_coverage(
        requested,
        episode_steps=episode_steps,
        n_symbols=len(fit_symbols),
        rollout_steps=A2C_ROLLOUT_STEPS,
        algorithm="A2C",
        require_single_symbol_coverage=True,
    )

    normalizer = (
        fit_ppo_feature_normalizer(
            dataset,
            feature_indices=indices,
            fit_symbol_indices=fit_symbols,
            start_index=start_index,
            stop_index=policy_stop,
        )
        if normalize_features
        else None
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=indices,
        symbol_indices=fit_symbols,
        information_symbol_indices=fit_symbols,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=gross_budget,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk_config=risk_config,
        feature_normalizer=normalizer,
        settle_terminal_position=settle_terminal_position,
    )
    try:
        sb3_module = importlib.import_module("stable_baselines3")
        a2c_class = getattr(sb3_module, "A2C")
        torch_module = importlib.import_module("torch")
        getattr(torch_module, "set_num_threads")(1)
        torch_nn = getattr(torch_module, "nn")
        torch_optim = getattr(torch_module, "optim")
        activation_fn = getattr(torch_nn, "Tanh")
        optimizer_class = getattr(torch_optim, "RMSprop")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
        features_extractor_class = getattr(torch_layers, "FlattenExtractor")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error

    policy_kwargs = {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": activation_fn,
        "ortho_init": True,
        "log_std_init": 0.0,
        "full_std": True,
        "use_expln": False,
        "squash_output": False,
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {},
        "share_features_extractor": True,
        "normalize_images": False,
        "optimizer_class": optimizer_class,
        "optimizer_kwargs": {
            "alpha": 0.99,
            "eps": _A2C_RMS_PROP_EPS,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "centered": False,
            "capturable": False,
            "foreach": None,
            "maximize": False,
            "differentiable": False,
        },
    }
    model = a2c_class(
        "MlpPolicy",
        env,
        learning_rate=_A2C_LEARNING_RATE,
        n_steps=A2C_ROLLOUT_STEPS,
        gamma=_A2C_GAMMA,
        gae_lambda=_A2C_GAE_LAMBDA,
        ent_coef=_A2C_ENT_COEF,
        vf_coef=_A2C_VF_COEF,
        max_grad_norm=_A2C_MAX_GRAD_NORM,
        rms_prop_eps=_A2C_RMS_PROP_EPS,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        normalize_advantage=False,
        stats_window_size=_A2C_STATS_WINDOW_SIZE,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        seed=actual_seed,
        device="cpu",
        verbose=0,
        _init_setup_model=True,
    )
    model.learn(total_timesteps=requested)
    actual_steps = getattr(model, "num_timesteps", None)
    expected_steps = effective_a2c_timesteps(requested)
    if (
        isinstance(actual_steps, bool)
        or not isinstance(actual_steps, int)
        or actual_steps != expected_steps
    ):
        raise RuntimeError(
            "A2C effective timestep count differs from the complete-rollout contract"
        )
    metadata = A2CFitMetadata(
        requested_timesteps=requested,
        effective_timesteps=actual_steps,
        rollout_steps=A2C_ROLLOUT_STEPS,
        step_rounding=A2C_STEP_ROUNDING,
        seed=actual_seed,
        fit_symbol_indices=fit_symbols,
        fit_symbols=tuple(dataset.symbols[index] for index in fit_symbols),
        episode_steps=episode_steps,
        required_coverage_timesteps=episode_steps * len(fit_symbols),
        nominal_full_episodes_per_symbol=actual_steps
        // (episode_steps * len(fit_symbols)),
    )
    return A2CIntentStrategy(
        cast(_PredictPolicy, model),
        feature_indices=indices,
        feature_names=tuple(dataset.feature_names[index] for index in indices),
        feature_normalizer=normalizer,
        fit_metadata=metadata,
    )


__all__ = [
    "A2CFitMetadata",
    "A2CIntentStrategy",
    "A2C_ROLLOUT_STEPS",
    "A2C_STEP_ROUNDING",
    "a2c_training_config",
    "effective_a2c_timesteps",
    "fit_a2c_strategy",
]
