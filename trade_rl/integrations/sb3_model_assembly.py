"""Typed Stable-Baselines3 policy and model assembly contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

from trade_rl.integrations.compact_rollout_buffer import (
    IndexBackedDictRolloutBuffer,
    SequenceRolloutReconstructor,
)
from trade_rl.rl.algorithm_configs import (
    AlgorithmConfig,
    CostCriticPPOConfig,
    LagrangianPPOConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
)
from trade_rl.rl.rollout_memory import (
    estimate_index_backed_ppo_rollout_buffer_bytes,
    estimate_ppo_rollout_buffer_bytes,
)
from trade_rl.rl.schedules import build_learning_rate_schedule
from trade_rl.rl.training import ResidualTrainingConfig


@dataclass(frozen=True, slots=True)
class SB3PolicyAssembly:
    """Resolved policy and rollout-buffer inputs for one SB3 model."""

    policy_identifier: object
    policy_kwargs: Mapping[str, object]
    rollout_buffer_bytes: int | None
    sequence_metadata: Mapping[str, Any] | None
    sequence_reconstructor: object | None
    uses_shared_asset_actor: bool
    observation_encoder: str = "flat_mlp"
    sequence_symbols: tuple[str, ...] | None = None
    sequence_action_names: tuple[str, ...] | None = None
    policy_actor_head: str | None = None
    hierarchical_gate_temperature: float | None = None
    rollout_buffer_class: object | None = None
    rollout_buffer_kwargs: Mapping[str, object] | None = None


def _action_size(identity: Mapping[str, object]) -> int:
    value = identity.get("action_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("training identity action_size must be a positive integer")
    return value


def _action_names(identity: Mapping[str, object]) -> tuple[str, ...]:
    value = identity.get("action_names")
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError("training identity action_names must be a string tuple")
    return value


def _is_universal_single_instrument(probe: object) -> bool:
    unwrapped = getattr(probe, "unwrapped", probe)
    return bool(getattr(unwrapped, "is_universal_single_instrument", False))


def _rollout_buffer_bytes(
    *,
    probe: object,
    identity: Mapping[str, object],
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
) -> int | None:
    if not isinstance(algorithm_config, PPOConfig):
        return None
    estimator = (
        estimate_index_backed_ppo_rollout_buffer_bytes
        if (
            config.observation_encoder == "hierarchical_sequence_v2"
            and not _is_universal_single_instrument(probe)
        )
        else estimate_ppo_rollout_buffer_bytes
    )
    observation_space = getattr(probe, "observation_space", None)
    if not isinstance(observation_space, gym.Space):
        raise ValueError("training probe must expose a Gymnasium observation space")
    estimated = estimator(
        observation_space,
        n_steps=algorithm_config.n_steps,
        n_envs=config.n_envs,
        action_dim=_action_size(identity),
    )
    if isinstance(algorithm_config, CostCriticPPOConfig):
        from trade_rl.integrations.cost_rollout_buffer import (
            estimate_cost_rollout_storage_bytes,
        )

        estimated += estimate_cost_rollout_storage_bytes(
            algorithm_config.n_steps,
            config.n_envs,
            len(algorithm_config.cost_schema.names),
        )
    if estimated > config.max_rollout_buffer_bytes:
        raise ValueError(
            "estimated PPO rollout buffer exceeds max_rollout_buffer_bytes: "
            f"{estimated} > {config.max_rollout_buffer_bytes}"
        )
    return estimated


def _sequence_policy_assembly(
    *,
    probe: object,
    identity: Mapping[str, object],
    config: ResidualTrainingConfig,
) -> tuple[
    object,
    dict[str, object],
    dict[str, object],
    SequenceRolloutReconstructor | None,
    bool,
]:
    from trade_rl.rl.policies import (
        SequenceAssetFeatureExtractor,
        SharedPerAssetActorCriticPolicy,
    )

    unwrapped: Any = getattr(probe, "unwrapped", probe)
    metadata = getattr(unwrapped, "sequence_layout_metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("sequence training requires environment sequence metadata")
    sequence_metadata = dict(metadata)
    action_names = _action_names(identity)
    universal = _is_universal_single_instrument(unwrapped)
    sequence_reconstructor: SequenceRolloutReconstructor | None = None
    if universal:
        if getattr(unwrapped, "policy_symbols", None) != ("INSTRUMENT",):
            raise ValueError(
                "Universal sequence training requires generic policy symbols"
            )
        if int(sequence_metadata["n_symbols"]) != 1 or _action_size(identity) != 1:
            raise ValueError("Universal sequence training requires one instrument")
        if action_names != ("target_weight:INSTRUMENT",):
            raise ValueError("Universal sequence action contract mismatch")
    else:
        dataset = getattr(unwrapped, "dataset", None)
        raw_symbols = getattr(dataset, "symbols", None)
        if not isinstance(raw_symbols, (tuple, list)) or not raw_symbols:
            raise ValueError("sequence training requires ordered dataset symbols")
        symbols = tuple(raw_symbols)
        if any(not isinstance(item, str) or not item for item in symbols):
            raise ValueError("sequence training requires ordered dataset symbols")
        expected = tuple(f"target_weight:{symbol}" for symbol in symbols)
        if (
            _action_size(identity) != int(sequence_metadata["n_symbols"])
            or action_names != expected
        ):
            raise ValueError("hierarchical sequence action/symbol order mismatch")
        builder = getattr(unwrapped, "sequence_observation_builder", None)
        if dataset is None or builder is None:
            raise ValueError(
                "sequence training requires dataset reconstruction metadata"
            )
        sequence_reconstructor = SequenceRolloutReconstructor(
            dataset=dataset,
            builder=builder,
            normalizer=getattr(unwrapped, "sequence_normalizer", None),
            expected_dataset_id=dataset.dataset_id,
            expected_layout_digest=builder.layout_digest(dataset),
            policy_plane=getattr(unwrapped, "sequence_policy_plane", None),
        )
    policy_kwargs: dict[str, object] = {
        "net_arch": {
            "pi": list(config.policy_net_arch),
            "vf": list(config.value_net_arch),
        },
        "features_extractor_class": SequenceAssetFeatureExtractor,
        "features_extractor_kwargs": {
            **sequence_metadata,
            "sequence_tcn_capacity": config.sequence_tcn_capacity,
            "d_model": config.sequence_d_model,
            "timeframe_attention_heads": config.sequence_timeframe_attention_heads,
            "timeframe_attention_layers": config.sequence_timeframe_attention_layers,
            "timeframe_ffn_multiplier": config.sequence_timeframe_ffn_multiplier,
            "timeframe_gate_bias": config.sequence_timeframe_gate_bias,
            "asset_attention_heads": config.sequence_asset_attention_heads,
            "asset_attention_layers": config.sequence_asset_attention_layers,
            "asset_ffn_multiplier": config.sequence_asset_ffn_multiplier,
            "asset_gate_bias": config.sequence_asset_gate_bias,
            "dropout": config.sequence_dropout,
        },
    }
    risk_config = getattr(getattr(unwrapped, "pre_trade_risk", None), "config", None)
    policy_kwargs.update(
        {
            "shared_actor_n_symbols": int(sequence_metadata["n_symbols"]),
            "shared_actor_d_model": config.sequence_d_model,
            "shared_actor_global_dim": 128,
            "shared_actor_net_arch": tuple(config.policy_net_arch),
            "shared_actor_head": config.policy_actor_head,
            "shared_actor_gate_temperature": config.hierarchical_gate_temperature,
            "shared_actor_gate_prediction_threshold": config.behavior_cloning_gate_prediction_threshold,
            "shared_actor_entry_threshold": float(
                getattr(risk_config, "entry_threshold", 0.0)
            ),
            "shared_actor_minimum_deterministic_change": float(
                getattr(risk_config, "no_trade_band", 0.0)
            ),
        }
    )
    return (
        SharedPerAssetActorCriticPolicy,
        policy_kwargs,
        sequence_metadata,
        sequence_reconstructor,
        True,
    )


def resolve_sb3_policy_assembly(
    *,
    probe: object,
    identity: Mapping[str, object],
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
) -> SB3PolicyAssembly:
    """Resolve policy, extractor, and rollout-buffer choices without training."""

    rollout_buffer_bytes = _rollout_buffer_bytes(
        probe=probe,
        identity=identity,
        config=config,
        algorithm_config=algorithm_config,
    )
    sequence_metadata: dict[str, object] | None = None
    sequence_reconstructor: object | None = None
    uses_shared_asset_actor = False
    sequence_symbols: tuple[str, ...] | None = None
    sequence_action_names: tuple[str, ...] | None = None
    rollout_buffer_class: object | None = None
    rollout_buffer_kwargs: dict[str, object] | None = None
    if config.observation_encoder == "hierarchical_sequence_v2":
        (
            policy_identifier,
            policy_kwargs,
            sequence_metadata,
            sequence_reconstructor,
            uses_shared_asset_actor,
        ) = _sequence_policy_assembly(
            probe=probe,
            identity=identity,
            config=config,
        )
        sequence_unwrapped: Any = getattr(probe, "unwrapped", probe)
        sequence_action_names = _action_names(identity)
        if _is_universal_single_instrument(sequence_unwrapped):
            if getattr(sequence_unwrapped, "policy_symbols", None) != ("INSTRUMENT",):
                raise ValueError(
                    "Universal sequence training requires generic policy symbols"
                )
            sequence_symbols = ("INSTRUMENT",)
            rollout_buffer_class = None
            rollout_buffer_kwargs = None
        else:
            dataset = getattr(sequence_unwrapped, "dataset", None)
            raw_symbols = getattr(dataset, "symbols", None)
            if not isinstance(raw_symbols, (tuple, list)) or not raw_symbols:
                raise ValueError("sequence training requires ordered dataset symbols")
            sequence_symbols = tuple(raw_symbols)
            rollout_buffer_class = IndexBackedDictRolloutBuffer
            rollout_buffer_kwargs = {
                "sequence_reconstructor": sequence_reconstructor,
                "sequence_transfer_mode": config.sequence_transfer_mode,
            }
    elif isinstance(algorithm_config, PPOConfig):
        policy_identifier = config.policy
        policy_kwargs = {
            "net_arch": {
                "pi": list(algorithm_config.policy_net_arch),
                "vf": list(algorithm_config.value_net_arch),
            }
        }
    else:
        policy_identifier = config.policy
        policy_kwargs = {
            "net_arch": {
                "pi": list(algorithm_config.policy_net_arch),
                "qf": list(algorithm_config.value_net_arch),
            }
        }
    if isinstance(algorithm_config, PPOConfig):
        policy_kwargs["log_std_init"] = algorithm_config.log_std_init
    if config.observation_encoder == "asset_set":
        from trade_rl.rl.policies import AssetSetFeatureExtractor

        unwrapped: Any = getattr(probe, "unwrapped", probe)
        layout = getattr(unwrapped, "layout", None)
        active_column = getattr(unwrapped, "asset_active_column", None)
        if layout is None or not isinstance(active_column, int):
            raise ValueError("asset-set training requires environment layout metadata")
        policy_kwargs.update(
            {
                "features_extractor_class": AssetSetFeatureExtractor,
                "features_extractor_kwargs": {
                    "n_symbols": layout.n_symbols,
                    "per_symbol_width": layout.per_symbol_width,
                    "global_width": layout.global_width,
                    "active_column": active_column,
                    "asset_embedding_dim": config.asset_embedding_dim,
                    "global_embedding_dim": config.global_embedding_dim,
                },
            }
        )
    return SB3PolicyAssembly(
        policy_identifier=policy_identifier,
        policy_kwargs=policy_kwargs,
        rollout_buffer_bytes=rollout_buffer_bytes,
        sequence_metadata=sequence_metadata,
        sequence_reconstructor=sequence_reconstructor,
        uses_shared_asset_actor=uses_shared_asset_actor,
        observation_encoder=config.observation_encoder,
        sequence_symbols=sequence_symbols,
        sequence_action_names=sequence_action_names,
        policy_actor_head=(
            config.policy_actor_head
            if config.observation_encoder == "hierarchical_sequence_v2"
            else None
        ),
        hierarchical_gate_temperature=(
            config.hierarchical_gate_temperature
            if config.observation_encoder == "hierarchical_sequence_v2"
            else None
        ),
        rollout_buffer_class=rollout_buffer_class,
        rollout_buffer_kwargs=rollout_buffer_kwargs,
    )


def _common_model_kwargs(
    *,
    seed: int,
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    verbose: int,
    output_root: Path,
) -> dict[str, object]:
    common: dict[str, object] = {
        "learning_rate": build_learning_rate_schedule(
            initial_rate=algorithm_config.learning_rate,
            final_ratio=algorithm_config.learning_rate_final_ratio,
            kind=algorithm_config.learning_rate_schedule,
        ),
        "gamma": algorithm_config.gamma,
        "policy_kwargs": policy.policy_kwargs,
        "seed": seed,
        "device": config.device,
        "verbose": verbose,
    }
    if config.tensorboard_enabled:
        common["tensorboard_log"] = str(output_root.parent / "tensorboard")
    return common


def build_sb3_model(
    *,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    verbose: int,
    output_root: Path,
    canonical_action_probe_evidence: object | None,
) -> Any:
    """Construct one SB3 algorithm from validated immutable assembly inputs."""

    import stable_baselines3

    common = _common_model_kwargs(
        seed=seed,
        config=config,
        algorithm_config=algorithm_config,
        policy=policy,
        verbose=verbose,
        output_root=output_root,
    )

    from trade_rl.rl.policy_identity import bind_sb3_policy_identity

    def _bind_identity(model: Any) -> Any:
        bind_sb3_policy_identity(model, policy)
        return model

    if isinstance(algorithm_config, PPOConfig):
        ppo_kwargs: dict[str, object] = {
            "n_steps": algorithm_config.n_steps,
            "batch_size": algorithm_config.batch_size,
            "n_epochs": algorithm_config.n_epochs,
            "gae_lambda": algorithm_config.gae_lambda,
            "clip_range": algorithm_config.clip_range,
            "normalize_advantage": algorithm_config.normalize_advantage,
            "ent_coef": algorithm_config.ent_coef,
            "vf_coef": algorithm_config.vf_coef,
            "max_grad_norm": algorithm_config.max_grad_norm,
            "target_kl": algorithm_config.target_kl,
            "use_sde": algorithm_config.use_sde,
            "sde_sample_freq": algorithm_config.sde_sample_freq,
            **common,
        }
        if policy.rollout_buffer_class is not None:
            ppo_kwargs["rollout_buffer_class"] = policy.rollout_buffer_class
        if policy.rollout_buffer_kwargs is not None:
            ppo_kwargs["rollout_buffer_kwargs"] = policy.rollout_buffer_kwargs
        constructor: Any
        if isinstance(algorithm_config, LagrangianPPOConfig):
            from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

            constructor = LagrangianPPO
            model: Any = constructor(
                policy.policy_identifier,
                environment,
                cost_schema=algorithm_config.cost_schema,
                cost_learning_rate=algorithm_config.cost_learning_rate,
                cost_n_epochs=algorithm_config.cost_n_epochs,
                cost_batch_size=algorithm_config.cost_batch_size,
                cost_continuous_hidden_dims=(
                    algorithm_config.cost_continuous_hidden_dims
                ),
                cost_event_hidden_dims=algorithm_config.cost_event_hidden_dims,
                cost_max_grad_norm=algorithm_config.cost_max_grad_norm,
                lagrangian_schema=algorithm_config.lagrangian_schema,
                canonical_action_probe_evidence=canonical_action_probe_evidence,
                **ppo_kwargs,
            )
            model.canonical_action_probe_evidence = canonical_action_probe_evidence
            return _bind_identity(model)
        if isinstance(algorithm_config, CostCriticPPOConfig):
            from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

            constructor = CostCriticPPO
            model = constructor(
                policy.policy_identifier,
                environment,
                cost_schema=algorithm_config.cost_schema,
                cost_learning_rate=algorithm_config.cost_learning_rate,
                cost_n_epochs=algorithm_config.cost_n_epochs,
                cost_batch_size=algorithm_config.cost_batch_size,
                cost_continuous_hidden_dims=(
                    algorithm_config.cost_continuous_hidden_dims
                ),
                cost_event_hidden_dims=algorithm_config.cost_event_hidden_dims,
                cost_max_grad_norm=algorithm_config.cost_max_grad_norm,
                **ppo_kwargs,
            )
            return _bind_identity(model)
        constructor = stable_baselines3.PPO
        return _bind_identity(
            constructor(policy.policy_identifier, environment, **ppo_kwargs)
        )

    off_policy: dict[str, object] = {
        "buffer_size": algorithm_config.buffer_size,
        "learning_starts": algorithm_config.learning_starts,
        "batch_size": algorithm_config.batch_size,
        "train_freq": algorithm_config.train_freq,
        "gradient_steps": algorithm_config.gradient_steps,
        **common,
    }
    constructor = None
    if isinstance(algorithm_config, SACConfig):
        constructor = stable_baselines3.SAC
        off_policy.update(
            {
                "use_sde": algorithm_config.use_sde,
                "sde_sample_freq": algorithm_config.sde_sample_freq,
            }
        )
    elif isinstance(algorithm_config, TD3Config):
        constructor = stable_baselines3.TD3
    else:
        try:
            from sb3_contrib import TQC
        except ImportError as error:
            raise RuntimeError(
                "TQC training requires the optional sb3-contrib package"
            ) from error
        constructor = TQC
        off_policy.update(
            {
                "use_sde": algorithm_config.use_sde,
                "sde_sample_freq": algorithm_config.sde_sample_freq,
            }
        )
    return _bind_identity(
        constructor(policy.policy_identifier, environment, **off_policy)
    )


__all__ = [
    "SB3PolicyAssembly",
    "build_sb3_model",
    "resolve_sb3_policy_assembly",
]
