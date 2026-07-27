from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_exact(
    path: str,
    old: str,
    new: str,
    *,
    expected_count: int = 1,
) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} anchors, found {count}: {old[:100]!r}"
        )
    updated = text.replace(old, new, expected_count)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    replace_exact(path, old, new, expected_count=1)


def update_model_assembly() -> None:
    path = "trade_rl/integrations/sb3_model_assembly.py"
    replace_once(
        path,
        """    uses_shared_asset_actor: bool
    rollout_buffer_class: object | None = None
""",
        """    uses_shared_asset_actor: bool
    observation_encoder: str = "flat_mlp"
    sequence_symbols: tuple[str, ...] | None = None
    sequence_action_names: tuple[str, ...] | None = None
    rollout_buffer_class: object | None = None
""",
    )
    replace_once(
        path,
        """    uses_shared_asset_actor = False
    rollout_buffer_class: object | None = None
""",
        """    uses_shared_asset_actor = False
    sequence_symbols: tuple[str, ...] | None = None
    sequence_action_names: tuple[str, ...] | None = None
    rollout_buffer_class: object | None = None
""",
    )
    replace_once(
        path,
        """        ) = _sequence_policy_assembly(
            probe=probe,
            identity=identity,
            config=config,
        )
        rollout_buffer_class = IndexBackedDictRolloutBuffer
""",
        """        ) = _sequence_policy_assembly(
            probe=probe,
            identity=identity,
            config=config,
        )
        unwrapped: Any = getattr(probe, "unwrapped", probe)
        dataset = getattr(unwrapped, "dataset", None)
        raw_symbols = getattr(dataset, "symbols", None)
        if (
            not isinstance(raw_symbols, (tuple, list))
            or not raw_symbols
            or any(not isinstance(item, str) or not item for item in raw_symbols)
        ):
            raise ValueError("sequence training requires ordered dataset symbols")
        sequence_symbols = tuple(raw_symbols)
        sequence_action_names = _action_names(identity)
        rollout_buffer_class = IndexBackedDictRolloutBuffer
""",
    )
    replace_once(
        path,
        """        uses_shared_asset_actor=uses_shared_asset_actor,
        rollout_buffer_class=rollout_buffer_class,
""",
        """        uses_shared_asset_actor=uses_shared_asset_actor,
        observation_encoder=config.observation_encoder,
        sequence_symbols=sequence_symbols,
        sequence_action_names=sequence_action_names,
        rollout_buffer_class=rollout_buffer_class,
""",
    )
    replace_once(
        path,
        """    common = _common_model_kwargs(
        seed=seed,
        config=config,
        algorithm_config=algorithm_config,
        policy=policy,
        verbose=verbose,
        output_root=output_root,
    )
    if isinstance(algorithm_config, PPOConfig):
""",
        """    common = _common_model_kwargs(
        seed=seed,
        config=config,
        algorithm_config=algorithm_config,
        policy=policy,
        verbose=verbose,
        output_root=output_root,
    )

    from trade_rl.integrations.sb3_policy_identity import bind_sb3_policy_identity

    def _bind_identity(model: Any) -> Any:
        bind_sb3_policy_identity(model, policy)
        return model

    if isinstance(algorithm_config, PPOConfig):
""",
    )
    replace_once(path, "            return model\n", "            return _bind_identity(model)\n")
    replace_once(
        path,
        """            constructor = CostCriticPPO
            return constructor(
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
""",
        """            constructor = CostCriticPPO
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
""",
    )
    replace_once(
        path,
        "        return constructor(policy.policy_identifier, environment, **ppo_kwargs)\n",
        """        return _bind_identity(
            constructor(policy.policy_identifier, environment, **ppo_kwargs)
        )
""",
    )
    replace_once(
        path,
        "    return constructor(policy.policy_identifier, environment, **off_policy)\n",
        """    return _bind_identity(
        constructor(policy.policy_identifier, environment, **off_policy)
    )
""",
    )


def update_checkpointing() -> None:
    path = "trade_rl/rl/checkpointing.py"
    replace_once(
        path,
        "def _model_algorithm_identity(model: SavablePolicy) -> dict[str, object] | None:\n",
        "def _model_algorithm_identity(model: object) -> dict[str, object] | None:\n",
    )
    anchor = """    payload = dict(raw)
    canonical_json_bytes(payload)
    return payload


@dataclass(frozen=True, slots=True)
"""
    replacement = """    payload = dict(raw)
    canonical_json_bytes(payload)
    return payload


def checkpoint_identity_payload_for_model(
    model: object,
) -> dict[str, object] | None:
    \"\"\"Compose the actual policy architecture with algorithm-specific identity.\"\"\"

    from trade_rl.integrations.sb3_policy_identity import model_sb3_policy_identity

    policy_identity = model_sb3_policy_identity(model)
    algorithm_identity = _model_algorithm_identity(model)
    if policy_identity is None:
        return algorithm_identity
    payload: dict[str, object] = {
        \"schema_version\": \"sb3_checkpoint_identity_v2\",
        \"policy\": policy_identity,
        \"algorithm\": algorithm_identity,
    }
    canonical_json_bytes(payload)
    return payload


@dataclass(frozen=True, slots=True)
"""
    replace_once(path, anchor, replacement)
    replace_exact(
        path,
        "    algorithm_identity = _model_algorithm_identity(model)\n",
        "    algorithm_identity = checkpoint_identity_payload_for_model(model)\n",
        expected_count=2,
    )
    replace_once(
        path,
        """    "build_checkpoint_callback",
    "checkpoint_manifests",
""",
        """    "build_checkpoint_callback",
    "checkpoint_identity_payload_for_model",
    "checkpoint_manifests",
""",
    )


def update_checkpoint_loader() -> None:
    path = "trade_rl/integrations/sb3_checkpoint_assembly.py"
    replace_once(
        path,
        """    CheckpointManifest,
    load_checkpoint_manifest,
    validate_checkpoint_algorithm_identity,
""",
        """    CheckpointManifest,
    checkpoint_identity_payload_for_model,
    load_checkpoint_manifest,
    validate_checkpoint_algorithm_identity,
""",
    )
    replace_once(
        path,
        """def _checkpoint_algorithm_identity(
    model: object,
    algorithm_config: AlgorithmConfig,
) -> dict[str, object] | None:
    if not isinstance(algorithm_config, CostCriticPPOConfig):
        return None
    provider = getattr(model, "checkpoint_identity_payload", None)
    if not callable(provider):
        raise TypeError("checkpoint_identity_payload must be callable")
    value = provider()
    if not isinstance(value, dict) or not value:
        raise ValueError("checkpoint algorithm identity must be a non-empty object")
    return value
""",
        """def _checkpoint_algorithm_identity(
    model: object,
    algorithm_config: AlgorithmConfig,
) -> dict[str, object] | None:
    del algorithm_config
    return checkpoint_identity_payload_for_model(model)
""",
    )
    replace_once(
        path,
        """    if int(model.num_timesteps) != manifest.observed_timestep:
        raise ValueError("checkpoint timestep identity mismatch")
    if isinstance(algorithm_config, CostCriticPPOConfig):
        loaded_identity = _checkpoint_algorithm_identity(model, algorithm_config)
        validate_checkpoint_algorithm_identity(manifest, loaded_identity)
""",
        """    if int(model.num_timesteps) != manifest.observed_timestep:
        raise ValueError("checkpoint timestep identity mismatch")
    from trade_rl.integrations.sb3_policy_identity import bind_sb3_policy_identity

    bind_sb3_policy_identity(model, policy)
    loaded_identity = _checkpoint_algorithm_identity(model, algorithm_config)
    validate_checkpoint_algorithm_identity(manifest, loaded_identity)
""",
    )


def main() -> None:
    update_model_assembly()
    update_checkpointing()
    update_checkpoint_loader()
    for path in (
        "trade_rl/integrations/sb3_model_assembly.py",
        "trade_rl/rl/checkpointing.py",
        "trade_rl/integrations/sb3_checkpoint_assembly.py",
    ):
        ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)


if __name__ == "__main__":
    main()
