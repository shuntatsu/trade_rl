from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.integrations.sb3_model_assembly import SB3PolicyAssembly
from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.training import ResidualTrainingConfig


def _config(algorithm: str = "ppo", **changes: object) -> ResidualTrainingConfig:
    payload: dict[str, object] = {
        "timesteps": 8,
        "gamma": 0.99,
        "seeds": (0,),
        "algorithm": algorithm,
        "observation_encoder": "flat_mlp",
        "device": "cpu",
    }
    payload.update(changes)
    return ResidualTrainingConfig(**payload)  # type: ignore[arg-type]


def _policy(config: ResidualTrainingConfig) -> SB3PolicyAssembly:
    value_key = "vf" if config.algorithm.endswith("ppo") else "qf"
    return SB3PolicyAssembly(
        policy_identifier=config.policy,
        policy_kwargs={
            "net_arch": {
                "pi": list(config.policy_net_arch),
                value_key: list(config.value_net_arch),
            }
        },
        rollout_buffer_bytes=None,
        sequence_metadata=None,
        sequence_reconstructor=None,
        uses_shared_asset_actor=False,
    )


def _constructor_spy(calls: list[tuple[tuple[object, ...], dict[str, object]]]):
    def constructor(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return SimpleNamespace()

    return constructor


def test_builds_ppo_with_exact_typed_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stable_baselines3

    import trade_rl.integrations.sb3_model_assembly as assembly_module
    from trade_rl.integrations.sb3_model_assembly import build_sb3_model

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(stable_baselines3, "PPO", _constructor_spy(calls))
    monkeypatch.setattr(
        assembly_module,
        "build_learning_rate_schedule",
        lambda **_: "schedule",
    )
    config = _config(
        tensorboard_enabled=True,
        use_sde=True,
        sde_sample_freq=4,
    )
    algorithm = build_algorithm_config(config)
    policy = _policy(config)
    environment = object()

    model = build_sb3_model(
        environment=environment,
        seed=7,
        config=config,
        algorithm_config=algorithm,
        policy=policy,
        verbose=2,
        output_root=tmp_path / "policy.zip",
        canonical_action_probe_evidence=None,
    )

    assert model is not None
    args, kwargs = calls[0]
    assert args == (config.policy, environment)
    assert kwargs == {
        "batch_size": config.batch_size,
        "clip_range": config.clip_range,
        "device": "cpu",
        "ent_coef": config.ent_coef,
        "gae_lambda": config.gae_lambda,
        "gamma": config.gamma,
        "learning_rate": "schedule",
        "max_grad_norm": config.max_grad_norm,
        "n_epochs": config.n_epochs,
        "n_steps": config.n_steps,
        "normalize_advantage": config.normalize_advantage,
        "policy_kwargs": policy.policy_kwargs,
        "sde_sample_freq": 4,
        "seed": 7,
        "target_kl": config.target_kl,
        "tensorboard_log": str(tmp_path / "tensorboard"),
        "use_sde": True,
        "verbose": 2,
        "vf_coef": config.vf_coef,
    }


def test_builds_cost_critic_ppo_with_cost_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.cost_critic_ppo as cost_module
    import trade_rl.integrations.sb3_model_assembly as assembly_module
    from trade_rl.integrations.sb3_model_assembly import build_sb3_model

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(cost_module, "CostCriticPPO", _constructor_spy(calls))
    monkeypatch.setattr(
        assembly_module,
        "build_learning_rate_schedule",
        lambda **_: "schedule",
    )
    config = _config("cost_critic_ppo")
    algorithm = build_algorithm_config(config)
    environment = object()

    build_sb3_model(
        environment=environment,
        seed=3,
        config=config,
        algorithm_config=algorithm,
        policy=_policy(config),
        verbose=0,
        output_root=tmp_path / "policy.zip",
        canonical_action_probe_evidence=None,
    )

    args, kwargs = calls[0]
    assert args == (config.policy, environment)
    assert kwargs["cost_schema"] == algorithm.cost_schema
    assert kwargs["cost_learning_rate"] == config.cost_learning_rate
    assert kwargs["cost_continuous_hidden_dims"] == config.cost_continuous_hidden_dims
    assert kwargs["cost_event_hidden_dims"] == config.cost_event_hidden_dims
    assert kwargs["cost_max_grad_norm"] == config.cost_max_grad_norm


def _lagrangian_config() -> ResidualTrainingConfig:
    count = len(canonical_cost_learning_schema().names)
    return _config(
        "lagrangian_ppo",
        lagrangian_budgets=(1.0,) * count,
        lagrangian_dual_learning_rates=(0.01,) * count,
        lagrangian_ema_betas=(0.9,) * count,
        lagrangian_initial_multipliers=(0.0,) * count,
        lagrangian_max_multipliers=(10.0,) * count,
        lagrangian_warmup_rollouts=(0,) * count,
        lagrangian_update_interval_rollouts=(1,) * count,
        lagrangian_minimum_completed_episodes=(1,) * count,
        lagrangian_probe_episodes=1,
        lagrangian_probe_max_steps_per_episode=4,
    )


def test_builds_lagrangian_ppo_and_binds_probe_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.lagrangian_ppo as lagrangian_module
    import trade_rl.integrations.sb3_model_assembly as assembly_module
    from trade_rl.integrations.sb3_model_assembly import build_sb3_model

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    model = SimpleNamespace()

    def constructor(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(lagrangian_module, "LagrangianPPO", constructor)
    monkeypatch.setattr(
        assembly_module,
        "build_learning_rate_schedule",
        lambda **_: "schedule",
    )
    config = _lagrangian_config()
    algorithm = build_algorithm_config(config)
    evidence = SimpleNamespace(digest="e" * 64)

    result = build_sb3_model(
        environment=object(),
        seed=5,
        config=config,
        algorithm_config=algorithm,
        policy=_policy(config),
        verbose=0,
        output_root=tmp_path / "policy.zip",
        canonical_action_probe_evidence=evidence,
    )

    assert result is model
    assert model.canonical_action_probe_evidence is evidence
    assert calls[0][1]["lagrangian_schema"] == algorithm.lagrangian_schema
    assert calls[0][1]["canonical_action_probe_evidence"] is evidence


@pytest.mark.parametrize("algorithm", ("sac", "td3", "tqc"))
def test_builds_off_policy_algorithm_with_exact_common_configuration(
    algorithm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stable_baselines3

    import trade_rl.integrations.sb3_model_assembly as assembly_module
    from trade_rl.integrations.sb3_model_assembly import build_sb3_model

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    constructor = _constructor_spy(calls)
    if algorithm == "tqc":
        import sb3_contrib

        monkeypatch.setattr(sb3_contrib, "TQC", constructor)
    else:
        monkeypatch.setattr(stable_baselines3, algorithm.upper(), constructor)
    monkeypatch.setattr(
        assembly_module,
        "build_learning_rate_schedule",
        lambda **_: "schedule",
    )
    config = _config(algorithm)
    typed = build_algorithm_config(config)
    environment = object()

    build_sb3_model(
        environment=environment,
        seed=11,
        config=config,
        algorithm_config=typed,
        policy=_policy(config),
        verbose=1,
        output_root=tmp_path / "policy.zip",
        canonical_action_probe_evidence=None,
    )

    args, kwargs = calls[0]
    assert args == (config.policy, environment)
    assert kwargs["batch_size"] == config.batch_size
    assert kwargs["buffer_size"] == config.buffer_size
    assert kwargs["gradient_steps"] == config.gradient_steps
    assert kwargs["learning_starts"] == config.learning_starts
    assert kwargs["train_freq"] == config.train_freq
    assert kwargs["learning_rate"] == "schedule"
    assert kwargs["policy_kwargs"] == _policy(config).policy_kwargs
    assert kwargs["seed"] == 11
    if algorithm in {"sac", "tqc"}:
        assert kwargs["use_sde"] is config.use_sde
        assert kwargs["sde_sample_freq"] == config.sde_sample_freq
    else:
        assert "use_sde" not in kwargs
        assert "sde_sample_freq" not in kwargs
