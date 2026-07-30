from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_model_assembly import SB3PolicyAssembly
from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**changes: object) -> ResidualTrainingConfig:
    payload: dict[str, object] = {
        "timesteps": 8,
        "gamma": 0.99,
        "seeds": (0,),
        "observation_encoder": "flat_mlp",
        "device": "cpu",
    }
    payload.update(changes)
    return ResidualTrainingConfig(**payload)  # type: ignore[arg-type]


def _policy(
    config: ResidualTrainingConfig,
    *,
    sequence_reconstructor: object | None = None,
) -> SB3PolicyAssembly:
    return SB3PolicyAssembly(
        policy_identifier=config.policy,
        policy_kwargs={"net_arch": {"pi": [128], "vf": [128]}},
        rollout_buffer_bytes=None,
        sequence_metadata=None,
        sequence_reconstructor=sequence_reconstructor,
        uses_shared_asset_actor=False,
    )


def _manifest(config: ResidualTrainingConfig, **changes: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "algorithm": config.algorithm,
        "seed": 7,
        "environment_digest": "e" * 64,
        "training_config_digest": content_digest(config.digest_payload()),
        "policy_path": Path("/tmp/policy.zip"),
        "observed_timestep": 12,
        "algorithm_identity": None,
        "algorithm_identity_digest": None,
    }
    payload.update(changes)
    return SimpleNamespace(**payload)


@contextmanager
def _passthrough_policy_copy(manifest: SimpleNamespace) -> Iterator[Path]:
    yield manifest.policy_path


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"algorithm": "sac"}, "checkpoint algorithm mismatch"),
        ({"seed": 8}, "checkpoint seed mismatch"),
        (
            {"environment_digest": "x" * 64},
            "checkpoint environment identity mismatch",
        ),
        (
            {"training_config_digest": "x" * 64},
            "checkpoint training configuration mismatch",
        ),
    ),
)
def test_checkpoint_loader_rejects_identity_mismatch(
    change: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_model,
    )

    config = _config()
    monkeypatch.setattr(
        assembly_module,
        "load_checkpoint_manifest",
        lambda _: _manifest(config, **change),
    )

    with pytest.raises(ValueError, match=message):
        load_sb3_checkpoint_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": "e" * 64},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )


def test_checkpoint_loader_uses_matching_algorithm_and_validates_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stable_baselines3

    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_model,
    )

    config = _config()
    manifest = _manifest(config)
    loaded_model = SimpleNamespace(num_timesteps=12)
    calls: list[tuple[str, object, str]] = []

    def load(path: str, *, env: object, device: str) -> object:
        calls.append((path, env, device))
        return loaded_model

    monkeypatch.setattr(
        stable_baselines3,
        "PPO",
        SimpleNamespace(load=load),
    )
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "verified_checkpoint_policy_copy",
        _passthrough_policy_copy,
    )
    monkeypatch.setattr(
        assembly_module,
        "validate_checkpoint_algorithm_identity",
        lambda manifest, identity: None,
    )
    environment = object()

    result = load_sb3_checkpoint_model(
        checkpoint_root=Path("checkpoint.json"),
        environment=environment,
        seed=7,
        config=config,
        identity={"environment_digest": "e" * 64},
        algorithm_config=build_algorithm_config(config),
        policy=_policy(config),
        fresh_model=SimpleNamespace(),
    )

    assert result.model is loaded_model
    assert result.manifest is manifest
    assert calls == [(str(manifest.policy_path), environment, "cpu")]


def test_checkpoint_loader_rebinds_sequence_reconstructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stable_baselines3

    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_model,
    )

    config = _config(
        observation_encoder=("hierarchical_sequence_v2"),
        policy="MultiInputPolicy",
    )
    manifest = _manifest(config)
    reconstructor = object()
    bound: list[tuple[object, str]] = []

    def bind_sequence_reconstructor(
        value: object, *, sequence_transfer_mode: str
    ) -> None:
        bound.append((value, sequence_transfer_mode))

    rollout_buffer = SimpleNamespace(
        bind_sequence_reconstructor=bind_sequence_reconstructor
    )
    loaded_model = SimpleNamespace(num_timesteps=12, rollout_buffer=rollout_buffer)
    monkeypatch.setattr(
        stable_baselines3,
        "PPO",
        SimpleNamespace(load=lambda *_, **__: loaded_model),
    )
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "verified_checkpoint_policy_copy",
        _passthrough_policy_copy,
    )
    monkeypatch.setattr(
        assembly_module,
        "validate_checkpoint_algorithm_identity",
        lambda manifest, identity: None,
    )

    result = load_sb3_checkpoint_model(
        checkpoint_root=Path("checkpoint.json"),
        environment=object(),
        seed=7,
        config=config,
        identity={"environment_digest": "e" * 64},
        algorithm_config=build_algorithm_config(config),
        policy=_policy(config, sequence_reconstructor=reconstructor),
        fresh_model=SimpleNamespace(),
    )

    assert result.model is loaded_model
    assert bound == [(reconstructor, config.sequence_transfer_mode)]
    assert loaded_model.rollout_buffer_kwargs == {
        "sequence_reconstructor": reconstructor,
        "sequence_transfer_mode": config.sequence_transfer_mode,
    }


def test_checkpoint_loader_rejects_loaded_timestep_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stable_baselines3

    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_model,
    )

    config = _config()
    manifest = _manifest(config)
    monkeypatch.setattr(
        stable_baselines3,
        "PPO",
        SimpleNamespace(load=lambda *_, **__: SimpleNamespace(num_timesteps=11)),
    )
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "verified_checkpoint_policy_copy",
        _passthrough_policy_copy,
    )
    monkeypatch.setattr(
        assembly_module,
        "validate_checkpoint_algorithm_identity",
        lambda manifest, identity: None,
    )

    with pytest.raises(ValueError, match="checkpoint timestep identity mismatch"):
        load_sb3_checkpoint_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": "e" * 64},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )
