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
        "observation_encoder": "hierarchical_sequence_v2",
        "policy": "MultiInputPolicy",
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
        uses_shared_asset_actor=True,
    )


def _checkpoint_identity(binding: str) -> dict[str, object]:
    return {
        "schema_version": "sb3_checkpoint_identity_v2",
        "policy": {
            "schema_version": "sb3_policy_identity_v4",
            "observation_encoder": "hierarchical_sequence_v2",
            "policy_architecture_digest": "a" * 64,
            "asset_binding_digest": binding * 64,
        },
        "algorithm": {"schema_version": "ppo_identity_v1"},
    }


def _manifest(config: ResidualTrainingConfig, **changes: object) -> SimpleNamespace:
    identity = _checkpoint_identity("b")
    payload: dict[str, object] = {
        "algorithm": config.algorithm,
        "seed": 7,
        "environment_digest": "s" * 64,
        "training_config_digest": content_digest(config.digest_payload()),
        "policy_path": Path("/tmp/policy.zip"),
        "observed_timestep": 12,
        "algorithm_identity": identity,
        "algorithm_identity_digest": content_digest(identity),
    }
    payload.update(changes)
    return SimpleNamespace(**payload)


@contextmanager
def _passthrough_policy_copy(manifest: SimpleNamespace) -> Iterator[Path]:
    yield manifest.policy_path


def test_transfer_loader_accepts_new_environment_with_compatible_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_transfer_model,
    )

    config = _config()
    manifest = _manifest(config)
    source_identity = manifest.algorithm_identity
    target_identity = _checkpoint_identity("c")
    reconstructor = object()
    bound: list[tuple[object, str]] = []

    def bind_sequence_reconstructor(
        value: object,
        *,
        sequence_transfer_mode: str,
    ) -> None:
        bound.append((value, sequence_transfer_mode))

    loaded_model = SimpleNamespace(
        num_timesteps=12,
        rollout_buffer=SimpleNamespace(
            bind_sequence_reconstructor=bind_sequence_reconstructor
        ),
    )
    compatibility_calls: list[tuple[object, object]] = []
    load_calls: list[tuple[str, object, str]] = []

    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "verified_checkpoint_policy_copy",
        _passthrough_policy_copy,
    )
    monkeypatch.setattr(
        assembly_module,
        "checkpoint_identity_payload_for_model",
        lambda _: target_identity,
    )
    monkeypatch.setattr(
        assembly_module,
        "validate_sb3_policy_architecture_compatibility",
        lambda observed, expected: compatibility_calls.append((observed, expected)),
    )
    monkeypatch.setattr(
        assembly_module,
        "bind_sb3_policy_identity",
        lambda model, policy: target_identity["policy"],
    )

    class Loader:
        @staticmethod
        def load(path: str, *, env: object, device: str) -> object:
            load_calls.append((path, env, device))
            return loaded_model

    monkeypatch.setattr(assembly_module, "_checkpoint_loader", lambda _: Loader)
    environment = object()

    result = load_sb3_checkpoint_transfer_model(
        checkpoint_root=Path("checkpoint.json"),
        environment=environment,
        seed=7,
        config=config,
        identity={"environment_digest": "t" * 64},
        algorithm_config=build_algorithm_config(config),
        policy=_policy(config, sequence_reconstructor=reconstructor),
        fresh_model=SimpleNamespace(),
    )

    assert result.model is loaded_model
    assert result.manifest is manifest
    assert load_calls == [(str(manifest.policy_path), environment, "cpu")]
    assert compatibility_calls == [
        (source_identity["policy"], target_identity["policy"]),
        (source_identity["policy"], target_identity["policy"]),
    ]
    assert bound == [(reconstructor, config.sequence_transfer_mode)]
    assert loaded_model.rollout_buffer_kwargs == {
        "sequence_reconstructor": reconstructor,
        "sequence_transfer_mode": config.sequence_transfer_mode,
    }


def test_transfer_loader_rejects_same_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_transfer_model,
    )

    config = _config()
    manifest = _manifest(config)
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)

    with pytest.raises(ValueError, match="different environment"):
        load_sb3_checkpoint_transfer_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": manifest.environment_digest},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"algorithm": "sac"}, "checkpoint algorithm mismatch"),
        ({"seed": 8}, "checkpoint seed mismatch"),
        (
            {"training_config_digest": "x" * 64},
            "checkpoint training configuration mismatch",
        ),
    ),
)
def test_transfer_loader_rejects_non_environment_identity_mismatch(
    change: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_transfer_model,
    )

    config = _config()
    monkeypatch.setattr(
        assembly_module,
        "load_checkpoint_manifest",
        lambda _: _manifest(config, **change),
    )

    with pytest.raises(ValueError, match=message):
        load_sb3_checkpoint_transfer_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": "t" * 64},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )


def test_transfer_loader_rejects_algorithm_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_transfer_model,
    )

    config = _config()
    manifest = _manifest(config)
    target_identity = _checkpoint_identity("c")
    target_identity["algorithm"] = {"schema_version": "other_algorithm_v1"}
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "checkpoint_identity_payload_for_model",
        lambda _: target_identity,
    )

    with pytest.raises(ValueError, match="algorithm identity mismatch"):
        load_sb3_checkpoint_transfer_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": "t" * 64},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )


def test_transfer_loader_rejects_policy_architecture_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.sb3_checkpoint_assembly as assembly_module
    from trade_rl.integrations.sb3_checkpoint_assembly import (
        load_sb3_checkpoint_transfer_model,
    )

    config = _config()
    manifest = _manifest(config)
    target_identity = _checkpoint_identity("c")
    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(
        assembly_module,
        "checkpoint_identity_payload_for_model",
        lambda _: target_identity,
    )
    monkeypatch.setattr(
        assembly_module,
        "validate_sb3_policy_architecture_compatibility",
        lambda observed, expected: (_ for _ in ()).throw(
            ValueError("SB3 policy architecture compatibility mismatch")
        ),
    )

    with pytest.raises(ValueError, match="architecture compatibility mismatch"):
        load_sb3_checkpoint_transfer_model(
            checkpoint_root=Path("checkpoint.json"),
            environment=object(),
            seed=7,
            config=config,
            identity={"environment_digest": "t" * 64},
            algorithm_config=build_algorithm_config(config),
            policy=_policy(config),
            fresh_model=SimpleNamespace(),
        )
