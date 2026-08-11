from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations import sb3_training


def test_publish_final_training_checkpoint_binds_completed_training_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations.sb3_training import _publish_final_training_checkpoint
    from trade_rl.rl import checkpointing

    observed: dict[str, object] = {}
    sentinel = object()

    def publish_checkpoint(**kwargs: object) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(checkpointing, "publish_checkpoint", publish_checkpoint)
    config = SimpleNamespace(
        algorithm="ppo",
        digest_payload=lambda: {"schema_version": "training-test-v1"},
    )
    model = SimpleNamespace(num_timesteps=40)
    environment_digest = content_digest("environment")

    result = _publish_final_training_checkpoint(
        model=model,
        output_root=tmp_path,
        config=config,
        seed=7,
        environment_digest=environment_digest,
        target_total_timesteps=32,
    )

    assert result is sentinel
    assert observed["model"] is model
    assert observed["checkpoint_root"] == tmp_path / "checkpoints"
    assert observed["algorithm"] == "ppo"
    assert observed["seed"] == 7
    assert observed["requested_timestep"] == 32
    assert observed["observed_timestep"] == 40
    assert observed["environment_digest"] == environment_digest
    assert observed["training_config_digest"] == content_digest(config.digest_payload())


def test_publish_final_training_checkpoint_rejects_incomplete_training(
    tmp_path: Path,
) -> None:
    from trade_rl.integrations.sb3_training import _publish_final_training_checkpoint

    with pytest.raises(RuntimeError, match="target training horizon"):
        _publish_final_training_checkpoint(
            model=SimpleNamespace(num_timesteps=31),
            output_root=tmp_path,
            config=SimpleNamespace(
                algorithm="ppo",
                digest_payload=lambda: {"schema_version": "training-test-v1"},
            ),
            seed=7,
            environment_digest=content_digest("environment"),
            target_total_timesteps=32,
        )


def test_sb3_training_publishes_final_checkpoint_before_final_policy_save() -> None:
    source = inspect.getsource(sb3_training.StableBaselines3Backend.train)
    learn = source.index("model.learn")
    final_checkpoint = source.index("_publish_final_training_checkpoint")
    final_policy = source.index("save_policy_without_runtime_state")

    assert learn < final_checkpoint < final_policy
