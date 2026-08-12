from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from trade_rl.integrations import sb3_training
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 32,
        "gamma": 1.0,
        "seeds": (7,),
        "n_steps": 8,
        "batch_size": 8,
        "behavior_cloning_epochs": 2,
        "behavior_cloning_teacher": "oracle",
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


class _SavablePolicy:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"policy")


def test_universal_pretraining_hook_is_fail_closed_and_persisted(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def hook(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": "universal_pretraining_evidence_v1",
            "passed": True,
            "teacher_artifact_digest": "a" * 64,
            "behavior_cloning_digest": "b" * 64,
            "critic_warm_start_digest": "c" * 64,
        }

    policy = _SavablePolicy()
    config = _config()
    result = sb3_training._apply_universal_pretraining_if_configured(
        hook=hook,
        policy=policy,
        config=config,
        behavior_cloning_seed=13,
        member_seed=7,
        output_root=tmp_path,
    )

    assert result is not None
    assert captured["policy"] is policy
    assert captured["config"] is config
    assert captured["behavior_cloning_seed"] == 13
    assert captured["member_seed"] == 7
    assert captured["output_root"] == tmp_path
    assert result["passed"] is True
    assert len(result["artifact_digest"]) == 64
    assert (tmp_path / "universal-pretraining.json").is_file()
    assert (tmp_path / "policy-stages/random/policy.zip").is_file()


def test_universal_pretraining_hook_accepts_causal_trend_teacher(
    tmp_path: Path,
) -> None:
    result = sb3_training._apply_universal_pretraining_if_configured(
        hook=lambda **_: {
            "schema_version": "universal_pretraining_evidence_v1",
            "passed": True,
            "teacher_artifact_digest": "a" * 64,
            "behavior_cloning_digest": "b" * 64,
            "critic_warm_start_digest": None,
        },
        policy=_SavablePolicy(),
        config=_config(behavior_cloning_teacher="trend_baseline"),
        behavior_cloning_seed=13,
        member_seed=7,
        output_root=tmp_path,
    )

    assert result is not None
    assert result["passed"] is True


def test_universal_pretraining_hook_rejects_failed_evidence(tmp_path: Path) -> None:
    def hook(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "schema_version": "universal_pretraining_evidence_v1",
            "passed": False,
        }

    with pytest.raises(RuntimeError, match="failed"):
        sb3_training._apply_universal_pretraining_if_configured(
            hook=hook,
            policy=_SavablePolicy(),
            config=_config(),
            behavior_cloning_seed=13,
            member_seed=7,
            output_root=tmp_path,
        )


def test_universal_pretraining_requires_behavior_cloning_enabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="behavior cloning"):
        sb3_training._apply_universal_pretraining_if_configured(
            hook=lambda **_: {
                "schema_version": "universal_pretraining_evidence_v1",
                "passed": True,
            },
            policy=object(),
            config=_config(behavior_cloning_epochs=0),
            behavior_cloning_seed=13,
            member_seed=7,
            output_root=tmp_path,
        )


def test_sb3_universal_hook_bypasses_single_dataset_teacher_path() -> None:
    source = inspect.getsource(sb3_training.StableBaselines3Backend.train)
    prefetch = source.index("prefetched_episode_batch")
    model_build = source.index("build_sb3_model")
    hook = source.index("_apply_universal_pretraining_if_configured")
    learn = source.index("model.learn")

    assert prefetch < model_build < hook < learn
    assert "self.universal_pretraining_hook is None" in source
