from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from trade_rl.artifacts.hashing import content_digest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_gpu_training_smoke", SMOKE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_config_preserves_the_maintained_cuda_training_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADE_RL_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("TRADE_RL_GIT_DIRTY", "false")
    config = _load_smoke().build_smoke_config(timesteps=128)

    assert config.training.device == "cuda"
    assert config.training.n_envs == 4
    assert config.training.policy == "MultiInputPolicy"
    assert config.training.sequence_encoder is True
    assert config.training.sequence_capacity == "compact"
    assert config.training.sequence_d_model == 128
    assert config.training.sequence_attention_heads == 4
    assert config.training.sequence_attention_layers == 1
    assert config.training.asset_set_encoder is False
    assert config.training.policy_net_arch == (128, 64)
    assert config.training.value_net_arch == (192, 96)
    assert config.training.n_epochs == 3
    assert config.training.max_policy_parameters == 2_500_000
    assert config.environment.structured_sequence_observation is True
    assert config.environment.resolved_sequence_windows == (
        ("15m", 96),
        ("1h", 168),
        ("4h", 120),
        ("1d", 60),
    )
    assert config.training.timesteps == 128
    assert config.training.behavior_cloning_epochs == 1
    assert config.training.checkpoint_interval_steps == 64
    assert config.training.max_checkpoints == 2
    assert config.action.mode == "target_weight"
    assert config.action.target_weight_count == 1
    assert config.git_commit == "a" * 40
    assert config.git_dirty is False


@pytest.mark.parametrize(
    ("commit", "dirty"),
    ((None, "false"), ("A" * 40, "false"), ("a" * 40, None), ("a" * 40, "0")),
)
def test_smoke_config_fails_closed_without_valid_packaged_git_provenance(
    monkeypatch: pytest.MonkeyPatch,
    commit: str | None,
    dirty: str | None,
) -> None:
    if commit is None:
        monkeypatch.delenv("TRADE_RL_GIT_COMMIT", raising=False)
    else:
        monkeypatch.setenv("TRADE_RL_GIT_COMMIT", commit)
    if dirty is None:
        monkeypatch.delenv("TRADE_RL_GIT_DIRTY", raising=False)
    else:
        monkeypatch.setenv("TRADE_RL_GIT_DIRTY", dirty)

    with pytest.raises(ValueError, match="TRADE_RL_GIT_"):
        _load_smoke().build_smoke_config(timesteps=128)


def _performance_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "training_performance_evidence_v1",
        "device_type": "cuda",
        "requested_environment_steps": 128,
        "observed_environment_steps": 128,
        "wall_clock_seconds": 2.0,
        "environment_steps_per_second": 64.0,
        "collect_rollouts_seconds": 1.0,
        "optimization_seconds": 0.8,
        "environment_step_seconds": 0.4,
        "feature_extraction_host_seconds": 0.5,
        "sequence_reconstruction_seconds": 0.2,
        "sequence_tensor_conversion_seconds": 0.1,
        "collect_rollouts_calls": 2,
        "optimization_calls": 2,
        "environment_step_calls": 16,
        "feature_extraction_calls": 24,
        "sequence_reconstruction_calls": 2,
        "sequence_tensor_conversion_calls": 2,
        "peak_cuda_allocated_bytes": 1_024,
        "peak_cuda_reserved_bytes": 2_048,
        "component_timers_overlap": True,
    }
    payload["digest"] = content_digest(payload)
    return payload


def test_load_training_performance_validates_schema_and_digest(tmp_path: Path) -> None:
    member = tmp_path / "member-000"
    member.mkdir()
    payload = _performance_payload()
    (member / "training-performance.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert _load_smoke()._load_training_performance(member) == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"schema_version": "wrong"}, "schema"),
        ({"digest": "0" * 64}, "digest"),
        ({"observed_environment_steps": 0}, "observed"),
        ({"wall_clock_seconds": 0.0}, "wall"),
    ),
)
def test_load_training_performance_fails_closed(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    member = tmp_path / "member-000"
    member.mkdir()
    payload = _performance_payload()
    payload.update(mutation)
    (member / "training-performance.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        _load_smoke()._load_training_performance(member)
