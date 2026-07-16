from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

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
    assert config.training.sequence_d_model == 336
    assert config.training.asset_set_encoder is False
    assert config.training.policy_net_arch == (384, 256, 128)
    assert config.training.value_net_arch == (512, 384, 256)
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
