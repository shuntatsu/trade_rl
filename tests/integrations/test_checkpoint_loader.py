from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from trade_rl.integrations.checkpoints import StableBaselines3CheckpointLoader
from trade_rl.workflows.checkpoints import PolicyCheckpoint


class _FakeModel:
    loaded: tuple[str, str] | None = None

    @classmethod
    def load(cls, path: str, *, device: str) -> object:
        cls.loaded = (path, device)
        return object()


def _module(**values: object) -> ModuleType:
    module = ModuleType("fake")
    for name, value in values.items():
        setattr(module, name, value)
    return module


def test_loader_loads_supported_algorithm_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3",
        _module(PPO=_FakeModel, SAC=_FakeModel, TD3=_FakeModel),
    )
    monkeypatch.setitem(sys.modules, "sb3_contrib", _module(TQC=_FakeModel))
    path = tmp_path / "policy.zip"
    path.write_bytes(b"model")

    model = StableBaselines3CheckpointLoader().load(
        PolicyCheckpoint(path=path, algorithm="ppo")
    )

    assert model is not None
    assert _FakeModel.loaded == (str(path), "cpu")


def test_loader_rejects_unsupported_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "policy.zip"
    path.write_bytes(b"model")
    with pytest.raises(ValueError, match="unsupported"):
        StableBaselines3CheckpointLoader().load(
            PolicyCheckpoint(path=path, algorithm="dqn")
        )
