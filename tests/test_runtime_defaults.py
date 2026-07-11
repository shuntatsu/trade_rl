from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_lite.serving.runtime_defaults import _load_policy


def test_load_policy_uses_ppo_for_single_bundle(monkeypatch) -> None:
    from stable_baselines3 import PPO

    sentinel = object()
    calls = []

    def fake_load(path, device="auto"):
        calls.append((path, device))
        return sentinel

    monkeypatch.setattr(PPO, "load", fake_load)
    bundle = SimpleNamespace(
        metadata={"model_kind": "single"},
        model_path=Path("/tmp/model.zip"),
    )

    assert _load_policy(bundle) is sentinel
    assert calls == [("/tmp/model.zip", "cpu")]


def test_load_policy_uses_seed_ensemble_for_ensemble_bundle(monkeypatch) -> None:
    from mars_lite.learning.policy_ensemble import SeedEnsemble

    sentinel = object()
    calls = []

    def fake_load(cls, path, device="cpu"):
        calls.append((Path(path), device))
        return sentinel

    monkeypatch.setattr(SeedEnsemble, "load", classmethod(fake_load))
    bundle = SimpleNamespace(
        metadata={"model_kind": "ensemble"},
        model_path=Path("/tmp/ensemble"),
    )

    assert _load_policy(bundle) is sentinel
    assert calls == [(Path("/tmp/ensemble"), "cpu")]


def test_load_policy_rejects_unknown_model_kind() -> None:
    bundle = SimpleNamespace(
        metadata={"model_kind": "pickle"},
        model_path=Path("/tmp/model.bin"),
    )

    with pytest.raises(ValueError, match="unsupported serving model_kind"):
        _load_policy(bundle)
