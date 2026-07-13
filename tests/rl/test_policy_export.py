from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from trade_rl.rl.export import export_policy_actor


class TinyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = np.array([self.steps / 8.0, -self.steps / 8.0], dtype=np.float32)
        reward = -float(np.square(action).sum())
        return observation, reward, self.steps >= 4, False, {}


def _checkpoint(path: Path) -> Path:
    model = PPO(
        "MlpPolicy",
        TinyEnv(),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=0,
        device="cpu",
    )
    model.learn(total_timesteps=8)
    model.save(str(path.with_suffix("")))
    return path


def test_torchscript_actor_export_matches_sb3_prediction(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy.zip")
    output = tmp_path / "exports"

    manifest = export_policy_actor(
        checkpoint_path=checkpoint,
        output_dir=output,
        algorithm="ppo",
        observation_size=2,
        action_size=1,
        action_spec_digest="a" * 64,
        normalizer_digest=None,
        onnx=False,
        torchscript=True,
        tolerance=1e-5,
    )

    record = next(item for item in manifest.exports if item.format == "torchscript")
    assert record.status == "verified"
    assert record.path == "policy.torchscript.pt"
    assert record.max_abs_error <= 1e-5
    scripted = torch.jit.load(str(output / record.path))
    observation = np.array([[0.25, -0.5]], dtype=np.float32)
    expected, _ = PPO.load(str(checkpoint), device="cpu").predict(
        observation, deterministic=True
    )
    actual = scripted(torch.from_numpy(observation)).detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0.0)
    assert (output / "export.json").is_file()


def test_torchscript_failure_is_recorded_as_unsupported(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "policy.zip")

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("unsupported policy graph")

    monkeypatch.setattr("trade_rl.rl.export._export_torchscript", fail)
    manifest = export_policy_actor(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exports",
        algorithm="ppo",
        observation_size=2,
        action_size=1,
        action_spec_digest="a" * 64,
        normalizer_digest=None,
        onnx=False,
        torchscript=True,
        tolerance=1e-5,
    )

    record = next(item for item in manifest.exports if item.format == "torchscript")
    assert record.status == "unsupported"
    assert record.reason == "unsupported policy graph"


def test_onnx_actor_export_matches_sb3_when_dependencies_exist(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = _checkpoint(tmp_path / "policy.zip")

    manifest = export_policy_actor(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "exports",
        algorithm="ppo",
        observation_size=2,
        action_size=1,
        action_spec_digest="a" * 64,
        normalizer_digest=None,
        onnx=True,
        torchscript=False,
        tolerance=1e-5,
    )

    record = next(item for item in manifest.exports if item.format == "onnx")
    assert record.status == "verified"
    assert record.max_abs_error <= 1e-5
