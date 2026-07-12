from __future__ import annotations

from pathlib import Path

import numpy as np


class ResidualActionEnsemble:
    """Average deterministic two-dimensional actions before composition."""

    def __init__(self, policies: list[object]):
        if not policies:
            raise ValueError("at least one residual policy is required")
        self.policies = list(policies)
        self.last_actions: np.ndarray | None = None

    def predict(self, observation, deterministic: bool = True):
        actions = []
        for policy in self.policies:
            action, _ = policy.predict(observation, deterministic=deterministic)
            value = np.asarray(action, dtype=np.float64)
            if value.shape[-1] != 2 or not np.all(np.isfinite(value)):
                raise ValueError("residual ensemble member returned an invalid action")
            actions.append(value)
        stacked = np.stack(actions, axis=0)
        self.last_actions = stacked
        return stacked.mean(axis=0).astype(np.float32), None

    def disagreement(self, observation) -> float:
        self.predict(observation, deterministic=True)
        assert self.last_actions is not None
        return float(np.mean(np.std(self.last_actions, axis=0)))

    def save(self, path: str | Path) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for index, policy in enumerate(self.policies):
            policy.save(str(root / f"seed_{index}"))
        return root

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        from stable_baselines3 import PPO

        root = Path(path)
        model_paths = sorted(root.glob("seed_*.zip"))
        if not model_paths:
            raise FileNotFoundError(f"no seed_*.zip models in {root}")
        return cls([PPO.load(model_path, device=device) for model_path in model_paths])
