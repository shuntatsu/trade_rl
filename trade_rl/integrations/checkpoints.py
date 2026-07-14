"""Stable-Baselines3 implementation of framework-neutral checkpoint loading."""

from __future__ import annotations

from typing import Any

from trade_rl.domain.checkpoints import PolicyCheckpoint


class StableBaselines3CheckpointLoader:
    """Load supported SB3-family checkpoints on CPU for evaluation."""

    def load(self, checkpoint: PolicyCheckpoint) -> Any:
        algorithm = checkpoint.algorithm.lower()
        if algorithm == "ppo":
            from stable_baselines3 import PPO

            return PPO.load(str(checkpoint.path), device="cpu")
        if algorithm == "sac":
            from stable_baselines3 import SAC

            return SAC.load(str(checkpoint.path), device="cpu")
        if algorithm == "td3":
            from stable_baselines3 import TD3

            return TD3.load(str(checkpoint.path), device="cpu")
        if algorithm == "tqc":
            from sb3_contrib import TQC

            return TQC.load(str(checkpoint.path), device="cpu")
        raise ValueError("unsupported walk-forward policy algorithm")


__all__ = ["StableBaselines3CheckpointLoader"]
