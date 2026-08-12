"""Framework-specific loading and RNG helpers for Universal stage audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trade_rl.rl.policies import SharedPerAssetActorCriticPolicy


def load_universal_policy_stage_model(
    path: Path,
    *,
    policy_only: bool,
    algorithm_identifier: str,
    device: str,
) -> Any:
    """Load one stage artifact without leaking model frameworks into workflows."""

    resolved = Path(path)
    if policy_only:
        return SharedPerAssetActorCriticPolicy.load(str(resolved), device=device)
    if algorithm_identifier == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(resolved), device=device)
    if algorithm_identifier == "lagrangian_ppo":
        from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

        return LagrangianPPO.load(str(resolved), device=device)
    raise ValueError(f"unsupported checkpoint algorithm: {algorithm_identifier}")


def seed_universal_policy_stage_torch(seed: int) -> None:
    """Seed the framework RNG used by stochastic policy-stage evaluation."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("stage evaluation seed must be a non-negative integer")
    import torch

    torch.manual_seed(seed)


__all__ = [
    "load_universal_policy_stage_model",
    "seed_universal_policy_stage_torch",
]
