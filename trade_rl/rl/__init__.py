"""Baseline-anchored residual reinforcement-learning core."""

from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    BaselineResidualComposer,
    ResidualAction,
    ResidualComposition,
)
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.training import (
    PolicyTrainingBackend,
    ResidualTrainingConfig,
    StableBaselines3PPOBackend,
    train_residual_ensemble,
)

__all__ = [
    "ACTION_SCHEMA",
    "BaselineResidualComposer",
    "PolicyTrainingBackend",
    "ResidualAction",
    "ResidualComposition",
    "ResidualMarketEnv",
    "ResidualMarketEnvConfig",
    "ResidualTrainingConfig",
    "StableBaselines3PPOBackend",
    "train_residual_ensemble",
]
