"""Baseline-anchored residual reinforcement-learning core."""

from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    BaselineResidualComposer,
    ResidualAction,
    ResidualComposition,
)
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.rl.rewards import (
    AbsoluteGrowthRewardConfig,
    RewardBreakdown,
    RewardContext,
)
from trade_rl.rl.training import (
    PolicyTrainingBackend,
    PolicyTrainingResult,
    ResidualTrainingConfig,
    StableBaselines3PPOBackend,
    train_residual_ensemble,
)

__all__ = [
    "ACTION_SCHEMA",
    "OBSERVATION_SCHEMA",
    "AbsoluteGrowthRewardConfig",
    "BaselineResidualComposer",
    "PolicyTrainingBackend",
    "PolicyTrainingResult",
    "ResidualAction",
    "ResidualComposition",
    "ResidualMarketEnv",
    "ResidualMarketEnvConfig",
    "ResidualTrainingConfig",
    "RewardBreakdown",
    "RewardContext",
    "StableBaselines3PPOBackend",
    "train_residual_ensemble",
]
