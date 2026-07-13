"""Baseline-anchored residual reinforcement-learning core."""

from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    ActionSpec,
    AlphaContract,
    AlphaSignalKind,
    ActionValidationMode,
    BaselineResidualComposer,
    ResidualAction,
    ResidualActionV2,
    ResidualComposition,
)
from trade_rl.rl.configuration import EnvironmentExperimentManifest
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.experiments import ActionAblation, ActionExperimentSpec
from trade_rl.rl.normalization import NORMALIZER_SCHEMA, ObservationNormalizer
from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationExecutionState,
    ObservationLayout,
)
from trade_rl.rl.rewards import (
    REWARD_SCHEMA,
    RewardBreakdown,
    RewardConfig,
    RewardTracker,
)
from trade_rl.rl.training import (
    PolicyTrainingBackend,
    PolicyTrainingResult,
    ResidualTrainingConfig,
    StableBaselines3Backend,
    StableBaselines3PPOBackend,
    train_residual_ensemble,
)

__all__ = [
    "ACTION_SCHEMA",
    "NORMALIZER_SCHEMA",
    "OBSERVATION_SCHEMA",
    "REWARD_SCHEMA",
    "ActionSpec",
    "AlphaContract",
    "AlphaSignalKind",
    "ActionValidationMode",
    "ActionAblation",
    "ActionExperimentSpec",
    "BaselineResidualComposer",
    "EnvironmentExperimentManifest",
    "ObservationExecutionState",
    "ObservationLayout",
    "ObservationNormalizer",
    "PolicyTrainingBackend",
    "PolicyTrainingResult",
    "ResidualAction",
    "ResidualActionV2",
    "ResidualComposition",
    "ResidualMarketEnv",
    "ResidualMarketEnvConfig",
    "ResidualTrainingConfig",
    "RewardBreakdown",
    "RewardConfig",
    "RewardTracker",
    "StableBaselines3Backend",
    "StableBaselines3PPOBackend",
    "train_residual_ensemble",
]
