"""Baseline-anchored residual reinforcement-learning core.

Optional Gymnasium, Torch and SB3 dependencies are loaded lazily so data,
evaluation and serving contracts remain importable in minimal installations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS = {
    "trade_rl.rl.actions": (
        "ACTION_SCHEMA",
        "ActionMode",
        "ActionSpec",
        "ActionValidationMode",
        "AlphaContract",
        "AlphaSignalKind",
        "BaselineResidualComposer",
        "ResidualAction",
        "ResidualActionV2",
        "ResidualComposition",
        "TargetWeightAction",
    ),
    "trade_rl.rl.normalization": ("NORMALIZER_SCHEMA", "ObservationNormalizer"),
    "trade_rl.rl.observations": (
        "OBSERVATION_SCHEMA",
        "ObservationBuilder",
        "ObservationExecutionState",
        "ObservationInput",
        "ObservationLayout",
    ),
    "trade_rl.rl.rewards": (
        "REWARD_SCHEMA",
        "AbsoluteGrowthRewardConfig",
        "RewardBreakdown",
        "RewardConfig",
        "RewardContext",
        "RewardTracker",
        "absolute_growth_reward",
        "build_reward_context",
        "drawdown_severity",
    ),
    "trade_rl.rl.environment": ("ResidualMarketEnv", "ResidualMarketEnvConfig"),
    "trade_rl.rl.configuration": ("EnvironmentExperimentManifest",),
    "trade_rl.rl.experiments": ("ActionAblation", "ActionExperimentSpec"),
    "trade_rl.rl.market_inputs": ("CausalMarketView", "MarketInputResolver"),
    "trade_rl.rl.training": (
        "PolicyTrainingBackend",
        "PolicyTrainingResult",
        "ResidualTrainingConfig",
        "StableBaselines3Backend",
        "StableBaselines3PPOBackend",
        "train_residual_ensemble",
    ),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(module), name)


__all__ = tuple(_EXPORTS)
