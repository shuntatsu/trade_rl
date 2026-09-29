"""Immutable contracts for controlled development experiments."""

from trade_rl.evaluation.experiments.contracts.comparison import ExperimentComparison
from trade_rl.evaluation.experiments.contracts.decision import (
    ExperimentDecision,
    ExperimentDecisionKind,
)
from trade_rl.evaluation.experiments.contracts.experiment import (
    ControlledFactor,
    ExperimentDefinition,
    ExperimentFailure,
)
from trade_rl.evaluation.experiments.contracts.research import (
    ConsumedEvidence,
    EvidenceKind,
    EvidenceUse,
    StudyResearchContext,
)
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.contracts.study import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    PPO_HOLDING_DURATION_HORIZONS,
    PPO_HOLDING_DURATION_MAX_DRAWDOWN,
    PPO_HOLDING_DURATION_RISK_CONFIG,
    PPO_HOLDING_DURATION_SEED_COUNT,
    StudyFreeze,
    StudyOutcome,
    StudyPlan,
    StudyProtocol,
)

__all__ = [
    "CANDIDATE_STRATEGY_NAMES",
    "CONTROL_STRATEGY_NAMES",
    "ConsumedEvidence",
    "ControlledFactor",
    "EvidenceKind",
    "EvidenceUse",
    "ExperimentComparison",
    "ExperimentDecision",
    "ExperimentDecisionKind",
    "ExperimentDefinition",
    "ExperimentFailure",
    "PPO_HOLDING_DURATION_HORIZONS",
    "PPO_HOLDING_DURATION_MAX_DRAWDOWN",
    "PPO_HOLDING_DURATION_RISK_CONFIG",
    "PPO_HOLDING_DURATION_SEED_COUNT",
    "ResolvedRunConfig",
    "StudyFreeze",
    "StudyOutcome",
    "StudyPlan",
    "StudyProtocol",
    "StudyResearchContext",
]
