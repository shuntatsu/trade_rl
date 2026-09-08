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
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.contracts.study import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    StudyFreeze,
    StudyOutcome,
    StudyPlan,
)

__all__ = [
    "CANDIDATE_STRATEGY_NAMES",
    "CONTROL_STRATEGY_NAMES",
    "ControlledFactor",
    "ExperimentComparison",
    "ExperimentDecision",
    "ExperimentDecisionKind",
    "ExperimentDefinition",
    "ExperimentFailure",
    "ResolvedRunConfig",
    "StudyFreeze",
    "StudyOutcome",
    "StudyPlan",
]
