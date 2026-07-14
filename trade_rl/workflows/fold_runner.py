"""Concrete nested fold execution with sealed outer-OOS discipline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.selection import PolicyMode, SelectionDecision
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnSeries
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestAccessRecord,
    SealedTestLedger,
)
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult

BASELINE_CONFIGURATION = "baseline"


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    """One predeclared residual configuration eligible for fold-local training."""

    name: str

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="candidate.name")
        if self.name == BASELINE_CONFIGURATION:
            raise ValueError("candidate name cannot use the reserved baseline identity")


@dataclass(frozen=True, slots=True)
class FoldExecutionConfig:
    """Immutable selection rule and candidate set for every nested fold."""

    dataset_id: str
    signal_digest: str
    candidates: tuple[CandidateConfiguration, ...]
    minimum_selection_uplift: float
    selected_at: datetime

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.signal_digest, field="signal_digest")
        if not self.candidates:
            raise ValueError("at least one residual candidate is required")
        names = tuple(candidate.name for candidate in self.candidates)
        if len(set(names)) != len(names):
            raise ValueError("candidate configuration names must be unique")
        if (
            not math.isfinite(self.minimum_selection_uplift)
            or self.minimum_selection_uplift < 0.0
        ):
            raise ValueError("minimum_selection_uplift must be finite and non-negative")
        require_aware_datetime(self.selected_at, field="selected_at")

    @property
    def experiment_plan_digest(self) -> str:
        return content_digest(
            {
                "candidates": tuple(candidate.name for candidate in self.candidates),
                "dataset_id": self.dataset_id,
                "minimum_selection_uplift": self.minimum_selection_uplift,
                "schema_version": "fold_execution_plan_v2",
                "signal_digest": self.signal_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class CandidateTrainingRequest:
    """Range-scoped request for one configuration's training and checkpoint choice."""

    dataset_id: str
    fold_index: int
    configuration: CandidateConfiguration
    train: IndexRange
    checkpoint_validation: IndexRange

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.train.stop > self.checkpoint_validation.start:
            raise ValueError("checkpoint validation must follow the training range")


@dataclass(frozen=True, slots=True)
class PolicyTrainingArtifact:
    """Frozen checkpoint identity selected within one residual configuration."""

    configuration: str
    policy_digest: str

    def __post_init__(self) -> None:
        require_non_empty(self.configuration, field="configuration")
        require_sha256(self.policy_digest, field="policy_digest")


class EvaluationPhase(StrEnum):
    CONFIGURATION_SELECTION = "configuration_selection"
    OUTER_TEST = "outer_test"


@dataclass(frozen=True, slots=True)
class CandidateEvaluationRequest:
    """A policy evaluation request restricted to exactly one declared range."""

    dataset_id: str
    fold_index: int
    phase: EvaluationPhase
    evaluation_range: IndexRange
    configuration: str
    policy_digest: str | None

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        require_non_empty(self.configuration, field="configuration")
        if self.configuration == BASELINE_CONFIGURATION:
            if self.policy_digest is not None:
                raise ValueError("baseline evaluation cannot contain a policy digest")
        elif self.policy_digest is None:
            raise ValueError("residual evaluation requires a policy digest")
        else:
            require_sha256(self.policy_digest, field="policy_digest")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Finite score, return series, and content identity for one evaluation."""

    score: float
    returns: ReturnSeries
    evaluation_digest: str
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("evaluation score must be finite")
        require_sha256(self.evaluation_digest, field="evaluation_digest")


class CandidateTrainer(Protocol):
    def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact: ...


class CandidateEvaluator(Protocol):
    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class FoldExecutionResult:
    """Identity-bound selection and sealed selected/baseline OOS evidence."""

    dataset_id: str
    fold_index: int
    selection: SelectionDecision
    selection_evaluation_digest: str
    test_evaluation_digest: str
    selected_oos: FoldOOSResult
    baseline_oos: FoldOOSResult
    sealed_test_access: SealedTestAccessRecord | None = None

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(
            self.selection_evaluation_digest,
            field="selection_evaluation_digest",
        )
        require_sha256(self.test_evaluation_digest, field="test_evaluation_digest")
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.selection.dataset_id != self.dataset_id:
            raise ValueError("selection dataset identity mismatch")
        if self.selection.evaluation_digest != self.selection_evaluation_digest:
            raise ValueError("selection evaluation identity mismatch")
        for name, result in (
            ("selected", self.selected_oos),
            ("baseline", self.baseline_oos),
        ):
            if result.fold_index != self.fold_index:
                raise ValueError(f"{name} OOS fold identity mismatch")
        selected_range = (self.selected_oos.start, self.selected_oos.stop)
        baseline_range = (self.baseline_oos.start, self.baseline_oos.stop)
        if selected_range != baseline_range:
            raise ValueError("selected and baseline OOS ranges must match")


class ConcreteFoldRunner:
    """Train candidates, select without test access, then evaluate sealed OOS once."""

    def __init__(
        self,
        *,
        config: FoldExecutionConfig,
        trainer: CandidateTrainer,
        evaluator: CandidateEvaluator,
    ) -> None:
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self._sealed_test_ledger = SealedTestLedger()

    @staticmethod
    def _require_range_length(
        evaluation: CandidateEvaluation,
        evaluation_range: IndexRange,
    ) -> None:
        if len(evaluation.returns.values) != evaluation_range.size:
            raise ValueError("evaluation return length does not match requested range")

    def _evaluate(
        self,
        *,
        fold: WalkForwardFold,
        phase: EvaluationPhase,
        evaluation_range: IndexRange,
        configuration: str,
        policy_digest: str | None,
    ) -> CandidateEvaluation:
        result = self.evaluator.evaluate(
            CandidateEvaluationRequest(
                dataset_id=self.config.dataset_id,
                fold_index=fold.fold_index,
                phase=phase,
                evaluation_range=evaluation_range,
                configuration=configuration,
                policy_digest=policy_digest,
            )
        )
        self._require_range_length(result, evaluation_range)
        return result

    @staticmethod
    def _fold_oos(
        fold: WalkForwardFold,
        evaluation: CandidateEvaluation,
    ) -> FoldOOSResult:
        return FoldOOSResult(
            fold_index=fold.fold_index,
            start=fold.test.start,
            stop=fold.test.stop,
            returns=evaluation.returns,
            diagnostics=evaluation.diagnostics,
        )

    def run_fold(self, fold: WalkForwardFold) -> FoldExecutionResult:
        artifacts: dict[str, PolicyTrainingArtifact] = {}
        for candidate in self.config.candidates:
            artifact = self.trainer.train(
                CandidateTrainingRequest(
                    dataset_id=self.config.dataset_id,
                    fold_index=fold.fold_index,
                    configuration=candidate,
                    train=fold.train,
                    checkpoint_validation=fold.checkpoint_validation,
                )
            )
            if artifact.configuration != candidate.name:
                raise ValueError("training artifact configuration identity mismatch")
            artifacts[candidate.name] = artifact

        baseline_selection = self._evaluate(
            fold=fold,
            phase=EvaluationPhase.CONFIGURATION_SELECTION,
            evaluation_range=fold.configuration_selection,
            configuration=BASELINE_CONFIGURATION,
            policy_digest=None,
        )
        candidate_selection: dict[str, CandidateEvaluation] = {}
        for candidate in self.config.candidates:
            artifact = artifacts[candidate.name]
            candidate_selection[candidate.name] = self._evaluate(
                fold=fold,
                phase=EvaluationPhase.CONFIGURATION_SELECTION,
                evaluation_range=fold.configuration_selection,
                configuration=candidate.name,
                policy_digest=artifact.policy_digest,
            )

        selection_evaluation_digest = content_digest(
            {
                "baseline": {
                    "digest": baseline_selection.evaluation_digest,
                    "score": baseline_selection.score,
                },
                "candidates": tuple(
                    {
                        "configuration": name,
                        "digest": evaluation.evaluation_digest,
                        "policy_digest": artifacts[name].policy_digest,
                        "score": evaluation.score,
                    }
                    for name, evaluation in sorted(candidate_selection.items())
                ),
                "dataset_id": self.config.dataset_id,
                "fold_index": fold.fold_index,
                "range": (
                    fold.configuration_selection.start,
                    fold.configuration_selection.stop,
                ),
            }
        )
        threshold = baseline_selection.score + self.config.minimum_selection_uplift
        eligible = tuple(
            (name, evaluation)
            for name, evaluation in candidate_selection.items()
            if evaluation.score > threshold
        )
        selected_candidate = (
            min(eligible, key=lambda item: (-item[1].score, item[0]))
            if eligible
            else None
        )

        if selected_candidate is None:
            mode = PolicyMode.BASELINE_ONLY
            selected_configuration = BASELINE_CONFIGURATION
            selected_policy_digest = None
            reasons = (
                "no residual candidate exceeded the baseline selection threshold",
            )
        else:
            selected_configuration = selected_candidate[0]
            selected_policy_digest = artifacts[selected_configuration].policy_digest
            mode = PolicyMode.RESIDUAL_POLICY
            reasons = (
                "selected residual candidate exceeded the baseline selection threshold",
            )

        selection = SelectionDecision(
            dataset_id=self.config.dataset_id,
            mode=mode,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
            signal_digest=self.config.signal_digest,
            evaluation_digest=selection_evaluation_digest,
            selected_at=self.config.selected_at,
            reasons=reasons,
        )

        sealed_test_access = self._sealed_test_ledger.authorize_once(
            experiment_plan_digest=self.config.experiment_plan_digest,
            dataset_id=self.config.dataset_id,
            fold_index=fold.fold_index,
            test_range=fold.test,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
        )
        baseline_test = self._evaluate(
            fold=fold,
            phase=EvaluationPhase.OUTER_TEST,
            evaluation_range=fold.test,
            configuration=BASELINE_CONFIGURATION,
            policy_digest=None,
        )
        if selected_candidate is None:
            selected_test = baseline_test
        else:
            selected_test = self._evaluate(
                fold=fold,
                phase=EvaluationPhase.OUTER_TEST,
                evaluation_range=fold.test,
                configuration=selected_configuration,
                policy_digest=selected_policy_digest,
            )

        test_evaluation_digest = content_digest(
            {
                "baseline_digest": baseline_test.evaluation_digest,
                "dataset_id": self.config.dataset_id,
                "fold_index": fold.fold_index,
                "range": (fold.test.start, fold.test.stop),
                "selected_configuration": selected_configuration,
                "selected_digest": selected_test.evaluation_digest,
                "selected_policy_digest": selected_policy_digest,
            }
        )
        baseline_oos = self._fold_oos(fold, baseline_test)
        selected_oos = (
            baseline_oos
            if selected_candidate is None
            else self._fold_oos(fold, selected_test)
        )
        return FoldExecutionResult(
            dataset_id=self.config.dataset_id,
            fold_index=fold.fold_index,
            selection=selection,
            selection_evaluation_digest=selection_evaluation_digest,
            test_evaluation_digest=test_evaluation_digest,
            selected_oos=selected_oos,
            baseline_oos=baseline_oos,
            sealed_test_access=sealed_test_access,
        )
