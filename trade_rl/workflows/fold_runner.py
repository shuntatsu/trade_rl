"""Concrete nested fold execution with sealed outer-OOS discipline."""

from __future__ import annotations

import math
import statistics
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
    experiment_plan_digest: str
    minimum_selection_score: float = 0.0
    minimum_seed_success_fraction: float = 0.0
    minimum_worst_seed_uplift: float | None = None
    maximum_seed_score_std: float | None = None
    maximum_selection_turnover_per_day: float | None = None
    maximum_selection_cost_fraction: float | None = None
    maximum_selection_drawdown: float | None = None

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(
            self.experiment_plan_digest,
            field="experiment_plan_digest",
        )
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
        if not math.isfinite(self.minimum_selection_score):
            raise ValueError("minimum_selection_score must be finite")
        if (
            not math.isfinite(self.minimum_seed_success_fraction)
            or not 0.0 <= self.minimum_seed_success_fraction <= 1.0
        ):
            raise ValueError("minimum_seed_success_fraction must be within [0, 1]")
        if self.minimum_worst_seed_uplift is not None and not math.isfinite(
            self.minimum_worst_seed_uplift
        ):
            raise ValueError("minimum_worst_seed_uplift must be finite")
        for field_name, value in (
            ("maximum_seed_score_std", self.maximum_seed_score_std),
            (
                "maximum_selection_turnover_per_day",
                self.maximum_selection_turnover_per_day,
            ),
            (
                "maximum_selection_cost_fraction",
                self.maximum_selection_cost_fraction,
            ),
            ("maximum_selection_drawdown", self.maximum_selection_drawdown),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative")
        require_aware_datetime(self.selected_at, field="selected_at")


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
class CheckpointPolicyEvaluation:
    """Checkpoint-validation evidence for one policy from one random seed."""

    seed: int
    policy_digest: str
    score: float
    evaluation_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        require_sha256(self.policy_digest, field="policy_digest")
        if not math.isfinite(self.score):
            raise ValueError("checkpoint score must be finite")
        require_sha256(self.evaluation_digest, field="evaluation_digest")


@dataclass(frozen=True, slots=True)
class SeedPolicyFinalist:
    """Auditable checkpoint-validation winner for one seed."""

    seed: int
    policy_digest: str
    checkpoint_score: float
    checkpoint_evaluation_digest: str

    def __post_init__(self) -> None:
        CheckpointPolicyEvaluation(
            seed=self.seed,
            policy_digest=self.policy_digest,
            score=self.checkpoint_score,
            evaluation_digest=self.checkpoint_evaluation_digest,
        )


@dataclass(frozen=True, slots=True)
class SeedFinalistSelection:
    """Configuration-selection evidence attached to a seed-local finalist."""

    configuration: str
    seed: int
    policy_digest: str
    checkpoint_score: float
    checkpoint_evaluation_digest: str
    selection_score: float
    selection_evaluation_digest: str

    def __post_init__(self) -> None:
        require_non_empty(self.configuration, field="configuration")
        SeedPolicyFinalist(
            seed=self.seed,
            policy_digest=self.policy_digest,
            checkpoint_score=self.checkpoint_score,
            checkpoint_evaluation_digest=self.checkpoint_evaluation_digest,
        )
        if not math.isfinite(self.selection_score):
            raise ValueError("selection score must be finite")
        require_sha256(
            self.selection_evaluation_digest,
            field="selection_evaluation_digest",
        )


@dataclass(frozen=True, slots=True)
class PolicyTrainingArtifact:
    """Checkpoint finalists plus the exact deterministic deployable ensemble identity."""

    configuration: str
    seed_finalists: tuple[SeedPolicyFinalist, ...]
    ensemble_policy_digest: str = ""

    def __post_init__(self) -> None:
        require_non_empty(self.configuration, field="configuration")
        if not self.seed_finalists:
            raise ValueError("training artifact requires seed finalists")
        ordered = tuple(
            sorted(
                self.seed_finalists,
                key=lambda item: (
                    item.seed,
                    -item.checkpoint_score,
                    item.policy_digest,
                ),
            )
        )
        policy_digests = tuple(item.policy_digest for item in ordered)
        if len(set(policy_digests)) != len(policy_digests):
            raise ValueError("seed finalists must have unique policy digests")
        object.__setattr__(self, "seed_finalists", ordered)
        resolved = self.ensemble_policy_digest
        if not resolved:
            members = self.deployment_members
            resolved = (
                members[0].policy_digest
                if len(members) == 1
                else content_digest(
                    {
                        "members": tuple(
                            {"policy_digest": item.policy_digest, "seed": item.seed}
                            for item in members
                        ),
                        "schema_version": "deployable_mean_policy_ensemble_v1",
                    }
                )
            )
            object.__setattr__(self, "ensemble_policy_digest", resolved)
        require_sha256(resolved, field="ensemble_policy_digest")

    @property
    def deployment_members(self) -> tuple[SeedPolicyFinalist, ...]:
        """Return the checkpoint-validation winner for each fixed seed."""

        winners: list[SeedPolicyFinalist] = []
        seen: set[int] = set()
        for item in self.seed_finalists:
            if item.seed not in seen:
                winners.append(item)
                seen.add(item.seed)
        return tuple(winners)


def select_seed_checkpoint_finalists(
    *,
    checkpoint_evaluations: tuple[CheckpointPolicyEvaluation, ...],
    finalists_per_seed: int = 1,
) -> tuple[SeedPolicyFinalist, ...]:
    """Select a fixed top-k of checkpoint-validation winners for every seed."""

    if not checkpoint_evaluations:
        raise ValueError("checkpoint evaluations cannot be empty")
    if (
        isinstance(finalists_per_seed, bool)
        or not isinstance(finalists_per_seed, int)
        or finalists_per_seed <= 0
    ):
        raise ValueError("finalists_per_seed must be a positive integer")
    checkpoint_digests = tuple(item.policy_digest for item in checkpoint_evaluations)
    if len(set(checkpoint_digests)) != len(checkpoint_digests):
        raise ValueError("checkpoint policy digests must be unique")
    seeds = sorted({item.seed for item in checkpoint_evaluations})
    return tuple(
        SeedPolicyFinalist(
            seed=winner.seed,
            policy_digest=winner.policy_digest,
            checkpoint_score=winner.score,
            checkpoint_evaluation_digest=winner.evaluation_digest,
        )
        for seed in seeds
        for winner in sorted(
            (item for item in checkpoint_evaluations if item.seed == seed),
            key=lambda item: (-item.score, item.policy_digest),
        )[:finalists_per_seed]
    )


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
    turnover_per_day: float = 0.0
    cost_fraction: float = 0.0
    maximum_drawdown: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("evaluation score must be finite")
        for field_name, value in (
            ("turnover_per_day", self.turnover_per_day),
            ("cost_fraction", self.cost_fraction),
            ("maximum_drawdown", self.maximum_drawdown),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"evaluation {field_name} must be non-negative")
        if self.maximum_drawdown > 1.0:
            raise ValueError("evaluation maximum_drawdown cannot exceed one")
        require_sha256(self.evaluation_digest, field="evaluation_digest")


class CandidateTrainer(Protocol):
    def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact: ...


class CandidateEvaluator(Protocol):
    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class CandidateSelectionAggregate:
    """Seed-distribution and execution evidence used for candidate eligibility."""

    configuration: str
    eligible: bool
    median_score: float
    worst_score: float
    score_std: float
    success_fraction: float
    maximum_turnover_per_day: float
    maximum_cost_fraction: float
    maximum_drawdown: float
    seed_count: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.configuration, field="aggregate.configuration")
        for name, value in (
            ("median_score", self.median_score),
            ("worst_score", self.worst_score),
            ("score_std", self.score_std),
            ("success_fraction", self.success_fraction),
            ("maximum_turnover_per_day", self.maximum_turnover_per_day),
            ("maximum_cost_fraction", self.maximum_cost_fraction),
            ("maximum_drawdown", self.maximum_drawdown),
        ):
            if not math.isfinite(value):
                raise ValueError(f"aggregate {name} must be finite")
        if self.score_std < 0.0 or not 0.0 <= self.success_fraction <= 1.0:
            raise ValueError("aggregate dispersion or success fraction is invalid")
        if (
            self.maximum_turnover_per_day < 0.0
            or self.maximum_cost_fraction < 0.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
        ):
            raise ValueError("aggregate execution diagnostics must be non-negative")
        if (
            isinstance(self.seed_count, bool)
            or not isinstance(self.seed_count, int)
            or self.seed_count <= 0
        ):
            raise ValueError("aggregate seed_count must be positive")
        if self.eligible != (not self.reasons):
            raise ValueError("aggregate eligibility must match rejection reasons")

    def digest_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "eligible": self.eligible,
            "maximum_cost_fraction": self.maximum_cost_fraction,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_turnover_per_day": self.maximum_turnover_per_day,
            "median_score": self.median_score,
            "reasons": self.reasons,
            "score_std": self.score_std,
            "seed_count": self.seed_count,
            "success_fraction": self.success_fraction,
            "worst_score": self.worst_score,
        }


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
    seed_finalists: tuple[SeedFinalistSelection, ...] = ()
    candidate_aggregates: tuple[CandidateSelectionAggregate, ...] = ()
    selected_member_policy_digests: tuple[str, ...] = ()
    selected_member_seeds: tuple[int, ...] = ()
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
        aggregate_names = tuple(
            item.configuration for item in self.candidate_aggregates
        )
        if len(set(aggregate_names)) != len(aggregate_names):
            raise ValueError("candidate aggregate configurations must be unique")
        if len(self.selected_member_policy_digests) != len(self.selected_member_seeds):
            raise ValueError(
                "selected ensemble member identities must align with seeds"
            )
        for digest in self.selected_member_policy_digests:
            require_sha256(digest, field="selected_member_policy_digest")
        if len(set(self.selected_member_policy_digests)) != len(
            self.selected_member_policy_digests
        ):
            raise ValueError("selected ensemble member policy digests must be unique")
        if len(set(self.selected_member_seeds)) != len(self.selected_member_seeds):
            raise ValueError("selected ensemble member seeds must be unique")
        if self.selection.mode is PolicyMode.BASELINE_ONLY and (
            self.selected_member_policy_digests or self.selected_member_seeds
        ):
            raise ValueError("baseline selection cannot declare ensemble members")
        if self.selection.mode is PolicyMode.RESIDUAL_POLICY and not (
            self.selected_member_policy_digests and self.selected_member_seeds
        ):
            raise ValueError("residual selection requires ensemble member evidence")


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
        candidate_winners: dict[str, str] = {}
        candidate_member_digests: dict[str, tuple[str, ...]] = {}
        candidate_member_seeds: dict[str, tuple[int, ...]] = {}
        candidate_aggregates: dict[str, CandidateSelectionAggregate] = {}
        seed_finalists: list[SeedFinalistSelection] = []
        threshold = max(
            baseline_selection.score + self.config.minimum_selection_uplift,
            self.config.minimum_selection_score,
        )
        for candidate in self.config.candidates:
            artifact = artifacts[candidate.name]
            evaluated_finalists: list[
                tuple[SeedFinalistSelection, CandidateEvaluation]
            ] = []
            for finalist in artifact.deployment_members:
                evaluation = self._evaluate(
                    fold=fold,
                    phase=EvaluationPhase.CONFIGURATION_SELECTION,
                    evaluation_range=fold.configuration_selection,
                    configuration=candidate.name,
                    policy_digest=finalist.policy_digest,
                )
                evidence = SeedFinalistSelection(
                    configuration=candidate.name,
                    seed=finalist.seed,
                    policy_digest=finalist.policy_digest,
                    checkpoint_score=finalist.checkpoint_score,
                    checkpoint_evaluation_digest=(
                        finalist.checkpoint_evaluation_digest
                    ),
                    selection_score=evaluation.score,
                    selection_evaluation_digest=evaluation.evaluation_digest,
                )
                evaluated_finalists.append((evidence, evaluation))
                seed_finalists.append(evidence)
            if not evaluated_finalists:
                raise ValueError("candidate produced no seed finalists")
            scores = [item[0].selection_score for item in evaluated_finalists]
            median_score = float(statistics.median(scores))
            score_std = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
            worst_score = min(scores)
            success_fraction = sum(score > threshold for score in scores) / len(scores)
            maximum_turnover_per_day = max(
                item[1].turnover_per_day for item in evaluated_finalists
            )
            maximum_cost_fraction = max(
                item[1].cost_fraction for item in evaluated_finalists
            )
            maximum_drawdown = max(
                item[1].maximum_drawdown for item in evaluated_finalists
            )
            ensemble_digest = artifact.ensemble_policy_digest
            matching = [
                evaluation
                for evidence, evaluation in evaluated_finalists
                if evidence.policy_digest == ensemble_digest
            ]
            ensemble_evaluation = (
                matching[0]
                if len(matching) == 1
                else self._evaluate(
                    fold=fold,
                    phase=EvaluationPhase.CONFIGURATION_SELECTION,
                    evaluation_range=fold.configuration_selection,
                    configuration=candidate.name,
                    policy_digest=ensemble_digest,
                )
            )
            maximum_turnover_per_day = max(
                maximum_turnover_per_day,
                ensemble_evaluation.turnover_per_day,
            )
            maximum_cost_fraction = max(
                maximum_cost_fraction,
                ensemble_evaluation.cost_fraction,
            )
            maximum_drawdown = max(
                maximum_drawdown,
                ensemble_evaluation.maximum_drawdown,
            )
            reasons: list[str] = []
            if median_score <= threshold:
                reasons.append("median_seed_score_below_threshold")
            if success_fraction + 1e-12 < self.config.minimum_seed_success_fraction:
                reasons.append("seed_success_fraction_below_threshold")
            if (
                self.config.minimum_worst_seed_uplift is not None
                and worst_score - baseline_selection.score
                < self.config.minimum_worst_seed_uplift
            ):
                reasons.append("worst_seed_uplift_below_threshold")
            if (
                self.config.maximum_seed_score_std is not None
                and score_std > self.config.maximum_seed_score_std
            ):
                reasons.append("seed_score_dispersion_above_limit")
            if (
                self.config.maximum_selection_turnover_per_day is not None
                and maximum_turnover_per_day
                > self.config.maximum_selection_turnover_per_day
            ):
                reasons.append("selection_turnover_above_limit")
            if (
                self.config.maximum_selection_cost_fraction is not None
                and maximum_cost_fraction > self.config.maximum_selection_cost_fraction
            ):
                reasons.append("selection_cost_above_limit")
            if (
                self.config.maximum_selection_drawdown is not None
                and maximum_drawdown > self.config.maximum_selection_drawdown
            ):
                reasons.append("selection_drawdown_above_limit")
            if ensemble_evaluation.score <= threshold:
                reasons.append("deployable_ensemble_score_below_threshold")
            candidate_winners[candidate.name] = ensemble_digest
            candidate_selection[candidate.name] = ensemble_evaluation
            candidate_member_digests[candidate.name] = tuple(
                item.policy_digest for item in artifact.deployment_members
            )
            candidate_member_seeds[candidate.name] = tuple(
                item.seed for item in artifact.deployment_members
            )
            candidate_aggregates[candidate.name] = CandidateSelectionAggregate(
                configuration=candidate.name,
                eligible=not reasons,
                maximum_cost_fraction=maximum_cost_fraction,
                maximum_drawdown=maximum_drawdown,
                maximum_turnover_per_day=maximum_turnover_per_day,
                median_score=median_score,
                reasons=tuple(reasons),
                score_std=score_std,
                seed_count=len(scores),
                success_fraction=success_fraction,
                worst_score=worst_score,
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
                        "policy_digest": candidate_winners[name],
                        "deployable_ensemble_score": evaluation.score,
                        "member_policy_digests": candidate_member_digests[name],
                        "member_seeds": candidate_member_seeds[name],
                        "aggregate": candidate_aggregates[name].digest_payload(),
                    }
                    for name, evaluation in sorted(candidate_selection.items())
                ),
                "seed_finalists": tuple(
                    {
                        "checkpoint_evaluation_digest": (
                            item.checkpoint_evaluation_digest
                        ),
                        "checkpoint_score": item.checkpoint_score,
                        "configuration": item.configuration,
                        "policy_digest": item.policy_digest,
                        "seed": item.seed,
                        "selection_evaluation_digest": item.selection_evaluation_digest,
                        "selection_score": item.selection_score,
                    }
                    for item in sorted(
                        seed_finalists,
                        key=lambda item: (item.configuration, item.seed),
                    )
                ),
                "dataset_id": self.config.dataset_id,
                "fold_index": fold.fold_index,
                "range": (
                    fold.configuration_selection.start,
                    fold.configuration_selection.stop,
                ),
            }
        )
        eligible = tuple(
            (name, candidate_aggregates[name])
            for name in candidate_selection
            if candidate_aggregates[name].eligible
        )
        selected_candidate = (
            min(
                eligible,
                key=lambda item: (-candidate_selection[item[0]].score, item[0]),
            )
            if eligible
            else None
        )

        selection_reasons: tuple[str, ...]
        if selected_candidate is None:
            mode = PolicyMode.BASELINE_ONLY
            selected_configuration = BASELINE_CONFIGURATION
            selected_policy_digest = None
            selection_reasons = (
                "no residual candidate exceeded the baseline selection threshold",
            )
        else:
            selected_configuration = selected_candidate[0]
            selected_policy_digest = candidate_winners[selected_configuration]
            mode = PolicyMode.RESIDUAL_POLICY
            selection_reasons = (
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
            reasons=selection_reasons,
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
            seed_finalists=tuple(
                sorted(seed_finalists, key=lambda item: (item.configuration, item.seed))
            ),
            candidate_aggregates=tuple(
                candidate_aggregates[name] for name in sorted(candidate_aggregates)
            ),
            selected_member_policy_digests=(
                ()
                if selected_candidate is None
                else candidate_member_digests[selected_configuration]
            ),
            selected_member_seeds=(
                ()
                if selected_candidate is None
                else candidate_member_seeds[selected_configuration]
            ),
            sealed_test_access=sealed_test_access,
        )
