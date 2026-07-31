"""Artifact-backed Stage A evaluation-cell implementation."""

from __future__ import annotations

from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAZeroShotEvaluationPlan,
)
from trade_rl.workflows.stage_a_execution_replay import (
    StageAExecutionCellIdentity,
)
from trade_rl.workflows.stage_a_execution_store import (
    StageAExecutionPromotionStore,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
    StageAEvaluationCellResult,
)


class ArtifactBackedStageAEvaluationCellEvaluator:
    """Return growth only from execution artifacts bound to the exact request."""

    def __init__(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        store: StageAExecutionPromotionStore,
        baseline_candidate_config_digest: str,
    ) -> None:
        require_sha256(
            baseline_candidate_config_digest,
            field="stage_a_baseline_candidate_config_digest",
        )
        self.plan = plan
        self.store = store
        self.baseline_candidate_config_digest = baseline_candidate_config_digest

    def _expected_candidate_config(
        self, request: StageAEvaluationCellRequest
    ) -> str:
        if request.plan_digest != self.plan.digest:
            raise ValueError("Stage A evaluator plan identity mismatch")
        if request.dataset_identity != self.plan.dataset_identity:
            raise ValueError("Stage A evaluator dataset identity mismatch")
        if request.feature_identity != self.plan.feature_identity:
            raise ValueError("Stage A evaluator feature identity mismatch")
        if request.execution_identity != self.plan.execution_identity:
            raise ValueError("Stage A evaluator execution identity mismatch")
        if request.evaluation_identity != self.plan.evaluation_identity:
            raise ValueError("Stage A evaluator evaluation identity mismatch")
        if request.fold not in self.plan.folds:
            raise ValueError("Stage A evaluator fold is not declared")
        if request.seed not in self.plan.seeds:
            raise ValueError("Stage A evaluator seed is not declared")
        if request.triplet_id not in self.plan.triplet_ids_for(request.split):
            raise ValueError("Stage A evaluator triplet is not declared for split")
        if request.is_baseline:
            return self.baseline_candidate_config_digest
        candidate_id = request.candidate_id
        checkpoint_digest = request.checkpoint_digest
        if candidate_id is None or checkpoint_digest is None:
            raise ValueError("Stage A evaluator policy identity is incomplete")
        candidate = self.plan.candidate(candidate_id)
        if checkpoint_digest != candidate.checkpoint_digest(request.seed):
            raise ValueError("Stage A evaluator checkpoint identity mismatch")
        return candidate.candidate_config_digest

    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult:
        """Validate a request, reload its replay, and return recomputed growth."""

        expected_config = self._expected_candidate_config(request)
        stored = self.store.load(request.digest)
        actual_config = stored.artifact.cell_identity.candidate_config_digest
        if actual_config != expected_config:
            if request.is_baseline:
                raise ValueError("Stage A evaluator baseline configuration mismatch")
            raise ValueError("Stage A evaluator candidate configuration mismatch")
        expected_identity = StageAExecutionCellIdentity.from_request(
            request,
            candidate_config_digest=expected_config,
        )
        if stored.artifact.cell_identity != expected_identity:
            raise ValueError("Stage A evaluator execution cell identity mismatch")
        return StageAEvaluationCellResult(
            request_digest=request.digest,
            execution_evidence_digest=stored.artifact.digest,
            log_growth=stored.artifact.log_growth,
        )


__all__ = ["ArtifactBackedStageAEvaluationCellEvaluator"]
