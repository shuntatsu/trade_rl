"""Deterministic Stage A validation and sealed-test orchestration."""

from __future__ import annotations

from typing import cast

from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAEvaluationObservation,
    StageAEvaluationSplit,
    StageAZeroShotEvaluationPlan,
    build_stage_a_evaluation_evidence,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    evaluate_stage_a_sealed_test,
    select_stage_a_validation_candidate,
)
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestLedger,
    SealedTestLedgerProtocol,
)
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellEvaluator,
    StageAEvaluationCellRequest,
    StageAEvaluationCellResult,
    StageASealedTestAccessRecord,
    StageASealedTestRun,
    StageATestSchedule,
    StageAValidationRun,
)


class StageAZeroShotEvaluationOrchestrator:
    """Build complete validation evidence before opening selected-only test data."""

    def __init__(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
        evaluator: StageAEvaluationCellEvaluator,
        test_schedule: StageATestSchedule | None = None,
        sealed_test_ledger: SealedTestLedgerProtocol | None = None,
    ) -> None:
        plan.validate_manifest(manifest)
        resolved_schedule = test_schedule or StageATestSchedule.from_manifest(
            plan=plan,
            manifest=manifest,
        )
        resolved_schedule.validate_manifest(plan, manifest)
        self.plan = plan
        self.manifest = manifest
        self.evaluator = evaluator
        self.test_schedule = resolved_schedule
        self._sealed_test_ledger = sealed_test_ledger or SealedTestLedger()

    def _request(
        self,
        *,
        split: StageAEvaluationSplit,
        triplet_id: str,
        fold: int,
        seed: int,
        candidate_id: str | None,
    ) -> StageAEvaluationCellRequest:
        checkpoint_digest = (
            None
            if candidate_id is None
            else self.plan.candidate(candidate_id).checkpoint_digest(seed)
        )
        return StageAEvaluationCellRequest(
            plan_digest=self.plan.digest,
            evaluation_dataset_manifest_digest=self.manifest.digest,
            split=split,
            triplet_id=triplet_id,
            fold=fold,
            seed=seed,
            candidate_id=candidate_id,
            checkpoint_digest=checkpoint_digest,
            dataset_id=self.manifest.dataset_id_for(split, triplet_id),
            evaluation_range=self.manifest.range_for(split, fold),
            feature_identity=self.plan.feature_identity,
            execution_identity=self.plan.execution_identity,
            evaluation_identity=self.plan.evaluation_identity,
        )

    def _evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult:
        result = self.evaluator.evaluate(request)
        if result.request_digest != request.digest:
            raise ValueError("Stage A evaluator result request identity mismatch")
        return result

    def _observations_for_split(
        self,
        *,
        split: StageAEvaluationSplit,
        candidate_ids: tuple[str, ...],
    ) -> tuple[StageAEvaluationObservation, ...]:
        observations: list[StageAEvaluationObservation] = []
        for triplet_id in self.plan.triplet_ids_for(split):
            for fold in self.plan.folds:
                for seed in self.plan.seeds:
                    baseline_request = self._request(
                        split=split,
                        triplet_id=triplet_id,
                        fold=fold,
                        seed=seed,
                        candidate_id=None,
                    )
                    baseline = self._evaluate(baseline_request)
                    for candidate_id in candidate_ids:
                        candidate = self.plan.candidate(candidate_id)
                        policy_request = self._request(
                            split=split,
                            triplet_id=triplet_id,
                            fold=fold,
                            seed=seed,
                            candidate_id=candidate_id,
                        )
                        if (
                            policy_request.evaluation_dataset_manifest_digest
                            != baseline_request.evaluation_dataset_manifest_digest
                            or policy_request.dataset_id != baseline_request.dataset_id
                            or policy_request.evaluation_range
                            != baseline_request.evaluation_range
                        ):
                            raise ValueError(
                                "Stage A policy and baseline request data identity mismatch"
                            )
                        policy = self._evaluate(policy_request)
                        observations.append(
                            StageAEvaluationObservation.create(
                                candidate_id=candidate_id,
                                split=split,
                                triplet_id=triplet_id,
                                fold=fold,
                                seed=seed,
                                checkpoint_digest=candidate.checkpoint_digest(seed),
                                evaluation_dataset_manifest_digest=(
                                    policy_request.evaluation_dataset_manifest_digest
                                ),
                                dataset_id=policy_request.dataset_id,
                                evaluation_range=policy_request.evaluation_range,
                                feature_identity=self.plan.feature_identity,
                                execution_identity=self.plan.execution_identity,
                                evaluation_identity=self.plan.evaluation_identity,
                                policy_execution_evidence_digest=(
                                    policy.execution_evidence_digest
                                ),
                                baseline_execution_evidence_digest=(
                                    baseline.execution_evidence_digest
                                ),
                                policy_log_growth=policy.log_growth,
                                baseline_log_growth=baseline.log_growth,
                            )
                        )
        return tuple(observations)

    def evaluate_validation(self) -> StageAValidationRun:
        observations = self._observations_for_split(
            split=cast(StageAEvaluationSplit, "validation"),
            candidate_ids=self.plan.candidate_ids,
        )
        evidence = build_stage_a_evaluation_evidence(
            plan=self.plan,
            manifest=self.manifest,
            split="validation",
            observations=observations,
        )
        selection = select_stage_a_validation_candidate(
            plan=self.plan,
            evidence=evidence,
        )
        return StageAValidationRun(evidence=evidence, selection=selection)

    def evaluate_sealed_test(
        self, validation_run: StageAValidationRun
    ) -> StageASealedTestRun:
        expected_selection = select_stage_a_validation_candidate(
            plan=self.plan,
            evidence=validation_run.evidence,
        )
        if validation_run.selection != expected_selection:
            raise ValueError("Stage A validation run does not match recomputation")
        selection = validation_run.selection
        selected_id = selection.selected_candidate_id
        if not selection.passed or selected_id is None:
            raise ValueError(
                "Stage A sealed test requires a passed validation selection"
            )
        self.test_schedule.validate_manifest(self.plan, self.manifest)
        selected_candidate = self.plan.candidate(selected_id)
        access_records: list[StageASealedTestAccessRecord] = []
        for triplet_id in self.plan.test_triplet_ids:
            dataset_id = self.manifest.dataset_id_for("test", triplet_id)
            for fold in self.plan.folds:
                test_range = self.test_schedule.range_for(fold)
                generic = self._sealed_test_ledger.authorize_once(
                    experiment_plan_digest=self.plan.digest,
                    dataset_id=dataset_id,
                    fold_index=fold,
                    test_range=test_range,
                    selected_configuration=selected_id,
                    selected_policy_digest=selected_candidate.digest,
                )
                access_records.append(
                    StageASealedTestAccessRecord(
                        evaluation_dataset_manifest_digest=self.manifest.digest,
                        triplet_id=triplet_id,
                        dataset_id=dataset_id,
                        fold=fold,
                        test_range=test_range,
                        ledger_record=generic,
                    )
                )
        observations = self._observations_for_split(
            split=cast(StageAEvaluationSplit, "test"),
            candidate_ids=(selected_id,),
        )
        evidence = build_stage_a_evaluation_evidence(
            plan=self.plan,
            manifest=self.manifest,
            split="test",
            observations=observations,
            candidate_ids=(selected_id,),
        )
        decision = evaluate_stage_a_sealed_test(
            plan=self.plan,
            validation_evidence=validation_run.evidence,
            selection=selection,
            evidence=evidence,
        )
        return StageASealedTestRun(
            validation_run=validation_run,
            access_records=tuple(access_records),
            evidence=evidence,
            decision=decision,
        )


__all__ = ["StageAZeroShotEvaluationOrchestrator"]
