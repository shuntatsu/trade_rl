"""Bind sealed outer-test authorization to verified allocation policy artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl._validation import require_sha256
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    ContinuousAllocationWalkForwardResult,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.policy_admission import (
    AllocationFoldPolicyArtifact,
    admit_allocation_fold_policies,
)
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.robustness.walk_forward.sealed_test import (
    SealedTestAccessRecord,
    SealedTestLedgerProtocol,
    build_sealed_test_access_record,
)


@dataclass(frozen=True, slots=True)
class SealedAllocationFoldPolicyArtifact:
    """One authorized outer-test policy artifact for one declared fold."""

    access: SealedTestAccessRecord
    artifact: AllocationFoldPolicyArtifact

    def __post_init__(self) -> None:
        if type(self.access) is not SealedTestAccessRecord:
            raise ValueError("sealed allocation binding requires an access record")
        if type(self.artifact) is not AllocationFoldPolicyArtifact:
            raise ValueError("sealed allocation binding requires a policy artifact")


def _canonical_access(record: SealedTestAccessRecord) -> SealedTestAccessRecord:
    rebuilt = build_sealed_test_access_record(
        experiment_plan_digest=record.experiment_plan_digest,
        dataset_id=record.dataset_id,
        fold_index=record.fold_index,
        test_range=record.test_range,
        selected_configuration=record.selected_configuration,
        selected_policy_digest=record.selected_policy_digest,
    )
    if rebuilt != record:
        raise ValueError("sealed outer-test access record digest is inconsistent")
    return record


def validate_sealed_allocation_fold_artifacts(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    bindings: tuple[SealedAllocationFoldPolicyArtifact, ...],
    *,
    experiment_plan_digest: str,
    access_ledger: SealedTestLedgerProtocol,
) -> None:
    """Validate sealed selection identity before policy artifact deserialization."""
    require_sha256(experiment_plan_digest, field="experiment_plan_digest")
    if (
        type(folds) is not tuple
        or not folds
        or any(type(fold) is not WalkForwardFold for fold in folds)
    ):
        raise ValueError("sealed allocation admission requires immutable folds")
    if (
        type(environments) is not tuple
        or len(environments) != len(folds)
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("sealed allocation admission requires one env per fold")
    if (
        type(bindings) is not tuple
        or len(bindings) != len(folds)
        or any(
            type(value) is not SealedAllocationFoldPolicyArtifact for value in bindings
        )
    ):
        raise ValueError("sealed allocation admission requires one binding per fold")

    records = access_ledger.records
    if type(records) is not tuple:
        raise ValueError(
            "sealed allocation access ledger must expose immutable records"
        )
    access_digests: list[str] = []
    for fold, env, binding in zip(folds, environments, bindings, strict=True):
        access = _canonical_access(binding.access)
        if access not in records:
            raise ValueError(
                "sealed allocation access was not authorized by this ledger"
            )
        artifact = binding.artifact
        if access.experiment_plan_digest != experiment_plan_digest:
            raise ValueError("sealed allocation experiment plan identity mismatch")
        if access.dataset_id != env.dataset.dataset_id:
            raise ValueError("sealed allocation Dataset identity mismatch")
        if (
            access.fold_index != fold.fold_index
            or artifact.fold_index != fold.fold_index
        ):
            raise ValueError("sealed allocation fold identity mismatch")
        if access.test_range != fold.test:
            raise ValueError("sealed allocation test range mismatch")
        if access.selected_policy_digest is None:
            raise ValueError("sealed allocation execution requires a selected policy")
        if access.selected_policy_digest != artifact.expected_digest:
            raise ValueError("sealed selected policy differs from admitted artifact")
        if artifact.expected_recipe_digest != env.recipe_digest:
            raise ValueError("sealed policy recipe differs from allocation runtime")
        access_digests.append(access.access_digest)
    if len(access_digests) != len(set(access_digests)):
        raise ValueError("sealed allocation access records must be unique")


def run_sealed_artifact_bound_continuous_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    bindings: tuple[SealedAllocationFoldPolicyArtifact, ...],
    *,
    experiment_plan_digest: str,
    access_ledger: SealedTestLedgerProtocol,
    reset_seed: int | None = None,
) -> ContinuousAllocationWalkForwardResult:
    """Verify sealed authorization and artifacts before the first OOS reset."""
    validate_sealed_allocation_fold_artifacts(
        folds,
        environments,
        bindings,
        experiment_plan_digest=experiment_plan_digest,
        access_ledger=access_ledger,
    )
    artifacts = tuple(value.artifact for value in bindings)
    policies = admit_allocation_fold_policies(folds, environments, artifacts)
    selected = tuple(value.access.selected_policy_digest for value in bindings)
    if (
        any(value is None for value in selected)
        or tuple(policy.policy_digest for policy in policies) != selected
    ):
        raise ValueError("admitted policies differ from sealed selected identities")
    return run_continuous_allocation_walk_forward(
        folds,
        environments,
        policies,
        reset_seed=reset_seed,
    )


__all__ = [
    "SealedAllocationFoldPolicyArtifact",
    "run_sealed_artifact_bound_continuous_allocation_walk_forward",
    "validate_sealed_allocation_fold_artifacts",
]
