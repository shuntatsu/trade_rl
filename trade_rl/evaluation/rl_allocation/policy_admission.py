"""Verified allocation-policy artifact admission before continuous OOS execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    ContinuousAllocationWalkForwardResult,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.strategies.rl.allocation_artifact import load_allocation_policy


@dataclass(frozen=True, slots=True)
class AllocationFoldPolicyArtifact:
    """One immutable policy bundle pin assigned to one declared OOS fold."""

    fold_index: int
    bundle_root: Path
    expected_digest: str
    expected_recipe_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be a non-negative integer")
        try:
            root = Path(self.bundle_root)
        except TypeError as error:
            raise ValueError("bundle_root must be path-like") from error
        object.__setattr__(self, "bundle_root", root)
        require_sha256(self.expected_digest, field="expected_digest")
        require_sha256(self.expected_recipe_digest, field="expected_recipe_digest")


def _preflight(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
) -> None:
    if (
        type(folds) is not tuple
        or not folds
        or any(type(fold) is not WalkForwardFold for fold in folds)
    ):
        raise ValueError("artifact admission requires immutable walk-forward folds")
    if (
        type(environments) is not tuple
        or len(environments) != len(folds)
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("artifact admission requires one allocation env per fold")
    if (
        type(artifacts) is not tuple
        or len(artifacts) != len(folds)
        or any(type(value) is not AllocationFoldPolicyArtifact for value in artifacts)
    ):
        raise ValueError("artifact admission requires one policy artifact per fold")
    for fold, env, artifact in zip(folds, environments, artifacts, strict=True):
        if artifact.fold_index != fold.fold_index:
            raise ValueError("policy artifact fold identity differs from declaration")
        if artifact.expected_recipe_digest != env.recipe_digest:
            raise ValueError("policy artifact recipe differs from allocation runtime")


def _bind_loaded_policy(
    loaded: Any, artifact: AllocationFoldPolicyArtifact
) -> AllocationFoldPolicy:
    manifest = getattr(loaded, "manifest", None)
    if not isinstance(manifest, dict):
        raise ValueError("loaded allocation policy has no canonical manifest")
    if (
        content_digest(manifest) != artifact.expected_digest
        or manifest.get("recipe_digest") != artifact.expected_recipe_digest
    ):
        raise ValueError("loaded allocation policy differs from its admitted identity")
    action = getattr(loaded, "action", None)
    if not callable(action):
        raise ValueError("loaded allocation policy has no inference action")

    def invoke(observation: object, runtime_recipe_digest: str) -> object:
        return action(
            observation,
            runtime_recipe_digest=runtime_recipe_digest,
        )

    return AllocationFoldPolicy(
        policy_digest=artifact.expected_digest,
        recipe_digest=artifact.expected_recipe_digest,
        action=invoke,
    )


def admit_allocation_fold_policies(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
) -> tuple[AllocationFoldPolicy, ...]:
    """Verify all immutable bundles before any outer-OOS account is reset."""
    _preflight(folds, environments, artifacts)
    admitted: list[AllocationFoldPolicy] = []
    for artifact in artifacts:
        loaded = load_allocation_policy(
            artifact.bundle_root,
            expected_digest=artifact.expected_digest,
            expected_recipe_digest=artifact.expected_recipe_digest,
        )
        admitted.append(_bind_loaded_policy(loaded, artifact))
    return tuple(admitted)


def run_artifact_bound_continuous_allocation_walk_forward(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
    *,
    reset_seed: int | None = None,
) -> ContinuousAllocationWalkForwardResult:
    """Verify every policy bundle first, then execute the continuous OOS chain."""
    policies = admit_allocation_fold_policies(folds, environments, artifacts)
    return run_continuous_allocation_walk_forward(
        folds,
        environments,
        policies,
        reset_seed=reset_seed,
    )


__all__ = [
    "AllocationFoldPolicyArtifact",
    "admit_allocation_fold_policies",
    "run_artifact_bound_continuous_allocation_walk_forward",
]
