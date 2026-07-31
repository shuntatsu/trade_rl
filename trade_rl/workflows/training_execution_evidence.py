"""Resolve one internally consistent execution-evidence package for training."""

from __future__ import annotations

from pathlib import Path

from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    execution_evidence_from_cost,
    load_execution_evidence,
    validate_execution_event_artifact,
)
from trade_rl.simulation.execution_replay import ExecutionEventArtifact


def resolve_training_execution_inputs(
    *,
    dataset_id: str,
    cost: ExecutionCostConfig,
    evidence_path: Path | None,
    event_artifact_path: Path | None,
) -> tuple[ExecutionEvidence, ExecutionEventArtifact | None]:
    """Load or derive evidence while preserving artifact binding and identity."""

    if evidence_path is None:
        evidence = execution_evidence_from_cost(
            dataset_id=dataset_id,
            cost=cost,
            order_event_artifact_path=event_artifact_path,
        )
    else:
        evidence = load_execution_evidence(evidence_path)

    if evidence.dataset_id != dataset_id:
        raise ValueError("execution evidence dataset identity mismatch")
    expected_policy_digest = cost.execution_policy_digest
    if evidence.execution_policy_digest != expected_policy_digest:
        raise ValueError("execution evidence policy digest mismatch")
    if event_artifact_path is None and evidence.order_event_artifact_digest is not None:
        raise ValueError("execution evidence requires its bound event artifact")

    artifact = (
        None
        if event_artifact_path is None
        else validate_execution_event_artifact(evidence, event_artifact_path)
    )
    if artifact is not None:
        if artifact.dataset_id != dataset_id:
            raise ValueError("execution replay dataset identity mismatch")
        if artifact.execution_policy_digest != expected_policy_digest:
            raise ValueError("execution replay policy digest mismatch")
    return evidence, artifact


__all__ = ["resolve_training_execution_inputs"]
