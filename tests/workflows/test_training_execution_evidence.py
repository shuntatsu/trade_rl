from __future__ import annotations

from pathlib import Path

import pytest

from tests.evaluation.replay_support import (
    COST,
    DATASET_ID,
    execution_artifact,
    write_execution_artifact,
)
from trade_rl.simulation.execution_promotion import (
    ExecutionPromotionError,
    execution_evidence_from_cost,
    write_execution_evidence,
)
from trade_rl.simulation.execution_replay import write_execution_event_artifact
from trade_rl.workflows.training_execution_evidence import (
    resolve_training_execution_inputs,
)


def test_no_loose_inputs_produce_default_unbound_evidence() -> None:
    evidence, artifact = resolve_training_execution_inputs(
        dataset_id=DATASET_ID,
        cost=COST,
        evidence_path=None,
        event_artifact_path=None,
    )

    assert artifact is None
    assert evidence.dataset_id == DATASET_ID
    assert evidence.execution_policy_digest == COST.execution_policy_digest
    assert evidence.order_event_count == 0
    assert not evidence.complete_order_evidence
    assert evidence.order_event_artifact_digest is None


def test_artifact_only_derives_evidence_bound_to_exact_bytes(tmp_path: Path) -> None:
    expected_artifact, artifact_path = write_execution_artifact(
        tmp_path / "order-events.json"
    )

    evidence, artifact = resolve_training_execution_inputs(
        dataset_id=DATASET_ID,
        cost=COST,
        evidence_path=None,
        event_artifact_path=artifact_path,
    )

    assert artifact == expected_artifact
    assert evidence.order_event_artifact_digest == expected_artifact.digest
    assert evidence.order_event_count == expected_artifact.order_event_count
    assert evidence.replay_identity_digest == expected_artifact.replay_identity.digest
    assert evidence.replay_evidence_digest == expected_artifact.replay_evidence.digest
    assert evidence.complete_order_evidence


def test_matching_loose_evidence_and_artifact_are_accepted(tmp_path: Path) -> None:
    expected_artifact, artifact_path = write_execution_artifact(
        tmp_path / "order-events.json"
    )
    expected_evidence = execution_evidence_from_cost(
        dataset_id=DATASET_ID,
        cost=COST,
        order_event_artifact_path=artifact_path,
    )
    evidence_path = tmp_path / "execution-evidence.json"
    write_execution_evidence(evidence_path, expected_evidence)

    evidence, artifact = resolve_training_execution_inputs(
        dataset_id=DATASET_ID,
        cost=COST,
        evidence_path=evidence_path,
        event_artifact_path=artifact_path,
    )

    assert evidence == expected_evidence
    assert artifact == expected_artifact


def test_loose_evidence_rejects_substituted_artifact(tmp_path: Path) -> None:
    _, bound_path = write_execution_artifact(tmp_path / "bound-events.json")
    evidence = execution_evidence_from_cost(
        dataset_id=DATASET_ID,
        cost=COST,
        order_event_artifact_path=bound_path,
    )
    evidence_path = tmp_path / "execution-evidence.json"
    write_execution_evidence(evidence_path, evidence)

    substituted = execution_artifact(candidate_config_digest="f" * 64)
    substituted_path = write_execution_event_artifact(
        tmp_path / "substituted-events.json",
        substituted,
    )

    with pytest.raises(ExecutionPromotionError, match="digest mismatch"):
        resolve_training_execution_inputs(
            dataset_id=DATASET_ID,
            cost=COST,
            evidence_path=evidence_path,
            event_artifact_path=substituted_path,
        )


def test_loose_evidence_requires_training_dataset_identity(tmp_path: Path) -> None:
    evidence = execution_evidence_from_cost(dataset_id="f" * 64, cost=COST)
    evidence_path = tmp_path / "execution-evidence.json"
    write_execution_evidence(evidence_path, evidence)

    with pytest.raises(ValueError, match="dataset identity"):
        resolve_training_execution_inputs(
            dataset_id=DATASET_ID,
            cost=COST,
            evidence_path=evidence_path,
            event_artifact_path=None,
        )
