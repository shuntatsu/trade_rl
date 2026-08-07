from __future__ import annotations

from pathlib import Path

from tests.workflows.test_stage_a_policy_producer_orchestration import _policy_producer
from trade_rl.simulation.funding_evidence import load_funding_evidence_artifact_bytes


def test_producer_publishes_funding_evidence_as_v3_sidecar(tmp_path: Path) -> None:
    producer, request, _, _ = _policy_producer(tmp_path)

    stored = producer.produce(request)

    assert stored.artifact.schema_version == "stage_a_execution_replay_v3"
    assert stored.funding_path is not None
    funding = load_funding_evidence_artifact_bytes(stored.funding_path.read_bytes())
    assert funding.dataset_id == request.dataset_id
    assert funding.execution_policy_digest == request.execution_identity
    assert funding.symbol_count == 1
    assert funding.boundaries == ()
