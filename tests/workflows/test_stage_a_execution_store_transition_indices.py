from __future__ import annotations

from pathlib import Path

from tests.workflows.test_stage_a_execution_store_funding import (
    _digest,
    _plan,
    _request,
    _source_paths,
)
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore


def test_store_publish_persists_transition_end_indices(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_path, evidence_path, funding_path = _source_paths(
        tmp_path / "source",
        request,
        candidate_config_digest=candidate_config_digest,
    )
    transition_end_indices = (request.evaluation_range.stop,)
    store = StageAExecutionPromotionStore(tmp_path / "store")

    published = store.publish(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("obs-0"), _digest("obs-1")),
        equity_curve=(1_000.0, 1_100.0),
        transition_end_indices=transition_end_indices,
        event_artifact_path=event_path,
        execution_evidence_path=evidence_path,
        funding_evidence_path=funding_path,
    )

    assert published.artifact.schema_version == "stage_a_execution_replay_v4"
    assert published.artifact.transition_end_indices == transition_end_indices
    assert store.load(request.digest) == published
