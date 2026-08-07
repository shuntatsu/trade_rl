from __future__ import annotations

from pathlib import Path

from tests.workflows.test_stage_a_execution_replay import (
    _digest,
    _promotion_bytes,
    _request,
)
from trade_rl.workflows.stage_a_execution_replay import (
    StageAExecutionReplayArtifact,
    build_stage_a_execution_replay_artifact,
)


def test_replay_artifact_persists_transition_end_indices(tmp_path: Path) -> None:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes, _ = _promotion_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )
    transition_end_indices = (request.evaluation_range.stop,)

    artifact = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, 1_100.0),
        transition_end_indices=transition_end_indices,
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
    )

    assert artifact.transition_end_indices == transition_end_indices
    assert (
        StageAExecutionReplayArtifact.from_json_bytes(artifact.raw_bytes)
        .transition_end_indices
        == transition_end_indices
    )
