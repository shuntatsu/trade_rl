from __future__ import annotations

from tests.workflows.test_stage_a_execution_replay import _digest, _request
from trade_rl.workflows.stage_a_execution_replay import (
    STAGE_A_EXECUTION_REPLAY_SCHEMA_V4,
    StageAExecutionCellIdentity,
    StageAExecutionReplayArtifact,
)
from trade_rl.workflows.stage_a_historical_interval_evidence import (
    build_stage_a_historical_interval_evidence,
)


def _two_transition_replay() -> StageAExecutionReplayArtifact:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    start = request.evaluation_range.start
    stop = request.evaluation_range.stop
    assert stop >= start + 2
    return StageAExecutionReplayArtifact(
        cell_identity=StageAExecutionCellIdentity.from_request(
            request,
            candidate_config_digest=candidate_config_digest,
        ),
        actions=((0.25,), (0.5,)),
        observation_digests=(
            _digest("observation-0"),
            _digest("observation-1"),
            _digest("observation-2"),
        ),
        equity_curve=(1_000.0, 1_010.0, 990.0),
        transition_end_indices=(start + 1, stop),
        event_artifact_digest=_digest("events"),
        event_artifact_size_bytes=1,
        execution_evidence_digest=_digest("evidence"),
        execution_evidence_sha256=_digest("evidence-bytes"),
        execution_evidence_size_bytes=1,
        schema_version=STAGE_A_EXECUTION_REPLAY_SCHEMA_V4,
    )


def test_historical_intervals_bind_step_end_indices_to_equity_curve() -> None:
    replay = _two_transition_replay()
    start = replay.cell_identity.evaluation_range.start
    stop = replay.cell_identity.evaluation_range.stop

    intervals = build_stage_a_historical_interval_evidence(replay)

    assert [
        (
            item.sequence,
            item.start_index,
            item.end_index,
            item.equity_before,
            item.equity_after,
        )
        for item in intervals
    ] == [
        (1, start, start + 1, 1_000.0, 1_010.0),
        (2, start + 1, stop, 1_010.0, 990.0),
    ]
