from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_stage_a_execution_replay import _digest, _request
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_execution_replay import (
    STAGE_A_EXECUTION_REPLAY_SCHEMA,
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


def _funding_boundary(index: int, *, timestamp_ns: int) -> FundingBoundaryEvidence:
    return FundingBoundaryEvidence(
        processing_index=index,
        timestamp_ns=timestamp_ns,
        funding_due=(True,),
        signed_quantities=(1.0,),
        mark_prices=(100.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-0.1,
        equity_before_funding=1_000.0,
        equity_after_funding=999.9,
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


def test_historical_intervals_require_transition_bound_replay_v4() -> None:
    replay = _two_transition_replay()
    legacy_replay = replace(
        replay,
        transition_end_indices=(),
        schema_version=STAGE_A_EXECUTION_REPLAY_SCHEMA,
        digest="",
    )

    with pytest.raises(ValueError, match="requires replay v4"):
        build_stage_a_historical_interval_evidence(legacy_replay)


def test_historical_intervals_assign_funding_boundaries_without_overlap() -> None:
    replay = _two_transition_replay()
    start = replay.cell_identity.evaluation_range.start
    shared = replay.transition_end_indices[0]
    stop = replay.cell_identity.evaluation_range.stop
    funding = (
        _funding_boundary(start, timestamp_ns=1),
        _funding_boundary(shared, timestamp_ns=2),
        _funding_boundary(stop, timestamp_ns=3),
    )

    intervals = build_stage_a_historical_interval_evidence(
        replay,
        funding_evidence=funding,
    )

    assert [
        tuple(boundary.processing_index for boundary in item.funding_boundaries)
        for item in intervals
    ] == [
        (start, shared),
        (stop,),
    ]
