from __future__ import annotations

from pathlib import Path

import pytest

from tests.workflows.test_stage_a_execution_replay_funding import (
    _digest,
    _request,
    _source_bytes,
)
from trade_rl.simulation.funding_evidence import build_funding_evidence_artifact
from trade_rl.workflows.stage_a_execution_replay import (
    build_stage_a_execution_replay_artifact,
)


def test_v3_replay_rejects_funding_symbol_count_different_from_terminal_book(
    tmp_path: Path,
) -> None:
    plan, request = _request()
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes = _source_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )
    funding = build_funding_evidence_artifact(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        symbol_count=2,
        boundaries=(),
    )

    with pytest.raises(ValueError, match="funding evidence symbol count mismatch"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(_digest("obs-0"), _digest("obs-1")),
            equity_curve=(1_000.0, 1_100.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
            funding_evidence_bytes=funding.raw_bytes,
        )
