from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.workflows.test_stage_a_policy_producer_orchestration import _policy_producer


def test_producer_persists_executor_transition_end_indices(tmp_path: Path) -> None:
    producer, request, _, executor = _policy_producer(tmp_path)
    transition_end_indices = (request.evaluation_range.stop,)
    executor.transform = lambda result: replace(
        result,
        transition_end_indices=transition_end_indices,
    )

    stored = producer.produce(request)

    assert stored.artifact.schema_version == "stage_a_execution_replay_v4"
    assert stored.artifact.transition_end_indices == transition_end_indices
