from __future__ import annotations

from tests.workflows.test_stage_a_sb3_evaluation import (
    _FakeEnvironment,
    _Policy,
    _dataset,
    _digest,
    _executor,
    _plan_and_manifest,
    _request,
)


def test_sb3_executor_records_transition_end_index_for_each_completed_step() -> None:
    request = _request(policy=True)
    dataset = _dataset(request)
    environment = _FakeEnvironment(request, dataset)
    executor, _, _ = _executor(
        request=request,
        dataset=dataset,
        environment=environment,
    )
    candidate_digest = (
        _plan_and_manifest()[0].candidate("candidate-a").candidate_config_digest
    )

    result = executor.execute(
        request,
        policy=_Policy(),
        policy_source_digest=_digest("policy-source"),
        candidate_config_digest=candidate_digest,
    )

    assert result.transition_end_indices == (request.evaluation_range.stop,)
