from __future__ import annotations

from tests.workflows import test_stage_a_sb3_evaluation as sb3_test


def test_sb3_executor_records_transition_end_index_for_each_completed_step() -> None:
    request = sb3_test._request(policy=True)
    dataset = sb3_test._dataset(request)
    environment = sb3_test._FakeEnvironment(request, dataset)
    executor, _, _ = sb3_test._executor(
        request=request,
        dataset=dataset,
        environment=environment,
    )
    candidate_digest = (
        sb3_test._plan_and_manifest()[0]
        .candidate("candidate-a")
        .candidate_config_digest
    )

    result = executor.execute(
        request,
        policy=sb3_test._Policy(),
        policy_source_digest=sb3_test._digest("policy-source"),
        candidate_config_digest=candidate_digest,
    )

    assert result.transition_end_indices == (request.evaluation_range.stop,)
