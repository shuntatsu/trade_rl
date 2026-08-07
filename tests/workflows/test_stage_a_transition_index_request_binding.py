from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_stage_a_execution_producer import (
    _digest,
    _plan,
    _request,
    _result,
)


def test_transition_end_index_cannot_exceed_authorized_evaluation_stop() -> None:
    request = _request(policy=True)
    result = replace(
        _result(policy=True),
        transition_end_indices=(request.evaluation_range.stop + 1,),
    )

    with pytest.raises(ValueError, match="transition end index outside request range"):
        result.validate_against(
            request,
            expected_policy_source_digest=_digest("policy-source"),
            expected_candidate_config_digest=(
                _plan().candidate("candidate-a").candidate_config_digest
            ),
        )
