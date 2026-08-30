from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.rollout_evaluation import ActionPathLifecycleTrace


def _trace() -> ActionPathLifecycleTrace:
    return ActionPathLifecycleTrace(
        submitted_targets=np.asarray([[0.1], [0.0]], dtype=np.float64),
        execution_intent_targets=np.asarray([[0.0], [0.1]], dtype=np.float64),
        final_risk_targets=np.asarray([[0.0], [0.1]], dtype=np.float64),
        applied_risk_scales=np.asarray([1.0, 0.8], dtype=np.float64),
        hard_risk_evidence_available=np.asarray([True, True], dtype=np.bool_),
        hard_risk_violations=np.asarray([False, False], dtype=np.bool_),
        risk_reasons=((), ("drawdown_deleveraging",)),
        transition_classes=("flat", "entry"),
        flatten_initiators=("not_applicable", "not_applicable"),
    )


def test_lifecycle_trace_round_trip_preserves_digest_and_arrays() -> None:
    trace = _trace()
    decoded = ActionPathLifecycleTrace.from_payload(trace.to_payload())

    assert decoded.digest == trace.digest
    np.testing.assert_array_equal(decoded.submitted_targets, trace.submitted_targets)
    np.testing.assert_array_equal(
        decoded.execution_intent_targets, trace.execution_intent_targets
    )
    np.testing.assert_array_equal(decoded.final_risk_targets, trace.final_risk_targets)
    assert decoded.risk_reasons == trace.risk_reasons


def test_lifecycle_trace_rejects_string_boolean_tampering() -> None:
    payload = _trace().to_payload()
    payload["hard_risk_evidence_available"] = ["true", "true"]

    with pytest.raises(ValueError, match="hard_risk_evidence_available.*boolean"):
        ActionPathLifecycleTrace.from_payload(payload)
