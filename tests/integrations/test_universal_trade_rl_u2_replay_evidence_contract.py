from __future__ import annotations

import math

import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _build_replay_fixture,
    _scope,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayRequest,
    UniversalTradeRLU2ReplayVariant,
)

_CHECKPOINT_DIGEST = content_digest({"fixture": "u2-replay-evidence-contract"})
_DIAGNOSTIC_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def evidence_fixture() -> ReplayIntegrationFixture:
    return _build_replay_fixture()


def _cash_evidence(fixture: ReplayIntegrationFixture):
    scope = _scope(fixture, cell="B")
    evidence = fixture.session.replay(
        UniversalTradeRLU2ReplayRequest(
            scope_digest=scope.digest,
            policy_variant=UniversalTradeRLU2ReplayVariant.CASH,
            evaluation_seed=0,
            paired_candidate_checkpoint_digest=_CHECKPOINT_DIGEST,
        )
    )
    return scope, evidence


def test_u2_replay_evidence_explicitly_binds_normative_scope_boundaries(
    evidence_fixture: ReplayIntegrationFixture,
) -> None:
    scope, evidence = _cash_evidence(evidence_fixture)

    assert evidence.schema_version == "universal_trade_rl_u2_replay_evidence_v1"
    assert evidence.outcome_start_bar_index == scope.outcome_start_bar_index
    assert (
        evidence.outcome_stop_bar_index_exclusive
        == scope.outcome_stop_bar_index_exclusive
    )
    assert evidence.evaluation_start_bar_index == scope.evaluation_start_bar_index
    assert evidence.evaluation_stop_bar_index == scope.evaluation_stop_bar_index

    assert evidence.evaluation_start_bar_index == evidence.outcome_start_bar_index - 1
    assert (
        evidence.evaluation_stop_bar_index
        == evidence.outcome_stop_bar_index_exclusive
    )
    assert evidence.runtime_start_bar_index == evidence.evaluation_start_bar_index
    assert evidence.runtime_end_bar_index == evidence.outcome_stop_bar_index_exclusive - 1
    assert evidence.final_current_bar_index == evidence.runtime_end_bar_index

    payload = evidence.to_payload(include_digest=False)
    for field in (
        "schema_version",
        "outcome_start_bar_index",
        "outcome_stop_bar_index_exclusive",
        "evaluation_start_bar_index",
        "evaluation_stop_bar_index",
    ):
        assert field in payload


def test_u2_replay_evidence_retains_step_aligned_lifecycle_inputs(
    evidence_fixture: ReplayIntegrationFixture,
) -> None:
    _scope_value, evidence = _cash_evidence(evidence_fixture)

    assert len(evidence.step_evidence) == evidence.observed_decision_count
    assert evidence.observed_decision_count == 2880

    previous_action = 0.0
    previous_exposure = 0.0
    target_change_count = 0
    sign_flip_count = 0
    hard_risk_violation_count = 0
    execution_rejection_count = 0
    fill_count = 0

    for step in evidence.step_evidence:
        assert math.isfinite(step.normalized_action)
        assert math.isfinite(step.submitted_target)
        assert math.isfinite(step.executed_target)
        assert math.isfinite(step.risk_projected_target)
        assert math.isfinite(step.realized_exposure)
        assert math.isfinite(step.requested_turnover)
        assert math.isfinite(step.filled_turnover)
        assert math.isfinite(step.requested_notional)
        assert math.isfinite(step.filled_notional)
        assert step.requested_turnover >= 0.0
        assert step.filled_turnover >= 0.0
        assert step.requested_notional >= 0.0
        assert step.filled_notional >= 0.0
        assert step.fill_count >= 0
        assert step.rejected_count >= 0
        assert len(step.rejection_reasons) == step.rejected_count
        assert all(reason for reason in step.rejection_reasons)

        hard_limit = max(
            step.max_abs_weight * step.risk_scale,
            step.max_gross * step.risk_scale,
        )
        assert hard_limit >= 0.0
        independently_violates = (
            abs(step.risk_projected_target)
            > step.max_abs_weight * step.risk_scale + step.fail_closed_tolerance
            or abs(step.risk_projected_target)
            > step.max_gross * step.risk_scale + step.fail_closed_tolerance
            or (
                step.risk_scale == 0.0
                and abs(step.risk_projected_target) > step.fail_closed_tolerance
            )
        )
        assert step.hard_risk_violation is independently_violates

        if abs(step.normalized_action - previous_action) > _DIAGNOSTIC_TOLERANCE:
            target_change_count += 1
        if (
            abs(previous_exposure) > _DIAGNOSTIC_TOLERANCE
            and abs(step.realized_exposure) > _DIAGNOSTIC_TOLERANCE
            and previous_exposure * step.realized_exposure < 0.0
        ):
            sign_flip_count += 1
            assert step.transition_class == "flip"

        previous_action = step.normalized_action
        previous_exposure = step.realized_exposure
        hard_risk_violation_count += int(step.hard_risk_violation)
        execution_rejection_count += step.rejected_count
        fill_count += step.fill_count

    assert evidence.target_change_count == target_change_count
    assert evidence.sign_flip_count == sign_flip_count
    assert evidence.hard_risk_violation_count == hard_risk_violation_count
    assert evidence.execution_rejection_count == execution_rejection_count
    assert evidence.fill_count == fill_count

    payload = evidence.to_payload(include_digest=False)
    assert len(payload["step_evidence"]) == evidence.observed_decision_count
    assert payload["target_change_count"] == target_change_count
    assert payload["sign_flip_count"] == sign_flip_count
    assert payload["hard_risk_violation_count"] == hard_risk_violation_count
    assert payload["execution_rejection_count"] == execution_rejection_count
    assert payload["fill_count"] == fill_count
