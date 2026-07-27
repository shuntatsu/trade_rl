from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    CausalScenarioC3Config,
    PersistedScenarioDecision,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    load_c3_decision_artifact,
    write_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_prediction import (
    create_c3_prediction_evidence,
)
from trade_rl.evaluation.causal_scenario_c3_report import build_c3_fold_report
from trade_rl.evaluation.causal_scenario_c3_runner import run_c3_query_comparison

_DAY_NS = 86_400_000_000_000


def _sha(char: str) -> str:
    return char * 64


def _identity(*, query_index: int, query_timestamp_ns: int) -> C3ReplayIdentity:
    return C3ReplayIdentity(
        dataset_id=_sha("a"),
        fold_digest=_sha("b"),
        environment_digest=_sha("c"),
        action_spec_digest=_sha("d"),
        observation_digest=_sha("e"),
        execution_policy_digest=_sha("f"),
        risk_digest=_sha("1"),
        initial_state_digest=_sha("2"),
        query_index=query_index,
        query_timestamp_ns=query_timestamp_ns,
        realized_stop_index=query_index + 96,
        aum=100_000.0,
    )


def _decision(
    *, query_index: int, query_timestamp_ns: int
) -> PersistedScenarioDecision:
    raw = np.asarray([[0.0], [1.0], [-1.0]], dtype=np.float64)
    projected = raw * 0.25
    candidate_digests = tuple(
        content_digest(
            {
                "candidate_index": index,
                "query_index": query_index,
                "schema_version": "c3_identity_candidate_v1",
            }
        )
        for index in range(3)
    )
    score = np.asarray([0.0, 0.03, -0.02], dtype=np.float64)
    regret = score.max() - score
    identity = _identity(
        query_index=query_index,
        query_timestamp_ns=query_timestamp_ns,
    )
    payload = {
        "action_spec_digest": identity.action_spec_digest,
        "candidate_digests": candidate_digests,
        "candidate_generator_digest": _sha("6"),
        "created_before_realized_replay": True,
        "dataset_id": identity.dataset_id,
        "environment_digest": identity.environment_digest,
        "execution_policy_digest": identity.execution_policy_digest,
        "fold_digest": identity.fold_digest,
        "observation_digest": identity.observation_digest,
        "projected_targets": projected.tolist(),
        "query_index": identity.query_index,
        "query_timestamp_ns": identity.query_timestamp_ns,
        "raw_candidate_actions": raw.tolist(),
        "realized_stop_index": identity.realized_stop_index,
        "regret": regret.tolist(),
        "risk_digest": identity.risk_digest,
        "scenario_library_digest": _sha("3"),
        "scenario_set_digest": _sha("4"),
        "schema_version": "causal_scenario_c3_decision_v1",
        "score": score.tolist(),
        "selected_candidate_digest": candidate_digests[1],
        "selected_candidate_index": 1,
        "starting_equity": identity.aum,
        "state_snapshot_digest": identity.initial_state_digest,
        "tie_candidate_indices": (1,),
        "value_result_digest": _sha("5"),
        "zero_candidate_index": 0,
    }
    return PersistedScenarioDecision(
        dataset_id=identity.dataset_id,
        fold_digest=identity.fold_digest,
        query_index=identity.query_index,
        query_timestamp_ns=identity.query_timestamp_ns,
        state_snapshot_digest=identity.initial_state_digest,
        observation_digest=identity.observation_digest,
        environment_digest=identity.environment_digest,
        action_spec_digest=identity.action_spec_digest,
        execution_policy_digest=identity.execution_policy_digest,
        risk_digest=identity.risk_digest,
        starting_equity=identity.aum,
        realized_stop_index=identity.realized_stop_index,
        scenario_library_digest=_sha("3"),
        scenario_set_digest=_sha("4"),
        candidate_generator_digest=_sha("6"),
        value_result_digest=_sha("5"),
        candidate_digests=candidate_digests,
        raw_candidate_actions=raw,
        projected_targets=projected,
        score=score,
        regret=regret,
        selected_candidate_index=1,
        zero_candidate_index=0,
        tie_candidate_indices=(1,),
        selected_candidate_digest=candidate_digests[1],
        created_before_realized_replay=True,
        decision_digest=content_digest(payload),
    )


def _prediction(decision: PersistedScenarioDecision):
    return create_c3_prediction_evidence(
        result_digest=decision.value_result_digest,
        scenario_library_digest=decision.scenario_library_digest,
        scenario_set_digest=decision.scenario_set_digest,
        candidate_digests=decision.candidate_digests,
        predicted_score=decision.score,
        predicted_mean_advantage=decision.score,
        predicted_loss_cvar=np.asarray([0.0, 0.01, 0.02]),
        predicted_expected_turnover=np.asarray([0.0, 0.25, 0.25]),
        scenario_anchor_indices=np.arange(64, dtype=np.int64),
        scenario_distances=np.linspace(0.0, 1.0, 64),
    )


def _outcome(kind: str, value: float) -> RealizedPolicyOutcome:
    payload = {
        "borrow_paid": 0.0,
        "cancel_replace_events": 0,
        "fees": 0.0001,
        "fill_count": 1,
        "fill_ratio": 1.0,
        "filled_turnover": 0.1,
        "funding_paid": 0.0,
        "gross_log_return": value,
        "impact_cost": 0.0001,
        "max_drawdown": 0.05,
        "pending_order_events": 0,
        "policy_kind": kind,
        "schema_version": "causal_scenario_c3_realized_outcome_v1",
        "spread_cost": 0.0001,
        "terminal_equity": 100_000.0 * math.exp(value),
        "termination_reason": "horizon",
    }
    return RealizedPolicyOutcome(
        policy_kind=kind,
        gross_log_return=value,
        filled_turnover=0.1,
        fees=0.0001,
        spread_cost=0.0001,
        impact_cost=0.0001,
        funding_paid=0.0,
        borrow_paid=0.0,
        fill_ratio=1.0,
        fill_count=1,
        pending_order_events=0,
        cancel_replace_events=0,
        max_drawdown=0.05,
        terminal_equity=100_000.0 * math.exp(value),
        termination_reason="horizon",
        outcome_digest=content_digest(payload),
    )


class CloneReplay:
    def __init__(
        self,
        identity: C3ReplayIdentity,
        starts: list[int],
        *,
        local_state: int = 0,
    ) -> None:
        self.identity = identity
        self.starts = starts
        self.local_state = local_state

    def clone_for_replay(self) -> CloneReplay:
        return CloneReplay(self.identity, self.starts)

    def run(
        self,
        raw_residual: np.ndarray,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
        policy_kind: str,
    ) -> RealizedPolicyOutcome:
        assert horizon_decisions == 96
        assert zero_residual_after_first is True
        self.starts.append(self.local_state)
        self.local_state += 1
        return _outcome(policy_kind, 0.01 + 0.02 * float(raw_residual[0]))


def _comparison(
    tmp_path: Path,
    *,
    query_index: int,
    query_timestamp_ns: int,
    starts: list[int] | None = None,
):
    created = _decision(
        query_index=query_index,
        query_timestamp_ns=query_timestamp_ns,
    )
    root = tmp_path / created.decision_digest
    write_c3_decision_artifact(root, created)
    replay = CloneReplay(created.replay_identity, starts if starts is not None else [])
    return run_c3_query_comparison(
        load_c3_decision_artifact(root),
        replay=replay,
        ppo_mean_action=np.asarray([0.5]),
        config=CausalScenarioC3Config(random_comparator_count=2),
        prediction_evidence=_prediction(created),
        execution_scenario="nominal",
    )


def test_identity_mismatch_aborts_before_any_replay(tmp_path: Path) -> None:
    created = _decision(query_index=10_000, query_timestamp_ns=10 * _DAY_NS)
    root = tmp_path / "decision"
    write_c3_decision_artifact(root, created)
    starts: list[int] = []
    mismatched = replace(created.replay_identity, aum=200_000.0)
    with pytest.raises(ValueError, match="replay identity"):
        run_c3_query_comparison(
            load_c3_decision_artifact(root),
            replay=CloneReplay(mismatched, starts),
            ppo_mean_action=np.asarray([0.5]),
            config=CausalScenarioC3Config(random_comparator_count=2),
            prediction_evidence=_prediction(created),
        )
    assert starts == []


def test_prediction_mismatch_aborts_before_any_replay(tmp_path: Path) -> None:
    created = _decision(query_index=10_000, query_timestamp_ns=10 * _DAY_NS)
    root = tmp_path / "decision"
    write_c3_decision_artifact(root, created)
    starts: list[int] = []
    bad = create_c3_prediction_evidence(
        result_digest=_sha("9"),
        scenario_library_digest=created.scenario_library_digest,
        scenario_set_digest=created.scenario_set_digest,
        candidate_digests=created.candidate_digests,
        predicted_score=created.score,
        predicted_mean_advantage=created.score,
        predicted_loss_cvar=np.asarray([0.0, 0.01, 0.02]),
        predicted_expected_turnover=np.asarray([0.0, 0.25, 0.25]),
        scenario_anchor_indices=np.arange(64, dtype=np.int64),
        scenario_distances=np.linspace(0.0, 1.0, 64),
    )
    with pytest.raises(ValueError, match="prediction result"):
        run_c3_query_comparison(
            load_c3_decision_artifact(root),
            replay=CloneReplay(created.replay_identity, starts),
            ppo_mean_action=np.asarray([0.5]),
            config=CausalScenarioC3Config(random_comparator_count=2),
            prediction_evidence=bad,
        )
    assert starts == []


def test_every_policy_uses_a_fresh_replay_clone(tmp_path: Path) -> None:
    starts: list[int] = []
    result = _comparison(
        tmp_path,
        query_index=10_000,
        query_timestamp_ns=10 * _DAY_NS,
        starts=starts,
    )
    assert (
        result.replay_identity_digest
        == _identity(
            query_index=10_000,
            query_timestamp_ns=10 * _DAY_NS,
        ).digest
    )
    assert result.prediction_result_digest == _sha("5")
    assert starts
    assert set(starts) == {0}


def test_fold_report_uses_daily_paired_log_growth(tmp_path: Path) -> None:
    comparisons = (
        _comparison(
            tmp_path / "a",
            query_index=10_000,
            query_timestamp_ns=10 * _DAY_NS + 1,
        ),
        _comparison(
            tmp_path / "b",
            query_index=10_100,
            query_timestamp_ns=10 * _DAY_NS + 2,
        ),
        _comparison(
            tmp_path / "c",
            query_index=10_200,
            query_timestamp_ns=11 * _DAY_NS + 1,
        ),
    )
    report = build_c3_fold_report(
        fold_id="fold-0",
        selection_days=2,
        comparisons=comparisons,
        required_adverse_passed=True,
        required_adverse_evidence_digest="a" * 64,
    )
    raw_uplift = np.asarray(
        [
            item.scenario_oracle.gross_log_return - item.trend.gross_log_return
            for item in comparisons
        ]
    )
    assert report.effective_days == 2
    np.testing.assert_array_equal(report.day_indices, np.asarray([10, 11]))
    np.testing.assert_allclose(
        report.uplift,
        np.asarray([raw_uplift[0] + raw_uplift[1], raw_uplift[2]]),
    )
