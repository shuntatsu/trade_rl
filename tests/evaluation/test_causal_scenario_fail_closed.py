from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_values import (
    CausalQuerySnapshot,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
)


def sha(char: str) -> str:
    return char * 64


def query() -> CausalQuerySnapshot:
    return CausalQuerySnapshot(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        train_start=0,
        train_stop=1000,
        query_index=1001,
        query_timestamp_ns=1,
        source_commit="c" * 40,
        query_digest=sha("1"),
        state_snapshot_digest=sha("2"),
        observation_digest=sha("3"),
        environment_digest=sha("4"),
        action_spec_digest=sha("5"),
        execution_policy_digest=sha("6"),
        risk_digest=sha("7"),
        trend_digest=sha("8"),
        starting_equity=100.0,
        baseline_target=np.asarray([0.0]),
    )


def scenarios() -> CausalScenarioSet:
    count = 64
    return CausalScenarioSet(
        scenario_ids=tuple(f"s-{i}" for i in range(count)),
        probabilities=np.full(count, 1.0 / count),
        anchor_indices=np.arange(count, dtype=np.int64),
        distances=np.arange(count, dtype=np.float64),
        query_condition=np.asarray([0.0]),
        anchor_conditions=np.zeros((count, 1)),
        library_digest=sha("9"),
    )


def projection(raw: np.ndarray) -> ProjectedResidualCandidate:
    target = 0.25 * raw
    execution = content_digest({"target": target.tolist()})
    digest = content_digest(
        {
            "execution_intent_digest": execution,
            "projected_target": target.tolist(),
            "schema_version": "projected_residual_candidate_v1",
        }
    )
    return ProjectedResidualCandidate(
        raw_action=raw,
        projected_target=target,
        execution_intent_digest=execution,
        candidate_digest=digest,
        expected_turnover_hint=float(np.abs(raw).sum()),
        is_zero=bool(np.all(raw == 0.0)),
    )


def valid_evidence() -> ScenarioRolloutEvidence:
    payload = {
        "feasible": True,
        "fill_ratio": 1.0,
        "filled_turnover": 0.0,
        "interval_cost": 0.0,
        "reported_log_return": 0.0,
        "schema_version": "scenario_rollout_evidence_v1",
        "terminal_equity": 100.0,
        "termination_reason": "horizon",
    }
    return ScenarioRolloutEvidence(
        terminal_equity=100.0,
        reported_log_return=0.0,
        filled_turnover=0.0,
        interval_cost=0.0,
        fill_ratio=1.0,
        feasible=True,
        termination_reason="horizon",
        evidence_digest=content_digest(payload),
    )


class Rollout:
    def __init__(self, evidence: ScenarioRolloutEvidence) -> None:
        self.evidence = evidence
        self.used = False

    def run(self, candidate, *, horizon_decisions, zero_residual_after_first):
        assert horizon_decisions == 96
        assert zero_residual_after_first
        if self.used:
            raise AssertionError("rollout object was reused")
        self.used = True
        return self.evidence


class Factory:
    def __init__(self, *, evidence: ScenarioRolloutEvidence | None = None) -> None:
        self.evidence = valid_evidence() if evidence is None else evidence
        self.created = 0

    def project_candidate(self, query, raw_action):
        return projection(np.asarray(raw_action, dtype=np.float64))

    def create_rollout(self, query, scenario_index, scenario_id):
        self.created += 1
        return Rollout(self.evidence)


def test_fresh_rollout_is_created_for_every_pair() -> None:
    factory = Factory()
    result = evaluate_causal_scenario_actions(
        query=query(),
        scenarios=scenarios(),
        config=CausalScenarioEvaluatorConfig(action_dimension=1),
        rollout_factory=factory,
    )
    assert factory.created == 64 * len(result.candidate_digests)


def test_rejects_reported_log_return_mismatch() -> None:
    payload = {
        "feasible": True,
        "fill_ratio": 1.0,
        "filled_turnover": 0.0,
        "interval_cost": 0.0,
        "reported_log_return": 0.1,
        "schema_version": "scenario_rollout_evidence_v1",
        "terminal_equity": 100.0,
        "termination_reason": "horizon",
    }
    bad = ScenarioRolloutEvidence(
        terminal_equity=100.0,
        reported_log_return=0.1,
        filled_turnover=0.0,
        interval_cost=0.0,
        fill_ratio=1.0,
        feasible=True,
        termination_reason="horizon",
        evidence_digest=content_digest(payload),
    )
    with pytest.raises(ValueError, match="log return"):
        evaluate_causal_scenario_actions(
            query=query(),
            scenarios=scenarios(),
            config=CausalScenarioEvaluatorConfig(action_dimension=1),
            rollout_factory=Factory(evidence=bad),
        )


def test_rollout_evidence_rejects_bad_fields_and_digest() -> None:
    with pytest.raises(ValueError, match="terminal_equity"):
        ScenarioRolloutEvidence(
            terminal_equity=0.0,
            reported_log_return=0.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=True,
            termination_reason="horizon",
            evidence_digest=sha("a"),
        )
    with pytest.raises(ValueError, match="non-negative"):
        ScenarioRolloutEvidence(
            terminal_equity=1.0,
            reported_log_return=0.0,
            filled_turnover=-1.0,
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=True,
            termination_reason="horizon",
            evidence_digest=sha("a"),
        )
    with pytest.raises(ValueError, match="fill_ratio"):
        ScenarioRolloutEvidence(
            terminal_equity=1.0,
            reported_log_return=0.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            fill_ratio=2.0,
            feasible=True,
            termination_reason="horizon",
            evidence_digest=sha("a"),
        )
    with pytest.raises(ValueError, match="termination_reason"):
        ScenarioRolloutEvidence(
            terminal_equity=1.0,
            reported_log_return=0.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=True,
            termination_reason="",
            evidence_digest=sha("a"),
        )
    with pytest.raises(ValueError, match="evidence_digest"):
        ScenarioRolloutEvidence(
            terminal_equity=1.0,
            reported_log_return=0.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=True,
            termination_reason="horizon",
            evidence_digest=sha("a"),
        )


def test_rejects_infeasible_rollout() -> None:
    payload = {
        "feasible": False,
        "fill_ratio": 1.0,
        "filled_turnover": 0.0,
        "interval_cost": 0.0,
        "reported_log_return": 0.0,
        "schema_version": "scenario_rollout_evidence_v1",
        "terminal_equity": 100.0,
        "termination_reason": "blocked",
    }
    bad = ScenarioRolloutEvidence(
        terminal_equity=100.0,
        reported_log_return=0.0,
        filled_turnover=0.0,
        interval_cost=0.0,
        fill_ratio=1.0,
        feasible=False,
        termination_reason="blocked",
        evidence_digest=content_digest(payload),
    )
    with pytest.raises(ValueError, match="feasible"):
        evaluate_causal_scenario_actions(
            query=query(),
            scenarios=scenarios(),
            config=CausalScenarioEvaluatorConfig(action_dimension=1),
            rollout_factory=Factory(evidence=bad),
        )


def test_projected_candidate_rejects_malformed_identity() -> None:
    with pytest.raises(ValueError, match="candidate_digest"):
        ProjectedResidualCandidate(
            raw_action=np.asarray([0.0]),
            projected_target=np.asarray([0.0]),
            execution_intent_digest=sha("a"),
            candidate_digest=sha("b"),
            expected_turnover_hint=0.0,
            is_zero=True,
        )
    with pytest.raises(ValueError, match="is_zero"):
        target = np.asarray([0.25])
        execution = sha("a")
        ProjectedResidualCandidate(
            raw_action=np.asarray([1.0]),
            projected_target=target,
            execution_intent_digest=execution,
            candidate_digest=content_digest(
                {
                    "execution_intent_digest": execution,
                    "projected_target": target.tolist(),
                    "schema_version": "projected_residual_candidate_v1",
                }
            ),
            expected_turnover_hint=1.0,
            is_zero=True,
        )


def test_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        CausalScenarioEvaluatorConfig(action_dimension=1, cvar_penalty=math.inf)
