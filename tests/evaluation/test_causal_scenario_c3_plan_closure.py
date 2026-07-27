from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.bootstrap import moving_block_mean_test
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
from trade_rl.evaluation.causal_scenario_c3_report import (
    build_c3_aggregate_report,
    build_c3_fold_report,
)
from trade_rl.evaluation.causal_scenario_c3_runner import run_c3_query_comparison


def _sha(char: str) -> str:
    return char * 64


def _decision() -> PersistedScenarioDecision:
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    projected = actions * 0.25
    candidate_digests = tuple(
        content_digest(
            {
                "candidate": index,
                "raw_action": action.tolist(),
                "schema_version": "c3_plan_closure_candidate_v1",
            }
        )
        for index, action in enumerate(actions)
    )
    score = np.asarray([0.0, 0.04, -0.03, 0.01], dtype=np.float64)
    regret = score.max() - score
    payload = {
        "action_spec_digest": _sha("d"),
        "candidate_digests": candidate_digests,
        "candidate_generator_digest": _sha("6"),
        "created_before_realized_replay": True,
        "dataset_id": _sha("a"),
        "environment_digest": _sha("c"),
        "execution_policy_digest": _sha("f"),
        "fold_digest": _sha("b"),
        "observation_digest": _sha("e"),
        "projected_targets": projected.tolist(),
        "query_index": 10_000,
        "query_timestamp_ns": 1_800_000_000_000_000_000,
        "raw_candidate_actions": actions.tolist(),
        "realized_stop_index": 10_096,
        "regret": regret.tolist(),
        "risk_digest": _sha("1"),
        "scenario_library_digest": _sha("3"),
        "scenario_set_digest": _sha("4"),
        "schema_version": "causal_scenario_c3_decision_v1",
        "score": score.tolist(),
        "selected_candidate_digest": candidate_digests[1],
        "selected_candidate_index": 1,
        "starting_equity": 100_000.0,
        "state_snapshot_digest": _sha("2"),
        "tie_candidate_indices": (1,),
        "value_result_digest": _sha("5"),
        "zero_candidate_index": 0,
    }
    return PersistedScenarioDecision(
        dataset_id=_sha("a"),
        fold_digest=_sha("b"),
        query_index=10_000,
        query_timestamp_ns=1_800_000_000_000_000_000,
        state_snapshot_digest=_sha("2"),
        observation_digest=_sha("e"),
        environment_digest=_sha("c"),
        action_spec_digest=_sha("d"),
        execution_policy_digest=_sha("f"),
        risk_digest=_sha("1"),
        starting_equity=100_000.0,
        realized_stop_index=10_096,
        scenario_library_digest=_sha("3"),
        scenario_set_digest=_sha("4"),
        candidate_generator_digest=_sha("6"),
        value_result_digest=_sha("5"),
        candidate_digests=candidate_digests,
        raw_candidate_actions=actions,
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


def _outcome(policy_kind: str, log_return: float) -> RealizedPolicyOutcome:
    payload = {
        "borrow_paid": 0.0,
        "fees": 0.0001,
        "fill_count": 1,
        "filled_turnover": 0.1,
        "funding_paid": 0.0,
        "gross_log_return": log_return,
        "impact_cost": 0.0001,
        "max_drawdown": 0.05,
        "pending_order_events": 0,
        "policy_kind": policy_kind,
        "schema_version": "causal_scenario_c3_realized_outcome_v1",
        "spread_cost": 0.0001,
        "terminal_equity": 100_000.0 * math.exp(log_return),
        "termination_reason": "horizon",
    }
    return RealizedPolicyOutcome(
        policy_kind=policy_kind,
        gross_log_return=log_return,
        filled_turnover=0.1,
        fees=0.0001,
        spread_cost=0.0001,
        impact_cost=0.0001,
        funding_paid=0.0,
        borrow_paid=0.0,
        fill_count=1,
        pending_order_events=0,
        max_drawdown=0.05,
        terminal_equity=100_000.0 * math.exp(log_return),
        termination_reason="horizon",
        outcome_digest=content_digest(payload),
    )


class _Replay:
    def __init__(self, identity: C3ReplayIdentity) -> None:
        self.identity = identity

    def clone_for_replay(self) -> _Replay:
        return _Replay(self.identity)

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
        value = 0.01 + 0.02 * float(raw_residual[0]) + 0.005 * float(
            raw_residual[1]
        )
        return _outcome(policy_kind, value)


def _comparison(tmp_path: Path, *, random_count: int = 8):
    created = _decision()
    root = tmp_path / "decision"
    write_c3_decision_artifact(root, created)
    return run_c3_query_comparison(
        load_c3_decision_artifact(root),
        replay=_Replay(created.replay_identity),
        ppo_mean_action=np.asarray([0.5, 0.0, 0.0]),
        config=CausalScenarioC3Config(random_comparator_count=random_count),
    )


def test_random_comparator_count_materializes_complete_evidence(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path, random_count=8)
    assert len(comparison.random_candidate_indices) == 8
    assert len(comparison.random_candidate_outcomes) == 8
    assert comparison.random_realized_regrets.shape == (8,)
    assert comparison.random_realized_regrets.flags.writeable is False
    assert comparison.random_candidate.outcome_digest == (
        comparison.random_candidate_outcomes[0].outcome_digest
    )
    assert comparison.random_realized_regret == float(
        comparison.random_realized_regrets.mean()
    )
    for index, outcome in zip(
        comparison.random_candidate_indices,
        comparison.random_candidate_outcomes,
        strict=True,
    ):
        assert outcome.outcome_digest == comparison.candidate_outcomes[index].outcome_digest


def test_explicit_moving_block_size_is_preserved() -> None:
    result = moving_block_mean_test(
        tuple(float(value) for value in np.linspace(-0.02, 0.03, 60)),
        n_bootstrap=128,
        seed=7,
        block_size=7,
    )
    assert result.block_size == 7


def test_c3_aggregate_report_binds_fixed_block_days(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path, random_count=8)
    folds = tuple(
        build_c3_fold_report(
            fold_id=f"fold-{index}",
            selection_days=30,
            comparisons=(comparison,),
            required_adverse_passed=True,
        )
        for index in range(6)
    )
    report = build_c3_aggregate_report(
        folds,
        bootstrap_resamples=128,
        bootstrap_block_days=7,
    )
    assert report.bootstrap_block_days == 7
