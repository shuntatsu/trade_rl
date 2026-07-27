from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.cli.extended import main
from trade_rl.evaluation.causal_scenario_c3_artifact import (
    load_c3_aggregate_report_artifact,
    load_phase_a_gate_artifact,
    write_c3_aggregate_report_artifact,
    write_phase_a_gate_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    CausalScenarioC3Config,
    PerfectInformationComparison,
    PersistedScenarioDecision,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    load_c3_decision_artifact,
    write_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_gate import evaluate_phase_a_entry_gate
from trade_rl.evaluation.causal_scenario_c3_perfect_information import (
    PerfectInformationCompatibilityEvidence,
    evaluate_perfect_information_compatibility,
)
from trade_rl.evaluation.causal_scenario_c3_prediction import (
    create_c3_prediction_evidence,
)
from trade_rl.evaluation.causal_scenario_c3_report import (
    build_c3_aggregate_report,
    build_c3_fold_report,
)
from trade_rl.evaluation.causal_scenario_c3_runner import run_c3_query_comparison
from trade_rl.workflows.causal_scenario.c3 import (
    C3BatchQuery,
    execute_c3_batch,
)

_DAY_NS = 86_400_000_000_000


def sha(char: str) -> str:
    return char * 64


def _decision(
    *, query_index: int = 10_000, fold_digest: str | None = None
) -> PersistedScenarioDecision:
    resolved_fold_digest = sha("b") if fold_digest is None else fold_digest
    query_timestamp_ns = (100 + query_index) * _DAY_NS
    raw = np.asarray([[0.0], [1.0], [-1.0]], dtype=np.float64)
    projected = raw * 0.25
    candidate_digests = tuple(
        content_digest(
            {
                "index": index,
                "query_index": query_index,
                "schema_version": "c3_integration_candidate_v1",
            }
        )
        for index in range(3)
    )
    score = np.asarray([0.0, 0.02, -0.02], dtype=np.float64)
    regret = score.max() - score
    payload = {
        "action_spec_digest": sha("d"),
        "candidate_digests": candidate_digests,
        "candidate_generator_digest": sha("6"),
        "created_before_realized_replay": True,
        "dataset_id": sha("a"),
        "environment_digest": sha("c"),
        "execution_policy_digest": sha("f"),
        "fold_digest": resolved_fold_digest,
        "observation_digest": sha("e"),
        "projected_targets": projected.tolist(),
        "query_index": query_index,
        "query_timestamp_ns": query_timestamp_ns,
        "raw_candidate_actions": raw.tolist(),
        "realized_stop_index": query_index + 96,
        "regret": regret.tolist(),
        "risk_digest": sha("1"),
        "scenario_library_digest": sha("3"),
        "scenario_set_digest": sha("4"),
        "schema_version": "causal_scenario_c3_decision_v1",
        "score": score.tolist(),
        "selected_candidate_digest": candidate_digests[1],
        "selected_candidate_index": 1,
        "starting_equity": 100_000.0,
        "state_snapshot_digest": sha("2"),
        "tie_candidate_indices": (1,),
        "value_result_digest": sha("5"),
        "zero_candidate_index": 0,
    }
    return PersistedScenarioDecision(
        dataset_id=sha("a"),
        fold_digest=resolved_fold_digest,
        query_index=query_index,
        query_timestamp_ns=query_timestamp_ns,
        state_snapshot_digest=sha("2"),
        observation_digest=sha("e"),
        environment_digest=sha("c"),
        action_spec_digest=sha("d"),
        execution_policy_digest=sha("f"),
        risk_digest=sha("1"),
        starting_equity=100_000.0,
        realized_stop_index=query_index + 96,
        scenario_library_digest=sha("3"),
        scenario_set_digest=sha("4"),
        candidate_generator_digest=sha("6"),
        value_result_digest=sha("5"),
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


def _prediction(created: PersistedScenarioDecision):
    return create_c3_prediction_evidence(
        result_digest=created.value_result_digest,
        scenario_library_digest=created.scenario_library_digest,
        scenario_set_digest=created.scenario_set_digest,
        candidate_digests=created.candidate_digests,
        predicted_score=created.score,
        predicted_mean_advantage=np.asarray([0.0, 0.018, -0.018]),
        predicted_loss_cvar=np.asarray([0.0, 0.005, 0.025]),
        predicted_expected_turnover=np.asarray([0.0, 0.25, 0.25]),
        scenario_anchor_indices=np.arange(64, dtype=np.int64),
        scenario_distances=np.linspace(0.0, 1.0, 64),
    )


def _outcome(
    kind: str,
    value: float,
    *,
    cost_multiplier: float = 1.0,
) -> RealizedPolicyOutcome:
    terminal = 100_000.0 * float(np.exp(value))
    cost = 0.0001 * cost_multiplier
    payload = {
        "borrow_paid": 0.0,
        "cancel_replace_events": (1 if cost_multiplier > 1.0 else 0),
        "fees": cost,
        "fill_count": 1,
        "fill_ratio": (0.9 if cost_multiplier > 1.0 else 1.0),
        "filled_turnover": 0.1,
        "funding_paid": 0.0,
        "gross_log_return": value,
        "impact_cost": cost,
        "max_drawdown": 0.08 if kind == "trend" else 0.09,
        "pending_order_events": (1 if cost_multiplier > 1.0 else 0),
        "policy_kind": kind,
        "schema_version": "causal_scenario_c3_realized_outcome_v1",
        "spread_cost": cost,
        "terminal_equity": terminal,
        "termination_reason": "horizon",
    }
    return RealizedPolicyOutcome(
        policy_kind=kind,
        gross_log_return=value,
        filled_turnover=0.1,
        fees=cost,
        spread_cost=cost,
        impact_cost=cost,
        funding_paid=0.0,
        borrow_paid=0.0,
        fill_ratio=(0.9 if cost_multiplier > 1.0 else 1.0),
        fill_count=1,
        pending_order_events=(1 if cost_multiplier > 1.0 else 0),
        cancel_replace_events=(1 if cost_multiplier > 1.0 else 0),
        max_drawdown=0.08 if kind == "trend" else 0.09,
        terminal_equity=terminal,
        termination_reason="horizon",
        outcome_digest=content_digest(payload),
    )


class Replay:
    def __init__(
        self,
        identity: C3ReplayIdentity,
        *,
        cost_multiplier: float = 1.0,
    ) -> None:
        self.identity = identity
        self.cost_multiplier = cost_multiplier

    def clone_for_replay(self) -> Replay:
        return Replay(self.identity, cost_multiplier=self.cost_multiplier)

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
        return _outcome(
            policy_kind,
            0.01 + 0.02 * float(raw_residual[0]),
            cost_multiplier=self.cost_multiplier,
        )


def _perfect_information(*, causal_log_return: float) -> PerfectInformationComparison:
    evidence = PerfectInformationCompatibilityEvidence(
        causal_period_digest=sha("7"),
        bound_period_digest=sha("7"),
        causal_return_matrix_digest=sha("8"),
        bound_return_matrix_digest=sha("8"),
        causal_initial_weights=np.asarray([0.0]),
        bound_initial_weights=np.asarray([0.0]),
        causal_aum=100_000.0,
        bound_aum=100_000.0,
        causal_max_abs_weight=np.asarray([0.45]),
        bound_max_abs_weight=np.asarray([0.50]),
        causal_max_gross=0.90,
        bound_max_gross=1.00,
        causal_max_net_exposure=0.40,
        bound_max_net_exposure=0.50,
        causal_transaction_cost_rate=np.asarray([0.0010]),
        bound_transaction_cost_rate=np.asarray([0.0005]),
        causal_liquidation_cost_rate=np.asarray([0.0010]),
        bound_liquidation_cost_rate=np.asarray([0.0005]),
        bound_result_digest=sha("9"),
        bound_log_return=0.08,
        causal_log_return=causal_log_return,
    )
    return evaluate_perfect_information_compatibility(evidence)


def _query_comparisons(tmp_path: Path, *, fold_index: int, day_index: int):
    fold_id = f"fold-{fold_index}"
    created = _decision(
        query_index=10_000 + fold_index * 100 + day_index,
        fold_digest=content_digest({"fold_id": fold_id}),
    )
    root = tmp_path / "decisions" / fold_id / str(day_index)
    write_c3_decision_artifact(root, created)
    loaded = load_c3_decision_artifact(root)
    evidence = _prediction(created)
    return (
        run_c3_query_comparison(
            loaded,
            replay=Replay(created.replay_identity),
            ppo_mean_action=np.asarray([0.5]),
            config=CausalScenarioC3Config(random_comparator_count=1),
            prediction_evidence=evidence,
            execution_scenario="nominal",
            perfect_information=_perfect_information(causal_log_return=0.03),
        ),
        run_c3_query_comparison(
            loaded,
            replay=Replay(created.replay_identity, cost_multiplier=2.0),
            ppo_mean_action=np.asarray([0.5]),
            config=CausalScenarioC3Config(random_comparator_count=1),
            prediction_evidence=evidence,
            execution_scenario="adverse_cost_2x",
            perfect_information=_perfect_information(causal_log_return=0.03),
        ),
    )


def _aggregate(tmp_path: Path):
    folds = tuple(
        build_c3_fold_report(
            fold_id=f"fold-{fold_index}",
            selection_days=30,
            comparisons=tuple(
                comparison
                for day_index in range(30)
                for comparison in _query_comparisons(
                    tmp_path,
                    fold_index=fold_index,
                    day_index=day_index,
                )
            ),
            required_adverse_passed=True,
        )
        for fold_index in range(6)
    )
    return build_c3_aggregate_report(folds, bootstrap_resamples=128)


def test_report_and_gate_artifacts_round_trip_and_fail_closed(tmp_path: Path) -> None:
    report = _aggregate(tmp_path)
    report_root = tmp_path / "report"
    report_digest = write_c3_aggregate_report_artifact(report_root, report)
    loaded_report = load_c3_aggregate_report_artifact(report_root)
    assert loaded_report.artifact_digest == report_digest
    assert loaded_report.report.digest == report.digest
    assert loaded_report.report.total_effective_days == 180
    assert loaded_report.report.calibration_buckets == report.calibration_buckets
    assert loaded_report.report.execution_summaries == report.execution_summaries

    gate = evaluate_phase_a_entry_gate(loaded_report.report)
    gate_root = tmp_path / "gate"
    gate_digest = write_phase_a_gate_artifact(gate_root, gate)
    loaded_gate = load_phase_a_gate_artifact(gate_root)
    assert loaded_gate.artifact_digest == gate_digest
    assert loaded_gate.gate.digest == gate.digest
    assert loaded_gate.gate.passed is True

    (report_root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure"):
        load_c3_aggregate_report_artifact(report_root)


def test_batch_publishes_report_and_gate_from_verified_decisions(
    tmp_path: Path,
) -> None:
    queries = []
    fold_days = {}
    adverse = {}
    for fold_index in range(6):
        fold_id = f"fold-{fold_index}"
        fold_days[fold_id] = 30
        adverse[fold_id] = True
        for day_index in range(30):
            created = _decision(
                query_index=10_000 + fold_index * 100 + day_index,
                fold_digest=content_digest({"fold_id": fold_id}),
            )
            decision_root = tmp_path / "batch-decisions" / fold_id / str(day_index)
            write_c3_decision_artifact(decision_root, created)
            evidence = _prediction(created)
            common = {
                "fold_id": fold_id,
                "decision_root": decision_root,
                "ppo_mean_action": np.asarray([0.5]),
                "prediction_evidence": evidence,
                "perfect_information": _perfect_information(causal_log_return=0.03),
            }
            queries.extend(
                (
                    C3BatchQuery(
                        **common,
                        replay=Replay(created.replay_identity),
                        execution_scenario="nominal",
                    ),
                    C3BatchQuery(
                        **common,
                        replay=Replay(created.replay_identity, cost_multiplier=2.0),
                        execution_scenario="adverse_cost_2x",
                    ),
                )
            )
    result = execute_c3_batch(
        tuple(queries),
        output_root=tmp_path / "batch",
        fold_selection_days=fold_days,
        required_adverse_passed=adverse,
        config=CausalScenarioC3Config(random_comparator_count=1),
    )
    assert result.gate.passed is True
    assert result.report.total_selection_days == 180
    assert result.report.total_effective_days == 180
    assert result.comparison_count == 360
    assert result.report_artifact_root.is_dir()
    assert result.gate_artifact_root.is_dir()
    assert result.production_status == "NO-GO"


def test_cli_publishes_gate_from_verified_report(tmp_path: Path) -> None:
    report_root = tmp_path / "report"
    write_c3_aggregate_report_artifact(report_root, _aggregate(tmp_path))
    gate_root = tmp_path / "gate"
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = main(
        [
            "causal-scenario",
            "gate",
            "--report",
            str(report_root),
            "--output",
            str(gate_root),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["passed"] is True
    assert payload["production_status"] == "NO-GO"
    assert payload["schema"] == "causal_scenario_gate_result_v1"
    assert load_phase_a_gate_artifact(gate_root).gate.passed is True
