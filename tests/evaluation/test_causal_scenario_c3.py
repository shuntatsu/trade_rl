from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    CausalScenarioC3Config,
    PerfectInformationComparison,
    PerfectInformationComparisonStatus,
    PersistedScenarioDecision,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    LoadedC3Decision,
    load_c3_decision_artifact,
    write_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_gate import evaluate_phase_a_entry_gate
from trade_rl.evaluation.causal_scenario_c3_report import (
    build_c3_aggregate_report,
    build_c3_fold_report,
)
from trade_rl.evaluation.causal_scenario_c3_runner import run_c3_query_comparison


def sha(char: str) -> str:
    return char * 64


def _decision_payload(
    *,
    selected_index: int,
    zero_index: int,
    candidate_digests: tuple[str, ...],
    raw_actions: np.ndarray,
    projected_targets: np.ndarray,
    score: np.ndarray,
    regret: np.ndarray,
) -> dict[str, object]:
    return {
        "candidate_digests": candidate_digests,
        "candidate_generator_digest": sha("6"),
        "created_before_realized_replay": True,
        "dataset_id": sha("a"),
        "fold_digest": sha("b"),
        "projected_targets": projected_targets.tolist(),
        "query_index": 10_000,
        "query_timestamp_ns": 1_800_000_000_000_000_000,
        "raw_candidate_actions": raw_actions.tolist(),
        "regret": regret.tolist(),
        "scenario_library_digest": sha("3"),
        "scenario_set_digest": sha("4"),
        "schema_version": "causal_scenario_c3_decision_v1",
        "score": score.tolist(),
        "selected_candidate_digest": candidate_digests[selected_index],
        "selected_candidate_index": selected_index,
        "state_snapshot_digest": sha("2"),
        "tie_candidate_indices": (selected_index,),
        "value_result_digest": sha("5"),
        "zero_candidate_index": zero_index,
    }


def decision() -> PersistedScenarioDecision:
    raw_actions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    projected = 0.25 * raw_actions
    candidate_digests = tuple(
        content_digest(
            {
                "candidate": index,
                "raw_action": action.tolist(),
                "schema_version": "test_candidate_v1",
            }
        )
        for index, action in enumerate(raw_actions)
    )
    score = np.asarray([0.0, 0.04, -0.03, 0.01], dtype=np.float64)
    regret = score.max() - score
    payload = _decision_payload(
        selected_index=1,
        zero_index=0,
        candidate_digests=candidate_digests,
        raw_actions=raw_actions,
        projected_targets=projected,
        score=score,
        regret=regret,
    )
    return PersistedScenarioDecision(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        query_index=10_000,
        query_timestamp_ns=1_800_000_000_000_000_000,
        state_snapshot_digest=sha("2"),
        scenario_library_digest=sha("3"),
        scenario_set_digest=sha("4"),
        candidate_generator_digest=sha("6"),
        value_result_digest=sha("5"),
        candidate_digests=candidate_digests,
        raw_candidate_actions=raw_actions,
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


def outcome(
    kind: str,
    *,
    gross_log_return: float,
    max_drawdown: float = 0.05,
) -> RealizedPolicyOutcome:
    payload = {
        "borrow_paid": 0.0,
        "fees": 0.0001,
        "fill_count": 1,
        "filled_turnover": 0.1,
        "funding_paid": 0.0,
        "gross_log_return": gross_log_return,
        "impact_cost": 0.0001,
        "max_drawdown": max_drawdown,
        "pending_order_events": 0,
        "policy_kind": kind,
        "schema_version": "causal_scenario_c3_realized_outcome_v1",
        "spread_cost": 0.0001,
        "terminal_equity": 100_000.0 * float(np.exp(gross_log_return)),
        "termination_reason": "horizon",
    }
    return RealizedPolicyOutcome(
        policy_kind=kind,
        gross_log_return=gross_log_return,
        filled_turnover=0.1,
        fees=0.0001,
        spread_cost=0.0001,
        impact_cost=0.0001,
        funding_paid=0.0,
        borrow_paid=0.0,
        fill_count=1,
        pending_order_events=0,
        max_drawdown=max_drawdown,
        terminal_equity=100_000.0 * float(np.exp(gross_log_return)),
        termination_reason="horizon",
        outcome_digest=content_digest(payload),
    )


class ArtificialReplay:
    def __init__(self) -> None:
        self.labels: list[str] = []

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
        self.labels.append(policy_kind)
        value = 0.01 + 0.02 * float(raw_residual[0]) + 0.005 * float(
            raw_residual[1]
        )
        return outcome(policy_kind, gross_log_return=value)


def test_c3_config_is_closed_and_rejects_boolean_counts() -> None:
    config = CausalScenarioC3Config()
    assert config.policy_order == (
        "trend",
        "scenario_oracle",
        "ppo_mean",
        "random_candidate",
        "perfect_information",
    )
    assert config.required_folds == 6
    assert config.required_selection_days == 180
    with pytest.raises(ValueError, match="positive integer"):
        CausalScenarioC3Config(random_comparator_count=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        CausalScenarioC3Config(schema_version="future")


def test_decision_is_immutable_and_digest_bound() -> None:
    created = decision()
    assert created.raw_candidate_actions.flags.writeable is False
    assert created.selected_candidate_digest == created.candidate_digests[1]
    with pytest.raises(ValueError, match="decision_digest"):
        replace(created, decision_digest=sha("f"))
    with pytest.raises(ValueError, match="selected_candidate_digest"):
        replace(created, selected_candidate_digest=created.candidate_digests[0])


def test_decision_artifact_is_exact_idempotent_and_tamper_evident(
    tmp_path: Path,
) -> None:
    created = decision()
    root = tmp_path / "decision"
    first = write_c3_decision_artifact(root, created)
    second = write_c3_decision_artifact(root, created)
    assert first == second
    loaded = load_c3_decision_artifact(root)
    assert isinstance(loaded, LoadedC3Decision)
    assert loaded.decision.decision_digest == created.decision_digest
    assert loaded.artifact_digest == first

    (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure"):
        load_c3_decision_artifact(root)


def test_runner_requires_loaded_decision_and_persists_before_replay(
    tmp_path: Path,
) -> None:
    replay = ArtificialReplay()
    with pytest.raises(TypeError, match="LoadedC3Decision"):
        run_c3_query_comparison(  # type: ignore[arg-type]
            decision(),
            replay=replay,
            ppo_mean_action=np.asarray([0.5, 0.0, 0.0]),
            config=CausalScenarioC3Config(random_comparator_count=1),
        )
    root = tmp_path / "decision"
    write_c3_decision_artifact(root, decision())
    loaded = load_c3_decision_artifact(root)
    result = run_c3_query_comparison(
        loaded,
        replay=replay,
        ppo_mean_action=np.asarray([0.5, 0.0, 0.0]),
        config=CausalScenarioC3Config(random_comparator_count=1),
    )
    assert result.scenario_oracle.gross_log_return > result.trend.gross_log_return
    assert result.selected_realized_regret == 0.0
    assert result.predicted_realized_spearman > 0.0
    assert replay.labels[:4] == [
        "candidate:0",
        "candidate:1",
        "candidate:2",
        "candidate:3",
    ]


def test_perfect_information_comparison_is_explicitly_not_comparable() -> None:
    comparison = PerfectInformationComparison.not_comparable("initial_weights_mismatch")
    assert comparison.status is PerfectInformationComparisonStatus.NOT_COMPARABLE
    assert comparison.gap is None
    with pytest.raises(ValueError, match="gap"):
        PerfectInformationComparison(
            status=PerfectInformationComparisonStatus.NOT_COMPARABLE,
            reason="mismatch",
            bound_log_return=None,
            causal_log_return=None,
            gap=0.0,
        )


def _comparison(tmp_path: Path, *, uplift: float, spearman_positive: bool = True):
    created = decision()
    root = tmp_path / created.decision_digest
    write_c3_decision_artifact(root, created)
    loaded = load_c3_decision_artifact(root)
    result = run_c3_query_comparison(
        loaded,
        replay=ArtificialReplay(),
        ppo_mean_action=np.asarray([0.5, 0.0, 0.0]),
        config=CausalScenarioC3Config(random_comparator_count=1),
    )
    trend = outcome("trend", gross_log_return=0.01, max_drawdown=0.08)
    oracle = outcome(
        "scenario_oracle", gross_log_return=0.01 + uplift, max_drawdown=0.09
    )
    random = outcome("random_candidate", gross_log_return=0.005)
    candidate_outcomes = list(result.candidate_outcomes)
    candidate_outcomes[0] = random
    scores = result.realized_candidate_advantages
    if not spearman_positive:
        scores = scores[::-1].copy()
    random_regrets = np.asarray([0.02], dtype=np.float64)
    return replace(
        result,
        trend=trend,
        scenario_oracle=oracle,
        random_candidate=random,
        random_candidate_indices=(0,),
        random_candidate_outcomes=(random,),
        random_realized_regrets=random_regrets,
        candidate_outcomes=tuple(candidate_outcomes),
        realized_candidate_advantages=scores,
        predicted_realized_spearman=(0.5 if spearman_positive else -0.5),
        selected_realized_regret=0.0,
        random_realized_regret=0.02,
        perfect_information=PerfectInformationComparison.comparable(
            bound_log_return=0.08,
            causal_log_return=0.01 + uplift,
        ),
    )


def test_six_fold_report_passes_all_phase_a_gates(tmp_path: Path) -> None:
    folds = []
    for fold_index in range(6):
        comparisons = tuple(
            _comparison(
                tmp_path / f"fold-{fold_index}",
                uplift=0.02 + 0.001 * query_index,
            )
            for query_index in range(32)
        )
        folds.append(
            build_c3_fold_report(
                fold_id=f"fold-{fold_index}",
                selection_days=30,
                comparisons=comparisons,
                required_adverse_passed=True,
            )
        )
    report = build_c3_aggregate_report(tuple(folds), bootstrap_resamples=256)
    gate = evaluate_phase_a_entry_gate(report)
    assert report.total_selection_days == 180
    assert report.positive_uplift_folds == 6
    assert report.uplift_lower_ci > 0.0
    assert report.spearman_lower_ci > 0.0
    assert report.regret_margin_lower_ci > 0.0
    assert gate.passed is True
    assert all(condition.passed for condition in gate.conditions)


def test_missing_adverse_evidence_fails_gate(tmp_path: Path) -> None:
    folds = []
    for fold_index in range(6):
        comparisons = tuple(
            _comparison(tmp_path / f"bad-{fold_index}", uplift=0.02) for _ in range(32)
        )
        folds.append(
            build_c3_fold_report(
                fold_id=f"fold-{fold_index}",
                selection_days=30,
                comparisons=comparisons,
                required_adverse_passed=fold_index != 5,
            )
        )
    gate = evaluate_phase_a_entry_gate(
        build_c3_aggregate_report(tuple(folds), bootstrap_resamples=256)
    )
    assert gate.passed is False
    assert "required_adverse_execution" in gate.failed_condition_names
