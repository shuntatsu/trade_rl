from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
)
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import SharedCashStrategyComparisonEntry
from trade_rl.evaluation.experiments import (
    ArtifactIntegrityError,
    ContractViolationError,
    ControlledFactor,
    ExperimentComparison,
    ExperimentDecisionKind,
    InvalidExperimentStateError,
    compare_experiment,
    create_study,
    decide_experiment,
    define_experiment,
    freeze_study,
    inspect_study,
    run_baseline,
    run_experiment,
    verify_experiment,
)
from trade_rl.evaluation.experiments.contracts import (
    StudyOutcome,
    StudyResearchContext,
)
from trade_rl.evaluation.experiments.protocols import (
    PPOHoldingDurationMetrics,
    ppo_holding_winner_digest,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.evaluation.runs.config import LEGACY_DATASET_EXECUTION_OVERLAY
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent

_HORIZONS = (72, 168, 336, 504)
_NOW = datetime(2026, 9, 29, tzinfo=UTC)


def _holding_final_window() -> dict[str, object]:
    return {
        "final_evaluation_start": "2026-01-02T00:00:00.000000000",
        "final_evaluation_stop_exclusive": "2026-01-02T02:00:00.000000000",
        "research_context": StudyResearchContext(
            parent_context_digests=(),
            consumed_evidence=(),
        ),
    }


def _holding_config():
    from trade_rl.risk import PreTradeRiskConfig
    from trade_rl.strategies.rl.ppo import PPO_OBSERVATION_SCHEMA_V3

    return replace(
        _config(),
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=True,
        pretrade_risk_config=PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.1,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
    )


def _fake_shared_cash_execute():
    def execute(dataset, spec):
        result = _fake_execute()(dataset, spec)
        replay = run_shared_cash_replay(
            dataset,
            tuple(ConstantIntentStrategy(PositionIntent.FLAT) for _ in dataset.symbols),
            start_index=spec.evaluation_start_index,
            stop_index=spec.evaluation_stop_index,
            gross_budget=spec.config.gross_budget,
            initial_capital=spec.config.initial_capital,
            execution_cost=ExecutionCostConfig.zero(),
            risk=(
                PreTradeRisk(spec.config.pretrade_risk_config)
                if spec.config.pretrade_risk_config is not None
                else None
            ),
            minimum_hold_bars=spec.config.ppo_minimum_hold_bars,
            settle_terminal_position=spec.config.ppo_settle_terminal_position,
            capture_ledger_evidence=True,
            capture_accounting_evidence=True,
            ohlc_drawdown_stress=True,
        )
        diagnostics = replay.diagnostics
        shared = SharedCashStrategyComparisonEntry(
            name="ppo",
            replay=replay,
            metrics=evaluate_performance(
                replay.returns,
                turnover_total=diagnostics.turnover_total,
                total_cost=diagnostics.total_cost,
                funding_pnl=diagnostics.funding_pnl,
                borrow_cost=diagnostics.borrow_cost,
                n_trades=diagnostics.n_trades,
                rebalance_events=diagnostics.rebalance_events,
                termination_count=diagnostics.termination_count,
            ),
        )
        comparison = replace(result.comparison, shared_cash_ppo=shared)
        return replace(result, comparison=comparison)

    return execute


def _holding_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    protocol: str = "ppo_holding_duration_v1",
):
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    execute = (
        _fake_shared_cash_execute()
        if protocol == "ppo_shared_cash_holding_duration_v2"
        else _fake_execute()
    )
    monkeypatch.setattr(evidence_module, "execute_candidate_run", execute)
    root = tmp_path / "study"
    baseline_config = _holding_config()
    if protocol == "ppo_shared_cash_holding_duration_v2":
        baseline_config = replace(baseline_config, initial_capital=100_000.0)
    snapshot = create_study(
        root,
        dataset_root=dataset_root,
        research_question="Does a PPO minimum holding period improve net returns?",
        baseline_config=baseline_config,
        ppo_seeds=(2, 5, 9, 13, 17),
        allowed_factors=(ControlledFactor.PPO_MINIMUM_HOLD,),
        max_experiments=len(_HORIZONS),
        n_bootstrap=32,
        bootstrap_seed=17,
        protocol=protocol,
        **_holding_final_window(),
    )
    return root, dataset_root, snapshot


def _holding_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, dataset_root, _ = _holding_study(tmp_path, monkeypatch)
    baseline = run_baseline(root, dataset_root=dataset_root)
    assert baseline.baseline is not None
    return root, dataset_root, baseline


def test_handwritten_review_record_is_not_an_authoritative_assurance_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, snapshot = _holding_study(tmp_path, monkeypatch)
    (root / "assurance-review.json").write_text(
        json.dumps(
            {
                "schema_version": "study_assurance_review_v1",
                "study_digest": snapshot.plan.digest,
                "dataset_artifact_digest": snapshot.plan.dataset_artifact_digest,
                "implementation_digest": snapshot.plan.implementation_digest,
                "runtime_environment_digest": snapshot.plan.runtime_environment_digest,
                "reviewer_identity": "caller-controlled",
                "review_reference": "https://github.com/example/fake-review",
                "review_evidence_digest": "a" * 64,
                "reviewer_independence_established": True,
                "result_blind": True,
                "g0": "PASS",
                "g1": "PASS",
                "g2": "PASS",
                "reviewed_at": "2026-09-29T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="unexpected Study root entries"):
        inspect_study(root)


def _define_all_horizons(root: Path, dataset_root: Path, baseline) -> None:
    assert baseline.baseline is not None
    config = _holding_config()
    if baseline.plan.is_ppo_shared_cash_holding_duration_study:
        config = replace(config, initial_capital=100_000.0)
    for horizon in _HORIZONS:
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis=f"PPO minimum hold is {horizon} one-hour bars.",
            factor=ControlledFactor.PPO_MINIMUM_HOLD,
            candidate_config=replace(config, ppo_minimum_hold_bars=horizon),
            baseline_evidence_digest=baseline.baseline.fingerprint,
        )


def test_holding_protocol_is_bound_into_the_immutable_study_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, snapshot = _holding_study(tmp_path, monkeypatch)

    assert snapshot.plan.schema_version == "controlled_study_plan_v5"
    assert snapshot.plan.protocol.value == "ppo_holding_duration_v1"
    assert snapshot.plan.allowed_factors == (ControlledFactor.PPO_MINIMUM_HOLD,)
    assert snapshot.plan.max_experiments == 4
    assert snapshot.plan.to_payload()["protocol"] == "ppo_holding_duration_v1"
    assert (
        "Primary score: median across the five registered seeds"
        in snapshot.plan.research_question
    )
    with pytest.raises(ContractViolationError, match="full selection rule"):
        replace(
            snapshot.plan, research_question="Does minimum holding improve returns?"
        )
    assert (root / "plan.json").is_file()


def test_shared_cash_holding_protocol_is_bound_to_a_separate_plan_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, snapshot = _holding_study(
        tmp_path,
        monkeypatch,
        protocol="ppo_shared_cash_holding_duration_v2",
    )

    assert snapshot.plan.schema_version == "controlled_study_plan_v6"
    assert snapshot.plan.protocol.value == "ppo_shared_cash_holding_duration_v2"
    assert snapshot.plan.is_ppo_holding_duration_study
    assert snapshot.plan.is_ppo_shared_cash_holding_duration_study
    assert snapshot.plan.to_payload()["protocol"] == (
        "ppo_shared_cash_holding_duration_v2"
    )
    assert "do not average independent per-symbol accounts" in (
        snapshot.plan.research_question
    )


def test_shared_cash_evidence_publication_fails_closed_on_source_binding_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    root, dataset_root, _ = _holding_study(
        tmp_path,
        monkeypatch,
        protocol="ppo_shared_cash_holding_duration_v2",
    )
    calls = 0

    validate_source_binding = (
        evidence_module._validate_shared_cash_candidate_source_binding
    )

    def mutate_source_then_validate(**kwargs) -> None:
        nonlocal calls
        calls += 1
        dataset = kwargs["dataset"]
        candidate_summary = kwargs["candidate_summary"]
        portfolio = candidate_summary["shared_cash_ppo"]
        ledger_evidence = portfolio["ledger_evidence"]
        ledger = ledger_evidence["payload"]
        transition = next(
            transition
            for interval in ledger["intervals"]
            for transition in interval["accounting_transitions"]
            if transition["transition_type"] == "mark_revaluation"
        )
        processing_index = transition["processing_index"]
        changed_open = dataset.open.copy()
        changed_open[processing_index, 0] += 1.0
        kwargs["dataset"] = replace(
            dataset,
            identity_payload_json=None,
            open=changed_open,
            high=np.maximum(dataset.high, changed_open),
            low=np.minimum(dataset.low, changed_open),
        )
        validate_source_binding(**kwargs)

    monkeypatch.setattr(
        evidence_module,
        "_validate_shared_cash_candidate_source_binding",
        mutate_source_then_validate,
    )

    with pytest.raises(ArtifactIntegrityError, match="source open row"):
        run_baseline(root, dataset_root=dataset_root)

    assert calls == 1
    assert not (root / "baseline").exists()


def test_shared_cash_study_reload_revalidates_resealed_borrow_overlay_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments.codec import _analysis_binding
    from trade_rl.evaluation.experiments.evidence import _evidence_fingerprint
    from trade_rl.evaluation.runs import inspect_candidate_run_artifact

    root, dataset_root, _ = _holding_study(
        tmp_path,
        monkeypatch,
        protocol="ppo_shared_cash_holding_duration_v2",
    )
    run_baseline(root, dataset_root=dataset_root)

    evidence_root = root / "baseline" / "evidence"
    run_root = evidence_root / "runs" / "seed-2"
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger_evidence = summary["shared_cash_ppo"]["ledger_evidence"]
    ledger = ledger_evidence["payload"]
    # This all-flat fixture has zero borrow amount, so changing the recorded
    # multiplier leaves the Run's internal arithmetic self-consistent.
    borrow_transitions = [
        transition
        for interval in ledger["intervals"]
        for transition in interval["accounting_transitions"]
        if transition["transition_type"] == "borrow_charge"
    ]
    assert borrow_transitions
    assert all(
        transition["evidence"]["borrow_amount"] == 0.0
        for transition in borrow_transitions
    )
    for transition in borrow_transitions:
        transition["evidence"]["borrow_rate_multiplier"] = 1.0
    ledger_evidence["digest"] = content_digest(ledger)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_digests = manifest["run_digests"]
    run_digests[0]["artifact_digest"] = inspect_candidate_run_artifact(
        run_root
    ).artifact_digest
    manifest["run_digests"] = run_digests
    manifest["fingerprint"] = _evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"]) for entry in run_digests
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    analysis_path = root / "baseline" / "analysis.json"
    binding = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis_path.write_text(
        json.dumps(
            _analysis_binding(
                evidence_fingerprint=manifest["fingerprint"],
                analysis=binding["analysis"],
            ),
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="borrow rate multiplier"):
        inspect_study(root)


def test_shared_cash_source_binding_rejects_fill_price_forged_across_linked_evidence(
    tmp_path: Path,
) -> None:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    replay = run_shared_cash_replay(
        dataset,
        (ConstantIntentStrategy(PositionIntent.LONG),),
        start_index=12,
        stop_index=20,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
        ohlc_drawdown_stress=True,
    )
    assert replay.ledger_evidence is not None
    ledger_payload = json.loads(json.dumps(replay.ledger_evidence.to_mapping()))
    ledger_evidence = {
        "schema_version": replay.ledger_evidence.schema_version,
        "digest": content_digest(ledger_payload),
        "payload": ledger_payload,
    }
    summary = {
        "dataset_id": dataset.dataset_id,
        "dataset_artifact": {"artifact_digest": dataset_artifact.artifact_digest},
        "evaluation": {"execution_overlay": LEGACY_DATASET_EXECUTION_OVERLAY},
        "shared_cash_ppo": {"ledger_evidence": ledger_evidence},
    }
    evidence_module._validate_shared_cash_candidate_source_binding(
        candidate_summary=summary,
        dataset=dataset,
        expected_dataset_artifact_digest=dataset_artifact.artifact_digest,
    )

    payload = ledger_evidence["payload"]
    fill_interval = next(
        interval
        for interval in payload["intervals"]
        if any(
            event["event_type"] in {"filled", "partial_fill"}
            for event in interval["order_events"]
        )
    )
    fill_event = next(
        event
        for event in fill_interval["order_events"]
        if event["event_type"] in {"filled", "partial_fill"}
    )
    fill_transition = next(
        transition
        for transition in fill_interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
        and transition["order_event_sequence"] == fill_event["sequence"]
    )
    symbol = fill_event["symbol_index"]
    processing_index = fill_event["processing_index"]
    original_price = fill_event["execution_price"]
    forged_price = original_price + 0.01
    assert dataset.low[processing_index, symbol] <= forged_price
    assert forged_price <= dataset.high[processing_index, symbol]
    forged_notional = (
        abs(fill_event["filled_quantity"])
        * forged_price
        * float(dataset.resolved_array("contract_multipliers")[symbol])
    )
    notional_delta = forged_notional - fill_event["filled_notional"]
    fill_event["execution_price"] = forged_price
    fill_event["filled_notional"] = forged_notional
    fill_evidence = fill_transition["evidence"]
    fill_evidence["execution_price"] = forged_price
    fill_evidence["filled_notional"] = forged_notional
    fill_evidence["turnover"] = (
        forged_notional / fill_interval["portfolio_value_before"]
    )
    fill_interval["turnover_total_after"] += (
        notional_delta / fill_interval["portfolio_value_before"]
    )
    ledger_evidence["digest"] = content_digest(payload)

    with pytest.raises(ArtifactIntegrityError):
        evidence_module._validate_shared_cash_candidate_source_binding(
            candidate_summary=summary,
            dataset=dataset,
            expected_dataset_artifact_digest=dataset_artifact.artifact_digest,
        )

    cost_payload = json.loads(json.dumps(ledger_evidence["payload"]))
    cost_interval = next(
        interval
        for interval in cost_payload["intervals"]
        if any(
            transition["transition_type"] == "fill"
            for transition in interval["accounting_transitions"]
        )
    )
    cost_transition = next(
        transition
        for transition in cost_interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
    )
    cost_transition["evidence"]["cost_amount"] += 0.01
    forged_cost_ledger = {
        **ledger_evidence,
        "digest": content_digest(cost_payload),
        "payload": cost_payload,
    }
    forged_cost_summary = {
        **summary,
        "shared_cash_ppo": {"ledger_evidence": forged_cost_ledger},
    }
    with pytest.raises(ArtifactIntegrityError, match="execution economics"):
        evidence_module._validate_shared_cash_candidate_source_binding(
            candidate_summary=forged_cost_summary,
            dataset=dataset,
            expected_dataset_artifact_digest=dataset_artifact.artifact_digest,
        )


def test_shared_cash_protocol_completes_comparison_decision_and_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, plan = _holding_study(
        tmp_path,
        monkeypatch,
        protocol="ppo_shared_cash_holding_duration_v2",
    )
    baseline = run_baseline(root, dataset_root=dataset_root)
    assert baseline.baseline is not None
    _define_all_horizons(root, dataset_root, baseline)

    for sequence in range(1, len(_HORIZONS) + 1):
        run_experiment(root, sequence, dataset_root=dataset_root)
        verification = verify_experiment(root, sequence)
        assert verification.status.value == "CONTROLLED"
        comparison = compare_experiment(root, sequence)
        assert comparison.to_payload()["factor_effect"]["schema_version"] == (
            "controlled_evidence_comparison_v4"
        )
        decision = decide_experiment(
            root,
            sequence,
            decision=ExperimentDecisionKind.KEEP_BASELINE,
            rationale="The shared-cash portfolio did not beat its paired baseline.",
            decided_by="test-mock",
            decided_at=_NOW,
        )
        assert decision.decision is ExperimentDecisionKind.KEEP_BASELINE

    frozen = freeze_study(
        root,
        outcome=StudyOutcome.NO_WINNER,
        rationale="No shared-cash arm passed the positive excess-return gate.",
        frozen_by="test-mock",
        frozen_at=_NOW,
    )
    inspected = inspect_study(root)
    assert frozen.outcome is StudyOutcome.NO_WINNER
    assert inspected.freeze == frozen
    assert inspected.plan.digest == plan.plan.digest


def test_new_holding_study_cannot_bypass_final_window_and_research_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    root = tmp_path / "study"

    with pytest.raises(ContractViolationError, match="final window.*research context"):
        create_study(
            root,
            dataset_root=dataset_root,
            research_question="A holding study must bind its final boundary.",
            baseline_config=_holding_config(),
            ppo_seeds=(2, 5, 9, 13, 17),
            allowed_factors=(ControlledFactor.PPO_MINIMUM_HOLD,),
            max_experiments=len(_HORIZONS),
            n_bootstrap=32,
            bootstrap_seed=17,
            protocol="ppo_holding_duration_v1",
        )

    assert not (root / "plan.json").exists()


def test_minimum_hold_factor_cannot_bypass_the_versioned_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())

    with pytest.raises(ContractViolationError, match="versioned Study protocol"):
        create_study(
            tmp_path / "study",
            dataset_root=dataset_root,
            research_question="Compare PPO minimum holding periods.",
            baseline_config=_config(),
            ppo_seeds=(2, 5, 9, 13, 17),
            allowed_factors=(ControlledFactor.PPO_MINIMUM_HOLD,),
            max_experiments=4,
            n_bootstrap=32,
            bootstrap_seed=17,
        )


@pytest.mark.parametrize(
    ("allowed_factors", "max_experiments"),
    [
        ((ControlledFactor.PPO_MINIMUM_HOLD, ControlledFactor.FEATURE_SET), 4),
        ((ControlledFactor.PPO_MINIMUM_HOLD,), 3),
    ],
)
def test_holding_protocol_rejects_a_noncanonical_roster_or_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowed_factors,
    max_experiments: int,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())

    with pytest.raises(ContractViolationError, match="holding-duration protocol"):
        create_study(
            tmp_path / "study",
            dataset_root=dataset_root,
            research_question="Invalid holding-duration plan.",
            baseline_config=_holding_config(),
            ppo_seeds=(2, 5, 9, 13, 17),
            allowed_factors=allowed_factors,
            max_experiments=max_experiments,
            n_bootstrap=32,
            bootstrap_seed=17,
            protocol="ppo_holding_duration_v1",
            **_holding_final_window(),
        )


def test_holding_protocol_preregisters_all_horizons_and_enforces_realized_drawdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, baseline = _holding_baseline(tmp_path, monkeypatch)
    assert baseline.baseline is not None
    config = _holding_config()

    with pytest.raises(ContractViolationError, match="72"):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="The first arm must use the registered three-day horizon.",
            factor=ControlledFactor.PPO_MINIMUM_HOLD,
            candidate_config=replace(config, ppo_minimum_hold_bars=168),
            baseline_evidence_digest=baseline.baseline.fingerprint,
        )

    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="PPO minimum hold is 72 one-hour bars.",
        factor=ControlledFactor.PPO_MINIMUM_HOLD,
        candidate_config=replace(config, ppo_minimum_hold_bars=72),
        baseline_evidence_digest=baseline.baseline.fingerprint,
    )
    with pytest.raises(InvalidExperimentStateError, match="all four.*registered"):
        run_experiment(root, 1, dataset_root=dataset_root)

    for horizon in _HORIZONS[1:]:
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis=f"PPO minimum hold is {horizon} one-hour bars.",
            factor=ControlledFactor.PPO_MINIMUM_HOLD,
            candidate_config=replace(config, ppo_minimum_hold_bars=horizon),
            baseline_evidence_digest=baseline.baseline.fingerprint,
        )

    evidence = run_experiment(root, 1, dataset_root=dataset_root)
    assert evidence.ppo_seeds == (2, 5, 9, 13, 17)
    verification = verify_experiment(root, 1)
    assert verification.status.value == "CONTROLLED"

    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments import inspection, workflow

    original_compare = workflow.compare_evidence_sets

    def over_limit_comparison(*args, **kwargs):
        result = original_compare(*args, **kwargs)
        cross_seed = dict(result["cross_seed"])
        ppo_summary = dict(cross_seed["ppo"])
        by_seed = dict(ppo_summary["by_seed"])
        first_seed, first_metrics = next(iter(by_seed.items()))
        first_worst_account = max(
            0.21,
            first_metrics["worst_baseline_max_drawdown"],
        )
        by_seed[first_seed] = {
            **first_metrics,
            "worst_candidate_max_drawdown": 0.21,
            "worst_account_max_drawdown": first_worst_account,
        }
        ppo_summary.update(
            worst_candidate_max_drawdown=0.21,
            worst_account_max_drawdown=max(
                0.21,
                ppo_summary["worst_baseline_max_drawdown"],
            ),
            by_seed=by_seed,
        )
        cross_seed["ppo"] = ppo_summary
        body = dict(result)
        body.pop("analysis_digest", None)
        body["cross_seed"] = cross_seed
        body["analysis_digest"] = content_digest(body)
        return body

    monkeypatch.setattr(workflow, "compare_evidence_sets", over_limit_comparison)
    monkeypatch.setattr(inspection, "compare_evidence_sets", over_limit_comparison)
    comparison = compare_experiment(root, 1)
    factor_effect = comparison.to_payload()["factor_effect"]
    assert factor_effect["schema_version"] == "controlled_evidence_comparison_v3"

    with pytest.raises(InvalidExperimentStateError, match="20% drawdown gate"):
        decide_experiment(
            root,
            1,
            decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
            rationale="Positive score cannot override the registered risk cap.",
            decided_by="researcher",
            decided_at=_NOW,
        )

    kept = decide_experiment(
        root,
        1,
        decision=ExperimentDecisionKind.KEEP_BASELINE,
        rationale="The arm exceeded the frozen realized drawdown gate.",
        decided_by="researcher",
        decided_at=_NOW,
    )
    assert kept.decision is ExperimentDecisionKind.KEEP_BASELINE


def test_holding_definition_rejects_unregistered_training_budget_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, baseline = _holding_baseline(tmp_path, monkeypatch)
    config = _holding_config()

    with pytest.raises(ContractViolationError, match="ppo_total_timesteps"):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="PPO minimum hold is 72 one-hour bars.",
            factor=ControlledFactor.PPO_MINIMUM_HOLD,
            candidate_config=replace(
                config,
                ppo_minimum_hold_bars=72,
                ppo_total_timesteps=config.ppo_total_timesteps + 1,
            ),
            baseline_evidence_digest=baseline.baseline.fingerprint,
        )

    assert not (root / "experiments" / "0001" / "definition.json").exists()


def test_candidate_execution_rechecks_persisted_factor_delta_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments import workflow
    from trade_rl.evaluation.experiments.codec import _candidate_config_payload

    root, dataset_root, baseline = _holding_baseline(tmp_path, monkeypatch)
    assert baseline.baseline is not None
    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="PPO minimum hold is 72 one-hour bars.",
        factor=ControlledFactor.PPO_MINIMUM_HOLD,
        candidate_config=replace(_holding_config(), ppo_minimum_hold_bars=72),
        baseline_evidence_digest=baseline.baseline.fingerprint,
    )
    for horizon in _HORIZONS[1:]:
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis=f"PPO minimum hold is {horizon} one-hour bars.",
            factor=ControlledFactor.PPO_MINIMUM_HOLD,
            candidate_config=replace(_holding_config(), ppo_minimum_hold_bars=horizon),
            baseline_evidence_digest=baseline.baseline.fingerprint,
        )
    tampered_config = replace(
        definition.candidate_config,
        ppo_total_timesteps=definition.candidate_config.ppo_total_timesteps + 1,
    )
    tampered_payload = definition.to_payload()
    tampered_payload["candidate_config"] = tampered_config.to_payload()
    tampered_payload["candidate_requested_config_digest"] = content_digest(
        _candidate_config_payload(
            tampered_config,
            resolved_schema_version=tampered_config.schema_version,
        )
    )
    definition_path = root / "experiments" / "0001" / "definition.json"
    definition_path.write_text(
        json.dumps(tampered_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    def fail_if_candidate_builds(*args, **kwargs) -> None:
        pytest.fail("candidate evidence builder ran before delta validation")

    monkeypatch.setattr(workflow, "_build_evidence_node", fail_if_candidate_builds)
    with pytest.raises(ContractViolationError, match="ppo_total_timesteps"):
        run_experiment(root, 1, dataset_root=dataset_root)

    assert not (root / "experiments" / "0001" / "candidate").exists()


def test_inspection_rejects_a_self_consistent_v2_comparison_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
    from trade_rl.evaluation.experiments.inspection import (
        _experiment_dir,
        _experiment_state,
        _find_evidence,
        _reconstruct,
    )
    from trade_rl.evaluation.experiments.store import StudyStore

    root, dataset_root, baseline = _holding_baseline(tmp_path, monkeypatch)
    _define_all_horizons(root, dataset_root, baseline)
    run_experiment(root, 1, dataset_root=dataset_root)
    verify_experiment(root, 1)

    store = StudyStore(root)
    state = _reconstruct(store)
    experiment = _experiment_state(state, 1)
    assert experiment.candidate is not None
    assert experiment.candidate_analysis is not None
    assert experiment.verification is not None
    baseline_evidence, baseline_analysis = _find_evidence(
        state,
        experiment.definition.baseline_evidence_digest,
    )
    factor_effect = compare_evidence_sets(
        baseline_evidence.runs,
        experiment.candidate.runs,
        n_bootstrap=state.plan.n_bootstrap,
        bootstrap_seed=state.plan.bootstrap_seed,
        schema_version="controlled_evidence_comparison_v2",
    )
    factor_digest = factor_effect["analysis_digest"]
    assert isinstance(factor_digest, str)
    downgraded = ExperimentComparison(
        study_digest=state.plan.digest,
        experiment_digest=experiment.definition.digest,
        baseline_evidence_digest=baseline_evidence.evidence.fingerprint,
        candidate_evidence_digest=experiment.candidate.evidence.fingerprint,
        verification_digest=experiment.verification.digest,
        baseline_analysis_digest=baseline_analysis.analysis_digest,
        candidate_analysis_digest=experiment.candidate_analysis.analysis_digest,
        factor_effect_digest=factor_digest,
        factor_effect=factor_effect,
    )
    store.publish_json_once(
        _experiment_dir(1) / "comparison.json",
        downgraded.to_payload(),
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="PPO holding-duration protocol requires factor-effect schema v3",
    ):
        inspect_study(root)


@pytest.mark.parametrize(
    ("score", "median_excess_return", "worst_max_drawdown"),
    [
        (0.01, 0.0, 0.10),
        (0.01, 0.01, 0.200001),
    ],
)
def test_holding_arm_is_ineligible_without_paired_improvement_or_drawdown_headroom(
    score: float,
    median_excess_return: float,
    worst_max_drawdown: float,
) -> None:
    metrics = PPOHoldingDurationMetrics(
        score=score,
        median_excess_return=median_excess_return,
        worst_max_drawdown=worst_max_drawdown,
        symbol_count=3,
    )

    assert not metrics.eligible


def test_holding_arm_can_rank_positive_pairwise_improvement_with_negative_absolute_return() -> (
    None
):
    metrics = PPOHoldingDurationMetrics(
        score=-0.02,
        median_excess_return=0.001,
        worst_max_drawdown=0.20,
        symbol_count=3,
    )

    assert metrics.eligible


def test_holding_arm_with_unsettled_terminal_account_is_ineligible() -> None:
    metrics = PPOHoldingDurationMetrics(
        score=0.02,
        median_excess_return=0.001,
        worst_max_drawdown=0.10,
        symbol_count=3,
        terminal_settlement_complete=False,
    )

    assert not metrics.eligible


def test_holding_arm_at_exactly_twenty_percent_is_eligible_and_ties_prefer_shorter():
    metrics = PPOHoldingDurationMetrics(
        score=0.01,
        median_excess_return=0.001,
        worst_max_drawdown=0.20,
        symbol_count=3,
    )

    assert metrics.eligible
    assert (
        ppo_holding_winner_digest(((504, 0.10, "long"), (72, 0.10, "short"))) == "short"
    )


def test_shared_cash_replay_matches_hand_calculated_dividend_and_interest() -> None:
    bars = 5
    shape = (bars, 1)
    close = np.full(shape, 100.0)
    dividend = np.zeros(shape)
    dividend[1, 0] = 1.0
    cash_rate = np.zeros(bars)
    cash_rate[1] = 0.05
    dataset = MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(bars) * np.timedelta64(1, "h"),
        features=np.zeros((bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((bars, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        dividend=dividend,
        cash_rate=cash_rate,
    )

    replay = run_shared_cash_replay(
        dataset,
        (ConstantIntentStrategy(PositionIntent.LONG),),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
        ohlc_drawdown_stress=True,
    )

    ledger = replay.ledger_evidence
    assert ledger is not None
    expected_dividend = 5.0
    expected_interest = (500.0 + expected_dividend) * 0.05 / 8_760
    expected_final_cash = 1_000.0 + expected_dividend + expected_interest
    assert sum(item.interval_dividend for item in ledger.intervals) == pytest.approx(
        expected_dividend
    )
    assert sum(
        item.interval_cash_interest for item in ledger.intervals
    ) == pytest.approx(expected_interest)
    assert ledger.final_cash == pytest.approx(expected_final_cash)
    assert replay.book.portfolio_value == pytest.approx(expected_final_cash)
    assert replay.book.quantities.tolist() == pytest.approx([0.0])
    assert ledger.terminal_exact_quantities == ("0",)
    assert ledger.active_order_remainders == ()
