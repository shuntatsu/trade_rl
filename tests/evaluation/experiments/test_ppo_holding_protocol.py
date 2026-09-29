from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
)
from trade_rl.data import publish_market_dataset_artifact
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
    inspect_study,
    run_baseline,
    run_experiment,
    verify_experiment,
)
from trade_rl.evaluation.experiments.protocols import (
    PPOHoldingDurationMetrics,
    ppo_holding_winner_digest,
)

_HORIZONS = (72, 168, 336, 504)
_NOW = datetime(2026, 9, 29, tzinfo=UTC)


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
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
    )


def _holding_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    root = tmp_path / "study"
    snapshot = create_study(
        root,
        dataset_root=dataset_root,
        research_question="Does a PPO minimum holding period improve net returns?",
        baseline_config=_holding_config(),
        ppo_seeds=(2, 5, 9, 13, 17),
        allowed_factors=(ControlledFactor.PPO_MINIMUM_HOLD,),
        max_experiments=len(_HORIZONS),
        n_bootstrap=32,
        bootstrap_seed=17,
        protocol="ppo_holding_duration_v1",
    )
    return root, dataset_root, snapshot


def _holding_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, dataset_root, _ = _holding_study(tmp_path, monkeypatch)
    baseline = run_baseline(root, dataset_root=dataset_root)
    assert baseline.baseline is not None
    return root, dataset_root, baseline


def _define_all_horizons(root: Path, dataset_root: Path, baseline) -> None:
    assert baseline.baseline is not None
    config = _holding_config()
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

    assert snapshot.plan.schema_version == "controlled_study_plan_v4"
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

    with pytest.raises(InvalidExperimentStateError, match="20% realized drawdown"):
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
