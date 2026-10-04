from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts.decision import (
    ExperimentDecision,
    ExperimentDecisionKind,
)
from trade_rl.evaluation.experiments.contracts.experiment import (
    ControlledFactor,
    ExperimentDefinition,
    ExperimentFailure,
)
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.contracts.study import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    PPO_HOLDING_DURATION_SELECTION_RULE,
    StudyFreeze,
    StudyOutcome,
    StudyPlan,
    StudyProtocol,
)
from trade_rl.evaluation.experiments.delta import (
    ControlledVerification,
    ControlledVerificationStatus,
)
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    ContractViolationError,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.rl.ppo import PPO_OBSERVATION_SCHEMA_V3


def resolved_config(*, ppo_seed: int = 2) -> ResolvedRunConfig:
    return ResolvedRunConfig(
        signal_name="signal",
        signal_index=0,
        feature_names=("signal", "volatility"),
        feature_indices=(0, 1),
        fit_symbol_names=("BTCUSDT", "ETHUSDT"),
        fit_symbol_indices=(0, 1),
        fit_cutoff="2026-01-01T00:00:00.000000000",
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=ppo_seed,
        evaluation_start="2026-02-01T00:00:00.000000000",
        evaluation_stop_exclusive="2026-03-01T00:00:00.000000000",
        gross_budget=0.5,
        initial_capital=100_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def study_plan(**overrides: object) -> StudyPlan:
    values: dict[str, object] = {
        "research_question": "Does one controlled feature family improve robustness?",
        "dataset_id": "a" * 64,
        "dataset_artifact_schema": "market_dataset_artifact_v3",
        "dataset_artifact_digest": "b" * 64,
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "baseline_config": resolved_config(),
        "ppo_seeds": (2, 5),
        "allowed_factors": (ControlledFactor.FEATURE_SET,),
        "max_experiments": 4,
        "n_bootstrap": 1_000,
        "bootstrap_seed": 7,
        "implementation_digest": "c" * 64,
        "runtime_environment_digest": "d" * 64,
    }
    values.update(overrides)
    return StudyPlan(**values)  # type: ignore[arg-type]


def test_resolved_run_config_is_frozen_and_digest_stable() -> None:
    config = resolved_config()
    assert config.digest == content_digest(config.to_payload())
    assert config.to_payload()["feature_names"] == ["signal", "volatility"]
    with pytest.raises(FrozenInstanceError):
        config.ppo_seed = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("make_contract", "error_type"),
    (
        (
            lambda: ExperimentDefinition(
                study_digest="a" * 64,
                sequence=1,
                hypothesis="test unknown schema rejection",
                baseline_evidence_digest="b" * 64,
                factor=ControlledFactor.FEATURE_SET,
                candidate_requested_config_digest="c" * 64,
                candidate_config=resolved_config(),
                schema_version="controlled_experiment_definition_v999",
            ),
            ContractViolationError,
        ),
        (
            lambda: ExperimentFailure(
                study_digest="a" * 64,
                experiment_digest="b" * 64,
                reason="test unknown schema rejection",
                recorded_by="test",
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
                schema_version="controlled_experiment_failure_v999",
            ),
            ContractViolationError,
        ),
        (
            lambda: ExperimentDecision(
                study_digest="a" * 64,
                experiment_digest="b" * 64,
                verification_digest="c" * 64,
                comparison_digest="d" * 64,
                decision=ExperimentDecisionKind.INCONCLUSIVE,
                rationale="test unknown schema rejection",
                decided_by="test",
                decided_at=datetime(2026, 1, 1, tzinfo=UTC),
                schema_version="controlled_experiment_decision_v999",
            ),
            ContractViolationError,
        ),
        (
            lambda: StudyFreeze(
                study_digest="a" * 64,
                experiment_decision_digests=("b" * 64,),
                outcome=StudyOutcome.NO_WINNER,
                selected_evidence_digest=None,
                selected_strategy=None,
                rationale="test unknown schema rejection",
                frozen_by="test",
                frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
                schema_version="controlled_study_freeze_v999",
            ),
            ContractViolationError,
        ),
        (
            lambda: ControlledVerification(
                study_digest="a" * 64,
                experiment_digest="b" * 64,
                baseline_evidence_digest="c" * 64,
                candidate_evidence_digest="d" * 64,
                factor=ControlledFactor.FEATURE_SET,
                status=ControlledVerificationStatus.INVALID,
                changed_paths=(),
                violations=("test unknown schema rejection",),
                schema_version="controlled_verification_v999",
            ),
            ArtifactIntegrityError,
        ),
    ),
)
def test_versioned_research_contracts_reject_unknown_schema_versions(
    make_contract,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="schema"):
        make_contract()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("research_question", "", "research_question"),
        ("ppo_seeds", (2,), "ppo_seeds"),
        ("ppo_seeds", (2, 2), "ppo_seeds"),
        ("ppo_seeds", (-1, 2), "ppo_seeds"),
        ("max_experiments", 0, "max_experiments"),
        ("max_experiments", 10_001, "max_experiments"),
        ("n_bootstrap", 0, "n_bootstrap"),
        ("bootstrap_seed", -1, "bootstrap_seed"),
        ("symbols", (), "symbols"),
        ("symbols", ("BTCUSDT", "BTCUSDT"), "symbols"),
        ("allowed_factors", (), "allowed_factors"),
        (
            "allowed_factors",
            (ControlledFactor.FEATURE_SET, ControlledFactor.FEATURE_SET),
            "allowed_factors",
        ),
    ],
)
def test_study_plan_rejects_invalid_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        study_plan(**{field: value})


def test_study_plan_requires_baseline_seed_to_equal_first_registered_seed() -> None:
    with pytest.raises(ContractViolationError, match="baseline.*ppo_seed"):
        study_plan(baseline_config=resolved_config(ppo_seed=5), ppo_seeds=(2, 5))


def test_ppo_holding_study_plan_requires_the_exact_preregistered_risk_profile() -> None:
    baseline = replace(
        resolved_config(ppo_seed=0),
        schema_version="resolved_run_config_v5",
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_minimum_hold_bars=0,
        ppo_settle_terminal_position=True,
        pretrade_risk_config=PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.1,
            max_turnover=1.0,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
    )

    with pytest.raises(
        ContractViolationError,
        match="violates the PPO holding-duration protocol",
    ):
        study_plan(
            research_question=(
                "Compare PPO hold durations.\n\n" + PPO_HOLDING_DURATION_SELECTION_RULE
            ),
            baseline_config=baseline,
            ppo_seeds=(0, 1, 2, 3, 4),
            allowed_factors=(ControlledFactor.PPO_MINIMUM_HOLD,),
            max_experiments=4,
            protocol=StudyProtocol.PPO_HOLDING_DURATION,
            schema_version="controlled_study_plan_v4",
        )


def test_study_plan_binds_fixed_candidate_and_control_rosters() -> None:
    plan = study_plan()
    assert plan.candidate_strategy_names == (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
    )
    assert plan.control_strategy_names == (
        "cash",
        "constant_long",
        "constant_short",
    )
    assert plan.candidate_strategy_names == CANDIDATE_STRATEGY_NAMES
    assert plan.control_strategy_names == CONTROL_STRATEGY_NAMES
    assert not set(plan.candidate_strategy_names) & set(plan.control_strategy_names)
    assert plan.digest == content_digest(plan.to_payload())


def test_study_plan_v2_binds_preregistered_final_window() -> None:
    plan = study_plan(
        final_evaluation_start="2026-03-01T00:00:00.000000000",
        final_evaluation_stop_exclusive="2026-04-01T00:00:00.000000000",
        schema_version="controlled_study_plan_v2",
    )

    assert plan.final_evaluation_start == "2026-03-01T00:00:00.000000000"
    assert plan.final_evaluation_stop_exclusive == "2026-04-01T00:00:00.000000000"
    payload = plan.to_payload()
    assert payload["schema_version"] == "controlled_study_plan_v2"
    assert payload["final_evaluation_start"] == plan.final_evaluation_start
    assert payload["final_evaluation_stop_exclusive"] == (
        plan.final_evaluation_stop_exclusive
    )
    assert plan.digest == content_digest(payload)


def test_study_plan_preserves_legacy_positional_schema_version() -> None:
    expected = study_plan(
        final_evaluation_start="2026-03-01T00:00:00.000000000",
        final_evaluation_stop_exclusive="2026-04-01T00:00:00.000000000",
        schema_version="controlled_study_plan_v2",
    )

    plan = StudyPlan(
        expected.research_question,
        expected.dataset_id,
        expected.dataset_artifact_schema,
        expected.dataset_artifact_digest,
        expected.symbols,
        expected.baseline_config,
        expected.ppo_seeds,
        expected.allowed_factors,
        expected.max_experiments,
        expected.n_bootstrap,
        expected.bootstrap_seed,
        expected.implementation_digest,
        expected.runtime_environment_digest,
        expected.final_evaluation_start,
        expected.final_evaluation_stop_exclusive,
        expected.research_context,
        expected.schema_version,
    )

    assert plan.schema_version == "controlled_study_plan_v2"
    assert plan.protocol is None


def test_study_plan_v1_forbids_final_window_fields() -> None:
    with pytest.raises(ContractViolationError, match="v1.*final|final.*v1"):
        study_plan(
            final_evaluation_start="2026-03-01T00:00:00.000000000",
            final_evaluation_stop_exclusive="2026-04-01T00:00:00.000000000",
        )


@pytest.mark.parametrize(
    ("start", "stop", "match"),
    (
        (None, "2026-04-01T00:00:00.000000000", "requires.*final|final.*requires"),
        ("2026-03-01T00:00:00.000000000", None, "requires.*final|final.*requires"),
        (
            "2026-02-28T23:00:00.000000000",
            "2026-04-01T00:00:00.000000000",
            "development|evaluation",
        ),
        (
            "2026-03-01T00:00:00.000000000",
            "2026-03-01T00:00:00.000000000",
            "strictly later|stop",
        ),
        (
            "2026-03-01T00:00:00",
            "2026-04-01T00:00:00.000000000",
            "canonical|nanosecond",
        ),
    ),
)
def test_study_plan_v2_rejects_invalid_final_window(
    start: str | None,
    stop: str | None,
    match: str,
) -> None:
    with pytest.raises(ContractViolationError, match=match):
        study_plan(
            final_evaluation_start=start,
            final_evaluation_stop_exclusive=stop,
            schema_version="controlled_study_plan_v2",
        )


def test_experiment_definition_is_digest_bound_and_sequence_positive() -> None:
    definition = ExperimentDefinition(
        study_digest="a" * 64,
        sequence=1,
        hypothesis="Adding volatility features improves development robustness.",
        baseline_evidence_digest="b" * 64,
        factor=ControlledFactor.FEATURE_SET,
        candidate_requested_config_digest="c" * 64,
        candidate_config=resolved_config(),
    )
    assert definition.digest == content_digest(definition.to_payload())
    with pytest.raises(ContractViolationError, match="sequence"):
        ExperimentDefinition(
            study_digest="a" * 64,
            sequence=0,
            hypothesis="hypothesis",
            baseline_evidence_digest="b" * 64,
            factor=ControlledFactor.FEATURE_SET,
            candidate_requested_config_digest="c" * 64,
            candidate_config=resolved_config(),
        )


def test_experiment_definition_rejects_empty_hypothesis() -> None:
    with pytest.raises(ContractViolationError, match="hypothesis"):
        ExperimentDefinition(
            study_digest="a" * 64,
            sequence=1,
            hypothesis="",
            baseline_evidence_digest="b" * 64,
            factor=ControlledFactor.FEATURE_SET,
            candidate_requested_config_digest="c" * 64,
            candidate_config=resolved_config(),
        )


def test_decision_is_immutable_and_requires_aware_time_and_non_empty_audit_fields() -> (
    None
):
    decision = ExperimentDecision(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        verification_digest="c" * 64,
        comparison_digest="d" * 64,
        decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
        rationale="Candidate improves the registered evidence without hidden drift.",
        decided_by="researcher",
        decided_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert decision.digest == content_digest(decision.to_payload())

    with pytest.raises(ContractViolationError, match="timezone-aware"):
        ExperimentDecision(
            study_digest="a" * 64,
            experiment_digest="b" * 64,
            verification_digest="c" * 64,
            comparison_digest="d" * 64,
            decision=ExperimentDecisionKind.KEEP_BASELINE,
            rationale="Keep the baseline.",
            decided_by="researcher",
            decided_at=datetime(2026, 9, 9),
        )
    with pytest.raises(ContractViolationError, match="rationale"):
        ExperimentDecision(
            study_digest="a" * 64,
            experiment_digest="b" * 64,
            verification_digest="c" * 64,
            comparison_digest="d" * 64,
            decision=ExperimentDecisionKind.INCONCLUSIVE,
            rationale="",
            decided_by="researcher",
            decided_at=datetime(2026, 9, 9, tzinfo=UTC),
        )


def test_failed_attempt_is_terminal_audit_record_with_aware_time() -> None:
    failure = ExperimentFailure(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        reason="candidate execution failed before a complete EvidenceSet was published",
        recorded_by="agent",
        recorded_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert failure.digest == content_digest(failure.to_payload())
    with pytest.raises(ContractViolationError, match="timezone-aware"):
        ExperimentFailure(
            study_digest="a" * 64,
            experiment_digest="b" * 64,
            reason="failure",
            recorded_by="agent",
            recorded_at=datetime(2026, 9, 9),
        )


def test_study_freeze_winner_requires_candidate_selection() -> None:
    freeze = StudyFreeze(
        study_digest="a" * 64,
        experiment_decision_digests=("b" * 64,),
        outcome=StudyOutcome.WINNER,
        selected_evidence_digest="c" * 64,
        selected_strategy="ridge24",
        rationale="The accepted candidate is the supported development winner.",
        frozen_by="researcher",
        frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert freeze.digest == content_digest(freeze.to_payload())

    with pytest.raises(ContractViolationError, match="candidate strategy"):
        StudyFreeze(
            study_digest="a" * 64,
            experiment_decision_digests=("b" * 64,),
            outcome=StudyOutcome.WINNER,
            selected_evidence_digest="c" * 64,
            selected_strategy="cash",
            rationale="invalid control winner",
            frozen_by="researcher",
            frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
        )


def test_study_freeze_no_winner_forbids_selected_fields() -> None:
    freeze = StudyFreeze(
        study_digest="a" * 64,
        experiment_decision_digests=("b" * 64,),
        outcome=StudyOutcome.NO_WINNER,
        selected_evidence_digest=None,
        selected_strategy=None,
        rationale="No candidate is supported by development evidence.",
        frozen_by="researcher",
        frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert freeze.selected_strategy is None

    with pytest.raises(ContractViolationError, match="NO_WINNER"):
        StudyFreeze(
            study_digest="a" * 64,
            experiment_decision_digests=("b" * 64,),
            outcome=StudyOutcome.NO_WINNER,
            selected_evidence_digest="c" * 64,
            selected_strategy="ridge24",
            rationale="invalid no-winner payload",
            frozen_by="researcher",
            frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
