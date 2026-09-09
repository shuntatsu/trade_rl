from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config, _dataset, _fake_execute
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.evaluation.experiments import (
    ArtifactIntegrityError,
    ControlledFactor,
    ControlledVerificationStatus,
    ExperimentBudgetExceededError,
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


def _created_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_experiments: int = 3,
):
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    root = tmp_path / "study"
    snapshot = create_study(
        root,
        dataset_root=dataset_root,
        research_question="Does one preregistered factor improve development evidence?",
        baseline_config=_config(),
        ppo_seeds=(2, 5),
        allowed_factors=tuple(ControlledFactor),
        max_experiments=max_experiments,
        n_bootstrap=32,
        bootstrap_seed=17,
    )
    return root, dataset_root, snapshot


def _with_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_experiments: int = 3,
):
    root, dataset_root, _ = _created_study(
        tmp_path,
        monkeypatch,
        max_experiments=max_experiments,
    )
    snapshot = run_baseline(root, dataset_root=dataset_root)
    assert snapshot.baseline is not None
    return root, dataset_root, snapshot


def test_create_study_pre_resolves_and_publishes_only_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import workflow as workflow_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    publish_market_dataset_artifact(dataset_root, dataset)
    seen: list[int] = []
    original = workflow_module.resolve_candidate_run_spec

    def recording_resolve(*args, **kwargs):
        spec = original(*args, **kwargs)
        seen.append(spec.config.ppo_seed)
        return spec

    monkeypatch.setattr(
        workflow_module, "resolve_candidate_run_spec", recording_resolve
    )
    root = tmp_path / "study"
    snapshot = create_study(
        root,
        dataset_root=dataset_root,
        research_question="Pre-resolve the frozen Study baseline.",
        baseline_config=_config(),
        ppo_seeds=(2, 5),
        allowed_factors=(ControlledFactor.PPO_TRAINING_BUDGET,),
        max_experiments=2,
        n_bootstrap=16,
        bootstrap_seed=3,
    )

    assert seen == [2]
    assert snapshot.baseline is None
    assert snapshot.experiment_sequences == ()
    assert snapshot.lineage_evidence_digests == ()
    assert {entry.name for entry in root.iterdir()} == {"plan.json", ".mutation.lock"}
    plan_payload = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    assert plan_payload["dataset_id"] == dataset.dataset_id
    assert plan_payload["ppo_seeds"] == [2, 5]


def test_baseline_is_published_once_with_separate_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, _ = _created_study(tmp_path, monkeypatch)

    snapshot = run_baseline(root, dataset_root=dataset_root)

    assert snapshot.baseline is not None
    assert snapshot.lineage_evidence_digests == (snapshot.baseline.fingerprint,)
    assert (root / "baseline" / "evidence" / "manifest.json").is_file()
    assert (root / "baseline" / "analysis.json").is_file()
    assert {entry.name for entry in (root / "baseline").iterdir()} == {
        "evidence",
        "analysis.json",
    }

    with pytest.raises(InvalidExperimentStateError, match="baseline"):
        run_baseline(root, dataset_root=dataset_root)


def test_definition_must_precede_execution_and_consumes_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    with pytest.raises(InvalidExperimentStateError, match="definition"):
        run_experiment(root, 1, dataset_root=dataset_root)

    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="More PPO training changes only PPO evidence.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )

    assert definition.sequence == 1
    experiment_root = root / "experiments" / "0001"
    assert (experiment_root / "definition.json").is_file()
    assert not (experiment_root / "candidate").exists()
    assert inspect_study(root).experiment_sequences == (1,)


def test_sequence_budget_and_gap_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(
        tmp_path,
        monkeypatch,
        max_experiments=2,
    )
    assert snapshot.baseline is not None

    for timesteps in (64, 96):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis=f"Registered attempt for {timesteps} PPO steps.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=timesteps),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        )

    assert inspect_study(root).experiment_sequences == (1, 2)
    with pytest.raises(ExperimentBudgetExceededError):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="Budget must be exhausted.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=128),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        )

    gap_root = tmp_path / "gap-study"
    root.rename(gap_root)
    (gap_root / "experiments" / "0002").rename(gap_root / "experiments" / "0003")
    with pytest.raises(ArtifactIntegrityError, match="contiguous|sequence"):
        inspect_study(gap_root)


def test_invalid_verification_is_terminal_visible_and_budgeted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(
        tmp_path,
        monkeypatch,
        max_experiments=1,
    )
    assert snapshot.baseline is not None
    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Deliberately declare the wrong factor to prove INVALID visibility.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), rule_entry_threshold=0.20),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    run_experiment(root, 1, dataset_root=dataset_root)

    verification = verify_experiment(root, 1)

    assert verification.status is ControlledVerificationStatus.INVALID
    assert (root / "experiments" / "0001" / "verification.json").is_file()
    with pytest.raises(InvalidExperimentStateError, match="CONTROLLED|INVALID"):
        compare_experiment(root, 1)
    with pytest.raises(ExperimentBudgetExceededError):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="INVALID still consumes the attempt.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=96),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        )


def test_operational_integrity_failure_does_not_publish_invalid_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import workflow as workflow_module

    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Operational failures are not research INVALID evidence.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    run_experiment(root, 1, dataset_root=dataset_root)

    def fail_integrity(**kwargs):
        del kwargs
        raise ArtifactIntegrityError("synthetic integrity failure")

    monkeypatch.setattr(workflow_module, "verify_controlled_delta", fail_integrity)
    with pytest.raises(ArtifactIntegrityError, match="synthetic integrity failure"):
        verify_experiment(root, 1)
    assert not (root / "experiments" / "0001" / "verification.json").exists()


def test_comparison_and_decision_ordering_are_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="More PPO training is one controlled factor.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    run_experiment(root, 1, dataset_root=dataset_root)

    with pytest.raises(InvalidExperimentStateError, match="verification"):
        compare_experiment(root, 1)

    verification = verify_experiment(root, 1)
    assert verification.status is ControlledVerificationStatus.CONTROLLED
    with pytest.raises(InvalidExperimentStateError, match="comparison"):
        decide_experiment(
            root,
            1,
            decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
            rationale="Too early.",
            decided_by="test",
            decided_at=datetime(2026, 9, 9, tzinfo=UTC),
        )

    comparison = compare_experiment(root, 1)
    assert comparison.verification_digest == verification.digest
    decision = decide_experiment(
        root,
        1,
        decision=ExperimentDecisionKind.ACCEPT_CANDIDATE,
        rationale="The registered evidence supports continuing this lineage.",
        decided_by="researcher",
        decided_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    assert decision.comparison_digest == comparison.digest
    with pytest.raises(InvalidExperimentStateError, match="decision"):
        decide_experiment(
            root,
            1,
            decision=ExperimentDecisionKind.KEEP_BASELINE,
            rationale="Cannot overwrite.",
            decided_by="researcher",
            decided_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
