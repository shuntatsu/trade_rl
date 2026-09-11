"""Append-only mutation orchestration for controlled development research."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    MarketDataset,
    PublishedDatasetArtifact,
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments.analysis import (
    analyze_evidence_set,
    compare_evidence_sets,
)
from trade_rl.evaluation.experiments.codec import (
    _analysis_binding,
    _as_string,
    _baseline_context_digest,
    _candidate_config_from_resolved,
    _candidate_config_payload,
    _resolved_contract,
    _semantic_payload_without_seed,
    _semantic_without_seed,
)
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ExperimentComparison,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentDefinition,
    ExperimentFailure,
    ResolvedRunConfig,
    StudyFreeze,
    StudyOutcome,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import (
    ControlledVerification,
    ControlledVerificationStatus,
    verify_controlled_delta,
)
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    ContractViolationError,
    ExperimentBudgetExceededError,
    InvalidExperimentStateError,
    StudyFrozenError,
)
from trade_rl.evaluation.experiments.evidence import (
    EvidenceSet,
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.inspection import (
    StudySnapshot,
    _experiment_dir,
    _experiment_state,
    _find_evidence,
    _reconstruct,
    _StudyState,
    inspect_study,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    build_candidate_run_provenance,
    resolve_candidate_run_spec,
)

_FACTOR_EFFECT_SCHEMA = "controlled_evidence_comparison_v1"
_STUDY_FIXED_FIELDS = (
    "fit_cutoff",
    "evaluation_start",
    "evaluation_stop_exclusive",
    "initial_capital",
    "execution_overlay",
)


def _validate_fixed_fields(plan: StudyPlan, config: ResolvedRunConfig) -> None:
    baseline = _semantic_without_seed(plan.baseline_config)
    candidate = _semantic_without_seed(config)
    for field in _STUDY_FIXED_FIELDS:
        if content_digest({"value": baseline[field]}) != content_digest(
            {"value": candidate[field]}
        ):
            raise ContractViolationError(
                f"candidate changes Study-fixed resolved field: {field}"
            )


def _validate_dataset_root(
    dataset_root: str | Path,
    plan: StudyPlan,
) -> tuple[MarketDataset, PublishedDatasetArtifact]:
    try:
        artifact = inspect_published_market_dataset_artifact(dataset_root)
        dataset = load_market_dataset_artifact(dataset_root)
    except ValueError as error:
        raise ArtifactIntegrityError(
            "Study dataset artifact cannot be trusted"
        ) from error
    if dataset.dataset_id != plan.dataset_id:
        raise ArtifactIntegrityError("Study dataset id differs from frozen plan")
    if (
        artifact.schema_version != plan.dataset_artifact_schema
        or artifact.artifact_digest != plan.dataset_artifact_digest
    ):
        raise ArtifactIntegrityError("Study dataset artifact differs from frozen plan")
    if tuple(dataset.symbols) != plan.symbols:
        raise ArtifactIntegrityError(
            "Study dataset symbol roster differs from frozen plan"
        )
    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != plan.implementation_digest:
        raise ArtifactIntegrityError(
            "current implementation provenance differs from frozen Study plan"
        )
    if provenance.get("runtime_environment_digest") != plan.runtime_environment_digest:
        raise ArtifactIntegrityError(
            "current runtime provenance differs from frozen Study plan"
        )
    return dataset, artifact


def _build_evidence_node(
    staging: Path,
    *,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
    research_context_digest: str,
    expected_semantic: dict[str, object],
) -> None:
    child_store = StudyStore(staging)
    execute_evidence_set(
        store=child_store,
        target=Path("evidence"),
        dataset_root=dataset_root,
        plan=plan,
        config=config,
        research_context_digest=research_context_digest,
    )
    loaded = load_evidence_set(staging / "evidence")
    if _semantic_payload_without_seed(loaded.semantic_config) != expected_semantic:
        raise ArtifactIntegrityError(
            "generated EvidenceSet does not match frozen resolved configuration"
        )
    analysis = analyze_evidence_set(
        loaded.runs,
        n_bootstrap=plan.n_bootstrap,
        bootstrap_seed=plan.bootstrap_seed,
    )
    child_store.publish_json_once(
        "analysis.json",
        _analysis_binding(
            evidence_fingerprint=loaded.evidence.fingerprint,
            analysis=analysis,
        ),
    )
    lock_path = staging / ".mutation.lock"
    if lock_path.exists():
        lock_path.unlink()


def _assert_mutable(state: _StudyState) -> None:
    if state.frozen is not None:
        raise StudyFrozenError("Study is frozen and cannot be mutated")


def create_study(
    root: str | Path,
    *,
    dataset_root: str | Path,
    research_question: str,
    baseline_config: CandidateRunConfig,
    ppo_seeds: tuple[int, ...],
    allowed_factors: tuple[ControlledFactor, ...],
    max_experiments: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> StudySnapshot:
    """Create one immutable Study plan without executing development evidence."""

    store = StudyStore(root)
    with store.mutation_lock():
        names = {entry.name for entry in store.root.iterdir()}
        if names != {".mutation.lock"}:
            raise InvalidExperimentStateError("Study root already contains artifacts")
        try:
            artifact = inspect_published_market_dataset_artifact(dataset_root)
            dataset = load_market_dataset_artifact(dataset_root)
        except ValueError as error:
            raise ArtifactIntegrityError(
                "dataset artifact cannot be trusted"
            ) from error
        spec = resolve_candidate_run_spec(
            dataset,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            config=baseline_config,
        )
        resolved = _resolved_contract(spec)
        provenance = build_candidate_run_provenance()
        plan = StudyPlan(
            research_question=research_question,
            dataset_id=dataset.dataset_id,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            symbols=tuple(dataset.symbols),
            baseline_config=resolved,
            ppo_seeds=ppo_seeds,
            allowed_factors=allowed_factors,
            max_experiments=max_experiments,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            implementation_digest=_as_string(
                provenance.get("implementation_digest"),
                field="implementation_digest",
            ),
            runtime_environment_digest=_as_string(
                provenance.get("runtime_environment_digest"),
                field="runtime_environment_digest",
            ),
        )
        store.publish_json_once("plan.json", plan.to_payload())
        return _reconstruct(store).snapshot(store.root)


def run_baseline(
    root: str | Path,
    *,
    dataset_root: str | Path,
) -> StudySnapshot:
    """Execute and atomically publish the frozen baseline EvidenceSet and analysis."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        if state.baseline is not None or (store.root / "baseline").exists():
            raise InvalidExperimentStateError("Study baseline is already published")
        _validate_dataset_root(dataset_root, state.plan)
        config = _candidate_config_from_resolved(state.plan.baseline_config)
        store.publish_directory_once(
            "baseline",
            lambda staging: _build_evidence_node(
                staging,
                dataset_root=dataset_root,
                plan=state.plan,
                config=config,
                research_context_digest=_baseline_context_digest(state.plan),
                expected_semantic=_semantic_without_seed(state.plan.baseline_config),
            ),
        )
        return _reconstruct(store).snapshot(store.root)


def define_experiment(
    root: str | Path,
    *,
    dataset_root: str | Path,
    hypothesis: str,
    factor: ControlledFactor,
    candidate_config: CandidateRunConfig,
    baseline_evidence_digest: str,
) -> ExperimentDefinition:
    """Pre-register one budgeted Experiment before any candidate execution."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        if state.baseline is None:
            raise InvalidExperimentStateError(
                "baseline must exist before Experiment definition"
            )
        if baseline_evidence_digest not in state.lineage_evidence_digests:
            raise InvalidExperimentStateError(
                "baseline EvidenceSet is not reachable accepted lineage"
            )
        if factor not in state.plan.allowed_factors:
            raise ContractViolationError(
                "controlled factor is not allowed by StudyPlan"
            )
        sequence = len(state.experiments) + 1
        if sequence > state.plan.max_experiments:
            raise ExperimentBudgetExceededError("Study Experiment budget is exhausted")
        if candidate_config.ppo_seed != state.plan.ppo_seeds[0]:
            raise ContractViolationError(
                "candidate config ppo_seed must equal first frozen Study seed"
            )
        dataset, artifact = _validate_dataset_root(dataset_root, state.plan)
        spec = resolve_candidate_run_spec(
            dataset,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            config=candidate_config,
        )
        resolved = _resolved_contract(spec)
        _validate_fixed_fields(state.plan, resolved)
        definition = ExperimentDefinition(
            study_digest=state.plan.digest,
            sequence=sequence,
            hypothesis=hypothesis,
            baseline_evidence_digest=baseline_evidence_digest,
            factor=factor,
            candidate_requested_config_digest=content_digest(
                _candidate_config_payload(candidate_config)
            ),
            candidate_config=resolved,
        )
        store.publish_json_once(
            _experiment_dir(sequence) / "definition.json",
            definition.to_payload(),
        )
        return definition


def run_experiment(
    root: str | Path,
    sequence: int,
    *,
    dataset_root: str | Path,
) -> EvidenceSet:
    """Execute and atomically publish the candidate EvidenceSet for one definition."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        experiment = _experiment_state(state, sequence)
        if experiment.failure is not None:
            raise InvalidExperimentStateError("FAILED experiment cannot be executed")
        if (
            experiment.candidate is not None
            or (store.root / _experiment_dir(sequence) / "candidate").exists()
        ):
            raise InvalidExperimentStateError("candidate evidence is already published")
        _validate_dataset_root(dataset_root, state.plan)
        config = _candidate_config_from_resolved(experiment.definition.candidate_config)
        target = _experiment_dir(sequence) / "candidate"
        store.publish_directory_once(
            target,
            lambda staging: _build_evidence_node(
                staging,
                dataset_root=dataset_root,
                plan=state.plan,
                config=config,
                research_context_digest=experiment.definition.digest,
                expected_semantic=_semantic_without_seed(
                    experiment.definition.candidate_config
                ),
            ),
        )
        rebuilt = _reconstruct(store)
        current = _experiment_state(rebuilt, sequence)
        assert current.candidate is not None
        return current.candidate.evidence


def verify_experiment(
    root: str | Path,
    sequence: int,
) -> ControlledVerification:
    """Verify the declared one-factor delta and publish CONTROLLED or INVALID once."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        experiment = _experiment_state(state, sequence)
        if experiment.failure is not None:
            raise InvalidExperimentStateError("FAILED experiment cannot be verified")
        if experiment.candidate is None:
            raise InvalidExperimentStateError(
                "candidate evidence is required before verification"
            )
        if experiment.verification is not None:
            raise InvalidExperimentStateError("verification is already published")
        baseline, _ = _find_evidence(
            state, experiment.definition.baseline_evidence_digest
        )
        verification = verify_controlled_delta(
            plan=state.plan,
            definition=experiment.definition,
            baseline=baseline,
            candidate=experiment.candidate,
        )
        store.publish_json_once(
            _experiment_dir(sequence) / "verification.json",
            verification.to_payload(),
        )
        return verification


def compare_experiment(
    root: str | Path,
    sequence: int,
) -> ExperimentComparison:
    """Publish one immutable factor-effect comparison after CONTROLLED verification."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        experiment = _experiment_state(state, sequence)
        if experiment.failure is not None:
            raise InvalidExperimentStateError("FAILED experiment cannot be compared")
        if experiment.verification is None:
            raise InvalidExperimentStateError(
                "verification is required before comparison"
            )
        if (
            experiment.verification.status
            is not ControlledVerificationStatus.CONTROLLED
        ):
            raise InvalidExperimentStateError(
                "comparison requires CONTROLLED verification; INVALID is terminal"
            )
        if experiment.comparison is not None:
            raise InvalidExperimentStateError("comparison is already published")
        if experiment.candidate is None or experiment.candidate_analysis is None:
            raise ArtifactIntegrityError(
                "CONTROLLED experiment lacks candidate evidence"
            )
        baseline, baseline_analysis = _find_evidence(
            state, experiment.definition.baseline_evidence_digest
        )
        factor_effect = compare_evidence_sets(
            baseline.runs,
            experiment.candidate.runs,
            n_bootstrap=state.plan.n_bootstrap,
            bootstrap_seed=state.plan.bootstrap_seed,
        )
        if factor_effect.get("schema_version") != _FACTOR_EFFECT_SCHEMA:
            raise ArtifactIntegrityError("unsupported factor-effect comparison schema")
        factor_digest = _as_string(
            factor_effect.get("analysis_digest"),
            field="factor-effect analysis_digest",
        )
        body = dict(factor_effect)
        body.pop("analysis_digest", None)
        if content_digest(body) != factor_digest:
            raise ArtifactIntegrityError("factor-effect comparison digest mismatch")
        comparison = ExperimentComparison(
            study_digest=state.plan.digest,
            experiment_digest=experiment.definition.digest,
            baseline_evidence_digest=baseline.evidence.fingerprint,
            candidate_evidence_digest=experiment.candidate.evidence.fingerprint,
            verification_digest=experiment.verification.digest,
            baseline_analysis_digest=baseline_analysis.analysis_digest,
            candidate_analysis_digest=experiment.candidate_analysis.analysis_digest,
            factor_effect_digest=factor_digest,
            factor_effect=factor_effect,
        )
        store.publish_json_once(
            _experiment_dir(sequence) / "comparison.json",
            comparison.to_payload(),
        )
        return comparison


def decide_experiment(
    root: str | Path,
    sequence: int,
    *,
    decision: ExperimentDecisionKind,
    rationale: str,
    decided_by: str,
    decided_at: datetime,
) -> ExperimentDecision:
    """Publish one irreversible research decision after comparison evidence exists."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        experiment = _experiment_state(state, sequence)
        if experiment.failure is not None:
            raise InvalidExperimentStateError("FAILED experiment cannot be decided")
        if experiment.verification is None or experiment.comparison is None:
            raise InvalidExperimentStateError("comparison is required before decision")
        if experiment.decision is not None:
            raise InvalidExperimentStateError("decision is already published")
        resolved = ExperimentDecision(
            study_digest=state.plan.digest,
            experiment_digest=experiment.definition.digest,
            verification_digest=experiment.verification.digest,
            comparison_digest=experiment.comparison.digest,
            decision=decision,
            rationale=rationale,
            decided_by=decided_by,
            decided_at=decided_at,
        )
        store.publish_json_once(
            _experiment_dir(sequence) / "decision.json",
            resolved.to_payload(),
        )
        return resolved


def record_experiment_failure(
    root: str | Path,
    sequence: int,
    *,
    reason: str,
    recorded_by: str,
    recorded_at: datetime,
) -> ExperimentFailure:
    """Record terminal operational failure before complete candidate evidence exists."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        experiment = _experiment_state(state, sequence)
        if experiment.candidate is not None:
            raise InvalidExperimentStateError(
                "complete candidate evidence must proceed through verification"
            )
        if experiment.failure is not None:
            raise InvalidExperimentStateError("failure is already published")
        if any(
            item is not None
            for item in (
                experiment.verification,
                experiment.comparison,
                experiment.decision,
            )
        ):
            raise InvalidExperimentStateError("later experiment state forbids FAILED")
        failure = ExperimentFailure(
            study_digest=state.plan.digest,
            experiment_digest=experiment.definition.digest,
            reason=reason,
            recorded_by=recorded_by,
            recorded_at=recorded_at,
        )
        store.publish_json_once(
            _experiment_dir(sequence) / "failure.json",
            failure.to_payload(),
        )
        return failure


def freeze_study(
    root: str | Path,
    *,
    outcome: StudyOutcome,
    selected_evidence_digest: str | None = None,
    selected_strategy: str | None = None,
    rationale: str,
    frozen_by: str,
    frozen_at: datetime,
) -> StudyFreeze:
    """Publish the one-shot terminal development Study outcome."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        _assert_mutable(state)
        if state.baseline is None:
            raise InvalidExperimentStateError(
                "Study baseline is required before freeze"
            )
        all_sequences = tuple(item.sequence for item in state.experiments)
        if state.terminal_sequences != all_sequences:
            raise InvalidExperimentStateError(
                "Study contains nonterminal Experiment and cannot freeze"
            )
        decision_digests = tuple(
            item.decision.digest
            for item in state.experiments
            if item.decision is not None
        )
        frozen = StudyFreeze(
            study_digest=state.plan.digest,
            experiment_decision_digests=decision_digests,
            outcome=outcome,
            selected_evidence_digest=selected_evidence_digest,
            selected_strategy=selected_strategy,
            rationale=rationale,
            frozen_by=frozen_by,
            frozen_at=frozen_at,
        )
        if outcome is StudyOutcome.WINNER:
            accepted_candidates = {
                item.candidate.evidence.fingerprint
                for item in state.experiments
                if item.decision is not None
                and item.decision.decision is ExperimentDecisionKind.ACCEPT_CANDIDATE
                and item.candidate is not None
            }
            if frozen.selected_evidence_digest not in accepted_candidates:
                raise InvalidExperimentStateError(
                    "WINNER selected evidence must come from an ACCEPT_CANDIDATE decision"
                )
        store.publish_json_once("freeze.json", frozen.to_payload())
        rebuilt = _reconstruct(store)
        if rebuilt.frozen is None:
            raise ArtifactIntegrityError("freeze publication did not reconstruct")
        return rebuilt.frozen


__all__ = [
    "StudySnapshot",
    "compare_experiment",
    "create_study",
    "decide_experiment",
    "define_experiment",
    "freeze_study",
    "inspect_study",
    "record_experiment_failure",
    "run_baseline",
    "run_experiment",
    "verify_experiment",
]
