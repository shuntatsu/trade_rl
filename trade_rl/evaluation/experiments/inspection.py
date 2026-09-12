"""Read-only reconstruction and tamper validation for controlled Studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from trade_rl.evaluation.experiments.codec import (
    _analysis_binding_from_payload,
    _AnalysisBinding,
    _as_string,
    _baseline_context_digest,
    _comparison_from_payload,
    _decision_from_payload,
    _definition_from_payload,
    _failure_from_payload,
    _freeze_from_payload,
    _semantic_payload_without_seed,
    _semantic_without_seed,
    _study_plan_from_payload,
    _verification_from_payload,
)
from trade_rl.evaluation.experiments.contracts import (
    ExperimentComparison,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentDefinition,
    ExperimentFailure,
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
    InvalidExperimentStateError,
)
from trade_rl.evaluation.experiments.evidence import (
    EvidenceSet,
    LoadedEvidenceSet,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore


@dataclass(frozen=True, slots=True)
class StudySnapshot:
    """Reconstructed public view of one immutable Study filesystem."""

    root: Path
    plan: StudyPlan
    baseline: EvidenceSet | None
    experiment_sequences: tuple[int, ...]
    terminal_sequences: tuple[int, ...]
    lineage_evidence_digests: tuple[str, ...]
    freeze: StudyFreeze | None

    @property
    def frozen(self) -> bool:
        return self.freeze is not None


@dataclass(frozen=True, slots=True)
class _ExperimentState:
    sequence: int
    definition: ExperimentDefinition
    candidate: LoadedEvidenceSet | None
    candidate_analysis: _AnalysisBinding | None
    verification: ControlledVerification | None
    comparison: ExperimentComparison | None
    decision: ExperimentDecision | None
    failure: ExperimentFailure | None


@dataclass(frozen=True, slots=True)
class _StudyState:
    plan: StudyPlan
    baseline: LoadedEvidenceSet | None
    baseline_analysis: _AnalysisBinding | None
    experiments: tuple[_ExperimentState, ...]
    lineage_evidence_digests: tuple[str, ...]
    terminal_sequences: tuple[int, ...]
    frozen: StudyFreeze | None

    def snapshot(self, root: Path) -> StudySnapshot:
        return StudySnapshot(
            root=root,
            plan=self.plan,
            baseline=None if self.baseline is None else self.baseline.evidence,
            experiment_sequences=tuple(item.sequence for item in self.experiments),
            terminal_sequences=self.terminal_sequences,
            lineage_evidence_digests=self.lineage_evidence_digests,
            freeze=self.frozen,
        )


def _load_evidence_node(
    store: StudyStore,
    relative: Path,
) -> tuple[LoadedEvidenceSet, _AnalysisBinding]:
    root = store.root / relative
    if root.is_symlink() or not root.is_dir():
        raise ArtifactIntegrityError(f"{relative} must be a regular directory")
    try:
        names = {item.name for item in root.iterdir()}
    except OSError as error:
        raise ArtifactIntegrityError(f"cannot read Study node: {relative}") from error
    if names != {"evidence", "analysis.json"}:
        raise ArtifactIntegrityError(
            f"{relative} must contain exactly evidence/ and analysis.json"
        )
    try:
        evidence = load_evidence_set(root / "evidence")
    except (ArtifactIntegrityError, ValueError) as error:
        raise ArtifactIntegrityError(f"{relative} EvidenceSet is invalid") from error
    analysis = _analysis_binding_from_payload(
        store.read_json(relative / "analysis.json"),
        expected_evidence_fingerprint=evidence.evidence.fingerprint,
    )
    return evidence, analysis


def _experiment_dir(sequence: int) -> Path:
    if isinstance(sequence, bool) or sequence <= 0 or sequence > 9_999:
        raise InvalidExperimentStateError("experiment sequence must be in 1..9999")
    return Path("experiments") / f"{sequence:04d}"


def _find_evidence(
    state: _StudyState,
    digest: str,
) -> tuple[LoadedEvidenceSet, _AnalysisBinding]:
    if (
        state.baseline is not None
        and state.baseline_analysis is not None
        and state.baseline.evidence.fingerprint == digest
    ):
        return state.baseline, state.baseline_analysis
    for experiment in state.experiments:
        if (
            experiment.candidate is not None
            and experiment.candidate_analysis is not None
            and experiment.candidate.evidence.fingerprint == digest
        ):
            return experiment.candidate, experiment.candidate_analysis
    raise ArtifactIntegrityError("referenced EvidenceSet is absent from Study")


def _read_plan(store: StudyStore) -> StudyPlan:
    if not (store.root / "plan.json").is_file():
        raise ArtifactIntegrityError("Study plan.json is missing")
    return _study_plan_from_payload(store.read_json("plan.json"))


def _reconstruct(store: StudyStore) -> _StudyState:
    root = store.root
    if root.is_symlink() or not root.is_dir():
        raise ArtifactIntegrityError("Study root must be a regular directory")
    allowed_root = {
        "plan.json",
        ".mutation.lock",
        "baseline",
        "experiments",
        "freeze.json",
    }
    try:
        root_names = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise ArtifactIntegrityError("Study root cannot be read") from error
    extras = root_names - allowed_root
    if extras:
        raise ArtifactIntegrityError(f"unexpected Study root entries: {sorted(extras)}")

    plan = _read_plan(store)
    baseline: LoadedEvidenceSet | None = None
    baseline_analysis: _AnalysisBinding | None = None
    evidence_by_digest: dict[str, LoadedEvidenceSet] = {}
    analysis_by_digest: dict[str, _AnalysisBinding] = {}
    lineage: list[str] = []
    baseline_root = root / "baseline"
    if baseline_root.exists() or baseline_root.is_symlink():
        baseline, baseline_analysis = _load_evidence_node(store, Path("baseline"))
        if _semantic_payload_without_seed(
            baseline.semantic_config
        ) != _semantic_without_seed(plan.baseline_config):
            raise ArtifactIntegrityError(
                "baseline EvidenceSet does not match frozen Study baseline config"
            )
        if baseline.evidence.research_context_digest != _baseline_context_digest(plan):
            raise ArtifactIntegrityError("baseline EvidenceSet context digest mismatch")
        evidence_by_digest[baseline.evidence.fingerprint] = baseline
        analysis_by_digest[baseline.evidence.fingerprint] = baseline_analysis
        lineage.append(baseline.evidence.fingerprint)

    experiment_states: list[_ExperimentState] = []
    experiments_root = root / "experiments"
    sequence_dirs: list[tuple[int, Path]] = []
    if experiments_root.exists() or experiments_root.is_symlink():
        if experiments_root.is_symlink() or not experiments_root.is_dir():
            raise ArtifactIntegrityError("experiments must be a regular directory")
        for entry in experiments_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                raise ArtifactIntegrityError("experiments may contain only directories")
            if len(entry.name) != 4 or not entry.name.isdigit() or entry.name == "0000":
                raise ArtifactIntegrityError(
                    "experiment directory sequence is malformed"
                )
            sequence_dirs.append((int(entry.name), entry))
        sequence_dirs.sort()
        actual = tuple(sequence for sequence, _ in sequence_dirs)
        expected = tuple(range(1, len(sequence_dirs) + 1))
        if actual != expected:
            raise ArtifactIntegrityError("experiment sequence must be contiguous")
        if len(sequence_dirs) > plan.max_experiments:
            raise ArtifactIntegrityError(
                "persisted experiment count exceeds Study budget"
            )
    if sequence_dirs and baseline is None:
        raise ArtifactIntegrityError("Study experiments require a published baseline")

    terminal: list[int] = []
    for sequence, entry in sequence_dirs:
        allowed = {
            "definition.json",
            "candidate",
            "verification.json",
            "comparison.json",
            "decision.json",
            "failure.json",
        }
        names = {item.name for item in entry.iterdir()}
        if names - allowed:
            raise ArtifactIntegrityError(
                f"experiment {sequence:04d} contains unexpected artifacts"
            )
        if "definition.json" not in names:
            raise ArtifactIntegrityError(
                f"experiment {sequence:04d} is missing definition.json"
            )
        base = _experiment_dir(sequence)
        definition = _definition_from_payload(store.read_json(base / "definition.json"))
        if definition.study_digest != plan.digest or definition.sequence != sequence:
            raise ArtifactIntegrityError("ExperimentDefinition identity mismatch")
        if definition.baseline_evidence_digest not in lineage:
            raise ArtifactIntegrityError(
                "ExperimentDefinition baseline is not reachable accepted lineage"
            )

        candidate: LoadedEvidenceSet | None = None
        candidate_analysis: _AnalysisBinding | None = None
        if "candidate" in names:
            candidate, candidate_analysis = _load_evidence_node(
                store, base / "candidate"
            )
            if _semantic_payload_without_seed(
                candidate.semantic_config
            ) != _semantic_without_seed(definition.candidate_config):
                raise ArtifactIntegrityError(
                    "candidate EvidenceSet differs from ExperimentDefinition"
                )
            if candidate.evidence.research_context_digest != definition.digest:
                raise ArtifactIntegrityError(
                    "candidate EvidenceSet context digest mismatch"
                )
            evidence_by_digest[candidate.evidence.fingerprint] = candidate
            analysis_by_digest[candidate.evidence.fingerprint] = candidate_analysis

        verification = (
            _verification_from_payload(store.read_json(base / "verification.json"))
            if "verification.json" in names
            else None
        )
        failure = (
            _failure_from_payload(store.read_json(base / "failure.json"))
            if "failure.json" in names
            else None
        )
        if candidate is not None and failure is not None:
            raise ArtifactIntegrityError(
                "FAILED experiment cannot contain candidate evidence"
            )
        if verification is not None:
            if candidate is None:
                raise ArtifactIntegrityError("verification requires candidate evidence")
            baseline_for_verification = evidence_by_digest.get(
                definition.baseline_evidence_digest
            )
            if baseline_for_verification is None:
                raise ArtifactIntegrityError(
                    "verification baseline evidence is missing"
                )
            expected_verification = verify_controlled_delta(
                plan=plan,
                definition=definition,
                baseline=baseline_for_verification,
                candidate=candidate,
            )
            if verification.to_payload() != expected_verification.to_payload():
                raise ArtifactIntegrityError("verification does not match evidence")

        comparison = (
            _comparison_from_payload(store.read_json(base / "comparison.json"))
            if "comparison.json" in names
            else None
        )
        if comparison is not None:
            if verification is None or candidate is None or candidate_analysis is None:
                raise ArtifactIntegrityError("comparison requires controlled evidence")
            if verification.status is not ControlledVerificationStatus.CONTROLLED:
                raise ArtifactIntegrityError(
                    "INVALID experiment cannot contain comparison"
                )
            baseline_for_comparison = evidence_by_digest.get(
                definition.baseline_evidence_digest
            )
            baseline_analysis_for_comparison = analysis_by_digest.get(
                definition.baseline_evidence_digest
            )
            if (
                baseline_for_comparison is None
                or baseline_analysis_for_comparison is None
            ):
                raise ArtifactIntegrityError("comparison baseline evidence is missing")
            comparison_payload = comparison.to_payload()
            persisted_factor_effect = comparison_payload.get("factor_effect")
            if not isinstance(persisted_factor_effect, dict):
                raise ArtifactIntegrityError("comparison factor_effect is malformed")
            factor_effect_schema = _as_string(
                persisted_factor_effect.get("schema_version"),
                field="factor-effect schema_version",
            )
            factor_effect = compare_evidence_sets(
                baseline_for_comparison.runs,
                candidate.runs,
                n_bootstrap=plan.n_bootstrap,
                bootstrap_seed=plan.bootstrap_seed,
                schema_version=factor_effect_schema,
            )
            factor_digest = _as_string(
                factor_effect.get("analysis_digest"),
                field="factor-effect analysis_digest",
            )
            expected_comparison = ExperimentComparison(
                study_digest=plan.digest,
                experiment_digest=definition.digest,
                baseline_evidence_digest=baseline_for_comparison.evidence.fingerprint,
                candidate_evidence_digest=candidate.evidence.fingerprint,
                verification_digest=verification.digest,
                baseline_analysis_digest=baseline_analysis_for_comparison.analysis_digest,
                candidate_analysis_digest=candidate_analysis.analysis_digest,
                factor_effect_digest=factor_digest,
                factor_effect=factor_effect,
            )
            if comparison.to_payload() != expected_comparison.to_payload():
                raise ArtifactIntegrityError("comparison does not match evidence")

        decision = (
            _decision_from_payload(store.read_json(base / "decision.json"))
            if "decision.json" in names
            else None
        )
        if decision is not None:
            if comparison is None or verification is None or candidate is None:
                raise ArtifactIntegrityError("decision requires controlled comparison")
            if (
                decision.study_digest != plan.digest
                or decision.experiment_digest != definition.digest
                or decision.verification_digest != verification.digest
                or decision.comparison_digest != comparison.digest
            ):
                raise ArtifactIntegrityError(
                    "decision digest references are inconsistent"
                )

        if failure is not None:
            if any(item is not None for item in (verification, comparison, decision)):
                raise ArtifactIntegrityError(
                    "FAILED experiment contains later-state artifacts"
                )
            if (
                failure.study_digest != plan.digest
                or failure.experiment_digest != definition.digest
            ):
                raise ArtifactIntegrityError(
                    "failure digest references are inconsistent"
                )
            terminal.append(sequence)
        elif (
            verification is not None
            and verification.status is ControlledVerificationStatus.INVALID
        ):
            if comparison is not None or decision is not None:
                raise ArtifactIntegrityError(
                    "INVALID experiment has post-verification artifacts"
                )
            terminal.append(sequence)
        elif decision is not None:
            terminal.append(sequence)
            if decision.decision is ExperimentDecisionKind.ACCEPT_CANDIDATE:
                if candidate is None:
                    raise ArtifactIntegrityError(
                        "accepted decision lacks candidate evidence"
                    )
                lineage.append(candidate.evidence.fingerprint)

        experiment_states.append(
            _ExperimentState(
                sequence=sequence,
                definition=definition,
                candidate=candidate,
                candidate_analysis=candidate_analysis,
                verification=verification,
                comparison=comparison,
                decision=decision,
                failure=failure,
            )
        )

    frozen = (
        _freeze_from_payload(store.read_json("freeze.json"))
        if "freeze.json" in root_names
        else None
    )
    if frozen is not None:
        if frozen.study_digest != plan.digest:
            raise ArtifactIntegrityError("freeze Study digest mismatch")
        decision_digests = tuple(
            item.decision.digest
            for item in experiment_states
            if item.decision is not None
        )
        if frozen.experiment_decision_digests != decision_digests:
            raise ArtifactIntegrityError("freeze decision digest lineage mismatch")
        all_sequences = tuple(item.sequence for item in experiment_states)
        if tuple(terminal) != all_sequences:
            raise ArtifactIntegrityError("frozen Study contains nonterminal Experiment")
        if frozen.outcome is StudyOutcome.WINNER:
            accepted_candidates = {
                item.candidate.evidence.fingerprint
                for item in experiment_states
                if item.decision is not None
                and item.decision.decision is ExperimentDecisionKind.ACCEPT_CANDIDATE
                and item.candidate is not None
            }
            if frozen.selected_evidence_digest not in accepted_candidates:
                raise ArtifactIntegrityError(
                    "WINNER freeze evidence is not an ACCEPT_CANDIDATE lineage node"
                )

    return _StudyState(
        plan=plan,
        baseline=baseline,
        baseline_analysis=baseline_analysis,
        experiments=tuple(experiment_states),
        lineage_evidence_digests=tuple(lineage),
        terminal_sequences=tuple(terminal),
        frozen=frozen,
    )


def _experiment_state(state: _StudyState, sequence: int) -> _ExperimentState:
    for experiment in state.experiments:
        if experiment.sequence == sequence:
            return experiment
    raise InvalidExperimentStateError(
        f"experiment {sequence:04d} definition does not exist"
    )


def inspect_study(root: str | Path) -> StudySnapshot:
    """Reconstruct and validate a Study from immutable on-disk evidence."""

    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError(
            "Study root must already exist as a regular directory"
        )
    store = StudyStore(path)
    with store.mutation_lock():
        return _reconstruct(store).snapshot(store.root)


__all__ = ["StudySnapshot", "inspect_study"]
