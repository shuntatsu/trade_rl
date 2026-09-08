"""Append-only workflow state machine for controlled development Studies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments.analysis import (
    analyze_evidence_set,
    compare_evidence_sets,
)
from trade_rl.evaluation.experiments.contracts import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    ControlledFactor,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentDefinition,
    ResolvedRunConfig,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import (
    ControlledVerification,
    ControlledVerificationStatus,
    verify_controlled_delta,
)
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    ExperimentBudgetExceededError,
    InvalidExperimentStateError,
)
from trade_rl.evaluation.experiments.evidence import (
    EvidenceSet,
    LoadedEvidenceSet,
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    ResolvedCandidateRunSpec,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance

_STUDY_COMPARISON_SCHEMA = "controlled_experiment_comparison_v1"


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    """One immutable comparison artifact bound to controlled verification."""

    study_digest: str
    experiment_digest: str
    verification_digest: str
    baseline_evidence_digest: str
    candidate_evidence_digest: str
    evidence_comparison: dict[str, object]
    schema_version: str = _STUDY_COMPARISON_SCHEMA

    def __post_init__(self) -> None:
        for value, field in (
            (self.study_digest, "study_digest"),
            (self.experiment_digest, "experiment_digest"),
            (self.verification_digest, "verification_digest"),
            (self.baseline_evidence_digest, "baseline_evidence_digest"),
            (self.candidate_evidence_digest, "candidate_evidence_digest"),
        ):
            try:
                require_sha256(value, field=field)
            except ValueError as error:
                raise ArtifactIntegrityError(str(error)) from error
        if self.schema_version != _STUDY_COMPARISON_SCHEMA:
            raise ArtifactIntegrityError("unsupported ExperimentComparison schema")
        _verify_derived_payload(self.evidence_comparison, field="evidence comparison")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_digest": self.experiment_digest,
            "verification_digest": self.verification_digest,
            "baseline_evidence_digest": self.baseline_evidence_digest,
            "candidate_evidence_digest": self.candidate_evidence_digest,
            "evidence_comparison": self.evidence_comparison,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class StudySnapshot:
    """Validated disk-reconstructed Study state exposed to callers."""

    root: Path
    plan: StudyPlan
    baseline: EvidenceSet | None
    experiment_sequences: tuple[int, ...]
    lineage_evidence_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExperimentState:
    definition: ExperimentDefinition
    candidate: LoadedEvidenceSet | None
    verification: ControlledVerification | None
    comparison: ExperimentComparison | None
    decision: ExperimentDecision | None


@dataclass(frozen=True, slots=True)
class _StudyState:
    snapshot: StudySnapshot
    baseline_loaded: LoadedEvidenceSet | None
    evidence_by_digest: Mapping[str, LoadedEvidenceSet]
    experiments: Mapping[int, _ExperimentState]


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArtifactIntegrityError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ArtifactIntegrityError(f"{field} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(f"{field} must be a number")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ArtifactIntegrityError(f"{field} must be finite")
    return resolved


def _texts(value: object, *, field: str) -> tuple[str, ...]:
    raw = _list(value, field=field)
    if any(not isinstance(item, str) or not item for item in raw):
        raise ArtifactIntegrityError(f"{field} must be a non-empty string array")
    return tuple(cast(str, item) for item in raw)


def _integers(value: object, *, field: str) -> tuple[int, ...]:
    raw = _list(value, field=field)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ArtifactIntegrityError(f"{field} must be an integer array")
    return tuple(cast(int, item) for item in raw)


def _exact_keys(raw: Mapping[str, object], expected: set[str], *, field: str) -> None:
    if set(raw) != expected:
        raise ArtifactIntegrityError(f"{field} keys do not match the supported schema")


def _datetime(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    try:
        resolved = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactIntegrityError(f"{field} must be an ISO datetime") from error
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ArtifactIntegrityError(f"{field} must be timezone-aware")
    return resolved


def _resolved_from_payload(value: object) -> ResolvedRunConfig:
    raw = _mapping(value, field="resolved run config")
    expected = {
        "schema_version",
        "signal_name",
        "signal_index",
        "feature_names",
        "feature_indices",
        "fit_symbol_names",
        "fit_symbol_indices",
        "fit_cutoff",
        "rule_entry_threshold",
        "rule_exit_threshold",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "ppo_total_timesteps",
        "ppo_seed",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "gross_budget",
        "initial_capital",
        "execution_overlay",
    }
    _exact_keys(raw, expected, field="resolved run config")
    try:
        return ResolvedRunConfig(
            signal_name=_text(raw.get("signal_name"), field="signal_name"),
            signal_index=_integer(raw.get("signal_index"), field="signal_index"),
            feature_names=_texts(raw.get("feature_names"), field="feature_names"),
            feature_indices=_integers(
                raw.get("feature_indices"), field="feature_indices"
            ),
            fit_symbol_names=_texts(
                raw.get("fit_symbol_names"), field="fit_symbol_names"
            ),
            fit_symbol_indices=_integers(
                raw.get("fit_symbol_indices"), field="fit_symbol_indices"
            ),
            fit_cutoff=_text(raw.get("fit_cutoff"), field="fit_cutoff"),
            rule_entry_threshold=_number(
                raw.get("rule_entry_threshold"), field="rule_entry_threshold"
            ),
            rule_exit_threshold=_number(
                raw.get("rule_exit_threshold"), field="rule_exit_threshold"
            ),
            forecast_entry_threshold=_number(
                raw.get("forecast_entry_threshold"), field="forecast_entry_threshold"
            ),
            forecast_exit_threshold=_number(
                raw.get("forecast_exit_threshold"), field="forecast_exit_threshold"
            ),
            ppo_total_timesteps=_integer(
                raw.get("ppo_total_timesteps"), field="ppo_total_timesteps"
            ),
            ppo_seed=_integer(raw.get("ppo_seed"), field="ppo_seed"),
            evaluation_start=_text(
                raw.get("evaluation_start"), field="evaluation_start"
            ),
            evaluation_stop_exclusive=_text(
                raw.get("evaluation_stop_exclusive"),
                field="evaluation_stop_exclusive",
            ),
            gross_budget=_number(raw.get("gross_budget"), field="gross_budget"),
            initial_capital=_number(
                raw.get("initial_capital"), field="initial_capital"
            ),
            execution_overlay=_text(
                raw.get("execution_overlay"), field="execution_overlay"
            ),
            schema_version=_text(raw.get("schema_version"), field="schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("resolved run config is invalid") from error


def _resolved_from_spec(spec: ResolvedCandidateRunSpec) -> ResolvedRunConfig:
    config = spec.config
    lean = spec.lean_config
    return ResolvedRunConfig(
        signal_name=config.signal_name,
        signal_index=lean.signal_index,
        feature_names=config.feature_names,
        feature_indices=lean.feature_indices,
        fit_symbol_names=config.fit_symbol_names,
        fit_symbol_indices=lean.fit_symbol_indices,
        fit_cutoff=str(lean.fit_cutoff),
        rule_entry_threshold=lean.rule_entry_threshold,
        rule_exit_threshold=lean.rule_exit_threshold,
        forecast_entry_threshold=lean.forecast_entry_threshold,
        forecast_exit_threshold=lean.forecast_exit_threshold,
        ppo_total_timesteps=lean.ppo_total_timesteps,
        ppo_seed=lean.ppo_seed,
        evaluation_start=str(config.evaluation_start),
        evaluation_stop_exclusive=str(config.evaluation_stop_exclusive),
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def _candidate_from_resolved(config: ResolvedRunConfig) -> CandidateRunConfig:
    return CandidateRunConfig(
        signal_name=config.signal_name,
        feature_names=config.feature_names,
        fit_symbol_names=config.fit_symbol_names,
        fit_cutoff=np.datetime64(config.fit_cutoff, "ns"),
        evaluation_start=np.datetime64(config.evaluation_start, "ns"),
        evaluation_stop_exclusive=np.datetime64(config.evaluation_stop_exclusive, "ns"),
        rule_entry_threshold=config.rule_entry_threshold,
        rule_exit_threshold=config.rule_exit_threshold,
        forecast_entry_threshold=config.forecast_entry_threshold,
        forecast_exit_threshold=config.forecast_exit_threshold,
        ppo_total_timesteps=config.ppo_total_timesteps,
        ppo_seed=config.ppo_seed,
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
    )


def _requested_config_payload(config: CandidateRunConfig) -> dict[str, object]:
    return {
        "signal_name": config.signal_name,
        "feature_names": list(config.feature_names),
        "fit_symbol_names": list(config.fit_symbol_names),
        "fit_cutoff": str(config.fit_cutoff),
        "evaluation_start": str(config.evaluation_start),
        "evaluation_stop_exclusive": str(config.evaluation_stop_exclusive),
        "rule_entry_threshold": config.rule_entry_threshold,
        "rule_exit_threshold": config.rule_exit_threshold,
        "forecast_entry_threshold": config.forecast_entry_threshold,
        "forecast_exit_threshold": config.forecast_exit_threshold,
        "ppo_total_timesteps": config.ppo_total_timesteps,
        "ppo_seed": config.ppo_seed,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
    }


def _plan_from_payload(raw: Mapping[str, object]) -> StudyPlan:
    expected = {
        "schema_version",
        "research_question",
        "dataset_id",
        "dataset_artifact_schema",
        "dataset_artifact_digest",
        "symbols",
        "baseline_config",
        "ppo_seeds",
        "allowed_factors",
        "max_experiments",
        "n_bootstrap",
        "bootstrap_seed",
        "implementation_digest",
        "runtime_environment_digest",
        "candidate_strategy_names",
        "control_strategy_names",
    }
    _exact_keys(raw, expected, field="Study plan")
    factors_raw = _list(raw.get("allowed_factors"), field="allowed_factors")
    try:
        factors = tuple(
            ControlledFactor(_text(item, field="allowed factor"))
            for item in factors_raw
        )
    except ValueError as error:
        raise ArtifactIntegrityError("Study allowed factor is unsupported") from error
    if (
        _texts(raw.get("candidate_strategy_names"), field="candidate_strategy_names")
        != CANDIDATE_STRATEGY_NAMES
    ):
        raise ArtifactIntegrityError("Study candidate strategy roster mismatch")
    if (
        _texts(raw.get("control_strategy_names"), field="control_strategy_names")
        != CONTROL_STRATEGY_NAMES
    ):
        raise ArtifactIntegrityError("Study control strategy roster mismatch")
    try:
        return StudyPlan(
            research_question=_text(
                raw.get("research_question"), field="research_question"
            ),
            dataset_id=_text(raw.get("dataset_id"), field="dataset_id"),
            dataset_artifact_schema=_text(
                raw.get("dataset_artifact_schema"), field="dataset_artifact_schema"
            ),
            dataset_artifact_digest=_text(
                raw.get("dataset_artifact_digest"), field="dataset_artifact_digest"
            ),
            symbols=_texts(raw.get("symbols"), field="symbols"),
            baseline_config=_resolved_from_payload(raw.get("baseline_config")),
            ppo_seeds=_integers(raw.get("ppo_seeds"), field="ppo_seeds"),
            allowed_factors=cast(tuple[object, ...], factors),
            max_experiments=_integer(
                raw.get("max_experiments"), field="max_experiments"
            ),
            n_bootstrap=_integer(raw.get("n_bootstrap"), field="n_bootstrap"),
            bootstrap_seed=_integer(raw.get("bootstrap_seed"), field="bootstrap_seed"),
            implementation_digest=_text(
                raw.get("implementation_digest"), field="implementation_digest"
            ),
            runtime_environment_digest=_text(
                raw.get("runtime_environment_digest"),
                field="runtime_environment_digest",
            ),
            schema_version=_text(raw.get("schema_version"), field="schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("Study plan is invalid") from error


def _definition_from_payload(raw: Mapping[str, object]) -> ExperimentDefinition:
    expected = {
        "schema_version",
        "study_digest",
        "sequence",
        "hypothesis",
        "baseline_evidence_digest",
        "factor",
        "candidate_requested_config_digest",
        "candidate_config",
    }
    _exact_keys(raw, expected, field="Experiment definition")
    try:
        factor = ControlledFactor(_text(raw.get("factor"), field="factor"))
        return ExperimentDefinition(
            study_digest=_text(raw.get("study_digest"), field="study_digest"),
            sequence=_integer(raw.get("sequence"), field="sequence"),
            hypothesis=_text(raw.get("hypothesis"), field="hypothesis"),
            baseline_evidence_digest=_text(
                raw.get("baseline_evidence_digest"), field="baseline_evidence_digest"
            ),
            factor=factor,
            candidate_requested_config_digest=_text(
                raw.get("candidate_requested_config_digest"),
                field="candidate_requested_config_digest",
            ),
            candidate_config=_resolved_from_payload(raw.get("candidate_config")),
            schema_version=_text(raw.get("schema_version"), field="schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("Experiment definition is invalid") from error


def _verification_from_payload(raw: Mapping[str, object]) -> ControlledVerification:
    expected = {
        "schema_version",
        "study_digest",
        "experiment_digest",
        "baseline_evidence_digest",
        "candidate_evidence_digest",
        "factor",
        "status",
        "changed_paths",
        "violations",
    }
    _exact_keys(raw, expected, field="controlled verification")
    changed_raw = _list(raw.get("changed_paths"), field="changed_paths")
    changed: list[tuple[str, ...]] = []
    for value in changed_raw:
        changed.append(_texts(value, field="changed path"))
    try:
        return ControlledVerification(
            study_digest=_text(raw.get("study_digest"), field="study_digest"),
            experiment_digest=_text(
                raw.get("experiment_digest"), field="experiment_digest"
            ),
            baseline_evidence_digest=_text(
                raw.get("baseline_evidence_digest"), field="baseline_evidence_digest"
            ),
            candidate_evidence_digest=_text(
                raw.get("candidate_evidence_digest"), field="candidate_evidence_digest"
            ),
            factor=ControlledFactor(_text(raw.get("factor"), field="factor")),
            status=ControlledVerificationStatus(
                _text(raw.get("status"), field="status")
            ),
            changed_paths=tuple(changed),
            violations=_texts(raw.get("violations"), field="violations"),
            schema_version=_text(raw.get("schema_version"), field="schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("controlled verification is invalid") from error


def _verify_derived_payload(payload: Mapping[str, object], *, field: str) -> None:
    digest = payload.get("analysis_digest")
    if not isinstance(digest, str):
        raise ArtifactIntegrityError(f"{field} digest is missing")
    body = dict(payload)
    body.pop("analysis_digest")
    if content_digest(body) != digest:
        raise ArtifactIntegrityError(f"{field} digest mismatch")


def _comparison_from_payload(raw: Mapping[str, object]) -> ExperimentComparison:
    expected = {
        "schema_version",
        "study_digest",
        "experiment_digest",
        "verification_digest",
        "baseline_evidence_digest",
        "candidate_evidence_digest",
        "evidence_comparison",
    }
    _exact_keys(raw, expected, field="Experiment comparison")
    return ExperimentComparison(
        study_digest=_text(raw.get("study_digest"), field="study_digest"),
        experiment_digest=_text(
            raw.get("experiment_digest"), field="experiment_digest"
        ),
        verification_digest=_text(
            raw.get("verification_digest"), field="verification_digest"
        ),
        baseline_evidence_digest=_text(
            raw.get("baseline_evidence_digest"), field="baseline_evidence_digest"
        ),
        candidate_evidence_digest=_text(
            raw.get("candidate_evidence_digest"), field="candidate_evidence_digest"
        ),
        evidence_comparison=_mapping(
            raw.get("evidence_comparison"), field="evidence_comparison"
        ),
        schema_version=_text(raw.get("schema_version"), field="schema_version"),
    )


def _decision_from_payload(raw: Mapping[str, object]) -> ExperimentDecision:
    expected = {
        "schema_version",
        "study_digest",
        "experiment_digest",
        "verification_digest",
        "comparison_digest",
        "decision",
        "rationale",
        "decided_by",
        "decided_at",
    }
    _exact_keys(raw, expected, field="Experiment decision")
    try:
        return ExperimentDecision(
            study_digest=_text(raw.get("study_digest"), field="study_digest"),
            experiment_digest=_text(
                raw.get("experiment_digest"), field="experiment_digest"
            ),
            verification_digest=_text(
                raw.get("verification_digest"), field="verification_digest"
            ),
            comparison_digest=_text(
                raw.get("comparison_digest"), field="comparison_digest"
            ),
            decision=ExperimentDecisionKind(
                _text(raw.get("decision"), field="decision")
            ),
            rationale=_text(raw.get("rationale"), field="rationale"),
            decided_by=_text(raw.get("decided_by"), field="decided_by"),
            decided_at=_datetime(raw.get("decided_at"), field="decided_at"),
            schema_version=_text(raw.get("schema_version"), field="schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("Experiment decision is invalid") from error


def _baseline_context_digest(plan: StudyPlan) -> str:
    return content_digest({"kind": "baseline", "study_digest": plan.digest})


def _check_dataset(plan: StudyPlan, dataset_root: str | Path):
    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    if dataset.dataset_id != plan.dataset_id:
        raise ArtifactIntegrityError("Study dataset identity mismatch")
    if artifact.schema_version != plan.dataset_artifact_schema:
        raise ArtifactIntegrityError("Study dataset artifact schema mismatch")
    if artifact.artifact_digest != plan.dataset_artifact_digest:
        raise ArtifactIntegrityError("Study dataset artifact digest mismatch")
    if tuple(dataset.symbols) != plan.symbols:
        raise ArtifactIntegrityError("Study dataset symbol roster mismatch")
    return artifact, dataset


def _read_analysis(path: Path, *, expected: dict[str, object], field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(f"{field} must be a regular file")
    try:
        import json

        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ArtifactIntegrityError(f"{field} is malformed") from error
    payload = _mapping(raw, field=field)
    _verify_derived_payload(payload, field=field)
    if payload != expected:
        raise ArtifactIntegrityError(f"{field} does not match reconstructed evidence")


def _load_bundle(
    root: Path,
    *,
    plan: StudyPlan,
    expected_context_digest: str,
    label: str,
) -> LoadedEvidenceSet:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactIntegrityError(f"{label} must be a regular directory")
    if {entry.name for entry in root.iterdir()} != {"evidence", "analysis.json"}:
        raise ArtifactIntegrityError(f"{label} artifact layout is invalid")
    loaded = load_evidence_set(root / "evidence")
    if loaded.evidence.ppo_seeds != plan.ppo_seeds:
        raise ArtifactIntegrityError(f"{label} PPO seed roster mismatch")
    if loaded.evidence.research_context_digest != expected_context_digest:
        raise ArtifactIntegrityError(f"{label} research context mismatch")
    expected_analysis = analyze_evidence_set(
        loaded.runs,
        n_bootstrap=plan.n_bootstrap,
        bootstrap_seed=plan.bootstrap_seed,
    )
    _read_analysis(
        root / "analysis.json",
        expected=expected_analysis,
        field=f"{label} analysis",
    )
    return loaded


def _publish_bundle(
    *,
    store: StudyStore,
    target: str | Path,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
    research_context_digest: str,
) -> EvidenceSet:
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        stage_store = StudyStore(staging)
        evidence = execute_evidence_set(
            store=stage_store,
            target="evidence",
            dataset_root=dataset_root,
            plan=plan,
            config=config,
            research_context_digest=research_context_digest,
        )
        loaded = load_evidence_set(staging / "evidence")
        analysis = analyze_evidence_set(
            loaded.runs,
            n_bootstrap=plan.n_bootstrap,
            bootstrap_seed=plan.bootstrap_seed,
        )
        atomic_write_bytes(staging / "analysis.json", canonical_json_bytes(analysis))
        lock_path = staging / ".mutation.lock"
        if lock_path.exists():
            lock_path.unlink()
        completed = evidence

    store.publish_directory_once(target, builder)
    if completed is None:
        raise ArtifactIntegrityError(
            "Study evidence bundle publication did not complete"
        )
    return completed


def _experiment_dirs(root: Path) -> tuple[int, ...]:
    experiments_root = root / "experiments"
    if not experiments_root.exists():
        return ()
    if experiments_root.is_symlink() or not experiments_root.is_dir():
        raise ArtifactIntegrityError(
            "Study experiments root must be a regular directory"
        )
    sequences: list[int] = []
    for entry in experiments_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            raise ArtifactIntegrityError("Experiment entries must be directories")
        if len(entry.name) != 4 or not entry.name.isdigit():
            raise ArtifactIntegrityError("Experiment directory sequence is malformed")
        sequences.append(int(entry.name))
    sequences.sort()
    expected = list(range(1, len(sequences) + 1))
    if sequences != expected:
        raise ArtifactIntegrityError("Experiment sequences must be contiguous")
    return tuple(sequences)


def _reconstruct(store: StudyStore) -> _StudyState:
    root = store.root
    allowed_root = {".mutation.lock", "plan.json", "baseline", "experiments"}
    unknown_root = {entry.name for entry in root.iterdir()} - allowed_root
    if unknown_root:
        raise ArtifactIntegrityError("Study root contains unsupported artifacts")
    plan = _plan_from_payload(store.read_json("plan.json"))

    baseline_loaded: LoadedEvidenceSet | None = None
    evidence_by_digest: dict[str, LoadedEvidenceSet] = {}
    lineage: list[str] = []
    baseline_root = root / "baseline"
    if baseline_root.exists() or baseline_root.is_symlink():
        baseline_loaded = _load_bundle(
            baseline_root,
            plan=plan,
            expected_context_digest=_baseline_context_digest(plan),
            label="baseline",
        )
        if (
            baseline_loaded.evidence.semantic_config_digest
            != plan.baseline_config.digest
        ):
            raise ArtifactIntegrityError(
                "baseline semantic config differs from Study plan"
            )
        evidence_by_digest[baseline_loaded.evidence.fingerprint] = baseline_loaded
        lineage.append(baseline_loaded.evidence.fingerprint)

    sequences = _experiment_dirs(root)
    if sequences and baseline_loaded is None:
        raise ArtifactIntegrityError("Study experiments require a published baseline")
    experiments: dict[int, _ExperimentState] = {}
    for sequence in sequences:
        prefix = Path("experiments") / f"{sequence:04d}"
        experiment_root = root / prefix
        allowed_names = {
            "definition.json",
            "candidate",
            "verification.json",
            "comparison.json",
            "decision.json",
        }
        names = {entry.name for entry in experiment_root.iterdir()}
        if names - allowed_names:
            raise ArtifactIntegrityError("Experiment contains unsupported artifacts")
        if "definition.json" not in names:
            raise ArtifactIntegrityError("Experiment definition is missing")
        definition = _definition_from_payload(
            store.read_json(prefix / "definition.json")
        )
        if definition.study_digest != plan.digest or definition.sequence != sequence:
            raise ArtifactIntegrityError("Experiment definition identity mismatch")
        if definition.baseline_evidence_digest not in lineage:
            raise ArtifactIntegrityError(
                "Experiment baseline is not reachable in lineage"
            )

        candidate: LoadedEvidenceSet | None = None
        candidate_root = experiment_root / "candidate"
        if candidate_root.exists() or candidate_root.is_symlink():
            candidate = _load_bundle(
                candidate_root,
                plan=plan,
                expected_context_digest=definition.digest,
                label=f"Experiment {sequence:04d} candidate",
            )
            evidence_by_digest[candidate.evidence.fingerprint] = candidate

        verification: ControlledVerification | None = None
        verification_path = experiment_root / "verification.json"
        if verification_path.exists() or verification_path.is_symlink():
            if candidate is None:
                raise ArtifactIntegrityError("verification requires candidate evidence")
            verification = _verification_from_payload(
                store.read_json(prefix / "verification.json")
            )
            baseline = evidence_by_digest.get(definition.baseline_evidence_digest)
            if baseline is None:
                raise ArtifactIntegrityError(
                    "verification baseline evidence is missing"
                )
            expected_verification = verify_controlled_delta(
                plan=plan,
                definition=definition,
                baseline=baseline,
                candidate=candidate,
            )
            if verification.to_payload() != expected_verification.to_payload():
                raise ArtifactIntegrityError("verification does not match evidence")

        comparison: ExperimentComparison | None = None
        comparison_path = experiment_root / "comparison.json"
        if comparison_path.exists() or comparison_path.is_symlink():
            if verification is None:
                raise ArtifactIntegrityError("comparison requires verification")
            if verification.status is not ControlledVerificationStatus.CONTROLLED:
                raise ArtifactIntegrityError(
                    "INVALID verification cannot have comparison"
                )
            if candidate is None:
                raise ArtifactIntegrityError("comparison requires candidate evidence")
            comparison = _comparison_from_payload(
                store.read_json(prefix / "comparison.json")
            )
            baseline = evidence_by_digest.get(definition.baseline_evidence_digest)
            if baseline is None:
                raise ArtifactIntegrityError("comparison baseline evidence is missing")
            expected_payload = compare_evidence_sets(
                baseline.runs,
                candidate.runs,
                n_bootstrap=plan.n_bootstrap,
                bootstrap_seed=plan.bootstrap_seed,
            )
            expected_comparison = ExperimentComparison(
                study_digest=plan.digest,
                experiment_digest=definition.digest,
                verification_digest=verification.digest,
                baseline_evidence_digest=baseline.evidence.fingerprint,
                candidate_evidence_digest=candidate.evidence.fingerprint,
                evidence_comparison=expected_payload,
            )
            if comparison.to_payload() != expected_comparison.to_payload():
                raise ArtifactIntegrityError("comparison does not match evidence")

        decision: ExperimentDecision | None = None
        decision_path = experiment_root / "decision.json"
        if decision_path.exists() or decision_path.is_symlink():
            if comparison is None or verification is None or candidate is None:
                raise ArtifactIntegrityError("decision requires controlled comparison")
            decision = _decision_from_payload(store.read_json(prefix / "decision.json"))
            if (
                decision.study_digest != plan.digest
                or decision.experiment_digest != definition.digest
                or decision.verification_digest != verification.digest
                or decision.comparison_digest != comparison.digest
            ):
                raise ArtifactIntegrityError("Experiment decision reference mismatch")
            if decision.decision is ExperimentDecisionKind.ACCEPT_CANDIDATE:
                if candidate.evidence.fingerprint not in lineage:
                    lineage.append(candidate.evidence.fingerprint)

        if (
            verification is not None
            and verification.status is ControlledVerificationStatus.INVALID
        ):
            if comparison is not None or decision is not None:
                raise ArtifactIntegrityError("INVALID attempt must be terminal")
        experiments[sequence] = _ExperimentState(
            definition=definition,
            candidate=candidate,
            verification=verification,
            comparison=comparison,
            decision=decision,
        )

    snapshot = StudySnapshot(
        root=root,
        plan=plan,
        baseline=None if baseline_loaded is None else baseline_loaded.evidence,
        experiment_sequences=sequences,
        lineage_evidence_digests=tuple(lineage),
    )
    return _StudyState(
        snapshot=snapshot,
        baseline_loaded=baseline_loaded,
        evidence_by_digest=evidence_by_digest,
        experiments=experiments,
    )


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
    """Create one immutable Study plan after shared Run Core pre-resolution."""

    store = StudyStore(root)
    with store.mutation_lock():
        names = {entry.name for entry in store.root.iterdir()}
        if names != {".mutation.lock"}:
            raise InvalidExperimentStateError("Study root already contains artifacts")
        artifact = inspect_published_market_dataset_artifact(dataset_root)
        dataset = load_market_dataset_artifact(dataset_root)
        spec = resolve_candidate_run_spec(
            dataset,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            config=baseline_config,
        )
        resolved = _resolved_from_spec(spec)
        provenance = build_candidate_run_provenance()
        plan = StudyPlan(
            research_question=research_question,
            dataset_id=dataset.dataset_id,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            symbols=tuple(dataset.symbols),
            baseline_config=resolved,
            ppo_seeds=ppo_seeds,
            allowed_factors=cast(tuple[object, ...], allowed_factors),
            max_experiments=max_experiments,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            implementation_digest=_text(
                provenance.get("implementation_digest"), field="implementation_digest"
            ),
            runtime_environment_digest=_text(
                provenance.get("runtime_environment_digest"),
                field="runtime_environment_digest",
            ),
        )
        store.publish_json_once("plan.json", plan.to_payload())
        return _reconstruct(store).snapshot


def inspect_study(root: str | Path) -> StudySnapshot:
    """Reconstruct and validate a Study entirely from immutable disk artifacts."""

    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError("Study root must be a regular directory")
    store = StudyStore(path)
    return _reconstruct(store).snapshot


def run_baseline(
    root: str | Path,
    *,
    dataset_root: str | Path,
) -> StudySnapshot:
    """Generate the Study-owned baseline EvidenceSet exactly once."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        if state.baseline_loaded is not None:
            raise InvalidExperimentStateError("Study baseline already exists")
        if state.snapshot.experiment_sequences:
            raise InvalidExperimentStateError("baseline must precede Experiments")
        _check_dataset(state.snapshot.plan, dataset_root)
        _publish_bundle(
            store=store,
            target="baseline",
            dataset_root=dataset_root,
            plan=state.snapshot.plan,
            config=_candidate_from_resolved(state.snapshot.plan.baseline_config),
            research_context_digest=_baseline_context_digest(state.snapshot.plan),
        )
        return _reconstruct(store).snapshot


def define_experiment(
    root: str | Path,
    *,
    dataset_root: str | Path,
    hypothesis: str,
    factor: ControlledFactor,
    candidate_config: CandidateRunConfig,
    baseline_evidence_digest: str,
) -> ExperimentDefinition:
    """Pre-register the next contiguous Experiment and consume its budget slot."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        plan = state.snapshot.plan
        if state.baseline_loaded is None:
            raise InvalidExperimentStateError("Experiment definition requires baseline")
        if len(state.snapshot.experiment_sequences) >= plan.max_experiments:
            raise ExperimentBudgetExceededError("Study Experiment budget is exhausted")
        if baseline_evidence_digest not in state.snapshot.lineage_evidence_digests:
            raise InvalidExperimentStateError(
                "baseline evidence is not reachable in lineage"
            )
        artifact, dataset = _check_dataset(plan, dataset_root)
        spec = resolve_candidate_run_spec(
            dataset,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            config=candidate_config,
        )
        resolved = _resolved_from_spec(spec)
        sequence = len(state.snapshot.experiment_sequences) + 1
        definition = ExperimentDefinition(
            study_digest=plan.digest,
            sequence=sequence,
            hypothesis=hypothesis,
            baseline_evidence_digest=baseline_evidence_digest,
            factor=factor,
            candidate_requested_config_digest=content_digest(
                _requested_config_payload(candidate_config)
            ),
            candidate_config=resolved,
        )
        store.publish_json_once(
            Path("experiments") / f"{sequence:04d}" / "definition.json",
            definition.to_payload(),
        )
        return definition


def run_experiment(
    root: str | Path,
    sequence: int,
    *,
    dataset_root: str | Path,
) -> EvidenceSet:
    """Execute one pre-registered candidate EvidenceSet exactly once."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        experiment = state.experiments.get(sequence)
        if experiment is None:
            raise InvalidExperimentStateError("Experiment definition does not exist")
        if experiment.candidate is not None:
            raise InvalidExperimentStateError("Experiment candidate already exists")
        if experiment.verification is not None:
            raise InvalidExperimentStateError("verified Experiment cannot be rerun")
        _check_dataset(state.snapshot.plan, dataset_root)
        return _publish_bundle(
            store=store,
            target=Path("experiments") / f"{sequence:04d}" / "candidate",
            dataset_root=dataset_root,
            plan=state.snapshot.plan,
            config=_candidate_from_resolved(experiment.definition.candidate_config),
            research_context_digest=experiment.definition.digest,
        )


def verify_experiment(root: str | Path, sequence: int) -> ControlledVerification:
    """Verify one candidate against its frozen baseline and publish once."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        experiment = state.experiments.get(sequence)
        if experiment is None:
            raise InvalidExperimentStateError("Experiment definition does not exist")
        if experiment.candidate is None:
            raise InvalidExperimentStateError(
                "verification requires candidate evidence"
            )
        if experiment.verification is not None:
            raise InvalidExperimentStateError("Experiment verification already exists")
        baseline = state.evidence_by_digest.get(
            experiment.definition.baseline_evidence_digest
        )
        if baseline is None:
            raise ArtifactIntegrityError("Experiment baseline evidence is missing")
        verification = verify_controlled_delta(
            plan=state.snapshot.plan,
            definition=experiment.definition,
            baseline=baseline,
            candidate=experiment.candidate,
        )
        store.publish_json_once(
            Path("experiments") / f"{sequence:04d}" / "verification.json",
            verification.to_payload(),
        )
        return verification


def compare_experiment(root: str | Path, sequence: int) -> ExperimentComparison:
    """Publish statistical comparison only after CONTROLLED verification."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        experiment = state.experiments.get(sequence)
        if experiment is None:
            raise InvalidExperimentStateError("Experiment definition does not exist")
        if experiment.verification is None:
            raise InvalidExperimentStateError("comparison requires verification")
        if (
            experiment.verification.status
            is not ControlledVerificationStatus.CONTROLLED
        ):
            raise InvalidExperimentStateError(
                "comparison requires CONTROLLED verification; attempt is INVALID"
            )
        if experiment.comparison is not None:
            raise InvalidExperimentStateError("Experiment comparison already exists")
        if experiment.candidate is None:
            raise ArtifactIntegrityError("comparison candidate evidence is missing")
        baseline = state.evidence_by_digest.get(
            experiment.definition.baseline_evidence_digest
        )
        if baseline is None:
            raise ArtifactIntegrityError("comparison baseline evidence is missing")
        payload = compare_evidence_sets(
            baseline.runs,
            experiment.candidate.runs,
            n_bootstrap=state.snapshot.plan.n_bootstrap,
            bootstrap_seed=state.snapshot.plan.bootstrap_seed,
        )
        comparison = ExperimentComparison(
            study_digest=state.snapshot.plan.digest,
            experiment_digest=experiment.definition.digest,
            verification_digest=experiment.verification.digest,
            baseline_evidence_digest=baseline.evidence.fingerprint,
            candidate_evidence_digest=experiment.candidate.evidence.fingerprint,
            evidence_comparison=payload,
        )
        store.publish_json_once(
            Path("experiments") / f"{sequence:04d}" / "comparison.json",
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
    """Publish one evidence-bound decision after controlled comparison."""

    store = StudyStore(root)
    with store.mutation_lock():
        state = _reconstruct(store)
        experiment = state.experiments.get(sequence)
        if experiment is None:
            raise InvalidExperimentStateError("Experiment definition does not exist")
        if experiment.comparison is None or experiment.verification is None:
            raise InvalidExperimentStateError("decision requires comparison")
        if experiment.decision is not None:
            raise InvalidExperimentStateError("Experiment decision already exists")
        resolved = ExperimentDecision(
            study_digest=state.snapshot.plan.digest,
            experiment_digest=experiment.definition.digest,
            verification_digest=experiment.verification.digest,
            comparison_digest=experiment.comparison.digest,
            decision=decision,
            rationale=rationale,
            decided_by=decided_by,
            decided_at=decided_at,
        )
        store.publish_json_once(
            Path("experiments") / f"{sequence:04d}" / "decision.json",
            resolved.to_payload(),
        )
        return resolved


__all__ = [
    "ExperimentComparison",
    "StudySnapshot",
    "compare_experiment",
    "create_study",
    "decide_experiment",
    "define_experiment",
    "inspect_study",
    "run_baseline",
    "run_experiment",
    "verify_experiment",
]
