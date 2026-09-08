"""Append-only Study lifecycle orchestration for controlled development research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

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

_ANALYSIS_BINDING_SCHEMA = "controlled_evidence_analysis_binding_v1"
_WITHIN_ANALYSIS_SCHEMA = "controlled_evidence_analysis_v1"
_FACTOR_EFFECT_SCHEMA = "controlled_evidence_comparison_v1"
_STUDY_FIXED_FIELDS = (
    "fit_cutoff",
    "evaluation_start",
    "evaluation_stop_exclusive",
    "initial_capital",
    "execution_overlay",
)


@dataclass(frozen=True, slots=True)
class StudySnapshot:
    """Reconstructed public view of one immutable Study filesystem."""

    root: Path
    plan: StudyPlan
    baseline: EvidenceSet | None
    experiment_sequences: tuple[int, ...]
    terminal_sequences: tuple[int, ...]
    lineage_evidence_digests: tuple[str, ...]
    frozen: bool


@dataclass(frozen=True, slots=True)
class _AnalysisBinding:
    evidence_fingerprint: str
    analysis: dict[str, object]
    analysis_digest: str


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
            frozen=self.frozen is not None,
        )


def _expect_keys(
    payload: dict[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ArtifactIntegrityError(
            f"{label} keys differ from contract: missing={missing}, extra={extra}"
        )


def _as_dict(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArtifactIntegrityError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _as_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ArtifactIntegrityError(f"{field} must be a JSON array")
    return cast(list[object], value)


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(f"{field} must be a non-empty string")
    return value


def _as_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field=field)


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError(f"{field} must be an integer")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(f"{field} must be numeric")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ArtifactIntegrityError(f"{field} must be finite")
    return resolved


def _as_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    values = _as_list(value, field=field)
    return tuple(_as_string(item, field=field) for item in values)


def _as_int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    values = _as_list(value, field=field)
    return tuple(_as_int(item, field=field) for item in values)


def _as_datetime(value: object, *, field: str) -> datetime:
    raw = _as_string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactIntegrityError(f"{field} must be an ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactIntegrityError(f"{field} must be timezone-aware")
    return parsed


def _resolved_from_payload(payload: object, *, field: str) -> ResolvedRunConfig:
    raw = _as_dict(payload, field=field)
    _expect_keys(
        raw,
        {
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
        },
        label=field,
    )
    try:
        return ResolvedRunConfig(
            signal_name=_as_string(raw["signal_name"], field=f"{field}.signal_name"),
            signal_index=_as_int(raw["signal_index"], field=f"{field}.signal_index"),
            feature_names=_as_string_tuple(
                raw["feature_names"], field=f"{field}.feature_names"
            ),
            feature_indices=_as_int_tuple(
                raw["feature_indices"], field=f"{field}.feature_indices"
            ),
            fit_symbol_names=_as_string_tuple(
                raw["fit_symbol_names"], field=f"{field}.fit_symbol_names"
            ),
            fit_symbol_indices=_as_int_tuple(
                raw["fit_symbol_indices"], field=f"{field}.fit_symbol_indices"
            ),
            fit_cutoff=_as_string(raw["fit_cutoff"], field=f"{field}.fit_cutoff"),
            rule_entry_threshold=_as_float(
                raw["rule_entry_threshold"], field=f"{field}.rule_entry_threshold"
            ),
            rule_exit_threshold=_as_float(
                raw["rule_exit_threshold"], field=f"{field}.rule_exit_threshold"
            ),
            forecast_entry_threshold=_as_float(
                raw["forecast_entry_threshold"],
                field=f"{field}.forecast_entry_threshold",
            ),
            forecast_exit_threshold=_as_float(
                raw["forecast_exit_threshold"],
                field=f"{field}.forecast_exit_threshold",
            ),
            ppo_total_timesteps=_as_int(
                raw["ppo_total_timesteps"], field=f"{field}.ppo_total_timesteps"
            ),
            ppo_seed=_as_int(raw["ppo_seed"], field=f"{field}.ppo_seed"),
            evaluation_start=_as_string(
                raw["evaluation_start"], field=f"{field}.evaluation_start"
            ),
            evaluation_stop_exclusive=_as_string(
                raw["evaluation_stop_exclusive"],
                field=f"{field}.evaluation_stop_exclusive",
            ),
            gross_budget=_as_float(raw["gross_budget"], field=f"{field}.gross_budget"),
            initial_capital=_as_float(
                raw["initial_capital"], field=f"{field}.initial_capital"
            ),
            execution_overlay=_as_string(
                raw["execution_overlay"], field=f"{field}.execution_overlay"
            ),
            schema_version=_as_string(
                raw["schema_version"], field=f"{field}.schema_version"
            ),
        )
    except ContractViolationError as error:
        raise ArtifactIntegrityError(
            f"{field} violates resolved-run contract"
        ) from error


def _study_plan_from_payload(payload: dict[str, object]) -> StudyPlan:
    _expect_keys(
        payload,
        {
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
        },
        label="plan.json",
    )
    candidates = _as_string_tuple(
        payload["candidate_strategy_names"], field="candidate_strategy_names"
    )
    controls = _as_string_tuple(
        payload["control_strategy_names"], field="control_strategy_names"
    )
    if candidates != CANDIDATE_STRATEGY_NAMES or controls != CONTROL_STRATEGY_NAMES:
        raise ArtifactIntegrityError("Study strategy roster differs from v1 contract")
    factors: list[ControlledFactor] = []
    for value in _as_list(payload["allowed_factors"], field="allowed_factors"):
        try:
            factors.append(ControlledFactor(_as_string(value, field="allowed_factors")))
        except ValueError as error:
            raise ArtifactIntegrityError(
                "Study contains unsupported controlled factor"
            ) from error
    try:
        return StudyPlan(
            research_question=_as_string(
                payload["research_question"], field="research_question"
            ),
            dataset_id=_as_string(payload["dataset_id"], field="dataset_id"),
            dataset_artifact_schema=_as_string(
                payload["dataset_artifact_schema"], field="dataset_artifact_schema"
            ),
            dataset_artifact_digest=_as_string(
                payload["dataset_artifact_digest"], field="dataset_artifact_digest"
            ),
            symbols=_as_string_tuple(payload["symbols"], field="symbols"),
            baseline_config=_resolved_from_payload(
                payload["baseline_config"], field="baseline_config"
            ),
            ppo_seeds=_as_int_tuple(payload["ppo_seeds"], field="ppo_seeds"),
            allowed_factors=tuple(factors),
            max_experiments=_as_int(
                payload["max_experiments"], field="max_experiments"
            ),
            n_bootstrap=_as_int(payload["n_bootstrap"], field="n_bootstrap"),
            bootstrap_seed=_as_int(payload["bootstrap_seed"], field="bootstrap_seed"),
            implementation_digest=_as_string(
                payload["implementation_digest"], field="implementation_digest"
            ),
            runtime_environment_digest=_as_string(
                payload["runtime_environment_digest"],
                field="runtime_environment_digest",
            ),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except ContractViolationError as error:
        raise ArtifactIntegrityError("plan.json violates StudyPlan contract") from error


def _candidate_config_payload(config: CandidateRunConfig) -> dict[str, object]:
    return {
        "signal_name": config.signal_name,
        "feature_names": list(config.feature_names),
        "fit_symbol_names": list(config.fit_symbol_names),
        "fit_cutoff": str(np.datetime64(config.fit_cutoff, "ns")),
        "evaluation_start": str(np.datetime64(config.evaluation_start, "ns")),
        "evaluation_stop_exclusive": str(
            np.datetime64(config.evaluation_stop_exclusive, "ns")
        ),
        "rule_entry_threshold": config.rule_entry_threshold,
        "rule_exit_threshold": config.rule_exit_threshold,
        "forecast_entry_threshold": config.forecast_entry_threshold,
        "forecast_exit_threshold": config.forecast_exit_threshold,
        "ppo_total_timesteps": config.ppo_total_timesteps,
        "ppo_seed": config.ppo_seed,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
    }


def _candidate_config_from_resolved(config: ResolvedRunConfig) -> CandidateRunConfig:
    try:
        return CandidateRunConfig(
            signal_name=config.signal_name,
            feature_names=config.feature_names,
            fit_symbol_names=config.fit_symbol_names,
            fit_cutoff=np.datetime64(config.fit_cutoff, "ns"),
            evaluation_start=np.datetime64(config.evaluation_start, "ns"),
            evaluation_stop_exclusive=np.datetime64(
                config.evaluation_stop_exclusive, "ns"
            ),
            rule_entry_threshold=config.rule_entry_threshold,
            rule_exit_threshold=config.rule_exit_threshold,
            forecast_entry_threshold=config.forecast_entry_threshold,
            forecast_exit_threshold=config.forecast_exit_threshold,
            ppo_total_timesteps=config.ppo_total_timesteps,
            ppo_seed=config.ppo_seed,
            gross_budget=config.gross_budget,
            initial_capital=config.initial_capital,
        )
    except ValueError as error:
        raise ArtifactIntegrityError(
            "resolved configuration cannot reconstruct CandidateRunConfig"
        ) from error


def _definition_from_payload(payload: dict[str, object]) -> ExperimentDefinition:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "sequence",
            "hypothesis",
            "baseline_evidence_digest",
            "factor",
            "candidate_requested_config_digest",
            "candidate_config",
        },
        label="ExperimentDefinition",
    )
    try:
        definition = ExperimentDefinition(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            sequence=_as_int(payload["sequence"], field="sequence"),
            hypothesis=_as_string(payload["hypothesis"], field="hypothesis"),
            baseline_evidence_digest=_as_string(
                payload["baseline_evidence_digest"], field="baseline_evidence_digest"
            ),
            factor=ControlledFactor(_as_string(payload["factor"], field="factor")),
            candidate_requested_config_digest=_as_string(
                payload["candidate_requested_config_digest"],
                field="candidate_requested_config_digest",
            ),
            candidate_config=_resolved_from_payload(
                payload["candidate_config"], field="candidate_config"
            ),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except (ContractViolationError, ValueError) as error:
        raise ArtifactIntegrityError(
            "definition.json violates ExperimentDefinition contract"
        ) from error
    reconstructed = _candidate_config_from_resolved(definition.candidate_config)
    if content_digest(_candidate_config_payload(reconstructed)) != (
        definition.candidate_requested_config_digest
    ):
        raise ArtifactIntegrityError(
            "definition requested-config digest does not match resolved configuration"
        )
    return definition


def _verification_from_payload(payload: dict[str, object]) -> ControlledVerification:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "experiment_digest",
            "baseline_evidence_digest",
            "candidate_evidence_digest",
            "factor",
            "status",
            "changed_paths",
            "violations",
        },
        label="ControlledVerification",
    )
    changed_paths: list[tuple[str, ...]] = []
    for path in _as_list(payload["changed_paths"], field="changed_paths"):
        changed_paths.append(_as_string_tuple(path, field="changed_paths"))
    try:
        return ControlledVerification(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            experiment_digest=_as_string(
                payload["experiment_digest"], field="experiment_digest"
            ),
            baseline_evidence_digest=_as_string(
                payload["baseline_evidence_digest"], field="baseline_evidence_digest"
            ),
            candidate_evidence_digest=_as_string(
                payload["candidate_evidence_digest"], field="candidate_evidence_digest"
            ),
            factor=ControlledFactor(_as_string(payload["factor"], field="factor")),
            status=ControlledVerificationStatus(
                _as_string(payload["status"], field="status")
            ),
            changed_paths=tuple(changed_paths),
            violations=_as_string_tuple(payload["violations"], field="violations"),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except (ArtifactIntegrityError, ValueError) as error:
        raise ArtifactIntegrityError(
            "verification.json violates ControlledVerification contract"
        ) from error


def _comparison_from_payload(payload: dict[str, object]) -> ExperimentComparison:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "experiment_digest",
            "baseline_evidence_digest",
            "candidate_evidence_digest",
            "verification_digest",
            "baseline_analysis_digest",
            "candidate_analysis_digest",
            "factor_effect_digest",
            "factor_effect",
        },
        label="ExperimentComparison",
    )
    try:
        return ExperimentComparison(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            experiment_digest=_as_string(
                payload["experiment_digest"], field="experiment_digest"
            ),
            baseline_evidence_digest=_as_string(
                payload["baseline_evidence_digest"], field="baseline_evidence_digest"
            ),
            candidate_evidence_digest=_as_string(
                payload["candidate_evidence_digest"], field="candidate_evidence_digest"
            ),
            verification_digest=_as_string(
                payload["verification_digest"], field="verification_digest"
            ),
            baseline_analysis_digest=_as_string(
                payload["baseline_analysis_digest"], field="baseline_analysis_digest"
            ),
            candidate_analysis_digest=_as_string(
                payload["candidate_analysis_digest"], field="candidate_analysis_digest"
            ),
            factor_effect_digest=_as_string(
                payload["factor_effect_digest"], field="factor_effect_digest"
            ),
            factor_effect=_as_dict(payload["factor_effect"], field="factor_effect"),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except ContractViolationError as error:
        raise ArtifactIntegrityError(
            "comparison.json violates ExperimentComparison contract"
        ) from error


def _decision_from_payload(payload: dict[str, object]) -> ExperimentDecision:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "experiment_digest",
            "verification_digest",
            "comparison_digest",
            "decision",
            "rationale",
            "decided_by",
            "decided_at",
        },
        label="ExperimentDecision",
    )
    try:
        return ExperimentDecision(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            experiment_digest=_as_string(
                payload["experiment_digest"], field="experiment_digest"
            ),
            verification_digest=_as_string(
                payload["verification_digest"], field="verification_digest"
            ),
            comparison_digest=_as_string(
                payload["comparison_digest"], field="comparison_digest"
            ),
            decision=ExperimentDecisionKind(
                _as_string(payload["decision"], field="decision")
            ),
            rationale=_as_string(payload["rationale"], field="rationale"),
            decided_by=_as_string(payload["decided_by"], field="decided_by"),
            decided_at=_as_datetime(payload["decided_at"], field="decided_at"),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except (ContractViolationError, ValueError) as error:
        raise ArtifactIntegrityError(
            "decision.json violates ExperimentDecision contract"
        ) from error


def _failure_from_payload(payload: dict[str, object]) -> ExperimentFailure:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "experiment_digest",
            "reason",
            "recorded_by",
            "recorded_at",
        },
        label="ExperimentFailure",
    )
    try:
        return ExperimentFailure(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            experiment_digest=_as_string(
                payload["experiment_digest"], field="experiment_digest"
            ),
            reason=_as_string(payload["reason"], field="reason"),
            recorded_by=_as_string(payload["recorded_by"], field="recorded_by"),
            recorded_at=_as_datetime(payload["recorded_at"], field="recorded_at"),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except ContractViolationError as error:
        raise ArtifactIntegrityError(
            "failure.json violates ExperimentFailure contract"
        ) from error


def _freeze_from_payload(payload: dict[str, object]) -> StudyFreeze:
    _expect_keys(
        payload,
        {
            "schema_version",
            "study_digest",
            "experiment_decision_digests",
            "outcome",
            "selected_evidence_digest",
            "selected_strategy",
            "rationale",
            "frozen_by",
            "frozen_at",
        },
        label="StudyFreeze",
    )
    try:
        return StudyFreeze(
            study_digest=_as_string(payload["study_digest"], field="study_digest"),
            experiment_decision_digests=_as_string_tuple(
                payload["experiment_decision_digests"],
                field="experiment_decision_digests",
            ),
            outcome=StudyOutcome(_as_string(payload["outcome"], field="outcome")),
            selected_evidence_digest=_as_optional_string(
                payload["selected_evidence_digest"], field="selected_evidence_digest"
            ),
            selected_strategy=_as_optional_string(
                payload["selected_strategy"], field="selected_strategy"
            ),
            rationale=_as_string(payload["rationale"], field="rationale"),
            frozen_by=_as_string(payload["frozen_by"], field="frozen_by"),
            frozen_at=_as_datetime(payload["frozen_at"], field="frozen_at"),
            schema_version=_as_string(
                payload["schema_version"], field="schema_version"
            ),
        )
    except (ContractViolationError, ValueError) as error:
        raise ArtifactIntegrityError(
            "freeze.json violates StudyFreeze contract"
        ) from error


def _resolved_contract(spec: ResolvedCandidateRunSpec) -> ResolvedRunConfig:
    config = spec.config
    lean = spec.lean_config
    return ResolvedRunConfig(
        signal_name=config.signal_name,
        signal_index=lean.signal_index,
        feature_names=config.feature_names,
        feature_indices=lean.feature_indices,
        fit_symbol_names=config.fit_symbol_names,
        fit_symbol_indices=lean.fit_symbol_indices,
        fit_cutoff=str(np.datetime64(lean.fit_cutoff, "ns")),
        rule_entry_threshold=lean.rule_entry_threshold,
        rule_exit_threshold=lean.rule_exit_threshold,
        forecast_entry_threshold=lean.forecast_entry_threshold,
        forecast_exit_threshold=lean.forecast_exit_threshold,
        ppo_total_timesteps=lean.ppo_total_timesteps,
        ppo_seed=lean.ppo_seed,
        evaluation_start=str(np.datetime64(config.evaluation_start, "ns")),
        evaluation_stop_exclusive=str(
            np.datetime64(config.evaluation_stop_exclusive, "ns")
        ),
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def _semantic_without_seed(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


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
):
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


def _baseline_context_digest(plan: StudyPlan) -> str:
    return content_digest({"kind": "baseline", "study_digest": plan.digest})


def _analysis_binding(
    *,
    evidence_fingerprint: str,
    analysis: dict[str, object],
) -> dict[str, object]:
    inner_digest = analysis.get("analysis_digest")
    if not isinstance(inner_digest, str):
        raise ArtifactIntegrityError("analysis payload is missing analysis_digest")
    body = dict(analysis)
    body.pop("analysis_digest", None)
    if content_digest(body) != inner_digest:
        raise ArtifactIntegrityError("analysis payload digest mismatch")
    if analysis.get("schema_version") != _WITHIN_ANALYSIS_SCHEMA:
        raise ArtifactIntegrityError("unsupported within-EvidenceSet analysis schema")
    payload: dict[str, object] = {
        "schema_version": _ANALYSIS_BINDING_SCHEMA,
        "evidence_fingerprint": evidence_fingerprint,
        "analysis": analysis,
    }
    return {**payload, "analysis_digest": content_digest(payload)}


def _analysis_binding_from_payload(
    payload: dict[str, object],
    *,
    expected_evidence_fingerprint: str,
) -> _AnalysisBinding:
    _expect_keys(
        payload,
        {"schema_version", "evidence_fingerprint", "analysis", "analysis_digest"},
        label="analysis binding",
    )
    if payload["schema_version"] != _ANALYSIS_BINDING_SCHEMA:
        raise ArtifactIntegrityError("unsupported analysis binding schema")
    fingerprint = _as_string(
        payload["evidence_fingerprint"], field="evidence_fingerprint"
    )
    if fingerprint != expected_evidence_fingerprint:
        raise ArtifactIntegrityError("analysis binding references another EvidenceSet")
    analysis = _as_dict(payload["analysis"], field="analysis")
    if analysis.get("schema_version") != _WITHIN_ANALYSIS_SCHEMA:
        raise ArtifactIntegrityError("unsupported within-EvidenceSet analysis schema")
    inner_digest = _as_string(analysis.get("analysis_digest"), field="analysis_digest")
    inner_body = dict(analysis)
    inner_body.pop("analysis_digest", None)
    if content_digest(inner_body) != inner_digest:
        raise ArtifactIntegrityError("within-EvidenceSet analysis digest mismatch")
    binding_digest = _as_string(
        payload["analysis_digest"], field="analysis binding digest"
    )
    binding_body = dict(payload)
    binding_body.pop("analysis_digest", None)
    if content_digest(binding_body) != binding_digest:
        raise ArtifactIntegrityError("analysis binding digest mismatch")
    return _AnalysisBinding(
        evidence_fingerprint=fingerprint,
        analysis=analysis,
        analysis_digest=binding_digest,
    )


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
    if loaded.semantic_config != expected_semantic:
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
        if baseline.semantic_config != _semantic_without_seed(plan.baseline_config):
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
            if candidate.semantic_config != _semantic_without_seed(
                definition.candidate_config
            ):
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
            factor_effect = compare_evidence_sets(
                baseline_for_comparison.runs,
                candidate.runs,
                n_bootstrap=plan.n_bootstrap,
                bootstrap_seed=plan.bootstrap_seed,
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
    if frozen is not None and frozen.study_digest != plan.digest:
        raise ArtifactIntegrityError("freeze Study digest mismatch")

    return _StudyState(
        plan=plan,
        baseline=baseline,
        baseline_analysis=baseline_analysis,
        experiments=tuple(experiment_states),
        lineage_evidence_digests=tuple(lineage),
        terminal_sequences=tuple(terminal),
        frozen=frozen,
    )


def _assert_mutable(state: _StudyState) -> None:
    if state.frozen is not None:
        raise StudyFrozenError("Study is frozen and cannot be mutated")


def _experiment_state(state: _StudyState, sequence: int) -> _ExperimentState:
    for experiment in state.experiments:
        if experiment.sequence == sequence:
            return experiment
    raise InvalidExperimentStateError(
        f"experiment {sequence:04d} definition does not exist"
    )


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


__all__ = [
    "StudySnapshot",
    "compare_experiment",
    "create_study",
    "decide_experiment",
    "define_experiment",
    "inspect_study",
    "record_experiment_failure",
    "run_baseline",
    "run_experiment",
    "verify_experiment",
]
