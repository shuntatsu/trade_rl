"""Persisted Study payload codecs and stable semantic identity helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
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
)
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    ContractViolationError,
)
from trade_rl.evaluation.runs import CandidateRunConfig

_ANALYSIS_BINDING_SCHEMA = "controlled_evidence_analysis_binding_v1"
_WITHIN_ANALYSIS_SCHEMA = "controlled_evidence_analysis_v1"


@dataclass(frozen=True, slots=True)
class _AnalysisBinding:
    evidence_fingerprint: str
    analysis: dict[str, object]
    analysis_digest: str


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
    schema_version = _as_string(
        raw.get("schema_version"), field=f"{field}.schema_version"
    )
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
    if schema_version == "resolved_run_config_v2":
        expected.update({"ppo_observation_schema", "ppo_global_feature_names"})
    elif schema_version != "resolved_run_config_v1":
        raise ArtifactIntegrityError("unsupported resolved-run config schema")
    _expect_keys(raw, expected, label=field)
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
            ppo_observation_schema=(
                None
                if schema_version == "resolved_run_config_v1"
                else _as_string(
                    raw["ppo_observation_schema"],
                    field=f"{field}.ppo_observation_schema",
                )
            ),
            ppo_global_feature_names=(
                ()
                if schema_version == "resolved_run_config_v1"
                else _as_string_tuple(
                    raw["ppo_global_feature_names"],
                    field=f"{field}.ppo_global_feature_names",
                )
            ),
            schema_version=schema_version,
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


def _semantic_without_seed(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


def _semantic_payload_without_seed(
    payload: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("ppo_seed", None)
    return normalized


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
