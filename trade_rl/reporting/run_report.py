"""Deterministic, read-only collection of persisted run evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)

RUN_REPORT_STAGE_ORDER = (
    "signal",
    "selection",
    "teacher_admission",
    "teacher_package",
    "behavior_cloning",
    "critic_warm_start",
    "ppo",
    "zero_shot",
    "sealed_evaluation",
)
_RUN_REPORT_SCHEMA = "run_report_v1"
_GENERIC_STAGE_SCHEMA = "run_report_stage_evidence_v1"
_GENERIC_STAGES = frozenset(RUN_REPORT_STAGE_ORDER[4:])

_SIGNAL_RESULT_FIELDS = frozenset(
    {
        "evidence",
        "fit_config_digest",
        "passed",
        "promotion_eligible",
        "schema_version",
        "unavailable_scope_contract_digests",
    }
)
_SIGNAL_EVIDENCE_FIELDS = frozenset(
    {
        "aggregation_mode",
        "artifact_digest",
        "direction_accuracy_excess",
        "expected_independent_episode_count",
        "expected_raw_scope_count",
        "gate_digest",
        "independence_unit",
        "independent_episode_count",
        "metric_digests",
        "passed",
        "promotion_eligible",
        "rank_ic",
        "raw_scope_count",
        "raw_scope_coverage",
        "rejection_reasons",
        "run_manifest_digest",
        "schema_version",
        "top_bottom_spread",
    }
)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "artifact_digest",
        "block_size",
        "lower_ci",
        "mean",
        "p_value",
        "schema_version",
        "upper_ci",
    }
)
_SIGNAL_REJECTION_FIELDS = frozenset(
    {"artifact_digest", "fit_results", "promotion_eligible", "schema_version"}
)
_SELECTION_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_digest",
        "candidate_evidence_digests",
        "freeze_digest",
        "promotion_eligible",
        "schema_version",
        "selected_candidate_digest",
    }
)
_SELECTION_REJECTION_FIELDS = frozenset(
    {"artifact_digest", "candidate_evidence_digests", "schema_version"}
)
_SELECTION_PROGRESS_FIELDS = frozenset(
    {
        "candidates",
        "completed_replay_count",
        "completion_fraction",
        "diagnostics_completed_count",
        "expected_replay_count",
        "fit_cache_hits",
        "fit_count",
        "promotion_eligible",
        "research_only",
        "schema_version",
        "symbols",
    }
)
_ADMISSION_EVIDENCE_FIELDS = frozenset(
    {
        "aggregate_gross_return",
        "aggregate_net_return",
        "artifact_digest",
        "base_admission_digest",
        "hard_risk_violation_count",
        "negative_gross_symbol_count",
        "passed",
        "promotion_eligible",
        "record_digests",
        "rejection_reasons",
        "schema_version",
        "total_trade_count",
        "unexplained_execution_rejection_count",
        "worst_symbol_net_return",
    }
)
_ADMISSION_REJECTION_FIELDS = frozenset(
    {
        "admission_digest",
        "artifact_digest",
        "promotion_eligible",
        "schema_version",
        "selected_candidate_digest",
    }
)
_TEACHER_PACKAGE_FIELDS = frozenset(
    {
        "admission_contract_digests",
        "artifact_digest",
        "batch_artifact_digests",
        "batch_digests",
        "freeze_digest",
        "generator_code_digest",
        "partition_digests",
        "promotion_eligible",
        "research_only",
        "run_manifest_digest",
        "sample_digests",
        "schema_version",
        "selected_candidate_digest",
        "selection_digest",
        "teacher_admission_digest",
        "teacher_admission_passed",
        "train_symbols",
    }
)
_GENERIC_STAGE_FIELDS = frozenset(
    {
        "artifact_digest",
        "artifact_digests",
        "metrics",
        "reasons",
        "schema_version",
        "stage",
        "status",
    }
)


class RunStageStatus(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_RUN = "NOT_RUN"
    MISSING = "MISSING"
    INVALID = "INVALID"


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RunStageReport:
    name: str
    status: RunStageStatus
    metrics: Mapping[str, object] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    artifact_digests: Mapping[str, str] = field(default_factory=dict)
    source_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in RUN_REPORT_STAGE_ORDER:
            raise ValueError("run report stage name is unsupported")
        if not isinstance(self.status, RunStageStatus):
            raise ValueError("run report stage status is invalid")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("run report stage reasons must be non-empty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("run report stage reasons must be unique")
        digests = dict(self.artifact_digests)
        if any(
            not isinstance(name, str) or not name or not _is_digest(digest)
            for name, digest in digests.items()
        ):
            raise ValueError("run report artifact digests are invalid")
        source_paths = tuple(self.source_paths)
        if any(not isinstance(path, str) or not path for path in source_paths):
            raise ValueError("run report source paths are invalid")
        object.__setattr__(self, "metrics", _mapping_copy(self.metrics))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "artifact_digests", MappingProxyType(digests))
        object.__setattr__(self, "source_paths", source_paths)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digests": dict(self.artifact_digests),
            "metrics": dict(self.metrics),
            "name": self.name,
            "reasons": list(self.reasons),
            "source_paths": list(self.source_paths),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    root: str
    identities: Mapping[str, object]
    stages: tuple[RunStageReport, ...]
    schema_version: str = _RUN_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root:
            raise ValueError("run report root must be non-empty")
        if self.schema_version != _RUN_REPORT_SCHEMA:
            raise ValueError("unsupported run report schema")
        stages = tuple(self.stages)
        if tuple(stage.name for stage in stages) != RUN_REPORT_STAGE_ORDER:
            raise ValueError("run report stage order is invalid")
        object.__setattr__(self, "identities", _mapping_copy(self.identities))
        object.__setattr__(self, "stages", stages)

    def to_payload(self) -> dict[str, object]:
        return {
            "identities": dict(self.identities),
            "root": self.root,
            "schema_version": self.schema_version,
            "stages": [stage.to_payload() for stage in self.stages],
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(raw)


def _strict_payload(
    raw: Mapping[str, Any],
    *,
    fields: frozenset[str],
    schema: str,
    label: str,
) -> dict[str, Any]:
    values = dict(raw)
    if set(values) != fields:
        missing = sorted(fields - set(values))
        unknown = sorted(set(values) - fields)
        raise ValueError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    if values.get("schema_version") != schema:
        raise ValueError(f"{label} schema is unsupported")
    return values


def _validate_content_digest(raw: Mapping[str, Any], *, label: str) -> str:
    values = dict(raw)
    digest = values.pop("artifact_digest", None)
    if not _is_digest(digest):
        raise ValueError(f"{label} artifact digest is invalid")
    if content_digest(values) != digest:
        raise ValueError(f"{label} artifact digest mismatch")
    return str(digest)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _missing(name: str) -> RunStageReport:
    return RunStageReport(name=name, status=RunStageStatus.MISSING)


def _not_run(
    name: str, *, reason: str = "blocked_by_upstream_rejection"
) -> RunStageReport:
    return RunStageReport(
        name=name,
        status=RunStageStatus.NOT_RUN,
        reasons=(reason,),
    )


def _invalid(
    name: str,
    *,
    reason: str,
    source_paths: tuple[str, ...] = (),
) -> RunStageReport:
    return RunStageReport(
        name=name,
        status=RunStageStatus.INVALID,
        reasons=(reason,),
        source_paths=source_paths,
    )


def _collect_identities(root: Path) -> tuple[dict[str, object], str | None]:
    paths = {
        "execution": root / "execution-identity.json",
        "manifest": root / "run-manifest.json",
        "config": root / "authored-config.json",
    }
    present = {name for name, path in paths.items() if path.is_file()}
    if not present:
        return {}, None
    if present != set(paths):
        return {}, "identity_artifact_set_incomplete"
    try:
        execution = CausalAlphaV3ExecutionIdentity.from_payload(
            _read_mapping(paths["execution"], label="V3 execution identity")
        )
        manifest = CausalAlphaV3RunManifestV2.from_payload(
            _read_mapping(paths["manifest"], label="V3 run manifest")
        )
        config = CausalAlphaV3ResearchConfig.from_mapping(
            _read_mapping(paths["config"], label="V3 authored config")
        )
        if manifest.execution_identity_digest != execution.digest:
            raise ValueError("V3 manifest execution identity mismatch")
        if manifest.training_contract_digest != execution.training_contract_digest:
            raise ValueError("V3 manifest training contract mismatch")
        if (
            manifest.instrument_context_schema_digest
            != execution.instrument_context_schema_digest
        ):
            raise ValueError("V3 manifest instrument context mismatch")
        if manifest.train_symbols != execution.train_symbols:
            raise ValueError("V3 manifest train symbol scope mismatch")
        if manifest.config_digest != config.digest:
            raise ValueError("V3 manifest authored config mismatch")
    except (KeyError, TypeError, ValueError) as error:
        return {}, f"identity_validation_failed:{type(error).__name__}"
    return {
        "catalog_digest": manifest.catalog_digest,
        "config_digest": manifest.config_digest,
        "dependency_lock_digest": execution.dependency_lock_digest,
        "execution_identity_digest": execution.digest,
        "feature_schema_digest": manifest.feature_schema_digest,
        "generator_code_digest": manifest.generator_code_digest,
        "instrument_context_schema_digest": manifest.instrument_context_schema_digest,
        "nested_partition_digest": manifest.nested_partition_digest,
        "partition_digest": manifest.partition_digest,
        "python_runtime_digest": execution.python_runtime_digest,
        "run_manifest_digest": manifest.digest,
        "shared_clock_digest": execution.shared_clock_digest,
        "source_tree_digest": execution.source_tree_digest,
        "split_manifest_digest": manifest.split_manifest_digest,
        "statistics_digest": manifest.statistics_digest,
        "train_symbols": execution.train_symbols,
        "training_contract_digest": manifest.training_contract_digest,
    }, None


def _known_config(root: Path) -> CausalAlphaV3ResearchConfig | None:
    path = root / "authored-config.json"
    if not path.is_file():
        return None
    return CausalAlphaV3ResearchConfig.from_mapping(
        _read_mapping(path, label="V3 authored config")
    )


def _signal_evidence_row(
    evidence: Mapping[str, Any],
    *,
    fit_config_digest: str,
    run_manifest_digest: object,
) -> dict[str, object]:
    values = _strict_payload(
        evidence,
        fields=_SIGNAL_EVIDENCE_FIELDS,
        schema="causal_alpha_v3_signal_gate_evidence_v2",
        label="V3 signal evidence",
    )
    digest = _validate_content_digest(values, label="V3 signal evidence")
    if values["promotion_eligible"] is not False:
        raise ValueError("V3 signal evidence promotion flag is invalid")
    if (
        run_manifest_digest is not None
        and values["run_manifest_digest"] != run_manifest_digest
    ):
        raise ValueError("V3 signal evidence run identity mismatch")
    passed = values["passed"]
    reasons = values["rejection_reasons"]
    if not isinstance(passed, bool) or not isinstance(reasons, list | tuple):
        raise ValueError("V3 signal pass state is invalid")
    resolved_reasons = tuple(str(item) for item in reasons)
    if passed == bool(resolved_reasons):
        raise ValueError("V3 signal pass state and rejection reasons disagree")

    bootstrap_metrics: dict[str, object] = {}
    for name in ("rank_ic", "top_bottom_spread", "direction_accuracy_excess"):
        raw = values[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"V3 signal {name} evidence is invalid")
        bootstrap = _strict_payload(
            raw,
            fields=_BOOTSTRAP_FIELDS,
            schema="causal_alpha_v3_bootstrap_evidence_v1",
            label=f"V3 signal {name} bootstrap",
        )
        _validate_content_digest(bootstrap, label=f"V3 signal {name} bootstrap")
        for field_name in ("mean", "lower_ci", "upper_ci", "p_value"):
            bootstrap_metrics[f"{name}_{field_name}"] = bootstrap[field_name]

    return {
        "artifact_digest": digest,
        "expected_independent_episode_count": values[
            "expected_independent_episode_count"
        ],
        "expected_raw_scope_count": values["expected_raw_scope_count"],
        "fit_config_digest": fit_config_digest,
        "independent_episode_count": values["independent_episode_count"],
        "passed": passed,
        "raw_scope_count": values["raw_scope_count"],
        "raw_scope_coverage": values["raw_scope_coverage"],
        "rejection_reasons": resolved_reasons,
        **bootstrap_metrics,
    }


def _collect_signal(
    root: Path,
    *,
    identities: Mapping[str, object],
    identity_error: str | None,
) -> RunStageReport:
    if identity_error is not None:
        return _invalid("signal", reason=identity_error)
    signal_root = root / "signal"
    rejection_path = signal_root / "rejection.json"
    fit_paths = tuple(
        sorted(
            path for path in signal_root.glob("*.json") if path.name != "rejection.json"
        )
    )
    if not rejection_path.is_file() and not fit_paths:
        return _missing("signal")

    source_paths: list[str] = []
    try:
        config = _known_config(root)
        valid_fit_digests = (
            None
            if config is None
            else {candidate.fit.digest for candidate in config.candidates}
        )
        rows: list[dict[str, object]] = []
        for path in fit_paths:
            raw = _strict_payload(
                _read_mapping(path, label="V3 signal fit result"),
                fields=_SIGNAL_RESULT_FIELDS,
                schema="causal_alpha_v3_fit_signal_result_v2",
                label="V3 signal fit result",
            )
            fit_digest = str(raw["fit_config_digest"])
            if not _is_digest(fit_digest):
                raise ValueError("V3 signal fit digest is invalid")
            if valid_fit_digests is not None and fit_digest not in valid_fit_digests:
                raise ValueError("V3 signal fit is outside authored config")
            if raw["promotion_eligible"] is not False or not isinstance(
                raw["passed"], bool
            ):
                raise ValueError("V3 signal fit result flags are invalid")
            evidence = raw["evidence"]
            if evidence is None:
                if raw["passed"] is not False:
                    raise ValueError("V3 signal fit passed without evidence")
                row: dict[str, object] = {
                    "fit_config_digest": fit_digest,
                    "passed": False,
                }
            elif isinstance(evidence, Mapping):
                row = _signal_evidence_row(
                    evidence,
                    fit_config_digest=fit_digest,
                    run_manifest_digest=identities.get("run_manifest_digest"),
                )
                if row["passed"] != raw["passed"]:
                    raise ValueError("V3 signal fit/evidence pass state mismatch")
            else:
                raise ValueError("V3 signal fit evidence is invalid")
            row["unavailable_scope_count"] = len(
                tuple(raw["unavailable_scope_contract_digests"])
            )
            rows.append(row)
            source_paths.append(_relative(root, path))

        passed_rows = tuple(row for row in rows if row.get("passed") is True)
        if rejection_path.is_file():
            rejection = _strict_payload(
                _read_mapping(rejection_path, label="V3 signal rejection"),
                fields=_SIGNAL_REJECTION_FIELDS,
                schema="causal_alpha_v3_signal_rejection_v2",
                label="V3 signal rejection",
            )
            rejection_digest = _validate_content_digest(
                rejection, label="V3 signal rejection"
            )
            if rejection["promotion_eligible"] is not False:
                raise ValueError("V3 signal rejection promotion flag is invalid")
            if passed_rows:
                raise ValueError("V3 signal rejection contradicts passed fit evidence")
            source_paths.append(_relative(root, rejection_path))
            return RunStageReport(
                name="signal",
                status=RunStageStatus.REJECT,
                metrics={"fit_count": len(tuple(rejection["fit_results"]))},
                reasons=("signal_gate_rejected",),
                artifact_digests={"signal_rejection": rejection_digest},
                source_paths=tuple(source_paths),
            )

        if not passed_rows:
            return RunStageReport(
                name="signal",
                status=RunStageStatus.IN_PROGRESS,
                metrics={"fit_count": len(rows), "fit_rows": tuple(rows)},
                source_paths=tuple(source_paths),
            )

        metrics: dict[str, object] = {
            "fit_count": len(rows),
            "fit_rows": tuple(rows),
            "passed_fit_count": len(passed_rows),
        }
        if len(passed_rows) == 1:
            metrics.update(
                {
                    key: value
                    for key, value in passed_rows[0].items()
                    if key
                    not in {
                        "artifact_digest",
                        "fit_config_digest",
                        "passed",
                        "rejection_reasons",
                        "unavailable_scope_count",
                    }
                }
            )
        return RunStageReport(
            name="signal",
            status=RunStageStatus.PASS,
            metrics=metrics,
            artifact_digests={
                "signal_evidence": str(passed_rows[0]["artifact_digest"])
            },
            source_paths=tuple(source_paths),
        )
    except (KeyError, TypeError, ValueError) as error:
        invalid_paths = tuple(_relative(root, path) for path in fit_paths)
        if rejection_path.is_file():
            invalid_paths += (_relative(root, rejection_path),)
        return _invalid(
            "signal",
            reason=f"signal_artifact_invalid:{type(error).__name__}",
            source_paths=invalid_paths,
        )


def _selection_progress_metrics(raw: Mapping[str, Any]) -> dict[str, object]:
    values = _strict_payload(
        raw,
        fields=_SELECTION_PROGRESS_FIELDS,
        schema="causal_alpha_v3_selection_progress_v1",
        label="V3 selection progress",
    )
    if values["promotion_eligible"] is not False or values["research_only"] is not True:
        raise ValueError("V3 selection progress safety flags are invalid")
    if not isinstance(values["candidates"], list | tuple) or not isinstance(
        values["symbols"], Mapping
    ):
        raise ValueError("V3 selection progress rows are invalid")
    return {
        "candidate_rows": tuple(dict(item) for item in values["candidates"]),
        "completed_replay_count": values["completed_replay_count"],
        "completion_fraction": values["completion_fraction"],
        "diagnostics_completed_count": values["diagnostics_completed_count"],
        "expected_replay_count": values["expected_replay_count"],
        "fit_cache_hits": values["fit_cache_hits"],
        "fit_count": values["fit_count"],
        "symbol_rows": {
            str(symbol): dict(row) for symbol, row in values["symbols"].items()
        },
    }


def _collect_selection(
    root: Path,
    *,
    signal: RunStageReport,
    identity_error: str | None,
) -> RunStageReport:
    if identity_error is not None:
        return _invalid("selection", reason=identity_error)
    evidence_path = root / "selection" / "evidence.json"
    rejection_path = root / "selection" / "rejection.json"
    progress_path = root / "selection" / "progress.json"
    existing = tuple(
        path
        for path in (evidence_path, rejection_path, progress_path)
        if path.is_file()
    )
    if signal.status is RunStageStatus.REJECT:
        if existing:
            return _invalid(
                "selection",
                reason="upstream_rejection_conflict",
                source_paths=tuple(_relative(root, path) for path in existing),
            )
        return _not_run("selection")
    if not existing:
        return _missing("selection")

    try:
        progress_metrics: dict[str, object] = {}
        source_paths: list[str] = []
        if progress_path.is_file():
            progress_metrics = _selection_progress_metrics(
                _read_mapping(progress_path, label="V3 selection progress")
            )
            source_paths.append(_relative(root, progress_path))
        if evidence_path.is_file() and rejection_path.is_file():
            raise ValueError("V3 selection terminal artifacts contradict each other")
        if evidence_path.is_file():
            evidence = _strict_payload(
                _read_mapping(evidence_path, label="V3 selection evidence"),
                fields=_SELECTION_EVIDENCE_FIELDS,
                schema="causal_alpha_v3_selection_evidence_v1",
                label="V3 selection evidence",
            )
            digest = _validate_content_digest(evidence, label="V3 selection evidence")
            if evidence["promotion_eligible"] is not False:
                raise ValueError("V3 selection evidence promotion flag is invalid")
            for key in ("freeze_digest", "selected_candidate_digest"):
                if not _is_digest(evidence[key]):
                    raise ValueError(f"V3 selection {key} is invalid")
            source_paths.append(_relative(root, evidence_path))
            return RunStageReport(
                name="selection",
                status=RunStageStatus.PASS,
                metrics={
                    **progress_metrics,
                    "candidate_evidence_count": len(
                        tuple(evidence["candidate_evidence_digests"])
                    ),
                    "freeze_digest": evidence["freeze_digest"],
                    "selected_candidate_digest": evidence["selected_candidate_digest"],
                    "selection_digest": digest,
                },
                artifact_digests={"selection_evidence": digest},
                source_paths=tuple(source_paths),
            )
        if rejection_path.is_file():
            rejection = _strict_payload(
                _read_mapping(rejection_path, label="V3 selection rejection"),
                fields=_SELECTION_REJECTION_FIELDS,
                schema="causal_alpha_v3_selection_rejection_v1",
                label="V3 selection rejection",
            )
            digest = _validate_content_digest(rejection, label="V3 selection rejection")
            source_paths.append(_relative(root, rejection_path))
            return RunStageReport(
                name="selection",
                status=RunStageStatus.REJECT,
                metrics={
                    **progress_metrics,
                    "candidate_evidence_count": len(
                        tuple(rejection["candidate_evidence_digests"])
                    ),
                },
                reasons=("selection_gate_rejected",),
                artifact_digests={"selection_rejection": digest},
                source_paths=tuple(source_paths),
            )
        return RunStageReport(
            name="selection",
            status=RunStageStatus.IN_PROGRESS,
            metrics=progress_metrics,
            source_paths=tuple(source_paths),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(
            "selection",
            reason=f"selection_artifact_invalid:{type(error).__name__}",
            source_paths=tuple(_relative(root, path) for path in existing),
        )


def _collect_admission(
    root: Path,
    *,
    signal: RunStageReport,
    selection: RunStageReport,
    identity_error: str | None,
) -> RunStageReport:
    if identity_error is not None:
        return _invalid("teacher_admission", reason=identity_error)
    evidence_path = root / "admission" / "evidence.json"
    rejection_path = root / "admission" / "rejection.json"
    existing = tuple(path for path in (evidence_path, rejection_path) if path.is_file())
    if (
        signal.status is RunStageStatus.REJECT
        or selection.status is RunStageStatus.REJECT
    ):
        if existing:
            return _invalid(
                "teacher_admission",
                reason="upstream_rejection_conflict",
                source_paths=tuple(_relative(root, path) for path in existing),
            )
        return _not_run("teacher_admission")
    if not existing:
        return _missing("teacher_admission")
    if not evidence_path.is_file():
        return _invalid(
            "teacher_admission",
            reason="admission_rejection_without_evidence",
            source_paths=tuple(_relative(root, path) for path in existing),
        )

    try:
        evidence = _strict_payload(
            _read_mapping(evidence_path, label="V3 admission evidence"),
            fields=_ADMISSION_EVIDENCE_FIELDS,
            schema="causal_alpha_v3_admission_evidence_v3",
            label="V3 admission evidence",
        )
        digest = _validate_content_digest(evidence, label="V3 admission evidence")
        if evidence["promotion_eligible"] is not False or not isinstance(
            evidence["passed"], bool
        ):
            raise ValueError("V3 admission safety flags are invalid")
        reasons_raw = evidence["rejection_reasons"]
        if not isinstance(reasons_raw, list | tuple):
            raise ValueError("V3 admission rejection reasons are invalid")
        reasons = tuple(str(item) for item in reasons_raw)
        if evidence["passed"] == bool(reasons):
            raise ValueError("V3 admission pass state and reasons disagree")

        source_paths = [_relative(root, evidence_path)]
        artifact_digests = {"admission_evidence": digest}
        if rejection_path.is_file():
            rejection = _strict_payload(
                _read_mapping(rejection_path, label="V3 admission rejection"),
                fields=_ADMISSION_REJECTION_FIELDS,
                schema="causal_alpha_v3_admission_rejection_v2",
                label="V3 admission rejection",
            )
            rejection_digest = _validate_content_digest(
                rejection, label="V3 admission rejection"
            )
            if (
                rejection["promotion_eligible"] is not False
                or rejection["admission_digest"] != digest
                or evidence["passed"] is not False
            ):
                raise ValueError("V3 admission rejection contradicts evidence")
            source_paths.append(_relative(root, rejection_path))
            artifact_digests["admission_rejection"] = rejection_digest
        elif evidence["passed"] is False:
            raise ValueError("V3 failed admission evidence lacks rejection marker")

        metrics = {
            "admission_digest": digest,
            "aggregate_gross_return": evidence["aggregate_gross_return"],
            "aggregate_net_return": evidence["aggregate_net_return"],
            "hard_risk_violation_count": evidence["hard_risk_violation_count"],
            "negative_gross_symbol_count": evidence["negative_gross_symbol_count"],
            "record_count": len(tuple(evidence["record_digests"])),
            "total_trade_count": evidence["total_trade_count"],
            "unexplained_execution_rejection_count": evidence[
                "unexplained_execution_rejection_count"
            ],
            "worst_symbol_net_return": evidence["worst_symbol_net_return"],
        }
        return RunStageReport(
            name="teacher_admission",
            status=(
                RunStageStatus.PASS if evidence["passed"] else RunStageStatus.REJECT
            ),
            metrics=metrics,
            reasons=reasons,
            artifact_digests=artifact_digests,
            source_paths=tuple(source_paths),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(
            "teacher_admission",
            reason=f"admission_artifact_invalid:{type(error).__name__}",
            source_paths=tuple(_relative(root, path) for path in existing),
        )


def _collect_teacher_package(
    root: Path,
    *,
    signal: RunStageReport,
    selection: RunStageReport,
    admission: RunStageReport,
    identities: Mapping[str, object],
    identity_error: str | None,
) -> RunStageReport:
    if identity_error is not None:
        return _invalid("teacher_package", reason=identity_error)
    path = root / "teacher" / "package.json"
    blocked = any(
        stage.status is RunStageStatus.REJECT
        for stage in (signal, selection, admission)
    )
    if blocked:
        if path.is_file():
            return _invalid(
                "teacher_package",
                reason="upstream_rejection_conflict",
                source_paths=(_relative(root, path),),
            )
        return _not_run("teacher_package")
    if not path.is_file():
        return _missing("teacher_package")

    try:
        raw = _strict_payload(
            _read_mapping(path, label="V3 teacher package"),
            fields=_TEACHER_PACKAGE_FIELDS,
            schema="universal_causal_alpha_v3_teacher_package_v2",
            label="V3 teacher package",
        )
        digest = _validate_content_digest(raw, label="V3 teacher package")
        if (
            raw["teacher_admission_passed"] is not True
            or raw["research_only"] is not True
            or raw["promotion_eligible"] is not False
        ):
            raise ValueError("V3 teacher package safety flags are invalid")
        run_manifest_digest = identities.get("run_manifest_digest")
        if (
            run_manifest_digest is not None
            and raw["run_manifest_digest"] != run_manifest_digest
        ):
            raise ValueError("V3 teacher package run identity mismatch")
        if admission.status is RunStageStatus.PASS:
            admission_digest = admission.metrics.get("admission_digest")
            if (
                admission_digest is not None
                and raw["teacher_admission_digest"] != admission_digest
            ):
                raise ValueError("V3 teacher package admission identity mismatch")
        if selection.status is RunStageStatus.PASS:
            selection_digest = selection.metrics.get("selection_digest")
            selected_candidate_digest = selection.metrics.get(
                "selected_candidate_digest"
            )
            if (
                selection_digest is not None
                and raw["selection_digest"] != selection_digest
            ):
                raise ValueError("V3 teacher package selection identity mismatch")
            if (
                selected_candidate_digest is not None
                and raw["selected_candidate_digest"] != selected_candidate_digest
            ):
                raise ValueError("V3 teacher package candidate identity mismatch")
        symbols = tuple(str(item) for item in raw["train_symbols"])
        return RunStageReport(
            name="teacher_package",
            status=RunStageStatus.PASS,
            metrics={
                "selected_candidate_digest": raw["selected_candidate_digest"],
                "teacher_admission_passed": True,
                "train_symbol_count": len(symbols),
                "train_symbols": symbols,
            },
            artifact_digests={"teacher_package": digest},
            source_paths=(_relative(root, path),),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(
            "teacher_package",
            reason=f"teacher_package_invalid:{type(error).__name__}",
            source_paths=(_relative(root, path),),
        )


def _generic_stage(
    root: Path,
    *,
    name: str,
    upstream_rejected: bool,
) -> RunStageReport:
    if name not in _GENERIC_STAGES:
        raise ValueError("generic run report stage is unsupported")
    path = root / "reporting" / "stages" / f"{name}.json"
    if not path.is_file():
        return _not_run(name) if upstream_rejected else _missing(name)

    try:
        raw = _strict_payload(
            _read_mapping(path, label=f"{name} stage evidence"),
            fields=_GENERIC_STAGE_FIELDS,
            schema=_GENERIC_STAGE_SCHEMA,
            label=f"{name} stage evidence",
        )
        _validate_content_digest(raw, label=f"{name} stage evidence")
        if raw["stage"] != name:
            raise ValueError("generic stage evidence path identity mismatch")
        try:
            status = RunStageStatus(str(raw["status"]))
        except ValueError as error:
            raise ValueError("generic stage evidence status is invalid") from error
        if status not in {
            RunStageStatus.PASS,
            RunStageStatus.REJECT,
            RunStageStatus.IN_PROGRESS,
            RunStageStatus.NOT_RUN,
        }:
            raise ValueError("generic stage evidence cannot claim reporter error state")
        metrics = raw["metrics"]
        artifact_digests = raw["artifact_digests"]
        reasons = raw["reasons"]
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(artifact_digests, Mapping)
            or not isinstance(reasons, list | tuple)
        ):
            raise ValueError("generic stage evidence payload types are invalid")
        resolved_reasons = tuple(str(item) for item in reasons)
        resolved_digests = {
            str(key): str(value) for key, value in artifact_digests.items()
        }
        if upstream_rejected and status in {
            RunStageStatus.PASS,
            RunStageStatus.IN_PROGRESS,
        }:
            return RunStageReport(
                name=name,
                status=RunStageStatus.INVALID,
                metrics=dict(metrics),
                reasons=tuple(
                    dict.fromkeys((*resolved_reasons, "upstream_rejection_conflict"))
                ),
                artifact_digests=resolved_digests,
                source_paths=(_relative(root, path),),
            )
        return RunStageReport(
            name=name,
            status=status,
            metrics=dict(metrics),
            reasons=resolved_reasons,
            artifact_digests=resolved_digests,
            source_paths=(_relative(root, path),),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(
            name,
            reason=f"stage_evidence_invalid:{type(error).__name__}",
            source_paths=(_relative(root, path),),
        )


def build_run_report(root: Path) -> RunReport:
    """Build a deterministic report from persisted evidence without re-evaluation."""

    source_root = Path(root)
    if not source_root.is_dir():
        raise ValueError("run report root must be an existing directory")
    identities, identity_error = _collect_identities(source_root)
    signal = _collect_signal(
        source_root,
        identities=identities,
        identity_error=identity_error,
    )
    selection = _collect_selection(
        source_root,
        signal=signal,
        identity_error=identity_error,
    )
    admission = _collect_admission(
        source_root,
        signal=signal,
        selection=selection,
        identity_error=identity_error,
    )
    teacher_package = _collect_teacher_package(
        source_root,
        signal=signal,
        selection=selection,
        admission=admission,
        identities=identities,
        identity_error=identity_error,
    )
    stages: list[RunStageReport] = [signal, selection, admission, teacher_package]
    upstream_rejected = any(stage.status is RunStageStatus.REJECT for stage in stages)
    for name in RUN_REPORT_STAGE_ORDER[4:]:
        stage = _generic_stage(
            source_root,
            name=name,
            upstream_rejected=upstream_rejected,
        )
        stages.append(stage)
        if stage.status is RunStageStatus.REJECT:
            upstream_rejected = True
    return RunReport(
        root=str(source_root),
        identities=identities,
        stages=tuple(stages),
    )


__all__ = [
    "RUN_REPORT_STAGE_ORDER",
    "RunReport",
    "RunStageReport",
    "RunStageStatus",
    "build_run_report",
]
