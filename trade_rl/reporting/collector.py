"""Read-only collection of persisted research artifacts into run reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.reporting.run_report import (
    RUN_REPORT_STAGE_ORDER,
    RunReport,
    RunStageReport,
    RunStageStatus,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)

_GENERIC_SCHEMA = "run_report_stage_evidence_v1"
_V3_COUNT = 4


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): value for key, value in raw.items()}


def _map(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _seq(value: object, field: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(value)


def _digest(payload: Mapping[str, Any], field: str) -> str:
    values = dict(payload)
    digest = values.pop("artifact_digest", None)
    if not _sha(digest) or digest != content_digest(values):
        raise ValueError(f"{field} digest mismatch")
    return str(digest)


def _reasons(value: object, field: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in _seq(value, field))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} are invalid")
    return result


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _missing(name: str) -> RunStageReport:
    return RunStageReport(name=name, status=RunStageStatus.MISSING)


def _not_run(name: str) -> RunStageReport:
    return RunStageReport(name=name, status=RunStageStatus.NOT_RUN)


def _invalid(name: str, reason: str, *paths: str) -> RunStageReport:
    return RunStageReport(
        name=name,
        status=RunStageStatus.INVALID,
        reasons=(reason,),
        source_paths=tuple(paths),
    )


def _identity(
    root: Path,
) -> tuple[
    dict[str, object],
    CausalAlphaV3RunManifestV2,
    CausalAlphaV3ResearchConfig,
]:
    execution = CausalAlphaV3ExecutionIdentity.from_payload(
        _load(root / "execution-identity.json")
    )
    manifest = CausalAlphaV3RunManifestV2.from_payload(_load(root / "run-manifest.json"))
    config = CausalAlphaV3ResearchConfig.from_mapping(_load(root / "authored-config.json"))
    if (
        manifest.execution_identity_digest != execution.digest
        or manifest.config_digest != config.digest
        or manifest.train_symbols != execution.train_symbols
        or manifest.training_contract_digest != execution.training_contract_digest
        or manifest.instrument_context_schema_digest
        != execution.instrument_context_schema_digest
    ):
        raise ValueError("V3 identity graph is inconsistent")
    return (
        {
            "catalog_digest": manifest.catalog_digest,
            "config_digest": manifest.config_digest,
            "dependency_lock_digest": execution.dependency_lock_digest,
            "execution_identity_digest": execution.digest,
            "feature_schema_digest": manifest.feature_schema_digest,
            "generator_code_digest": manifest.generator_code_digest,
            "partition_digest": manifest.partition_digest,
            "python_runtime_digest": execution.python_runtime_digest,
            "run_manifest_digest": manifest.digest,
            "source_tree_digest": execution.source_tree_digest,
            "split_manifest_digest": manifest.split_manifest_digest,
            "statistics_digest": manifest.statistics_digest,
            "train_symbols": manifest.train_symbols,
            "training_contract_digest": manifest.training_contract_digest,
        },
        manifest,
        config,
    )


def _signal(
    root: Path,
    manifest: CausalAlphaV3RunManifestV2,
    config: CausalAlphaV3ResearchConfig,
) -> RunStageReport:
    signal_root = root / "signal"
    rejection = signal_root / "rejection.json"
    fits = tuple(
        path for path in sorted(signal_root.glob("*.json")) if path.name != "rejection.json"
    )
    rows: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}
    passing = 0
    authored_fits = {candidate.fit.digest for candidate in config.candidates}
    for path in fits:
        payload = _load(path)
        if payload.get("schema_version") != "causal_alpha_v3_fit_signal_result_v2":
            raise ValueError("signal fit schema is unsupported")
        fit_digest = payload.get("fit_config_digest")
        if fit_digest not in authored_fits or path.stem != fit_digest:
            raise ValueError("signal fit identity mismatch")
        passed = payload.get("passed")
        if not isinstance(passed, bool) or payload.get("promotion_eligible") is not False:
            raise ValueError("signal fit status is invalid")
        row: dict[str, object] = {
            "fit_config_digest": fit_digest,
            "passed": passed,
            "unavailable_scope_count": len(
                _seq(payload.get("unavailable_scope_contract_digests"), "signal unavailable")
            ),
        }
        evidence_raw = payload.get("evidence")
        if evidence_raw is not None:
            evidence = _map(evidence_raw, "signal evidence")
            if (
                evidence.get("schema_version") != "causal_alpha_v3_signal_gate_evidence_v2"
                or evidence.get("run_manifest_digest") != manifest.digest
                or evidence.get("gate_digest") != config.signal_gate.digest
                or evidence.get("promotion_eligible") is not False
                or evidence.get("passed") is not passed
            ):
                raise ValueError("signal evidence identity mismatch")
            evidence_digest = _digest(evidence, "signal evidence")
            reasons = _reasons(evidence.get("rejection_reasons"), "signal reasons")
            if passed == bool(reasons):
                raise ValueError("signal evidence status is inconsistent")
            row.update(
                {
                    "independent_episode_count": evidence.get("independent_episode_count"),
                    "raw_scope_count": evidence.get("raw_scope_count"),
                    "raw_scope_coverage": evidence.get("raw_scope_coverage"),
                    "rank_ic_lower_ci": _map(evidence.get("rank_ic"), "rank_ic").get(
                        "lower_ci"
                    ),
                    "top_bottom_spread_lower_ci": _map(
                        evidence.get("top_bottom_spread"), "top_bottom_spread"
                    ).get("lower_ci"),
                    "direction_accuracy_excess_lower_ci": _map(
                        evidence.get("direction_accuracy_excess"),
                        "direction_accuracy_excess",
                    ).get("lower_ci"),
                }
            )
            artifacts[f"signal_evidence:{fit_digest}"] = evidence_digest
        elif passed:
            raise ValueError("passing signal fit is missing evidence")
        rows.append(row)
        passing += int(passed)
    if rejection.is_file():
        payload = _load(rejection)
        if (
            payload.get("schema_version") != "causal_alpha_v3_signal_rejection_v2"
            or payload.get("promotion_eligible") is not False
            or passing
        ):
            raise ValueError("signal rejection contract is invalid")
        artifacts["signal_rejection"] = _digest(payload, "signal rejection")
        return RunStageReport(
            name="signal",
            status=RunStageStatus.REJECT,
            metrics={
                "fit_count": len(_seq(payload.get("fit_results"), "signal fit results")),
                "fit_rows": tuple(rows),
                "passing_fit_count": 0,
            },
            reasons=("no_passing_fit",),
            artifact_digests=artifacts,
            source_paths=tuple(
                [str(path.relative_to(root)) for path in fits]
                + ["signal/rejection.json"]
            ),
        )
    if not fits:
        return _missing("signal")
    return RunStageReport(
        name="signal",
        status=RunStageStatus.PASS if passing else RunStageStatus.IN_PROGRESS,
        metrics={"fit_count": len(fits), "fit_rows": tuple(rows), "passing_fit_count": passing},
        artifact_digests=artifacts,
        source_paths=tuple(str(path.relative_to(root)) for path in fits),
    )


def _selection(root: Path, config: CausalAlphaV3ResearchConfig) -> RunStageReport:
    evidence = root / "selection" / "evidence.json"
    rejection = root / "selection" / "rejection.json"
    progress = root / "selection" / "progress.json"
    if evidence.is_file() and rejection.is_file():
        return _invalid(
            "selection",
            "conflicting_terminal_evidence",
            "selection/evidence.json",
            "selection/rejection.json",
        )
    if evidence.is_file():
        payload = _load(evidence)
        selected = payload.get("selected_candidate_digest")
        if (
            payload.get("schema_version") != "causal_alpha_v3_selection_evidence_v1"
            or payload.get("promotion_eligible") is not False
            or selected not in {candidate.digest for candidate in config.candidates}
        ):
            raise ValueError("selection evidence contract is invalid")
        digest = _digest(payload, "selection evidence")
        candidates = _seq(payload.get("candidate_evidence_digests"), "selection candidates")
        if any(not _sha(item) for item in candidates):
            raise ValueError("selection candidate digest is invalid")
        return RunStageReport(
            name="selection",
            status=RunStageStatus.PASS,
            metrics={"candidate_count": len(candidates), "selected_candidate_digest": selected},
            artifact_digests={"selection_evidence": digest},
            source_paths=("selection/evidence.json",),
        )
    if rejection.is_file():
        payload = _load(rejection)
        if payload.get("schema_version") != "causal_alpha_v3_selection_rejection_v1":
            raise ValueError("selection rejection schema is unsupported")
        digest = _digest(payload, "selection rejection")
        return RunStageReport(
            name="selection",
            status=RunStageStatus.REJECT,
            reasons=("no_admissible_candidate",),
            artifact_digests={"selection_rejection": digest},
            source_paths=("selection/rejection.json",),
        )
    if progress.is_file():
        payload = _load(progress)
        if (
            payload.get("schema_version") != "causal_alpha_v3_selection_progress_v1"
            or payload.get("research_only") is not True
            or payload.get("promotion_eligible") is not False
        ):
            raise ValueError("selection progress contract is invalid")
        completed = payload.get("completed_replay_count")
        expected = payload.get("expected_replay_count")
        if not isinstance(completed, int) or not isinstance(expected, int) or expected < completed:
            raise ValueError("selection replay counts are invalid")
        fraction = _number(payload.get("completion_fraction"), "selection completion")
        if abs(fraction - (0.0 if expected == 0 else completed / expected)) > 1e-12:
            raise ValueError("selection completion fraction is inconsistent")
        return RunStageReport(
            name="selection",
            status=RunStageStatus.IN_PROGRESS,
            metrics={
                "candidate_rows": tuple(
                    dict(_map(item, "selection candidate"))
                    for item in _seq(payload.get("candidates"), "selection candidates")
                ),
                "completed_replay_count": completed,
                "completion_fraction": fraction,
                "diagnostics_completed_count": payload.get("diagnostics_completed_count"),
                "expected_replay_count": expected,
                "fit_cache_hits": payload.get("fit_cache_hits"),
                "fit_count": payload.get("fit_count"),
                "symbol_rows": {
                    symbol: dict(_map(item, f"selection symbol {symbol}"))
                    for symbol, item in _map(payload.get("symbols"), "selection symbols").items()
                },
            },
            source_paths=("selection/progress.json",),
        )
    return _missing("selection")


def _admission(root: Path, selection: RunStageReport) -> RunStageReport:
    evidence = root / "admission" / "evidence.json"
    rejection = root / "admission" / "rejection.json"
    if not evidence.is_file() and not rejection.is_file():
        return _missing("teacher_admission")
    if rejection.is_file() and not evidence.is_file():
        return _invalid("teacher_admission", "rejection_without_evidence", "admission/rejection.json")
    payload = _load(evidence)
    if (
        payload.get("schema_version") != "causal_alpha_v3_admission_evidence_v3"
        or payload.get("promotion_eligible") is not False
        or not isinstance(payload.get("passed"), bool)
    ):
        raise ValueError("teacher admission evidence contract is invalid")
    digest = _digest(payload, "teacher admission evidence")
    passed = bool(payload["passed"])
    reasons = _reasons(payload.get("rejection_reasons"), "teacher admission reasons")
    if passed == bool(reasons):
        raise ValueError("teacher admission status is inconsistent")
    artifacts = {"teacher_admission_evidence": digest}
    sources = ["admission/evidence.json"]
    if rejection.is_file():
        rejected = _load(rejection)
        if (
            rejected.get("schema_version") != "causal_alpha_v3_admission_rejection_v2"
            or rejected.get("promotion_eligible") is not False
            or passed
            or rejected.get("admission_digest") != digest
        ):
            raise ValueError("teacher admission rejection contract is invalid")
        selected = selection.metrics.get("selected_candidate_digest")
        if selected is not None and rejected.get("selected_candidate_digest") != selected:
            raise ValueError("teacher admission selected candidate mismatch")
        artifacts["teacher_admission_rejection"] = _digest(
            rejected, "teacher admission rejection"
        )
        sources.append("admission/rejection.json")
    return RunStageReport(
        name="teacher_admission",
        status=RunStageStatus.PASS if passed else RunStageStatus.REJECT,
        metrics={
            "aggregate_gross_return": _number(
                payload.get("aggregate_gross_return"), "admission aggregate gross"
            ),
            "aggregate_net_return": _number(
                payload.get("aggregate_net_return"), "admission aggregate net"
            ),
            "hard_risk_violation_count": payload.get("hard_risk_violation_count"),
            "negative_gross_symbol_count": payload.get("negative_gross_symbol_count"),
            "total_trade_count": payload.get("total_trade_count"),
            "unexplained_execution_rejection_count": payload.get(
                "unexplained_execution_rejection_count"
            ),
            "worst_symbol_net_return": _number(
                payload.get("worst_symbol_net_return"), "admission worst symbol net"
            ),
        },
        reasons=reasons,
        artifact_digests=artifacts,
        source_paths=tuple(sources),
    )


def _teacher_package(
    root: Path,
    manifest: CausalAlphaV3RunManifestV2,
    selection: RunStageReport,
    admission: RunStageReport,
) -> RunStageReport:
    path = root / "teacher" / "package.json"
    if not path.is_file():
        return _missing("teacher_package")
    payload = _load(path)
    if (
        payload.get("schema_version") != "universal_causal_alpha_v3_teacher_package_v2"
        or payload.get("run_manifest_digest") != manifest.digest
        or payload.get("teacher_admission_passed") is not True
        or payload.get("research_only") is not True
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("teacher package contract is invalid")
    selected = selection.metrics.get("selected_candidate_digest")
    if selected is not None and payload.get("selected_candidate_digest") != selected:
        raise ValueError("teacher package selected candidate mismatch")
    admission_digest = admission.artifact_digests.get("teacher_admission_evidence")
    if admission_digest is not None and payload.get("teacher_admission_digest") != admission_digest:
        raise ValueError("teacher package admission identity mismatch")
    symbols = tuple(str(item) for item in _seq(payload.get("train_symbols"), "teacher symbols"))
    return RunStageReport(
        name="teacher_package",
        status=RunStageStatus.PASS,
        metrics={"batch_count": len(symbols), "train_symbols": symbols},
        artifact_digests={"teacher_package": _digest(payload, "teacher package")},
        source_paths=("teacher/package.json",),
    )


def _generic(root: Path, name: str) -> RunStageReport | None:
    path = root / "reporting" / "stages" / f"{name}.json"
    if not path.is_file():
        return None
    source = str(path.relative_to(root))
    try:
        payload = _load(path)
        if payload.get("schema_version") != _GENERIC_SCHEMA or payload.get("stage") != name:
            raise ValueError("generic stage identity is invalid")
        status = RunStageStatus(str(payload.get("status")))
        if status not in {RunStageStatus.PASS, RunStageStatus.REJECT, RunStageStatus.IN_PROGRESS}:
            raise ValueError("generic persisted status is invalid")
        reasons = _reasons(payload.get("reasons"), f"{name} reasons")
        if status is RunStageStatus.PASS and reasons:
            raise ValueError("passing stage cannot contain reasons")
        if status is RunStageStatus.REJECT and not reasons:
            raise ValueError("rejected stage requires reasons")
        digests = _map(payload.get("artifact_digests"), f"{name} digests")
        if any(not _sha(item) for item in digests.values()):
            raise ValueError("generic artifact digest is invalid")
        artifacts = {key: str(value) for key, value in digests.items()}
        artifacts["stage_evidence"] = _digest(payload, f"{name} stage evidence")
        return RunStageReport(
            name=name,
            status=status,
            metrics=_map(payload.get("metrics"), f"{name} metrics"),
            reasons=reasons,
            artifact_digests=artifacts,
            source_paths=(source,),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return _invalid(name, "invalid_stage_evidence", source)


def collect_run_report(root: Path) -> RunReport:
    source = Path(root)
    stages = [_missing(name) for name in RUN_REPORT_STAGE_ORDER]
    identities: dict[str, object] = {}
    identity_paths = tuple(
        source / name
        for name in ("execution-identity.json", "run-manifest.json", "authored-config.json")
    )
    present = tuple(path.is_file() for path in identity_paths)
    manifest: CausalAlphaV3RunManifestV2 | None = None
    config: CausalAlphaV3ResearchConfig | None = None
    if any(present):
        if not all(present):
            for index in range(_V3_COUNT):
                stages[index] = _invalid(RUN_REPORT_STAGE_ORDER[index], "partial_v3_identity")
        else:
            try:
                identities, manifest, config = _identity(source)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                for index in range(_V3_COUNT):
                    stages[index] = _invalid(RUN_REPORT_STAGE_ORDER[index], "invalid_v3_identity")
    elif any((source / name).exists() for name in ("signal", "selection", "admission", "teacher")):
        for index in range(_V3_COUNT):
            stages[index] = _invalid(RUN_REPORT_STAGE_ORDER[index], "missing_v3_identity")

    if manifest is not None and config is not None:
        try:
            stages[0] = _signal(source, manifest, config)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            stages[0] = _invalid("signal", "invalid_signal_artifact")
        if stages[0].status is RunStageStatus.REJECT:
            stages[1:4] = [_not_run(name) for name in RUN_REPORT_STAGE_ORDER[1:4]]
        elif stages[0].status is RunStageStatus.INVALID:
            stages[1:4] = [
                _invalid(name, "invalid_v3_identity_chain") for name in RUN_REPORT_STAGE_ORDER[1:4]
            ]
        else:
            try:
                stages[1] = _selection(source, config)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                stages[1] = _invalid("selection", "invalid_selection_artifact")
            if stages[1].status is RunStageStatus.REJECT:
                stages[2] = _not_run("teacher_admission")
                stages[3] = _not_run("teacher_package")
            elif stages[1].status is RunStageStatus.INVALID:
                stages[2] = _invalid("teacher_admission", "invalid_selection_chain")
                stages[3] = _invalid("teacher_package", "invalid_selection_chain")
            else:
                try:
                    stages[2] = _admission(source, stages[1])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                    stages[2] = _invalid("teacher_admission", "invalid_admission_artifact")
                if stages[2].status is RunStageStatus.REJECT:
                    stages[3] = _not_run("teacher_package")
                elif stages[2].status is RunStageStatus.INVALID:
                    stages[3] = _invalid("teacher_package", "invalid_admission_chain")
                else:
                    try:
                        stages[3] = _teacher_package(source, manifest, stages[1], stages[2])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                        stages[3] = _invalid("teacher_package", "invalid_teacher_package")

    blocked = any(stage.status is RunStageStatus.REJECT for stage in stages[:_V3_COUNT])
    for index, name in enumerate(RUN_REPORT_STAGE_ORDER[4:], start=4):
        evidence = _generic(source, name)
        if evidence is not None:
            stages[index] = (
                _invalid(name, "upstream_rejection_conflict", *evidence.source_paths)
                if blocked
                else evidence
            )
        elif blocked:
            stages[index] = _not_run(name)
        if stages[index].status is RunStageStatus.REJECT:
            blocked = True
    return RunReport(root=str(source), identities=identities, stages=tuple(stages))


__all__ = ["collect_run_report"]
