"""Public contracts and state validation for deterministic run reports."""

from __future__ import annotations

from pathlib import Path

from trade_rl.reporting._run_report_impl import (
    RUN_REPORT_STAGE_ORDER,
    RunReport,
    RunStageReport,
    RunStageStatus,
)
from trade_rl.reporting._run_report_impl import (
    build_run_report as _build_run_report,
)
from trade_rl.reporting.source_validation import validate_run_report_source_shapes


def _apply_source_shape_validation(report: RunReport, root: Path) -> RunReport:
    invalid_sources = validate_run_report_source_shapes(root)
    if not invalid_sources:
        return report
    stages = list(report.stages)
    indices = {name: index for index, name in enumerate(RUN_REPORT_STAGE_ORDER)}
    for name, source_paths in invalid_sources.items():
        index = indices[name]
        if stages[index].status is RunStageStatus.INVALID:
            continue
        stages[index] = RunStageReport(
            name=name,
            status=RunStageStatus.INVALID,
            reasons=("artifact_shape_invalid",),
            source_paths=source_paths,
        )
    return RunReport(
        root=report.root,
        identities=report.identities,
        stages=tuple(stages),
        schema_version=report.schema_version,
    )


def _upstream_not_passed(stage: RunStageReport) -> RunStageReport:
    if stage.status is RunStageStatus.INVALID:
        return stage
    return RunStageReport(
        name=stage.name,
        status=RunStageStatus.INVALID,
        metrics=stage.metrics,
        reasons=tuple(dict.fromkeys((*stage.reasons, "upstream_not_passed_conflict"))),
        artifact_digests=stage.artifact_digests,
        source_paths=stage.source_paths,
    )


def _validate_v3_stage_transitions(report: RunReport) -> RunReport:
    stages = list(report.stages)
    signal = stages[0]
    selection = stages[1]

    if (
        signal.status not in {RunStageStatus.PASS, RunStageStatus.REJECT}
        and selection.source_paths
    ):
        selection = _upstream_not_passed(selection)
        stages[1] = selection

    admission = stages[2]
    v3_selection_rejected = any(
        stage.status is RunStageStatus.REJECT for stage in (signal, selection)
    )
    if (
        not v3_selection_rejected
        and (
            signal.status is not RunStageStatus.PASS
            or selection.status is not RunStageStatus.PASS
        )
        and admission.source_paths
    ):
        admission = _upstream_not_passed(admission)
        stages[2] = admission

    teacher_package = stages[3]
    v3_admission_rejected = any(
        stage.status is RunStageStatus.REJECT
        for stage in (signal, selection, admission)
    )
    if (
        not v3_admission_rejected
        and any(
            stage.status is not RunStageStatus.PASS
            for stage in (signal, selection, admission)
        )
        and teacher_package.source_paths
    ):
        stages[3] = _upstream_not_passed(teacher_package)

    resolved = tuple(stages)
    if resolved == report.stages:
        return report
    return RunReport(
        root=report.root,
        identities=report.identities,
        stages=resolved,
        schema_version=report.schema_version,
    )


def build_run_report(root: Path) -> RunReport:
    """Build and fail closed on malformed or impossible persisted V3 states."""

    report = _apply_source_shape_validation(_build_run_report(root), root)
    return _validate_v3_stage_transitions(report)


__all__ = [
    "RUN_REPORT_STAGE_ORDER",
    "RunReport",
    "RunStageReport",
    "RunStageStatus",
    "build_run_report",
]
