"""Pure Markdown rendering for deterministic run reports."""

from __future__ import annotations

from collections.abc import Mapping

from trade_rl.reporting.run_report import RunReport, RunStageReport


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _percentage(value: object) -> str:
    return f"{float(value) * 100.0:.6f}%"


def _render_candidate_rows(rows: object) -> list[str]:
    if not isinstance(rows, tuple | list) or not rows:
        return []
    lines = [
        "### Selection candidates",
        "",
        "| candidate | scopes | mean gross | mean net | worst net | turnover/day | rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(raw.get("name", "")),
                    str(raw.get("completed_scope_count", "")),
                    _percentage(raw.get("mean_gross_return", 0.0)),
                    _percentage(raw.get("mean_net_return", 0.0)),
                    _percentage(raw.get("worst_net_return", 0.0)),
                    _scalar(raw.get("mean_turnover_per_day", "")),
                    _scalar(raw.get("irrecoverably_rejected", "")),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_symbol_rows(rows: object) -> list[str]:
    if not isinstance(rows, Mapping) or not rows:
        return []
    lines = [
        "### Selection symbols",
        "",
        "| symbol | scopes | mean gross | mean net | turnover/day | trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for symbol, raw in rows.items():
        if not isinstance(raw, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(symbol),
                    str(raw.get("completed_scope_count", "")),
                    _percentage(raw.get("mean_gross_return", 0.0)),
                    _percentage(raw.get("mean_net_return", 0.0)),
                    _scalar(raw.get("mean_turnover_per_day", "")),
                    str(raw.get("total_trade_count", "")),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_stage_details(stage: RunStageReport) -> list[str]:
    scalar_metrics = {
        key: value
        for key, value in stage.metrics.items()
        if key not in {"candidate_rows", "symbol_rows"}
        and isinstance(value, str | int | float | bool)
    }
    if not scalar_metrics and not stage.reasons and not stage.artifact_digests:
        return []
    lines = [f"## {stage.name}", ""]
    if scalar_metrics:
        lines.extend(("| metric | value |", "| --- | ---: |"))
        for key in sorted(scalar_metrics):
            lines.append(f"| {key} | {_scalar(scalar_metrics[key])} |")
        lines.append("")
    if stage.reasons:
        lines.append("Reasons: `" + "`, `".join(stage.reasons) + "`")
        lines.append("")
    if stage.artifact_digests:
        lines.extend(("| artifact | digest |", "| --- | --- |"))
        for key in sorted(stage.artifact_digests):
            lines.append(f"| {key} | `{stage.artifact_digests[key]}` |")
        lines.append("")
    return lines


def render_run_report_markdown(report: RunReport) -> str:
    if not isinstance(report, RunReport):
        raise TypeError("Markdown run reporter requires RunReport")
    lines = [
        "# Machine Run Report",
        "",
        f"Root: `{report.root}`",
        f"Schema: `{report.schema_version}`",
        "",
    ]
    if report.identities:
        lines.extend(("## Identities", "", "| identity | value |", "| --- | --- |"))
        for key in sorted(report.identities):
            lines.append(f"| {key} | `{_scalar(report.identities[key])}` |")
        lines.append("")
    lines.extend(("## Stages", "", "| stage | status | reasons |", "| --- | --- | --- |"))
    for stage in report.stages:
        reasons = ", ".join(stage.reasons)
        lines.append(f"| {stage.name} | {stage.status.value} | {reasons} |")
    lines.append("")
    for stage in report.stages:
        lines.extend(_render_stage_details(stage))
        if stage.name == "selection":
            lines.extend(_render_candidate_rows(stage.metrics.get("candidate_rows")))
            lines.extend(_render_symbol_rows(stage.metrics.get("symbol_rows")))
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_run_report_markdown"]
