"""Deterministic Markdown read model over verified C3 core artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_c3_artifact import (
    load_c3_aggregate_report_artifact,
    load_phase_a_gate_artifact,
)

C3_MARKDOWN_ARTIFACT_SCHEMA: Final = "causal_scenario_c3_markdown_artifact_v1"
PRODUCTION_STATUS: Final = "NO-GO"
_FILES: Final = frozenset({"manifest.json", "report.md"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, field: str) -> str:
    return require_sha256(str(value), field=field)


@dataclass(frozen=True, slots=True)
class VerifiedC3Evidence:
    report: object
    gate: object
    report_artifact_digest: str
    gate_artifact_digest: str
    report_digest: str
    gate_digest: str
    passed: bool
    failed_condition_names: tuple[str, ...]
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        for name in (
            "report_artifact_digest",
            "gate_artifact_digest",
            "report_digest",
            "gate_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), field=name))
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be boolean")
        failures = tuple(str(item).strip() for item in self.failed_condition_names)
        if any(not item for item in failures) or len(set(failures)) != len(failures):
            raise ValueError("failed_condition_names must be unique and non-empty")
        object.__setattr__(self, "failed_condition_names", failures)
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 evidence production status must remain NO-GO")


@dataclass(frozen=True, slots=True)
class LoadedC3MarkdownArtifact:
    root: Path
    artifact_digest: str
    report_artifact_digest: str
    gate_artifact_digest: str
    report_digest: str
    gate_digest: str
    passed: bool
    failed_condition_names: tuple[str, ...]
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for name in (
            "artifact_digest",
            "report_artifact_digest",
            "gate_artifact_digest",
            "report_digest",
            "gate_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), field=name))
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be boolean")
        failures = tuple(str(item).strip() for item in self.failed_condition_names)
        if any(not item for item in failures) or len(set(failures)) != len(failures):
            raise ValueError("failed_condition_names must be unique and non-empty")
        object.__setattr__(self, "failed_condition_names", failures)
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 Markdown production status must remain NO-GO")


def verify_c3_evidence(
    *,
    report_root: str | Path,
    gate_root: str | Path,
) -> VerifiedC3Evidence:
    """Load the authoritative artifacts and verify their identity binding."""

    loaded_report = load_c3_aggregate_report_artifact(report_root)
    loaded_gate = load_phase_a_gate_artifact(gate_root)
    report = loaded_report.report
    gate = loaded_gate.gate
    if gate.report_digest != report.digest:
        raise ValueError("C3 gate does not bind the verified aggregate report")
    return VerifiedC3Evidence(
        report=report,
        gate=gate,
        report_artifact_digest=loaded_report.artifact_digest,
        gate_artifact_digest=loaded_gate.artifact_digest,
        report_digest=report.digest,
        gate_digest=gate.digest,
        passed=gate.passed,
        failed_condition_names=gate.failed_condition_names,
    )


def render_c3_markdown(report: object, gate: object) -> str:
    """Render a deterministic human-readable view of authoritative C3 evidence."""

    report_digest = _digest(getattr(report, "digest"), field="report.digest")
    gate_report_digest = _digest(
        getattr(gate, "report_digest"), field="gate.report_digest"
    )
    if gate_report_digest != report_digest:
        raise ValueError("C3 gate does not bind the aggregate report")
    passed = getattr(gate, "passed")
    if not isinstance(passed, bool):
        raise ValueError("C3 gate pass state must be boolean")
    lines = [
        "# Causal Scenario C3 Evaluation Report",
        "",
        f"Report digest: `{report_digest}`",
        f"Gate digest: `{_digest(getattr(gate, 'digest'), field='gate.digest')}`",
        f"Gate config digest: `{_digest(getattr(gate, 'config_digest'), field='gate.config_digest')}`",
        f"Production status: **{PRODUCTION_STATUS}**",
        f"Phase A gate: **{'PASS' if passed else 'BLOCKED'}**",
        "",
        "## Aggregate evidence",
        "",
        f"- Fold support: {getattr(report, 'fold_count')}",
        (
            f"- Selection/effective days: {getattr(report, 'total_selection_days')}/"
            f"{getattr(report, 'total_effective_days')}"
        ),
        f"- Positive uplift folds: {getattr(report, 'positive_uplift_folds')}",
        (
            f"- Uplift: {getattr(report, 'mean_uplift'):.12g} "
            f"[{getattr(report, 'uplift_lower_ci'):.12g}, "
            f"{getattr(report, 'uplift_upper_ci'):.12g}], "
            f"p={getattr(report, 'uplift_p_value'):.12g}"
        ),
        (
            f"- Predicted-realized Spearman: {getattr(report, 'mean_spearman'):.12g} "
            f"[{getattr(report, 'spearman_lower_ci'):.12g}, "
            f"{getattr(report, 'spearman_upper_ci'):.12g}]"
        ),
        (
            f"- Regret improvement: {getattr(report, 'mean_regret_margin'):.12g} "
            f"[{getattr(report, 'regret_margin_lower_ci'):.12g}, "
            f"{getattr(report, 'regret_margin_upper_ci'):.12g}]"
        ),
        (
            f"- Worst drawdown, scenario oracle/trend: "
            f"{getattr(report, 'worst_scenario_oracle_drawdown'):.12g}/"
            f"{getattr(report, 'worst_trend_drawdown'):.12g}"
        ),
        "",
        "## Fold evidence",
        "",
        "| Fold | Days | Uplift | Spearman | Regret margin | Adverse | Perfect info |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for fold in getattr(report, "folds"):
        lines.append(
            f"| {fold.fold_id} | {fold.effective_days}/{fold.selection_days} | "
            f"{fold.mean_uplift:.12g} | {fold.mean_spearman:.12g} | "
            f"{fold.mean_regret_margin:.12g} | "
            f"{'pass' if fold.required_adverse_passed else 'fail'} | "
            f"{'valid' if fold.perfect_information_valid else 'invalid'} |"
        )
    lines.extend(
        [
            "",
            "## Execution scenarios",
            "",
            "| Scenario | Policy | Observations | Gross return | Economic cost | Fill ratio | Max DD |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in getattr(report, "execution_summaries"):
        lines.append(
            f"| {item.execution_scenario} | {item.policy_kind} | "
            f"{item.observation_count} | {item.mean_gross_log_return:.12g} | "
            f"{item.mean_total_economic_cost:.12g} | {item.mean_fill_ratio:.12g} | "
            f"{item.maximum_drawdown:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            (
                f"- Neighbor distance p50/p90/p99: "
                f"{getattr(report, 'neighbor_distance_p50'):.12g}/"
                f"{getattr(report, 'neighbor_distance_p90'):.12g}/"
                f"{getattr(report, 'neighbor_distance_p99'):.12g}"
            ),
            (
                f"- Anchors unique/effective/max-share: "
                f"{getattr(report, 'unique_anchor_count')}/"
                f"{getattr(report, 'effective_anchor_count'):.12g}/"
                f"{getattr(report, 'anchor_max_share'):.12g}"
            ),
            f"- Historical coverage: {getattr(report, 'historical_coverage_fraction'):.12g}",
            f"- Calibration buckets: {len(getattr(report, 'calibration_buckets'))}",
            "",
            "## Phase A conditions",
            "",
        ]
    )
    for condition in getattr(gate, "conditions"):
        lines.append(
            f"- [{'x' if condition.passed else ' '}] `{condition.name}` — {condition.detail}"
        )
    lines.extend(["", "## Failure reasons", ""])
    failures = tuple(getattr(report, "failure_reasons"))
    if failures:
        lines.extend(f"- {reason}" for reason in failures)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "A passing Phase A gate authorizes only the next evaluation phase and does not authorize production.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_closure(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"C3 Markdown artifact directory is missing: {root}")
    names: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("C3 Markdown artifact contains an invalid file entry")
        names.add(entry.name)
    if names != set(_FILES):
        raise ValueError("C3 Markdown artifact file closure mismatch")


def _artifact_payload(evidence: VerifiedC3Evidence) -> tuple[dict[str, bytes], str]:
    markdown = render_c3_markdown(evidence.report, evidence.gate).encode("utf-8")
    base: dict[str, object] = {
        "gate_artifact_digest": evidence.gate_artifact_digest,
        "gate_digest": evidence.gate_digest,
        "markdown_sha256": _sha256(markdown),
        "passed": evidence.passed,
        "production_status": PRODUCTION_STATUS,
        "report_artifact_digest": evidence.report_artifact_digest,
        "report_digest": evidence.report_digest,
        "schema_version": C3_MARKDOWN_ARTIFACT_SCHEMA,
    }
    artifact_digest = content_digest(base)
    manifest = dict(base)
    manifest["artifact_digest"] = artifact_digest
    return {
        "manifest.json": canonical_json_bytes(manifest),
        "report.md": markdown,
    }, artifact_digest


def _publish_exact(root: Path, expected: dict[str, bytes]) -> None:
    if root.exists() and any(root.iterdir()):
        actual_names = {entry.name for entry in root.iterdir()}
        if actual_names != set(expected) or any(
            not (root / name).is_file()
            or (root / name).is_symlink()
            or (root / name).read_bytes() != payload
            for name, payload in expected.items()
        ):
            raise FileExistsError(
                f"conflicting C3 Markdown artifact already exists: {root}"
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        _atomic_write(root / name, expected[name])


def write_c3_markdown_artifact(
    root: str | Path,
    *,
    report_root: str | Path,
    gate_root: str | Path,
) -> LoadedC3MarkdownArtifact:
    evidence = verify_c3_evidence(report_root=report_root, gate_root=gate_root)
    destination = Path(root)
    expected, _ = _artifact_payload(evidence)
    _publish_exact(destination, expected)
    return load_c3_markdown_artifact(
        destination,
        report_root=report_root,
        gate_root=gate_root,
    )


def load_c3_markdown_artifact(
    root: str | Path,
    *,
    report_root: str | Path,
    gate_root: str | Path,
) -> LoadedC3MarkdownArtifact:
    source = Path(root)
    _verify_closure(source)
    evidence = verify_c3_evidence(report_root=report_root, gate_root=gate_root)
    manifest_bytes = (source / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C3 Markdown manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or any(
        not isinstance(key, str) for key in manifest
    ):
        raise ValueError("C3 Markdown manifest must be an object")
    expected_fields = {
        "artifact_digest",
        "gate_artifact_digest",
        "gate_digest",
        "markdown_sha256",
        "passed",
        "production_status",
        "report_artifact_digest",
        "report_digest",
        "schema_version",
    }
    if set(manifest) != expected_fields:
        raise ValueError("C3 Markdown manifest field closure mismatch")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("C3 Markdown manifest must be canonical JSON")
    if manifest["schema_version"] != C3_MARKDOWN_ARTIFACT_SCHEMA:
        raise ValueError("unsupported C3 Markdown artifact schema")
    if manifest["production_status"] != PRODUCTION_STATUS:
        raise ValueError("C3 Markdown production status must remain NO-GO")
    artifact_digest = _digest(manifest["artifact_digest"], field="artifact_digest")
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("C3 Markdown artifact digest mismatch")
    expected_identity = {
        "report_artifact_digest": evidence.report_artifact_digest,
        "gate_artifact_digest": evidence.gate_artifact_digest,
        "report_digest": evidence.report_digest,
        "gate_digest": evidence.gate_digest,
        "passed": evidence.passed,
    }
    if any(manifest[name] != value for name, value in expected_identity.items()):
        raise ValueError("C3 Markdown artifact identity mismatch")
    markdown = (source / "report.md").read_bytes()
    if _sha256(markdown) != _digest(
        manifest["markdown_sha256"], field="markdown_sha256"
    ):
        raise ValueError("C3 Markdown file digest mismatch")
    expected_markdown = render_c3_markdown(evidence.report, evidence.gate).encode(
        "utf-8"
    )
    if markdown != expected_markdown:
        raise ValueError("C3 Markdown content does not match verified evidence")
    return LoadedC3MarkdownArtifact(
        root=source,
        artifact_digest=artifact_digest,
        report_artifact_digest=evidence.report_artifact_digest,
        gate_artifact_digest=evidence.gate_artifact_digest,
        report_digest=evidence.report_digest,
        gate_digest=evidence.gate_digest,
        passed=evidence.passed,
        failed_condition_names=evidence.failed_condition_names,
    )


__all__ = [
    "LoadedC3MarkdownArtifact",
    "VerifiedC3Evidence",
    "load_c3_markdown_artifact",
    "render_c3_markdown",
    "verify_c3_evidence",
    "write_c3_markdown_artifact",
]
