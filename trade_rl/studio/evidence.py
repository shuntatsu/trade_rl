"""Read-only evidence-chain inspection for Studio run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.studio.contracts import (
    EvidenceNode,
    EvidenceReport,
    FileIntegritySummary,
)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _raw_manifest(root: Path) -> Mapping[str, Any]:
    path = root / "run.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return _mapping(value) or {}


def _declared_files(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    files = raw.get("files")
    if not isinstance(files, list):
        return result
    for item in files:
        mapped = _mapping(item)
        path = None if mapped is None else mapped.get("path")
        if mapped is not None and isinstance(path, str):
            result[path] = mapped
    return result


def _file_node(
    *,
    root: Path,
    declared: Mapping[str, Mapping[str, Any]],
    key: str,
    label: str,
    candidates: tuple[str, ...],
    required: bool,
    valid_manifest: bool,
) -> EvidenceNode:
    path = next((name for name in candidates if (root / name).is_file()), None)
    if path is None:
        return EvidenceNode(
            key=key,
            label=label,
            status="INVALID" if required else "ABSENT",
            required=required,
            detail="required artifact is missing"
            if required
            else "optional artifact is absent",
        )
    declared_item = declared.get(path)
    digest = None if declared_item is None else declared_item.get("digest")
    status: Literal["VERIFIED", "PRESENT"] = (
        "VERIFIED" if valid_manifest and declared_item is not None else "PRESENT"
    )
    return EvidenceNode(
        key=key,
        label=label,
        status=status,
        required=required,
        digest=digest if isinstance(digest, str) else None,
        path=path,
        detail="declared file closure verified"
        if status == "VERIFIED"
        else "artifact is present",
    )


def _bound_node(
    *,
    key: str,
    label: str,
    digest: object,
    required: bool,
) -> EvidenceNode:
    value = digest if isinstance(digest, str) and digest else None
    if value is None:
        return EvidenceNode(
            key=key,
            label=label,
            status="INVALID" if required else "ABSENT",
            required=required,
            detail="required identity is missing"
            if required
            else "optional identity is absent",
        )
    return EvidenceNode(
        key=key,
        label=label,
        status="PRESENT",
        required=required,
        digest=value,
        detail="identity is bound by the run manifest",
    )


def inspect_run_evidence(root: Path) -> EvidenceReport:
    """Return a structured report while preserving validator failures."""

    raw = _raw_manifest(root)
    validation_error: str | None = None
    manifest = None
    try:
        manifest = validate_training_run_directory(root)
    except (OSError, ValueError, TypeError) as error:
        validation_error = str(error)
    valid = manifest is not None
    run_id = (
        manifest.run_id if manifest is not None else str(raw.get("run_id") or root.name)
    )
    run_kind = (
        manifest.run_kind
        if manifest is not None
        else str(raw.get("run_kind") or "unknown")
    )
    selected_final = run_kind == "research_selected_final"
    declared = _declared_files(raw)

    nodes: list[EvidenceNode] = [
        EvidenceNode(
            key="run_manifest",
            label="Run manifest",
            status="VERIFIED" if valid else "INVALID",
            required=True,
            digest=(manifest.digest if manifest is not None else None),
            path="run.json" if (root / "run.json").is_file() else None,
            detail="manifest and file closure verified"
            if valid
            else (validation_error or "manifest is invalid"),
        ),
        _file_node(
            root=root,
            declared=declared,
            key="dataset_reference",
            label="Dataset reference",
            candidates=("dataset-reference.json",),
            required=True,
            valid_manifest=valid,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="configuration",
            label="Research configuration",
            candidates=("training-config.json", "walk-forward-config.json"),
            required=True,
            valid_manifest=valid,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="policy_ensemble",
            label="Policy ensemble",
            candidates=("ensemble.json", "walk-forward.json"),
            required=True,
            valid_manifest=valid,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="selection_proposal",
            label="Selection proposal",
            candidates=("selection-proposal.json",),
            required=selected_final,
            valid_manifest=valid,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="selection_authorization",
            label="Selection authorization",
            candidates=("selection-authorization.json",),
            required=selected_final,
            valid_manifest=valid,
        ),
    ]
    walk_digest = (
        manifest.walk_forward_run_digest
        if manifest is not None
        else raw.get("walk_forward_run_digest")
    )
    gate_digest = (
        manifest.gate_evidence_digest
        if manifest is not None
        else raw.get("gate_evidence_digest")
    )
    nodes.extend(
        (
            _bound_node(
                key="walk_forward",
                label="Walk-forward evidence",
                digest=walk_digest,
                required=selected_final,
            ),
            _bound_node(
                key="gate_evidence",
                label="Gate evidence",
                digest=gate_digest,
                required=selected_final,
            ),
            _file_node(
                root=root,
                declared=declared,
                key="confirmation_evidence",
                label="Confirmation evidence",
                candidates=("confirmation-evidence.json",),
                required=False,
                valid_manifest=valid,
            ),
            EvidenceNode(
                key="serving_bundle",
                label="Serving bundle",
                status="ABSENT",
                required=False,
                detail="serving bundle linkage is optional for research runs",
            ),
        )
    )

    declared_count = len(declared)
    total_size = sum(
        int(item.get("size_bytes", 0))
        for item in declared.values()
        if isinstance(item.get("size_bytes", 0), int)
    )
    files = FileIntegritySummary(
        status="VERIFIED" if valid else "INVALID",
        declared_count=declared_count,
        verified_count=declared_count if valid else 0,
        total_size_bytes=total_size,
    )
    required_invalid = any(
        node.required and node.status in {"ABSENT", "INVALID"} for node in nodes
    )
    status: Literal["VALID", "INVALID"] = (
        "VALID" if valid and not required_invalid else "INVALID"
    )
    return EvidenceReport(
        run_id=run_id,
        run_kind=run_kind,
        status=status,
        nodes=tuple(nodes),
        files=files,
        validation_error=validation_error,
    )


__all__ = ["inspect_run_evidence"]
