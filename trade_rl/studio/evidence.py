"""Read-only evidence coverage and internal identity inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    validate_training_run_directory,
)
from trade_rl.studio.contracts import EvidenceNode, EvidenceReport, FileIntegritySummary
from trade_rl.workflows.selection_authorization import (
    load_selection_authorization,
    load_selection_proposal,
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


def _selection_nodes(
    root: Path,
    manifest: TrainingRunManifest | None,
    *,
    required: bool,
) -> tuple[EvidenceNode, EvidenceNode, bool]:
    proposal_path = root / "selection-proposal.json"
    authorization_path = root / "selection-authorization.json"
    if not proposal_path.is_file():
        proposal_node = EvidenceNode(
            key="selection_proposal",
            label="Selection proposal",
            status="INVALID" if required else "ABSENT",
            required=required,
            detail="required artifact is missing"
            if required
            else "optional artifact is absent",
        )
        proposal = None
    else:
        try:
            proposal = load_selection_proposal(proposal_path)
            if manifest is None:
                raise ValueError("run manifest is invalid")
            if proposal.digest != manifest.selection_proposal_digest:
                raise ValueError("proposal digest differs from run manifest")
            if proposal.dataset_id != manifest.dataset_id:
                raise ValueError("proposal dataset identity differs from run manifest")
            if proposal.walk_forward_run_digest != manifest.walk_forward_run_digest:
                raise ValueError(
                    "proposal walk-forward identity differs from run manifest"
                )
            if proposal.gate_evidence_digest != manifest.gate_evidence_digest:
                raise ValueError(
                    "proposal gate evidence identity differs from run manifest"
                )
            proposal_node = EvidenceNode(
                key="selection_proposal",
                label="Selection proposal",
                status="VERIFIED",
                required=required,
                digest=proposal.digest,
                path=proposal_path.name,
                detail="proposal content and manifest bindings verified",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            proposal = None
            proposal_node = EvidenceNode(
                key="selection_proposal",
                label="Selection proposal",
                status="INVALID",
                required=required,
                path=proposal_path.name,
                detail=str(error),
            )

    if not authorization_path.is_file():
        authorization_node = EvidenceNode(
            key="selection_authorization",
            label="Selection authorization",
            status="INVALID" if required else "ABSENT",
            required=required,
            detail="required artifact is missing"
            if required
            else "optional artifact is absent",
        )
        authorization_valid = False
    else:
        try:
            authorization = load_selection_authorization(authorization_path)
            if manifest is None:
                raise ValueError("run manifest is invalid")
            if (
                authorization.authorization_digest
                != manifest.selection_authorization_digest
            ):
                raise ValueError("authorization digest differs from run manifest")
            if proposal is None or authorization.proposal_digest != proposal.digest:
                raise ValueError("authorization proposal binding is invalid")
            authorization_node = EvidenceNode(
                key="selection_authorization",
                label="Selection authorization",
                status="VERIFIED",
                required=required,
                digest=authorization.authorization_digest,
                path=authorization_path.name,
                detail="authorization content and proposal binding verified; signature trust remains external",
            )
            authorization_valid = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            authorization_node = EvidenceNode(
                key="selection_authorization",
                label="Selection authorization",
                status="INVALID",
                required=required,
                path=authorization_path.name,
                detail=str(error),
            )
            authorization_valid = False
    return (
        proposal_node,
        authorization_node,
        proposal is not None and authorization_valid,
    )


def inspect_run_evidence(root: Path, *, run_resource_id: str) -> EvidenceReport:
    """Return a structured report while preserving validator and binding failures."""

    raw = _raw_manifest(root)
    validation_error: str | None = None
    manifest = None
    try:
        manifest = validate_training_run_directory(root)
    except (OSError, ValueError, TypeError) as error:
        validation_error = str(error)
    valid_manifest = manifest is not None
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

    proposal_node, authorization_node, selection_valid = _selection_nodes(
        root,
        manifest,
        required=selected_final,
    )
    walk_digest = None if manifest is None else manifest.walk_forward_run_digest
    gate_digest = None if manifest is None else manifest.gate_evidence_digest
    bound_status: Literal["VERIFIED", "PRESENT", "ABSENT", "INVALID"]
    if selected_final:
        bound_status = "VERIFIED" if selection_valid else "INVALID"
    else:
        bound_status = "ABSENT"

    nodes: list[EvidenceNode] = [
        EvidenceNode(
            key="run_manifest",
            label="Run manifest",
            status="VERIFIED" if valid_manifest else "INVALID",
            required=True,
            digest=(manifest.digest if manifest is not None else None),
            path="run.json" if (root / "run.json").is_file() else None,
            detail=(
                "manifest and file closure verified"
                if valid_manifest
                else (validation_error or "manifest is invalid")
            ),
        ),
        _file_node(
            root=root,
            declared=declared,
            key="dataset_reference",
            label="Dataset reference",
            candidates=("dataset-reference.json",),
            required=True,
            valid_manifest=valid_manifest,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="configuration",
            label="Research configuration",
            candidates=("training-config.json", "walk-forward-config.json"),
            required=True,
            valid_manifest=valid_manifest,
        ),
        _file_node(
            root=root,
            declared=declared,
            key="policy_ensemble",
            label="Policy ensemble",
            candidates=("ensemble.json", "walk-forward.json"),
            required=True,
            valid_manifest=valid_manifest,
        ),
        proposal_node,
        authorization_node,
        EvidenceNode(
            key="walk_forward",
            label="Walk-forward evidence",
            status=bound_status,
            required=selected_final,
            digest=walk_digest,
            detail=(
                "proposal and manifest bind the walk-forward identity"
                if bound_status == "VERIFIED"
                else "walk-forward identity is not fully verified"
                if selected_final
                else "optional identity is absent"
            ),
        ),
        EvidenceNode(
            key="gate_evidence",
            label="Gate evidence",
            status=bound_status,
            required=selected_final,
            digest=gate_digest,
            detail=(
                "proposal and manifest bind the gate evidence identity"
                if bound_status == "VERIFIED"
                else "gate evidence identity is not fully verified"
                if selected_final
                else "optional identity is absent"
            ),
        ),
        _file_node(
            root=root,
            declared=declared,
            key="confirmation_evidence",
            label="Confirmation evidence",
            candidates=("confirmation-evidence.json",),
            required=False,
            valid_manifest=valid_manifest,
        ),
        EvidenceNode(
            key="serving_bundle",
            label="Serving bundle",
            status="ABSENT",
            required=False,
            detail="serving bundle linkage is optional for research runs",
        ),
    ]

    declared_count = len(declared)
    total_size = sum(
        int(item.get("size_bytes", 0))
        for item in declared.values()
        if isinstance(item.get("size_bytes", 0), int)
    )
    files = FileIntegritySummary(
        status="VERIFIED" if valid_manifest else "INVALID",
        declared_count=declared_count,
        verified_count=declared_count if valid_manifest else 0,
        total_size_bytes=total_size,
    )
    required_invalid = any(
        node.required and node.status in {"ABSENT", "INVALID"} for node in nodes
    )
    status: Literal["VALID", "INVALID"] = (
        "VALID" if valid_manifest and not required_invalid else "INVALID"
    )
    if status == "INVALID" and validation_error is None:
        validation_error = "required evidence binding is invalid"
    return EvidenceReport(
        run_resource_id=run_resource_id,
        run_id=run_id,
        run_kind=run_kind,
        status=status,
        nodes=tuple(nodes),
        files=files,
        validation_error=validation_error,
    )


__all__ = ["inspect_run_evidence"]
