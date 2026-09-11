"""Immutable persistence for canonical runtime performance evidence."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import cast

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.simulation.diagnostics.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    approve_runtime_performance_evidence,
)


def _artifact_mapping(evidence: RuntimePerformanceEvidence) -> dict[str, object]:
    return {"evidence_digest": evidence.digest, **evidence.to_mapping()}


def _policy_artifact_mapping(
    policy: RuntimePerformanceApprovalPolicy,
) -> dict[str, object]:
    return {"policy_digest": policy.digest, **policy.to_mapping()}


def _publish_immutable_bytes(
    path: str | Path,
    payload: bytes,
    *,
    artifact_name: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite immutable {artifact_name}: {target}"
            )
        return target

    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.write_bytes(payload)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            if target.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to overwrite immutable {artifact_name}: {target}"
                ) from error
        return target
    finally:
        temporary.unlink(missing_ok=True)


def write_runtime_performance_evidence(
    path: str | Path,
    evidence: RuntimePerformanceEvidence,
) -> Path:
    """Persist one canonical performance evidence artifact without overwriting drift."""

    return _publish_immutable_bytes(
        path,
        canonical_json_bytes(_artifact_mapping(evidence)),
        artifact_name="evidence",
    )


def write_runtime_performance_policy(
    path: str | Path,
    policy: RuntimePerformanceApprovalPolicy,
) -> Path:
    """Persist one reviewed performance policy without overwriting threshold drift."""

    return _publish_immutable_bytes(
        path,
        canonical_json_bytes(_policy_artifact_mapping(policy)),
        artifact_name="policy",
    )


def load_runtime_performance_evidence(
    path: str | Path,
) -> RuntimePerformanceEvidence:
    """Load persisted performance evidence and revalidate its complete identity."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime performance evidence must be an object")
    supplied_digest = raw.pop("evidence_digest", None)
    if not isinstance(supplied_digest, str):
        raise ValueError("runtime performance evidence digest is required")
    require_sha256(supplied_digest, field="evidence_digest")
    evidence = RuntimePerformanceEvidence.from_mapping(raw)
    if supplied_digest != evidence.digest:
        raise ValueError("runtime performance evidence digest mismatch")
    return evidence


def load_runtime_performance_policy(
    path: str | Path,
) -> RuntimePerformanceApprovalPolicy:
    """Load a persisted performance policy and revalidate its complete identity."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime performance policy must be an object")
    supplied_digest = raw.pop("policy_digest", None)
    if not isinstance(supplied_digest, str):
        raise ValueError("runtime performance policy digest is required")
    require_sha256(supplied_digest, field="policy_digest")
    required = {
        "max_elapsed_slowdown_ratio",
        "max_peak_process_tree_rss_ratio",
        "minimum_max_timesteps",
        "minimum_workloads",
        "review_reference",
        "reviewed",
        "schema_version",
    }
    if set(raw) != required:
        raise ValueError("runtime performance policy field closure mismatch")
    policy = RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=cast(float, raw["max_elapsed_slowdown_ratio"]),
        max_peak_process_tree_rss_ratio=cast(
            float,
            raw["max_peak_process_tree_rss_ratio"],
        ),
        minimum_workloads=cast(int, raw["minimum_workloads"]),
        minimum_max_timesteps=cast(int, raw["minimum_max_timesteps"]),
        reviewed=cast(bool, raw["reviewed"]),
        review_reference=cast(str | None, raw["review_reference"]),
        schema_version=cast(str, raw["schema_version"]),
    )
    if supplied_digest != policy.digest:
        raise ValueError("runtime performance policy digest mismatch")
    return policy


def materialize_runtime_performance_approval(
    *,
    evidence_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
    approval_note: str,
) -> RuntimePerformanceEvidence:
    """Approve retained observational evidence and persist the revalidated result."""

    evidence = load_runtime_performance_evidence(evidence_path)
    policy = load_runtime_performance_policy(policy_path)
    approved = approve_runtime_performance_evidence(
        evidence=evidence,
        policy=policy,
        approval_note=approval_note,
    )
    target = write_runtime_performance_evidence(output_path, approved)
    return load_runtime_performance_evidence(target)


__all__ = [
    "load_runtime_performance_evidence",
    "load_runtime_performance_policy",
    "materialize_runtime_performance_approval",
    "write_runtime_performance_evidence",
    "write_runtime_performance_policy",
]
