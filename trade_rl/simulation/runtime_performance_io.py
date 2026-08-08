"""Immutable persistence for canonical runtime performance evidence."""

from __future__ import annotations

import json
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.runtime_performance import RuntimePerformanceEvidence


def _artifact_mapping(evidence: RuntimePerformanceEvidence) -> dict[str, object]:
    return {"evidence_digest": evidence.digest, **evidence.to_mapping()}


def write_runtime_performance_evidence(
    path: str | Path,
    evidence: RuntimePerformanceEvidence,
) -> Path:
    """Persist one canonical performance evidence artifact without overwriting drift."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(_artifact_mapping(evidence))
    if target.exists():
        if target.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite immutable evidence: {target}")
        return target
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(encoded)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


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


__all__ = [
    "load_runtime_performance_evidence",
    "write_runtime_performance_evidence",
]
