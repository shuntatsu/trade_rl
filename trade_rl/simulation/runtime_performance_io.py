"""Immutable persistence for canonical runtime performance evidence."""

from __future__ import annotations

import json
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.simulation.runtime_performance import RuntimePerformanceEvidence


def write_runtime_performance_evidence(
    path: str | Path,
    evidence: RuntimePerformanceEvidence,
) -> Path:
    """Persist one canonical performance evidence artifact without overwriting drift."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(evidence.to_mapping())
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
    """Load persisted performance evidence and revalidate all derived summaries."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime performance evidence must be an object")
    return RuntimePerformanceEvidence.from_mapping(raw)


__all__ = [
    "load_runtime_performance_evidence",
    "write_runtime_performance_evidence",
]
