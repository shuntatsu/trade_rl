"""Evaluation-only C3 execution boundary over a strict reporting contract."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from trade_rl.evaluation.causal_scenario_c3_reporting import (
    C3AggregateSummary,
    PhaseAGateEvidence,
    evaluate_phase_a_gate,
    load_c3_aggregate_summary,
    write_c3_report_artifact,
    write_phase_a_gate_artifact,
)

PRODUCTION_STATUS: Final = "NO-GO"
_BACKEND_MODULE: Final = "trade_rl.workflows.causal_scenario.c3_core"
_BACKEND_FUNCTION: Final = "execute_c3_core_request"

C3CoreBackend = Callable[[Path], Path]


class C3CoreBackendUnavailable(RuntimeError):
    """Raised when lane B has not provided the C3 evaluation backend."""


@dataclass(frozen=True, slots=True)
class C3ExecutionResult:
    source_summary_path: Path
    summary: C3AggregateSummary
    gate: PhaseAGateEvidence
    report_artifact_root: Path
    report_artifact_digest: str
    gate_artifact_root: Path
    gate_artifact_digest: str
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_summary_path", Path(self.source_summary_path))
        object.__setattr__(self, "report_artifact_root", Path(self.report_artifact_root))
        object.__setattr__(self, "gate_artifact_root", Path(self.gate_artifact_root))
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 execution production status must remain NO-GO")


def _backend_from_module(module: ModuleType) -> Callable[..., Path]:
    backend = getattr(module, _BACKEND_FUNCTION, None)
    if not callable(backend):
        raise C3CoreBackendUnavailable(
            f"lane B backend {_BACKEND_MODULE}.{_BACKEND_FUNCTION} is unavailable"
        )
    return backend


def _resolve_backend() -> Callable[..., Path]:
    try:
        module = importlib.import_module(_BACKEND_MODULE)
    except (ImportError, ModuleNotFoundError) as error:
        raise C3CoreBackendUnavailable(
            "lane B C3 core backend is unavailable; merge lane B before evaluate"
        ) from error
    return _backend_from_module(module)


def _request_file(path: str | Path) -> Path:
    request = Path(path)
    if request.is_symlink() or not request.is_file():
        raise FileNotFoundError(f"C3 request file is missing: {request}")
    return request


def _summary_file(core_root: Path, returned: str | Path) -> Path:
    candidate = Path(returned)
    resolved_root = core_root.resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (core_root / candidate).resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError("C3 backend returned a summary outside the core output root")
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"C3 backend summary file is missing: {resolved}")
    return resolved


def execute_c3_evaluation_request(
    request_path: str | Path,
    *,
    output_root: str | Path,
    backend: Callable[..., Path] | None = None,
) -> C3ExecutionResult:
    """Execute lane B through one backend contract and publish lane C evidence."""

    request = _request_file(request_path)
    destination = Path(output_root)
    core_root = destination / "core"
    core_root.mkdir(parents=True, exist_ok=True)
    resolved_backend = _resolve_backend() if backend is None else backend
    returned = resolved_backend(request, output_root=core_root)
    summary_path = _summary_file(core_root, returned)
    summary = load_c3_aggregate_summary(summary_path)
    gate = evaluate_phase_a_gate(summary)
    report = write_c3_report_artifact(destination / "report", summary, gate)
    gate_artifact = write_phase_a_gate_artifact(
        destination / "gate",
        gate,
        report_artifact_digest=report.artifact_digest,
    )
    return C3ExecutionResult(
        source_summary_path=summary_path,
        summary=summary,
        gate=gate,
        report_artifact_root=report.root,
        report_artifact_digest=report.artifact_digest,
        gate_artifact_root=gate_artifact.root,
        gate_artifact_digest=gate_artifact.artifact_digest,
    )


__all__ = [
    "C3CoreBackendUnavailable",
    "C3ExecutionResult",
    "execute_c3_evaluation_request",
]
