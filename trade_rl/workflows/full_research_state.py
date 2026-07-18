"""Typed phase and state handling for the maintained full research workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class ResearchPhase(StrEnum):
    DEVELOP = "develop"
    TRAIN_SELECTED = "train-selected"
    FINALIZE = "finalize"


class FullResearchStatus(StrEnum):
    AWAITING_SELECTION_AUTHORIZATION = "awaiting_selection_authorization"
    AWAITING_FRESH_CONFIRMATION = "awaiting_fresh_confirmation"
    AWAITING_RELEASE_APPROVAL = "awaiting_release_approval"
    COMPLETE_NO_GO = "complete_no_go"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


_SUCCESS_STATUSES = frozenset(
    {
        FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION,
        FullResearchStatus.AWAITING_FRESH_CONFIRMATION,
        FullResearchStatus.AWAITING_RELEASE_APPROVAL,
    }
)


@dataclass(frozen=True, slots=True)
class ResearchPhaseOutcome:
    status: FullResearchStatus
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.summary.get("production_status") != "NO-GO":
            raise ValueError("full research status must retain production_status NO-GO")


@dataclass(frozen=True, slots=True)
class FullResearchResult:
    status: FullResearchStatus
    exit_code: int
    summary_path: Path
    summary: Mapping[str, object]


class ResearchStages(Protocol):
    def run(self, phase: ResearchPhase, work_root: Path) -> ResearchPhaseOutcome: ...


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _exit_code(status: FullResearchStatus) -> int:
    if status in _SUCCESS_STATUSES:
        return 0
    if status is FullResearchStatus.COMPLETE_NO_GO:
        return 2
    return 3


def require_separate_cache_root(cache_root: Path, work_root: Path) -> Path:
    """Resolve roots and reject either directory containing the other."""

    cache = cache_root.resolve()
    work = work_root.resolve()
    if cache == work or cache.is_relative_to(work) or work.is_relative_to(cache):
        raise ValueError("cache root must be outside the run generation")
    return cache


def run_research_phase(
    *,
    phase: ResearchPhase,
    work_root: Path,
    stages: ResearchStages,
) -> FullResearchResult:
    """Execute one phase and persist an explicit fail-closed state record."""

    resolved_root = work_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    try:
        outcome = stages.run(phase, resolved_root)
        summary = {
            **dict(outcome.summary),
            "phase": phase.value,
            "production_status": "NO-GO",
            "schema_version": "full_research_state_v1",
            "status": outcome.status.value,
        }
        status = outcome.status
    except Exception as error:
        status = FullResearchStatus.INFRASTRUCTURE_ERROR
        summary = {
            "error": str(error),
            "error_type": type(error).__name__,
            "phase": phase.value,
            "production_status": "NO-GO",
            "schema_version": "full_research_state_v1",
            "status": status.value,
        }
    summary_path = resolved_root / "summary.json"
    _write_json_atomic(summary_path, summary)
    return FullResearchResult(
        status=status,
        exit_code=_exit_code(status),
        summary_path=summary_path,
        summary=summary,
    )


__all__ = [
    "FullResearchResult",
    "FullResearchStatus",
    "ResearchPhase",
    "ResearchPhaseOutcome",
    "ResearchStages",
    "require_separate_cache_root",
    "run_research_phase",
]
