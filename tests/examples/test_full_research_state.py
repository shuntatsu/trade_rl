from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.full_research_state import (
    FullResearchStatus,
    ResearchPhase,
    ResearchPhaseOutcome,
    run_research_phase,
)


class _Stages:
    def __init__(self, outcome: ResearchPhaseOutcome) -> None:
        self.outcome = outcome
        self.calls: list[ResearchPhase] = []

    def run(self, phase: ResearchPhase, work_root: Path) -> ResearchPhaseOutcome:
        self.calls.append(phase)
        assert work_root.is_absolute()
        return self.outcome


@pytest.mark.parametrize(
    ("phase", "status"),
    (
        (ResearchPhase.DEVELOP, FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION),
        (ResearchPhase.TRAIN_SELECTED, FullResearchStatus.AWAITING_FRESH_CONFIRMATION),
        (ResearchPhase.FINALIZE, FullResearchStatus.AWAITING_RELEASE_APPROVAL),
    ),
)
def test_waiting_states_are_successful(
    tmp_path: Path,
    phase: ResearchPhase,
    status: FullResearchStatus,
) -> None:
    stages = _Stages(
        ResearchPhaseOutcome(
            status=status,
            summary={"production_status": "NO-GO", "phase": phase.value},
        )
    )

    result = run_research_phase(
        phase=phase, work_root=tmp_path / "generation", stages=stages
    )

    assert result.exit_code == 0
    assert result.status is status
    assert stages.calls == [phase]
    persisted = json.loads((tmp_path / "generation" / "summary.json").read_text())
    assert persisted["status"] == status.value
    assert persisted["production_status"] == "NO-GO"


def test_research_rejection_is_distinct_nonzero_state(tmp_path: Path) -> None:
    stages = _Stages(
        ResearchPhaseOutcome(
            status=FullResearchStatus.COMPLETE_NO_GO,
            summary={"production_status": "NO-GO", "reason": "gate rejected"},
        )
    )

    result = run_research_phase(
        phase=ResearchPhase.DEVELOP,
        work_root=tmp_path / "generation",
        stages=stages,
    )

    assert result.exit_code == 2
    assert result.status is FullResearchStatus.COMPLETE_NO_GO


def test_infrastructure_error_is_persisted_and_nonzero(tmp_path: Path) -> None:
    class _Broken:
        def run(self, phase: ResearchPhase, work_root: Path) -> ResearchPhaseOutcome:
            raise RuntimeError("disk unavailable")

    result = run_research_phase(
        phase=ResearchPhase.DEVELOP,
        work_root=tmp_path / "generation",
        stages=_Broken(),
    )

    assert result.exit_code == 3
    assert result.status is FullResearchStatus.INFRASTRUCTURE_ERROR
    persisted = json.loads((tmp_path / "generation" / "summary.json").read_text())
    assert persisted["error_type"] == "RuntimeError"
    assert "disk unavailable" in persisted["error"]


@pytest.mark.parametrize(
    ("cache_relative", "work_relative"),
    (
        ("shared", "shared"),
        ("shared/cache", "shared"),
        ("shared", "shared/run"),
    ),
)
def test_cache_and_generation_roots_must_not_contain_each_other(
    tmp_path: Path,
    cache_relative: str,
    work_relative: str,
) -> None:
    from trade_rl.workflows.full_research_state import require_separate_cache_root

    with pytest.raises(ValueError, match="cache root"):
        require_separate_cache_root(
            tmp_path / cache_relative,
            tmp_path / work_relative,
        )


def test_separate_cache_and_generation_roots_are_accepted(tmp_path: Path) -> None:
    from trade_rl.workflows.full_research_state import require_separate_cache_root

    assert (
        require_separate_cache_root(
            tmp_path / "cache",
            tmp_path / "runs" / "generation",
        )
        == (tmp_path / "cache").resolve()
    )
