from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v4_pipeline import (
    CausalAlphaV4AdmissionRejected,
    CausalAlphaV4ResearchPackage,
    CausalAlphaV4SelectionRejected,
    CausalAlphaV4SignalRejected,
    run_universal_causal_alpha_v4_research_pipeline,
)


def _digest(char: str) -> str:
    return char * 64


@dataclass(frozen=True)
class _Evidence:
    name: str
    passed: bool

    @property
    def digest(self) -> str:
        return content_digest({"name": self.name, "passed": self.passed})

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "name": self.name,
            "passed": self.passed,
            "schema_version": f"test_{self.name}_v1",
        }


def _store(tmp_path: Path) -> CausalAlphaV4ArtifactStore:
    return CausalAlphaV4ArtifactStore(
        tmp_path,
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        generator_code_digest=_digest("d"),
    )


def test_v4_pipeline_runs_stages_in_order_and_returns_research_package(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def prepare() -> object:
        calls.append("prepare")
        return {"prepared": True}

    def signal(prepared: object) -> _Evidence:
        assert prepared == {"prepared": True}
        calls.append("signal")
        return _Evidence("signal", True)

    def selection(prepared: object, signal_evidence: _Evidence) -> _Evidence:
        assert prepared == {"prepared": True}
        assert signal_evidence.passed
        calls.append("selection")
        return _Evidence("selection", True)

    def admission(
        prepared: object,
        signal_evidence: _Evidence,
        selection_evidence: _Evidence,
    ) -> _Evidence:
        assert prepared == {"prepared": True}
        assert signal_evidence.passed and selection_evidence.passed
        calls.append("admission")
        return _Evidence("admission", True)

    package = run_universal_causal_alpha_v4_research_pipeline(
        store=_store(tmp_path),
        prepare_stage=prepare,
        signal_stage=signal,
        selection_stage=selection,
        admission_stage=admission,
    )

    assert isinstance(package, CausalAlphaV4ResearchPackage)
    assert package.research_only is True
    assert package.promotion_eligible is False
    assert calls == ["prepare", "signal", "selection", "admission"]


def test_v4_pipeline_signal_rejection_stops_later_stages(tmp_path: Path) -> None:
    calls: list[str] = []

    def prepare() -> object:
        calls.append("prepare")
        return object()

    def signal(_prepared: object) -> _Evidence:
        calls.append("signal")
        return _Evidence("signal", False)

    def unexpected_selection(*_args: object) -> _Evidence:
        calls.append("selection")
        return _Evidence("selection", True)

    def unexpected_admission(*_args: object) -> _Evidence:
        calls.append("admission")
        return _Evidence("admission", True)

    with pytest.raises(CausalAlphaV4SignalRejected):
        run_universal_causal_alpha_v4_research_pipeline(
            store=_store(tmp_path),
            prepare_stage=prepare,
            signal_stage=signal,
            selection_stage=unexpected_selection,
            admission_stage=unexpected_admission,
        )

    assert calls == ["prepare", "signal"]


def test_v4_pipeline_selection_and_admission_rejections_are_distinct(
    tmp_path: Path,
) -> None:
    def prepare() -> object:
        return object()

    def signal(_prepared: object) -> _Evidence:
        return _Evidence("signal", True)

    def selection_rejected(_prepared: object, _signal: _Evidence) -> _Evidence:
        return _Evidence("selection", False)

    with pytest.raises(CausalAlphaV4SelectionRejected):
        run_universal_causal_alpha_v4_research_pipeline(
            store=_store(tmp_path / "selection"),
            prepare_stage=prepare,
            signal_stage=signal,
            selection_stage=selection_rejected,
            admission_stage=lambda *_: _Evidence("admission", True),
        )

    def selection_passed(_prepared: object, _signal: _Evidence) -> _Evidence:
        return _Evidence("selection", True)

    def admission_rejected(
        _prepared: object, _signal: _Evidence, _selection: _Evidence
    ) -> _Evidence:
        return _Evidence("admission", False)

    with pytest.raises(CausalAlphaV4AdmissionRejected):
        run_universal_causal_alpha_v4_research_pipeline(
            store=_store(tmp_path / "admission"),
            prepare_stage=prepare,
            signal_stage=signal,
            selection_stage=selection_passed,
            admission_stage=admission_rejected,
        )
