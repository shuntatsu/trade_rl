from __future__ import annotations

import hashlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v7_artifact_store import (
    CausalAlphaV7ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v7_pipeline import (
    CausalAlphaV7ResearchPackage,
    CausalAlphaV7StageRejected,
    run_universal_causal_alpha_v7_research_pipeline,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Evidence:
    def __init__(
        self,
        name: str,
        *,
        passed: bool,
        selected: CausalAlphaV7Candidate | None = None,
    ) -> None:
        self.name = name
        self.passed = passed
        self.digest = _digest(name)
        self.selected_candidate = selected
        self.selected_config_digest = (
            _digest("config") if selected is not None else None
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "name": self.name,
            "passed": self.passed,
        }


def _store(tmp_path: Any) -> CausalAlphaV7ArtifactStore:
    return CausalAlphaV7ArtifactStore(
        tmp_path,
        run_manifest_digest=_digest("run"),
        v4_context_manifest_digest=_digest("context"),
        config_digest=_digest("config"),
        generator_code_digest=_digest("generator"),
    )


def test_v7_pipeline_stops_and_persists_selection_rejection(tmp_path: Any) -> None:
    calls: list[str] = []

    def stage(
        name: str,
        evidence: _Evidence,
    ) -> Callable[..., _Evidence]:
        def run(*_args: object) -> _Evidence:
            calls.append(name)
            return evidence

        return run

    with pytest.raises(CausalAlphaV7StageRejected) as caught:
        run_universal_causal_alpha_v7_research_pipeline(
            store=_store(tmp_path),
            prepare_stage=lambda: SimpleNamespace(),
            signal_stage=stage("signal", _Evidence("signal", passed=True)),
            selection_stage=stage("selection", _Evidence("selection", passed=False)),
            admission_stage=stage("admission", _Evidence("admission", passed=True)),
        )

    assert caught.value.stage == "selection"
    assert caught.value.exit_code == 3
    assert calls == ["signal", "selection"]
    assert (tmp_path / "signal" / "evidence.json").exists()
    assert (tmp_path / "selection" / "evidence.json").exists()
    assert (tmp_path / "result.json").exists()
    assert not (tmp_path / "admission" / "evidence.json").exists()


def test_v7_pipeline_packages_only_after_admission(tmp_path: Any) -> None:
    selected = CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    result = run_universal_causal_alpha_v7_research_pipeline(
        store=_store(tmp_path),
        prepare_stage=lambda: SimpleNamespace(),
        signal_stage=lambda _prepared: _Evidence("signal", passed=True),
        selection_stage=lambda _prepared, _signal: _Evidence(
            "selection", passed=True, selected=selected
        ),
        admission_stage=lambda _prepared, _signal, _selection: _Evidence(
            "admission", passed=True
        ),
    )

    assert isinstance(result, CausalAlphaV7ResearchPackage)
    assert result.selected_candidate is selected
    assert (tmp_path / "package.json").exists()
    assert (tmp_path / "result.json").exists()
