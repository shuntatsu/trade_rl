from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v5_artifact_store import (
    CausalAlphaV5ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
    CausalAlphaV5StageRejected,
    run_universal_causal_alpha_v5_research_pipeline,
)


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


def _store(path: Path) -> CausalAlphaV5ArtifactStore:
    return CausalAlphaV5ArtifactStore(
        path,
        run_manifest_digest="1" * 64,
        v4_context_manifest_digest="2" * 64,
        config_digest="3" * 64,
        generator_code_digest="4" * 64,
    )


def _run(path: Path, rejected: str | None, calls: list[str]):
    def stage(name: str):
        def run(*_args: object) -> _Evidence:
            calls.append(name)
            return _Evidence(name, rejected != name)

        return run

    return run_universal_causal_alpha_v5_research_pipeline(
        store=_store(path),
        prepare_stage=lambda: calls.append("prepare") or object(),
        calibration_stage=stage("calibration"),
        signal_stage=stage("signal"),
        selection_stage=stage("selection"),
        admission_stage=stage("admission"),
    )


def test_v5_pipeline_publishes_only_after_all_stages_pass(tmp_path: Path) -> None:
    calls: list[str] = []
    package = _run(tmp_path, None, calls)
    assert isinstance(package, CausalAlphaV5ResearchPackage)
    assert calls == ["prepare", "calibration", "signal", "selection", "admission"]
    assert (tmp_path / "package.json").is_file()
    assert package.research_only and not package.promotion_eligible


@pytest.mark.parametrize(
    "rejected", ["calibration", "signal", "selection", "admission"]
)
def test_v5_pipeline_stops_at_rejection_and_publishes_no_package(
    tmp_path: Path, rejected: str
) -> None:
    calls: list[str] = []
    with pytest.raises(CausalAlphaV5StageRejected) as error:
        _run(tmp_path, rejected, calls)
    assert error.value.stage == rejected
    order = ["prepare", "calibration", "signal", "selection", "admission"]
    assert calls == order[: order.index(rejected) + 1]
    assert not (tmp_path / "package.json").exists()
    assert (tmp_path / "result.json").is_file()
