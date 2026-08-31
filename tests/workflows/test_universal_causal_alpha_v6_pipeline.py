from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v6_artifact_store import (
    CausalAlphaV6ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v6_pipeline import (
    CausalAlphaV6ResearchPackage,
    CausalAlphaV6StageRejected,
    run_universal_causal_alpha_v6_research_pipeline,
)


@dataclass(frozen=True)
class _Evidence:
    name: str
    passed: bool
    selected_candidate: CausalAlphaV6Candidate | None = None
    selected_config_digest: str | None = None

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "name": self.name,
                "passed": self.passed,
                "selected_candidate": (
                    None
                    if self.selected_candidate is None
                    else self.selected_candidate.value
                ),
                "selected_config_digest": self.selected_config_digest,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "name": self.name,
            "passed": self.passed,
            "schema_version": f"test_{self.name}_v1",
            "selected_candidate": (
                None
                if self.selected_candidate is None
                else self.selected_candidate.value
            ),
            "selected_config_digest": self.selected_config_digest,
        }


def _store(path: Path) -> CausalAlphaV6ArtifactStore:
    return CausalAlphaV6ArtifactStore(
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
            if name == "selection":
                return _Evidence(
                    name,
                    rejected != name,
                    selected_candidate=(
                        CausalAlphaV6Candidate.FAST_ONLY if rejected != name else None
                    ),
                    selected_config_digest=("3" * 64 if rejected != name else None),
                )
            return _Evidence(name, rejected != name)

        return run

    return run_universal_causal_alpha_v6_research_pipeline(
        store=_store(path),
        prepare_stage=lambda: calls.append("prepare") or object(),
        signal_stage=stage("signal"),
        selection_stage=stage("selection"),
        admission_stage=stage("admission"),
    )


def test_v6_pipeline_publishes_only_after_admission_passes(tmp_path: Path) -> None:
    calls: list[str] = []
    package = _run(tmp_path, None, calls)
    assert isinstance(package, CausalAlphaV6ResearchPackage)
    assert calls == ["prepare", "signal", "selection", "admission"]
    assert package.selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
    assert package.selected_config_digest == "3" * 64
    assert package.research_only and not package.promotion_eligible
    assert (tmp_path / "signal" / "evidence.json").is_file()
    assert (tmp_path / "selection" / "evidence.json").is_file()
    assert (tmp_path / "admission" / "evidence.json").is_file()
    assert (tmp_path / "package.json").is_file()
    assert (tmp_path / "result.json").is_file()


@pytest.mark.parametrize("rejected", ["signal", "selection", "admission"])
def test_v6_pipeline_stops_at_rejection_and_never_publishes_package(
    tmp_path: Path,
    rejected: str,
) -> None:
    calls: list[str] = []
    with pytest.raises(CausalAlphaV6StageRejected) as error:
        _run(tmp_path, rejected, calls)
    order = ["prepare", "signal", "selection", "admission"]
    assert calls == order[: order.index(rejected) + 1]
    assert error.value.stage == rejected
    assert (
        error.value.exit_code == {"signal": 2, "selection": 3, "admission": 4}[rejected]
    )
    assert (tmp_path / rejected / "evidence.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert not (tmp_path / "package.json").exists()


def test_v6_pipeline_rejects_reuse_of_immutable_output_root(tmp_path: Path) -> None:
    _run(tmp_path, None, [])
    with pytest.raises(FileExistsError, match="already exists"):
        _run(tmp_path, None, [])
