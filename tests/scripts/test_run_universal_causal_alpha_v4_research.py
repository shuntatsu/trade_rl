from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_pipeline import (
    CausalAlphaV4AdmissionRejected,
    CausalAlphaV4ResearchPackage,
    CausalAlphaV4SelectionRejected,
    CausalAlphaV4SignalRejected,
)
from trade_rl.workflows.universal_causal_alpha_v4_runner import cli_main


def _digest(char: str) -> str:
    return char * 64


class _Evidence:
    def __init__(self, name: str, passed: bool) -> None:
        self.name = name
        self.passed = passed
        self.digest = content_digest({"name": name, "passed": passed})

    def to_payload(self) -> dict[str, object]:
        body: dict[str, object] = {
            "name": self.name,
            "passed": self.passed,
            "schema_version": f"test_{self.name}_v1",
        }
        return {**body, "artifact_digest": content_digest(body)}


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "v4.json"),
        "--run-config",
        str(tmp_path / "run.json"),
        "--runtime-manifest",
        str(tmp_path / "runtime-manifest.json"),
        "--v4-context-manifest",
        str(tmp_path / "v4-context-manifest.json"),
        "--frozen-metadata-root",
        str(tmp_path / "metadata"),
        "--output-root",
        str(tmp_path / "output"),
    ]


def test_v4_cli_requires_context_manifest(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    index = argv.index("--v4-context-manifest")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as error:
        cli_main(argv, run_from_paths=lambda **_: None)

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("kind", "expected"),
    (("signal", 2), ("selection", 3), ("admission", 4)),
)
def test_v4_cli_maps_research_rejections_to_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    expected: int,
) -> None:
    def run_from_paths(**_: object) -> object:
        evidence = _Evidence(kind, False)
        if kind == "signal":
            raise CausalAlphaV4SignalRejected(evidence)
        if kind == "selection":
            raise CausalAlphaV4SelectionRejected(evidence)
        raise CausalAlphaV4AdmissionRejected(evidence)

    status = cli_main(_argv(tmp_path), run_from_paths=run_from_paths)
    payload = json.loads(capsys.readouterr().out)

    assert status == expected
    assert payload["status"] == f"{kind}_rejected"
    assert payload["promotion_eligible"] is False


def test_v4_cli_returns_zero_only_for_research_admission(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = CausalAlphaV4ResearchPackage(
        signal_evidence_digest=_digest("a"),
        selection_evidence_digest=_digest("b"),
        admission_evidence_digest=_digest("c"),
        run_manifest_digest=_digest("d"),
        v4_context_manifest_digest=_digest("e"),
        config_digest=_digest("f"),
    )

    status = cli_main(_argv(tmp_path), run_from_paths=lambda **_: package)
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["status"] == "admitted"
    assert payload["research_only"] is True
    assert payload["promotion_eligible"] is False


def test_v4_cli_entrypoint_file_exists() -> None:
    source = Path("scripts/run_universal_causal_alpha_v4_research.py")

    assert source.is_file()
    text = source.read_text(encoding="utf-8")
    assert "cli_main" in text
