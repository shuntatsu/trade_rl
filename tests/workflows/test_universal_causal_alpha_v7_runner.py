from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v7_pipeline import (
    CausalAlphaV7ResearchPackage,
    CausalAlphaV7StageRejected,
)
from trade_rl.workflows.universal_causal_alpha_v7_runner import (
    CausalAlphaV7ResearchConfig,
    cli_main,
)


@dataclass(frozen=True)
class _Evidence:
    name: str
    passed: bool

    @property
    def digest(self) -> str:
        return content_digest({"name": self.name, "passed": self.passed})

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed}


def _source() -> dict[str, object]:
    return json.loads(
        Path("examples/binance/universal-causal-alpha-v7-research.json").read_text()
    )


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args() -> list[str]:
    return [
        "--config",
        "c",
        "--run-config",
        "r",
        "--runtime-manifest",
        "m",
        "--v4-context-manifest",
        "v",
        "--frozen-metadata-root",
        "f",
        "--output-root",
        "o",
    ]


def test_v7_runner_loads_exact_fixed_calibration_and_target(tmp_path: Path) -> None:
    config = CausalAlphaV7ResearchConfig.from_json(
        _write(tmp_path / "valid.json", _source())
    )
    assert config.calibration.calibration_fraction == 0.5
    assert config.calibration.working_memory_rows == 4096
    assert config.target.maximum_absolute_target == 0.25
    assert config.digest == config.stage_config_digest


@pytest.mark.parametrize("section", ["calibration", "target"])
def test_v7_runner_rejects_unknown_config_fields(tmp_path: Path, section: str) -> None:
    payload = _source()
    raw_values = payload[section]
    assert isinstance(raw_values, dict)
    values = dict(raw_values)
    values["posthoc_threshold"] = 1
    payload[section] = values
    with pytest.raises(ValueError):
        CausalAlphaV7ResearchConfig.from_json(_write(tmp_path / "bad.json", payload))


@pytest.mark.parametrize(
    ("stage", "code"),
    (("signal", 2), ("selection", 3), ("admission", 4)),
)
def test_v7_cli_has_stable_rejection_exit_codes(stage: str, code: int) -> None:
    def rejected(**_kwargs: object) -> object:
        raise CausalAlphaV7StageRejected(stage, _Evidence(stage, False))

    assert cli_main(_args(), run_from_paths=rejected) == code


def test_v7_cli_returns_zero_only_for_research_package() -> None:
    package = CausalAlphaV7ResearchPackage(
        signal_evidence_digest="1" * 64,
        selection_evidence_digest="2" * 64,
        admission_evidence_digest="3" * 64,
        selected_candidate=CausalAlphaV7Candidate.CAUSAL_CALIBRATED,
        selected_config_digest="4" * 64,
        run_manifest_digest="5" * 64,
        v4_context_manifest_digest="6" * 64,
        generator_code_digest="7" * 64,
    )
    assert cli_main(_args(), run_from_paths=lambda **_: package) == 0
    with pytest.raises(TypeError, match="invalid research package"):
        cli_main(_args(), run_from_paths=lambda **_: object())
