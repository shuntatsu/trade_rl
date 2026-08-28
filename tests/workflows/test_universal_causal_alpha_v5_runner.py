from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
    CausalAlphaV5StageRejected,
)
from trade_rl.workflows.universal_causal_alpha_v5_runner import (
    CausalAlphaV5ResearchConfig,
    cli_main,
)
from trade_rl.workflows.universal_causal_alpha_v5_stage_execution import (
    execute_causal_alpha_v5_stage_callbacks,
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
        Path("examples/binance/universal-causal-alpha-v5-research.json").read_text()
    )


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v5_runner_loads_only_the_exact_ordered_frozen_config(tmp_path: Path) -> None:
    config = CausalAlphaV5ResearchConfig.from_json(
        _write(tmp_path / "valid.json", _source())
    )
    assert config.calibration.minimum_active_coverage == 0.25
    assert config.epsilon == 1e-12


@pytest.mark.parametrize(
    "kind", ["missing", "unknown", "reordered", "boolean", "changed"]
)
def test_v5_runner_rejects_config_drift(tmp_path: Path, kind: str) -> None:
    payload = _source()
    calibration = dict(payload["calibration"])
    if kind == "missing":
        calibration.pop("edge_margin")
    elif kind == "unknown":
        calibration["new_threshold"] = 1
    elif kind == "reordered":
        calibration = {key: calibration[key] for key in reversed(calibration)}
    elif kind == "boolean":
        calibration["forward_block_count"] = True
    else:
        calibration["minimum_active_coverage"] = 0.3
    payload["calibration"] = calibration
    with pytest.raises(ValueError):
        CausalAlphaV5ResearchConfig.from_json(
            _write(tmp_path / f"{kind}.json", payload)
        )


def test_v5_stage_callbacks_stop_before_replay_or_admission() -> None:
    calls: list[str] = []

    def stage(name: str, passed: bool = True):
        def run(*_args: object) -> _Evidence:
            calls.append(name)
            return _Evidence(name, passed)

        return run

    result = execute_causal_alpha_v5_stage_callbacks(
        prepare_v4=stage("prepare"),
        fit_calibration=stage("calibration"),
        build_selective_signal=stage("signal", False),
        replay_and_select=stage("selection"),
        untouched_admission=stage("admission"),
    )
    assert calls == ["prepare", "calibration", "signal"]
    assert result[3:] == (None, None)


@pytest.mark.parametrize(
    ("stage", "code"),
    [("signal", 2), ("selection", 3), ("admission", 4), ("calibration", 5)],
)
def test_v5_cli_has_stable_rejection_exit_codes(stage: str, code: int) -> None:
    def rejected(**_kwargs: object):
        raise CausalAlphaV5StageRejected(stage, _Evidence(stage, False))

    args = [
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
    assert cli_main(args, run_from_paths=rejected) == code


def test_v5_cli_returns_zero_only_for_research_package() -> None:
    package = CausalAlphaV5ResearchPackage(
        calibration_evidence_digest="1" * 64,
        signal_evidence_digest="2" * 64,
        selection_evidence_digest="3" * 64,
        admission_evidence_digest="4" * 64,
        run_manifest_digest="5" * 64,
        v4_context_manifest_digest="6" * 64,
        config_digest="7" * 64,
    )
    args = [
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
    assert cli_main(args, run_from_paths=lambda **_: package) == 0
