from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v6_pipeline import (
    CausalAlphaV6ResearchPackage,
    CausalAlphaV6StageRejected,
)
from trade_rl.workflows.universal_causal_alpha_v6_runner import (
    CausalAlphaV6ResearchConfig,
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
        Path("examples/binance/universal-causal-alpha-v6-research.json").read_text()
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


def test_v6_runner_loads_only_exact_ordered_fixed_config(tmp_path: Path) -> None:
    config = CausalAlphaV6ResearchConfig.from_json(
        _write(tmp_path / "valid.json", _source())
    )
    assert config.target.maximum_absolute_target == 0.25
    assert config.target.fast_rebalance_decisions == 4
    assert config.target.slow_context_decisions == 16


@pytest.mark.parametrize(
    "kind", ["missing", "unknown", "reordered", "boolean", "changed"]
)
def test_v6_runner_rejects_config_drift(tmp_path: Path, kind: str) -> None:
    payload = _source()
    target = dict(payload["target"])
    if kind == "missing":
        target.pop("edge_margin")
    elif kind == "unknown":
        target["new_threshold"] = 1
    elif kind == "reordered":
        target = {key: target[key] for key in reversed(target)}
    elif kind == "boolean":
        target["fast_rebalance_decisions"] = True
    else:
        target["maximum_absolute_target"] = 0.5
    payload["target"] = target
    with pytest.raises(ValueError):
        CausalAlphaV6ResearchConfig.from_json(
            _write(tmp_path / f"{kind}.json", payload)
        )


@pytest.mark.parametrize(
    ("stage", "code"),
    [("signal", 2), ("selection", 3), ("admission", 4)],
)
def test_v6_cli_has_stable_rejection_exit_codes(stage: str, code: int) -> None:
    def rejected(**_kwargs: object):
        raise CausalAlphaV6StageRejected(stage, _Evidence(stage, False))

    assert cli_main(_args(), run_from_paths=rejected) == code


def test_v6_cli_uses_exit_five_for_invalid_config_or_preparation() -> None:
    def invalid(**_kwargs: object):
        raise ValueError("identity drifted")

    assert cli_main(_args(), run_from_paths=invalid) == 5


def test_v6_cli_returns_zero_only_for_research_package() -> None:
    package = CausalAlphaV6ResearchPackage(
        signal_evidence_digest="1" * 64,
        selection_evidence_digest="2" * 64,
        admission_evidence_digest="3" * 64,
        selected_candidate=CausalAlphaV6Candidate.FAST_ONLY,
        selected_config_digest="4" * 64,
        run_manifest_digest="5" * 64,
        v4_context_manifest_digest="6" * 64,
        generator_code_digest="7" * 64,
    )
    assert cli_main(_args(), run_from_paths=lambda **_: package) == 0
    with pytest.raises(TypeError, match="invalid research package"):
        cli_main(_args(), run_from_paths=lambda **_: object())
