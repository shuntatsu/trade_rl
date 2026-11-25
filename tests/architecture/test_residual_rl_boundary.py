from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RESIDUAL = PACKAGE / "evaluation" / "residual_rl"
MODULE = "trade_rl.evaluation.residual_rl"
# Frozen evidence from a09134fbd87b2e2712bfbe86f75d817efc3c9969.
# The original result-checking tests remain available at that revision.
EVIDENCE = {
    "issue810-residual-rl-synthetic-20261030-protocol.json": "05bb1997787d000e6d2f730799f28bf531c2986ef2767678104b6ae6912f3d42",
    "issue810-residual-rl-synthetic-20261030-provenance.json": "517ca5bbf77b6fc0977b31925ba5e7d6d0bd2f3a186a166233f406be0bab12d7",
    "issue810-residual-rl-synthetic-20261030-results.json": "ee7d5512ba8cb3d175010e5bf0460c2897da504a71a982605f868f6078f7b6b3",
}


def test_closed_residual_rl_source_is_absent() -> None:
    assert not RESIDUAL.exists()
    assert importlib.util.find_spec(MODULE) is None


def test_residual_rl_is_not_part_of_the_evaluation_public_api() -> None:
    assert "residual_rl" not in " ".join(evaluation.__all__)
    init_text = (PACKAGE / "evaluation" / "__init__.py").read_text(encoding="utf-8")
    assert "residual_rl" not in init_text


def test_core_has_no_static_import_of_retired_study() -> None:
    collector = ImportCollector(PACKAGE)
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in collector.collect(path):
            assert not within_module(name, MODULE), (path, name)


def test_study_artifacts_remain_byte_identical() -> None:
    report = ROOT / "report"
    assert {
        p.name for p in report.glob("issue810-residual-rl-synthetic-20261030*.json")
    } == set(EVIDENCE)
    for name, expected in EVIDENCE.items():
        raw = (report / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        assert b"\r" not in raw, name


def test_residual_rl_is_documented_as_synthetic_only() -> None:
    boundaries = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "evaluation/residual_rl/" in boundaries
    assert "never a profitability claim" in boundaries
    status = (ROOT / "docs" / "research" / "current-status.md").read_text(
        encoding="utf-8"
    )
    assert "synthetic residual-PPO over the cost-aware allocator" in status
    assert (
        "real-data RL remains blocked behind the fee-only learnability check" in status
    )
