from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CANONICAL_OWNERS = {
    "binance-public-data": "guides/binance-data.md",
    "execution-model": "reference/execution-robustness.md",
    "licensing": "legal/licensing.md",
    "licensing-provenance": "legal/licensing-provenance.md",
    "multi-timeframe-research-boundary": "research/multi-timeframe.md",
    "nautilus-compatibility": "reference/nautilus-migration.md",
    "operations-causal-scenario-c3": "operations/causal-scenario-c3-execution.md",
    "operations-docker-gpu-training": "operations/docker-gpu-full-training.md",
    "quickstart-workflow": "getting-started/quickstart.md",
    "research-status": "research/status.md",
    "reward-semantics": "reference/reward-objective.md",
    "run-reporting-contract": "reference/run-reporting.md",
    "runtime-architecture": "reference/architecture.md",
    "single-symbol-contract": "reference/single-symbol.md",
    "training-configuration": "reference/configuration.md",
    "universal-trade-rl-contract": "reference/universal-trade-rl.md",
    "universal-training-contract": "reference/universal-training.md",
}


def test_real_repository_documentation_validation_is_clean() -> None:
    from scripts.docs.validate import validate_documents

    assert validate_documents(ROOT) == ()


def test_real_repository_has_exact_canonical_authority_map() -> None:
    from scripts.docs.manifest import build_manifest

    manifest = build_manifest(ROOT)
    assert manifest["canonical_owners"] == EXPECTED_CANONICAL_OWNERS


def test_representative_source_paths_route_to_current_canonical_docs() -> None:
    from scripts.docs.context import documents_for_path

    workflow_docs = documents_for_path(
        ROOT,
        "trade_rl/workflows/universal_trade_rl_u2_training.py",
    )
    assert workflow_docs[0].path.as_posix() in {
        "reference/universal-trade-rl.md",
        "reference/universal-training.md",
    }
    assert all(document.lifecycle == "current" for document in workflow_docs)

    reporting_docs = documents_for_path(ROOT, "trade_rl/reporting/run_report.py")
    assert reporting_docs[0].path.as_posix() == "reference/run-reporting.md"
