from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT

PACKAGE_ROOT = PYTHON_SOURCE_ROOT


def _top_level_functions(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def test_market_walk_forward_orchestration_has_one_owner() -> None:
    public = PACKAGE_ROOT / "workflows/market_walk_forward.py"
    core = PACKAGE_ROOT / "workflows/_market_walk_forward_core.py"

    assert "execute_market_walk_forward" in _top_level_functions(public)
    assert "execute_market_walk_forward" not in _top_level_functions(core)


def test_market_walk_forward_core_does_not_own_publication_or_ledger_adapters() -> None:
    core = (
        PACKAGE_ROOT / "workflows/_market_walk_forward_core.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "ArtifactStore",
        "PostgresSealedTestReservationStore",
        "PostgresSealedTestLedger",
        "capture_runtime_provenance",
        "load_market_dataset_artifact",
        "WalkForwardRunManifest",
        "write_walk_forward_run_manifest",
    ):
        assert forbidden not in core
