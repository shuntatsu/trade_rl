from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.import_references import scan_import_references
from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT

PACKAGE_ROOT = PYTHON_SOURCE_ROOT
WORKFLOWS_ROOT = PACKAGE_ROOT / "workflows"


def _top_level_functions(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def _function_owners(root: Path, name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in root.rglob("*.py")
            if name in _top_level_functions(path)
        )
    )


def _import_targets(path: Path, *, module_name: str) -> frozenset[str]:
    return frozenset(
        reference.target
        for reference in scan_import_references(path, module_name=module_name)
        if reference.target is not None
    )


def test_market_walk_forward_orchestration_has_one_workflow_owner() -> None:
    assert _function_owners(WORKFLOWS_ROOT, "execute_market_walk_forward") == (
        "workflows/market_walk_forward.py",
    )


def test_market_walk_forward_cli_remains_a_public_workflow_adapter() -> None:
    path = PACKAGE_ROOT / "cli/extended.py"
    targets = _import_targets(path, module_name="trade_rl.cli.extended")

    assert (
        "trade_rl.workflows.market_walk_forward.execute_market_walk_forward" in targets
    )
    assert not any(
        target == "trade_rl.workflows._market_walk_forward_core"
        or target.startswith("trade_rl.workflows._market_walk_forward_core.")
        for target in targets
    )


def test_market_walk_forward_core_does_not_own_publication_or_ledger_adapters() -> None:
    core = PACKAGE_ROOT / "workflows/_market_walk_forward_core.py"
    targets = _import_targets(
        core,
        module_name="trade_rl.workflows._market_walk_forward_core",
    )

    forbidden = {
        "trade_rl.artifacts.provenance.capture_runtime_provenance",
        "trade_rl.artifacts.run_manifest.WalkForwardRunManifest",
        "trade_rl.artifacts.run_manifest.validate_walk_forward_run_directory",
        "trade_rl.artifacts.run_manifest.write_walk_forward_run_manifest",
        "trade_rl.artifacts.store.ArtifactStore",
        "trade_rl.catalog.postgres_sealed_test.PostgresSealedTestReservationStore",
        "trade_rl.catalog.sealed_test.PostgresSealedTestLedger",
        "trade_rl.data.load_market_dataset_artifact",
    }
    assert targets.isdisjoint(forbidden)
