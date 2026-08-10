from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG_MODULE = ROOT / "trade_rl" / "catalog" / "stored_instrument_catalog.py"
LEGACY_WORKFLOW_MODULE = (
    ROOT / "trade_rl" / "workflows" / "stored_instrument_catalog.py"
)
PARTITION_MODULE = (
    ROOT / "trade_rl" / "workflows" / "universal_instrument_partition.py"
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def test_stored_instrument_contract_is_owned_by_catalog_layer() -> None:
    assert CATALOG_MODULE.is_file()
    assert not LEGACY_WORKFLOW_MODULE.exists()


def test_universal_partition_depends_on_catalog_contract() -> None:
    assert PARTITION_MODULE.is_file()
    imports = _imports(PARTITION_MODULE)
    assert "trade_rl.catalog.stored_instrument_catalog" in imports
    assert "trade_rl.workflows.stored_instrument_catalog" not in imports
