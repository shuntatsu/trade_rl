import ast
from pathlib import Path

from trade_rl.strategies import allocation_snapshot

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "trade_rl/strategies/allocation_snapshot.py"


def test_snapshot_contract_remains_below_accounting_and_evaluation():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    project_imports = {name for name in imports if name.startswith("trade_rl.")}
    assert project_imports <= {
        "trade_rl._validation",
        "trade_rl.artifacts",
        "trade_rl.artifacts.canonical",
        "trade_rl.artifacts.hashing",
    }
    assert allocation_snapshot.__all__ == ["AllocationAccountSnapshot"]


def test_declared_mapping_oracle_is_independent_of_native_producers():
    source = ROOT / "tests/strategies/test_allocation_snapshot_contract.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(
                ("trade_rl.evaluation", "trade_rl.simulation")
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "__import__"}
