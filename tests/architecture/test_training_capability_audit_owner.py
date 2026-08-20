from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT

SCRIPT = REPOSITORY_ROOT / "scripts/run_training_capability_audit.py"
PUBLIC = PYTHON_SOURCE_ROOT / "operations/training_capability_audit.py"
PRIVATE = PYTHON_SOURCE_ROOT / "operations/_training_capability_audit_impl.py"

_FORBIDDEN_SCRIPT_IMPORT_PREFIXES = (
    "gymnasium",
    "numpy",
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "trade_rl.data",
    "trade_rl.integrations.binance",
    "trade_rl.integrations.sb3_training",
    "trade_rl.rl.actions",
    "trade_rl.rl.environment",
    "trade_rl.rl.export",
    "trade_rl.rl.observations",
    "trade_rl.rl.training",
    "trade_rl.simulation.execution",
    "trade_rl.strategies.trend",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_modules(path: Path) -> frozenset[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return frozenset(modules)


def _top_level_definitions(path: Path) -> frozenset[str]:
    return frozenset(
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    )


def test_training_capability_audit_has_package_owned_public_boundary() -> None:
    assert PUBLIC.is_file()
    assert PRIVATE.is_file()
    public_source = PUBLIC.read_text(encoding="utf-8")
    assert "def run_training_capability_audit(" in public_source
    assert "_training_capability_audit_impl" in public_source
    assert _top_level_definitions(PUBLIC) == frozenset(
        {"run_training_capability_audit"}
    )


def test_training_capability_audit_script_is_a_thin_adapter() -> None:
    imports = _import_modules(SCRIPT)
    assert "trade_rl.operations.training_capability_audit" in imports
    for prefix in _FORBIDDEN_SCRIPT_IMPORT_PREFIXES:
        assert not any(
            module == prefix or module.startswith(f"{prefix}.") for module in imports
        ), prefix
    assert _top_level_definitions(SCRIPT) == frozenset({"main"})


def test_lower_production_layers_do_not_import_training_capability_audit() -> None:
    forbidden = {
        "trade_rl.operations.training_capability_audit",
        "trade_rl.operations._training_capability_audit_impl",
    }
    for path in sorted(PYTHON_SOURCE_ROOT.rglob("*.py")):
        if path in {PUBLIC, PRIVATE}:
            continue
        assert _import_modules(path).isdisjoint(forbidden), path
