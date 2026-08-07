from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT

APPROVED_INTEGRATION_MODULES = {
    "oracle_bellman_torch.py",
    "oracle_transition_torch.py",
}


def _imports_torch(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "torch" or alias.name.startswith("torch.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                return True
    return False


def test_learning_remains_torch_free() -> None:
    learning = PYTHON_SOURCE_ROOT / "learning"
    torch_modules = {
        path.name for path in learning.glob("*.py") if _imports_torch(path)
    }
    assert torch_modules == set()

    contract = (REPOSITORY_ROOT / ".importlinter").read_text(encoding="utf-8")
    block = contract.split("[importlinter:contract:learning-frameworks]", maxsplit=1)[
        1
    ].split("[importlinter:", maxsplit=1)[0]
    assert "ignore_imports" not in block
    assert "oracle_bellman_torch" not in block
    assert "oracle_transition_torch" not in block


def test_oracle_torch_backend_is_confined_to_integrations() -> None:
    integrations = PYTHON_SOURCE_ROOT / "integrations"
    torch_modules = {
        path.name
        for path in integrations.glob("oracle_*_torch.py")
        if _imports_torch(path)
    }
    assert torch_modules == APPROVED_INTEGRATION_MODULES


def test_catalog_reusable_artifacts_remains_below_learning() -> None:
    source = (PYTHON_SOURCE_ROOT / "catalog" / "reusable_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "trade_rl.learning" not in source
