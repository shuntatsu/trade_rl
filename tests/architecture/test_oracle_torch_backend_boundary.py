from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPROVED_TORCH_MODULES = {
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


def test_learning_torch_dependency_is_limited_to_explicit_oracle_backend() -> None:
    learning = ROOT / "trade_rl" / "learning"
    torch_modules = {
        path.name for path in learning.glob("*.py") if _imports_torch(path)
    }
    assert torch_modules == APPROVED_TORCH_MODULES

    contract = (ROOT / ".importlinter").read_text(encoding="utf-8")
    block = contract.split("[importlinter:contract:learning-frameworks]", maxsplit=1)[
        1
    ].split("[importlinter:", maxsplit=1)[0]
    assert "allow_indirect_imports = True" in block
    for module in APPROVED_TORCH_MODULES:
        qualified = f"trade_rl.learning.{module.removesuffix('.py')} -> torch"
        assert qualified in block


def test_catalog_reusable_artifacts_remains_below_learning() -> None:
    source = (ROOT / "trade_rl" / "catalog" / "reusable_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "trade_rl.learning" not in source
