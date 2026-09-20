import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL_TEST_ROOT = ROOT / "trade_rl" / "evaluation" / "final_test"
FORBIDDEN_IMPORT_PREFIXES = (
    "trade_rl.data",
    "trade_rl.integrations",
    "trade_rl.strategies",
    "trade_rl.evaluation.replay",
    "trade_rl.evaluation.runs",
    "trade_rl.evaluation.robustness",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_final_test_authorization_does_not_import_execution_or_data_paths() -> None:
    python_files = sorted(FINAL_TEST_ROOT.glob("*.py"))
    assert python_files

    violations: list[str] = []
    for path in python_files:
        for imported in sorted(_imports(path)):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            ):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")

    assert violations == []
