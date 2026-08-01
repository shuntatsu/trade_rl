from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _import_targets(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.append(node.module)
    return tuple(targets)


def test_release_packaging_is_owned_by_workflows() -> None:
    assert (ROOT / "trade_rl/workflows/release_packaging.py").is_file()
    assert not (ROOT / "trade_rl/serving/package.py").exists()


def test_no_python_imports_the_removed_serving_package() -> None:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        if "trade_rl.serving.package" in _import_targets(path):
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []
