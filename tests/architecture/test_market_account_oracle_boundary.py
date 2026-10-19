"""Independent market facts and accounting cannot reuse native execution answers."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "tools.market_account_oracle"


def _imports(path: Path, *, require_absolute: bool = False) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if require_absolute:
                assert node.level == 0, path
            if node.level == 0:
                result.add(node.module)
    return result


def test_oracle_uses_only_its_own_arithmetic_and_the_standard_library() -> None:
    sources = tuple((ROOT / "tools/market_account_oracle").rglob("*.py"))
    assert sources
    for path in sources:
        for module in _imports(path, require_absolute=True):
            assert (
                module.split(".", 1)[0] in sys.stdlib_module_names
                or module == OWNER
                or module.startswith(OWNER + ".")
            ), (path.name, module)
    for path in (ROOT / "trade_rl").rglob("*.py"):
        assert not any(
            module == OWNER or module.startswith(OWNER + ".")
            for module in _imports(path)
        ), path.relative_to(ROOT)


def test_oracle_is_checked_in_both_research_typing_routes() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert workflow.count("tools/market_account_oracle") == 2
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tools/market_account_oracle/*.py text eol=lf" in attributes
