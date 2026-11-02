"""Protect the unique financial-objective implementation during #810 integration.

The older evaluation/objectives sources must not be merged as independent
implementations of contracts already owned by simulation/objectives. An
import-only facade may re-export existing types without a second type identity.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY_LOCATION = ROOT / "trade_rl" / "evaluation" / "objectives"
CONTRACT_OWNERS = {
    "contract.py": frozenset(
        {"CapitalContract", "ObjectiveContract", "net_equity_increment"}
    ),
    "clock.py": frozenset({"FinancialClockContract"}),
    "binding.py": frozenset({"BoundObjectiveClock"}),
}
FINANCIAL_SYMBOLS = frozenset().union(*CONTRACT_OWNERS.values())


def _module_scope_statements(path: Path) -> Iterator[ast.stmt]:
    """Follow executable module suites, never local function or class scopes."""

    def visit(node: ast.AST) -> Iterator[ast.stmt]:
        if isinstance(node, ast.stmt):
            yield node
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                return
        # Compound suites (if/try/for/with/match) share module bindings.
        # Expressions and comprehension-local names are not module statements.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                yield from visit(child)

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        yield from visit(statement)


def _binding_names(target: ast.AST) -> set[str]:
    return {
        name.id
        for name in ast.walk(target)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    }


def _module_expression_bindings(node: ast.stmt) -> set[str]:
    """Find walrus bindings in module expressions, but not local lambda scopes."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    found: set[str] = set()

    def inspect(expr: ast.expr) -> None:
        if isinstance(expr, ast.Lambda):
            return
        if isinstance(expr, ast.NamedExpr):
            found.update(_binding_names(expr.target))
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                inspect(child)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            inspect(child)
    return found


def _top_level_bindings(path: Path) -> frozenset[str]:
    """Detect module-visible definitions and shadowing, including compound suites."""
    result: set[str] = set()
    for node in _module_scope_statements(path):
        result.update(_module_expression_bindings(node))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            result.update(_binding_names(node.name))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                result.update(_binding_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            result.update(_binding_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            result.update(_binding_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    result.update(_binding_names(item.optional_vars))
        elif isinstance(node, (ast.Try, ast.TryStar)):
            for handler in node.handlers:
                if handler.name:
                    result.add(handler.name)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                for pattern in ast.walk(case.pattern):
                    if (
                        isinstance(pattern, (ast.MatchAs, ast.MatchStar))
                        and pattern.name
                    ):
                        result.add(pattern.name)
    return frozenset(result)


def _legacy_owner_imports(path: Path) -> frozenset[str]:
    """Allow direct canonical re-exports, but reject foreign or shadowing imports."""
    aliases: set[str] = set()
    canonical = "trade_rl.simulation.objectives"
    for node in _module_scope_statements(path):
        if isinstance(node, ast.Import):
            for symbol in node.names:
                # Without "as", Python binds the first dotted name only.
                bound = symbol.asname or symbol.name.split(".", 1)[0]
                if bound in FINANCIAL_SYMBOLS:
                    aliases.add(bound)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_module = node.module if node.level == 0 else None
        trusted = imported_module == canonical
        if imported_module is not None:
            trusted = trusted or imported_module.startswith(canonical + ".")
        for symbol in node.names:
            if symbol.name == "*":
                aliases.update(FINANCIAL_SYMBOLS)
            elif trusted:
                if symbol.asname in FINANCIAL_SYMBOLS and symbol.asname != symbol.name:
                    aliases.add(symbol.asname)
            else:
                for name in (symbol.name, symbol.asname):
                    if name in FINANCIAL_SYMBOLS:
                        aliases.add(name)
    return frozenset(aliases)


def _duplicate_owners(directory: Path) -> dict[str, frozenset[str]]:
    if not directory.exists():
        return {}
    duplicates: dict[str, frozenset[str]] = {}
    for source in sorted(directory.rglob("*.py")):
        definitions = _top_level_bindings(source)
        noncanonical = _legacy_owner_imports(source)
        overlap = (definitions | noncanonical) & FINANCIAL_SYMBOLS
        if overlap:
            duplicates[source.relative_to(directory).as_posix()] = overlap
    return duplicates


def test_evaluation_cannot_declare_second_financial_objective_owners() -> None:
    duplicates = _duplicate_owners(LEGACY_LOCATION)
    assert duplicates == {}, (
        "Legacy evaluation/objectives cannot redefine the simulation financial "
        f"contracts; preserve a single class/receipt authority: {duplicates}"
    )


@pytest.mark.parametrize(
    "definition",
    [
        "class CapitalContract: pass\n",
        "def net_equity_increment(): pass\n",
        "ObjectiveContract = object\n",
        "FinancialClockContract: object = object\n",
        "class BoundObjectiveClock: pass\n",
    ],
)
def test_legacy_duplicate_detection_rejects_definitions_and_shadowing(
    tmp_path: Path, definition: str
) -> None:
    (tmp_path / "contract.py").write_text(definition, encoding="utf-8")
    assert _duplicate_owners(tmp_path)


def test_legacy_import_only_facade_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text(
        "from trade_rl.simulation.objectives import CapitalContract, "
        "FinancialClockContract\n",
        encoding="utf-8",
    )
    assert _duplicate_owners(tmp_path) == {}


@pytest.mark.parametrize(
    "import_source",
    [
        "from trade_rl.evaluation.objectives.contract import CapitalContract\n",
        "from trade_rl.evaluation.objectives.clock import FinancialClockContract\n",
        "from legacy_module import ObjectiveContract as AlternateObjective\n",
        "from legacy_module import *\n",
        "from trade_rl.simulation.objectives import *\n",
        "from trade_rl.simulation.objectives import ObjectiveContract as CapitalContract\n",
        "from legacy_module import strange as CapitalContract\n",
    ],
)
def test_import_only_legacy_source_is_not_mistaken_for_canonical_facade(
    tmp_path: Path, import_source: str
) -> None:
    (tmp_path / "__init__.py").write_text(import_source, encoding="utf-8")
    assert _duplicate_owners(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "import third_party as CapitalContract\n",
        "import third_party.contract as ObjectiveContract\n",
        "if True:\n    class CapitalContract: pass\n",
        "try:\n    from third_party import FinancialClockContract\nexcept ImportError:\n    pass\n",
        "for item in ():\n    BoundObjectiveClock = item\n",
        "with ignored_context():\n    ObjectiveContract = object\n",
        "match 1:\n    case 1:\n        def net_equity_increment(): pass\n",
        "for CapitalContract in ():\n    pass\n",
        "with ignored_context() as ObjectiveContract:\n    pass\n",
        "try:\n    pass\nexcept Exception as FinancialClockContract:\n    pass\n",
        "match 1:\n    case BoundObjectiveClock:\n        pass\n",
    ],
)
def test_module_scope_nested_or_plain_import_cannot_hide_objective_owner(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "contract.py").write_text(source, encoding="utf-8")
    assert _duplicate_owners(tmp_path)


def test_nested_local_function_or_class_definition_is_not_module_owner(
    tmp_path: Path,
) -> None:
    (tmp_path / "contract.py").write_text(
        "def unrelated():\n"
        "    CapitalContract = object\n"
        "class Helper:\n"
        "    FinancialClockContract = object\n",
        encoding="utf-8",
    )
    assert _duplicate_owners(tmp_path) == {}


def test_nested_legacy_subpackage_is_checked(tmp_path: Path) -> None:
    subdir = tmp_path / "other"
    subdir.mkdir()
    (subdir / "contract.py").write_text(
        "class CapitalContract: pass\n", encoding="utf-8"
    )
    assert _duplicate_owners(tmp_path)


@pytest.mark.parametrize(
    "declaration",
    [
        "(CapitalContract := object())\n",
        "if (ObjectiveContract := False):\n    pass\n",
        "type FinancialClockContract = object\n",
    ],
)
def test_module_named_expression_or_type_alias_cannot_hide_owner(
    tmp_path: Path, declaration: str
) -> None:
    (tmp_path / "contract.py").write_text(declaration, encoding="utf-8")
    assert _duplicate_owners(tmp_path)


@pytest.mark.parametrize(
    "usage",
    [
        "registry[CapitalContract] = handler\n",
        "CapitalContract.attr = item\n",
        "run = lambda: (CapitalContract := object())\n",
        "def unrelated():\n    (CapitalContract := object())\n",
    ],
)
def test_nonbinding_uses_and_local_walrus_are_not_module_owners(
    tmp_path: Path, usage: str
) -> None:
    (tmp_path / "contract.py").write_text(usage, encoding="utf-8")
    assert _duplicate_owners(tmp_path) == {}
