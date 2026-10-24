"""Protect the unique financial-objective implementation during #810 integration.

The older evaluation/objectives sources must not be merged as independent
implementations of contracts already owned by simulation/objectives. An
import-only facade may re-export existing types without a second type identity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_OWNER = ROOT / "trade_rl" / "simulation" / "objectives"
LEGACY_LOCATION = ROOT / "trade_rl" / "evaluation" / "objectives"
CONTRACT_OWNERS = {
    "contract.py": frozenset(
        {"CapitalContract", "ObjectiveContract", "net_equity_increment"}
    ),
    "clock.py": frozenset({"FinancialClockContract"}),
    "binding.py": frozenset({"BoundObjectiveClock"}),
}
FINANCIAL_SYMBOLS = frozenset().union(*CONTRACT_OWNERS.values())


def _top_level_bindings(path: Path) -> frozenset[str]:
    """Detect real definitions and shadowing assignments, not import facades."""
    result: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                result.update(
                    name.id for name in ast.walk(target) if isinstance(name, ast.Name)
                )
        elif isinstance(node, ast.AnnAssign):
            result.update(
                name.id for name in ast.walk(node.target) if isinstance(name, ast.Name)
            )
    return frozenset(result)


def _legacy_owner_imports(path: Path) -> frozenset[str]:
    """Allow direct canonical imports, not aliases from another owner."""
    aliases: set[str] = set()
    canonical = "trade_rl.simulation.objectives"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
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
    for source in sorted(directory.glob("*.py")):
        definitions = _top_level_bindings(source)
        noncanonical = _legacy_owner_imports(source)
        overlap = (definitions | noncanonical) & FINANCIAL_SYMBOLS
        if overlap:
            duplicates[source.name] = overlap
    return duplicates


def test_canonical_financial_objective_declarations_stay_in_simulation() -> None:
    for filename, expected in CONTRACT_OWNERS.items():
        source = CANONICAL_OWNER / filename
        assert source.is_file(), f"missing financial-objective owner: {source}"
        assert expected <= _top_level_bindings(source)


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
