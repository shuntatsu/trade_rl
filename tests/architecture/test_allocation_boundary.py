from __future__ import annotations

import ast
import sys
from pathlib import Path

import trade_rl.strategies as strategies
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
ALLOCATION = PACKAGE / "strategies" / "allocation"
MODULES = ("solver.py", "cost_aware.py", "buy_and_hold.py", "costs.py")


def _imports(path: Path) -> set[str]:
    return ImportCollector(PACKAGE).collect(path)


def _trade_rl_imports(path: Path) -> set[str]:
    return {name for name in _imports(path) if within_module(name, "trade_rl")}


def _third_party(path: Path) -> set[str]:
    return {
        name.split(".", 1)[0]
        for name in _imports(path)
        if name.split(".", 1)[0] not in sys.stdlib_module_names
        and not within_module(name, "trade_rl")
    }


def test_allocation_package_layout_and_facade() -> None:
    assert (ALLOCATION / "__init__.py").is_file()
    for module in MODULES:
        assert (ALLOCATION / module).is_file(), module
    assert {path.name for path in ALLOCATION.glob("*.py")} == {
        "__init__.py",
        *MODULES,
    }
    assert "allocation" not in strategies.__all__
    init_tree = ast.parse((ALLOCATION / "__init__.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in init_tree.body
    )


def test_allocation_modules_only_use_numpy_and_the_detached_portfolio_contract() -> (
    None
):
    assert _trade_rl_imports(ALLOCATION / "solver.py") == set()
    assert _trade_rl_imports(ALLOCATION / "buy_and_hold.py") == {
        "trade_rl.strategies.portfolio"
    }
    assert _trade_rl_imports(ALLOCATION / "cost_aware.py") == {
        "trade_rl.strategies.allocation.solver",
        "trade_rl.strategies.portfolio",
    }
    for module in MODULES:
        assert _third_party(ALLOCATION / module) <= {"numpy"}, module


def test_only_the_cost_helper_reads_the_simulation_execution_profile() -> None:
    assert _trade_rl_imports(ALLOCATION / "costs.py") == {
        "trade_rl.simulation.execution"
    }
    forbidden = (
        "trade_rl.evaluation",
        "trade_rl.risk",
        "trade_rl.simulation",
        "trade_rl.data",
        "trade_rl.integrations",
        "trade_rl.strategies.rl",
        "trade_rl.strategies.forecasts",
    )
    for module in MODULES:
        for imported in _trade_rl_imports(ALLOCATION / module):
            if module == "costs.py":
                continue
            assert not any(within_module(imported, prefix) for prefix in forbidden), (
                module,
                imported,
            )


def test_allocation_policies_own_no_account_or_ledger_and_no_randomness() -> None:
    forbidden_names = {"BookState", "MarketExecutor", "SharedPortfolioSession"}
    for module in MODULES:
        tree = ast.parse((ALLOCATION / module).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not forbidden_names & names, module
        assert "random" not in names, module
