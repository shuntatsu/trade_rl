from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.simulation as simulation

ROOT = Path(__file__).resolve().parents[2]
SIMULATION = ROOT / "trade_rl" / "simulation"

EXPECTED_ROOT_FILES = {
    "__init__.py",
    "accounting.py",
    "bar_path.py",
    "execution.py",
    "liquidity.py",
}

EXPECTED_PACKAGE_FILES = {
    "orders": {"__init__.py", "model.py", "admission.py", "reconciliation.py"},
    "stateful": {
        "__init__.py",
        "runtime.py",
        "execution.py",
        "bar_lifecycle.py",
        "order_transitions.py",
        "symbol_fills.py",
    },
    "targets": {"__init__.py", "execution.py", "exposure_controller.py"},
    "diagnostics": {
        "__init__.py",
        "execution_stress.py",
        "funding.py",
        "runtime_performance.py",
        "runtime_performance_io.py",
    },
}

RETIRED_FLAT_FILES = {
    "orders.py",
    "order_admission.py",
    "order_reconciliation.py",
    "stateful_runtime.py",
    "stateful_execution.py",
    "stateful_bar_lifecycle.py",
    "stateful_order_transitions.py",
    "stateful_symbol_fills.py",
    "target_execution.py",
    "target_exposure_controller.py",
    "execution_stress.py",
    "funding_evidence.py",
    "runtime_performance.py",
    "runtime_performance_io.py",
}

RETIRED_MODULES = {
    "trade_rl.simulation.orders",
    "trade_rl.simulation.order_admission",
    "trade_rl.simulation.order_reconciliation",
    "trade_rl.simulation.stateful_runtime",
    "trade_rl.simulation.stateful_execution",
    "trade_rl.simulation.stateful_bar_lifecycle",
    "trade_rl.simulation.stateful_order_transitions",
    "trade_rl.simulation.stateful_symbol_fills",
    "trade_rl.simulation.target_execution",
    "trade_rl.simulation.target_exposure_controller",
    "trade_rl.simulation.execution_stress",
    "trade_rl.simulation.funding_evidence",
    "trade_rl.simulation.runtime_performance",
    "trade_rl.simulation.runtime_performance_io",
}

EXPECTED_PUBLIC = {
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionEnvironmentStress",
    "ExecutionResult",
    "MarketExecutor",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_simulation_root_contains_only_core_economic_owners_and_subpackages() -> None:
    observed_files = {path.name for path in SIMULATION.glob("*.py")}
    assert observed_files == EXPECTED_ROOT_FILES

    observed_directories = {
        path.name
        for path in SIMULATION.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert observed_directories == set(EXPECTED_PACKAGE_FILES)


def test_simulation_subpackage_layout_matches_approved_ownership() -> None:
    for package, expected_files in EXPECTED_PACKAGE_FILES.items():
        package_root = SIMULATION / package
        assert package_root.is_dir(), package
        observed = {path.name for path in package_root.glob("*.py")}
        assert observed == expected_files, package


def test_retired_flat_simulation_paths_are_absent() -> None:
    assert not ({path.name for path in SIMULATION.glob("*.py")} & RETIRED_FLAT_FILES)
    for filename in RETIRED_FLAT_FILES:
        assert not (SIMULATION / filename).exists(), filename


def test_simulation_package_public_api_is_preserved_exactly() -> None:
    assert set(simulation.__all__) == EXPECTED_PUBLIC
    for name in EXPECTED_PUBLIC:
        assert getattr(simulation, name) is not None


def test_simulation_tree_has_no_retired_private_imports_or_upward_dependencies() -> None:
    offenders: list[tuple[str, list[str]]] = []
    for path in SIMULATION.rglob("*.py"):
        modules = _imported_modules(path)
        invalid = sorted(
            module
            for module in modules
            if module in RETIRED_MODULES
            or module.startswith("trade_rl.strategies")
            or module.startswith("trade_rl.evaluation")
        )
        if invalid:
            offenders.append((str(path.relative_to(ROOT)), invalid))
    assert offenders == []
