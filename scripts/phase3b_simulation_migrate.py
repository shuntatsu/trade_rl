from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "trade_rl" / "simulation"
ARCHITECTURE_TEST = ROOT / "tests" / "architecture" / "test_lean_simulation_layout.py"

MOVES = {
    "orders.py": "orders/model.py",
    "order_admission.py": "orders/admission.py",
    "order_reconciliation.py": "orders/reconciliation.py",
    "stateful_runtime.py": "stateful/runtime.py",
    "stateful_execution.py": "stateful/execution.py",
    "stateful_bar_lifecycle.py": "stateful/bar_lifecycle.py",
    "stateful_order_transitions.py": "stateful/order_transitions.py",
    "stateful_symbol_fills.py": "stateful/symbol_fills.py",
    "target_execution.py": "targets/execution.py",
    "target_exposure_controller.py": "targets/exposure_controller.py",
    "execution_stress.py": "diagnostics/execution_stress.py",
    "funding_evidence.py": "diagnostics/funding.py",
    "runtime_performance.py": "diagnostics/runtime_performance.py",
    "runtime_performance_io.py": "diagnostics/runtime_performance_io.py",
}

IMPORT_OWNER_MAP = {
    "trade_rl.simulation.orders": "trade_rl.simulation.orders.model",
    "trade_rl.simulation.order_admission": "trade_rl.simulation.orders.admission",
    "trade_rl.simulation.order_reconciliation": "trade_rl.simulation.orders.reconciliation",
    "trade_rl.simulation.stateful_runtime": "trade_rl.simulation.stateful.runtime",
    "trade_rl.simulation.stateful_execution": "trade_rl.simulation.stateful.execution",
    "trade_rl.simulation.stateful_bar_lifecycle": "trade_rl.simulation.stateful.bar_lifecycle",
    "trade_rl.simulation.stateful_order_transitions": "trade_rl.simulation.stateful.order_transitions",
    "trade_rl.simulation.stateful_symbol_fills": "trade_rl.simulation.stateful.symbol_fills",
    "trade_rl.simulation.target_execution": "trade_rl.simulation.targets.execution",
    "trade_rl.simulation.target_exposure_controller": "trade_rl.simulation.targets.exposure_controller",
    "trade_rl.simulation.execution_stress": "trade_rl.simulation.diagnostics.execution_stress",
    "trade_rl.simulation.funding_evidence": "trade_rl.simulation.diagnostics.funding",
    "trade_rl.simulation.runtime_performance": "trade_rl.simulation.diagnostics.runtime_performance",
    "trade_rl.simulation.runtime_performance_io": "trade_rl.simulation.diagnostics.runtime_performance_io",
}

EXPECTED_ROOT_BEFORE = {
    "__init__.py",
    "accounting.py",
    "bar_path.py",
    "execution.py",
    "execution_stress.py",
    "funding_evidence.py",
    "liquidity.py",
    "order_admission.py",
    "order_reconciliation.py",
    "orders.py",
    "runtime_performance.py",
    "runtime_performance_io.py",
    "stateful_bar_lifecycle.py",
    "stateful_execution.py",
    "stateful_order_transitions.py",
    "stateful_runtime.py",
    "stateful_symbol_fills.py",
    "target_execution.py",
    "target_exposure_controller.py",
}
EXPECTED_ROOT_AFTER = {
    "__init__.py",
    "accounting.py",
    "bar_path.py",
    "execution.py",
    "liquidity.py",
}
EXPECTED_PUBLIC = {
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionEnvironmentStress",
    "ExecutionResult",
    "MarketExecutor",
}
PACKAGE_FILES = {
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


def _all_export_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError(f"missing __all__: {path}")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_preconditions() -> None:
    observed = {path.name for path in SIMULATION.glob("*.py")}
    if observed != EXPECTED_ROOT_BEFORE:
        raise RuntimeError(
            "simulation root inventory changed before Phase 3B: "
            f"observed={sorted(observed)} expected={sorted(EXPECTED_ROOT_BEFORE)}"
        )
    if len(MOVES) != 14 or len(IMPORT_OWNER_MAP) != 14:
        raise RuntimeError("Phase 3B requires exactly 14 MOVE owners")
    if _all_export_names(SIMULATION / "__init__.py") != EXPECTED_PUBLIC:
        raise RuntimeError("simulation public API changed before migration")
    for package in PACKAGE_FILES:
        if (SIMULATION / package).exists():
            raise RuntimeError(f"simulation owner package already exists: {package}")


def _move_files() -> None:
    for old, new in MOVES.items():
        old_path = SIMULATION / old
        new_path = SIMULATION / new
        if not old_path.is_file() or new_path.exists():
            raise RuntimeError(f"invalid move precondition: {old} -> {new}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)


def _rewrite_owner_references() -> None:
    # Two-stage placeholders make the owner map atomic. This prevents a new
    # owner such as ``orders.admission`` from being rewritten again by the
    # shorter retired ``orders`` key into ``orders.model.admission``.
    placeholders = {
        old: f"__PHASE3B_SIM_OWNER_{index}__"
        for index, old in enumerate(IMPORT_OWNER_MAP)
    }
    for root_name in ("trade_rl", "tests", "scripts"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() in {Path(__file__).resolve(), ARCHITECTURE_TEST.resolve()}:
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, placeholder in placeholders.items():
                updated = updated.replace(old, placeholder)
            for old, new in IMPORT_OWNER_MAP.items():
                updated = updated.replace(placeholders[old], new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def _write_package_markers() -> None:
    descriptions = {
        "orders": "Order model, admission, and reconciliation ownership.",
        "stateful": "Stateful execution lifecycle ownership.",
        "targets": "Target execution and exposure-control ownership.",
        "diagnostics": "Simulation evidence, stress, and performance diagnostics.",
    }
    for package, description in descriptions.items():
        path = SIMULATION / package / "__init__.py"
        if path.exists():
            raise RuntimeError(f"unexpected package marker: {path}")
        path.write_text(f'"""{description}"""\n', encoding="utf-8")


def _assert_final_tree() -> None:
    observed_root = {path.name for path in SIMULATION.glob("*.py")}
    if observed_root != EXPECTED_ROOT_AFTER:
        raise RuntimeError(f"unexpected final simulation root: {sorted(observed_root)}")
    for package, expected_files in PACKAGE_FILES.items():
        observed = {path.name for path in (SIMULATION / package).glob("*.py")}
        if observed != expected_files:
            raise RuntimeError(
                f"unexpected {package} files: observed={sorted(observed)} "
                f"expected={sorted(expected_files)}"
            )
    if _all_export_names(SIMULATION / "__init__.py") != EXPECTED_PUBLIC:
        raise RuntimeError("simulation public API changed during migration")

    stale = set(IMPORT_OWNER_MAP)
    offenders: list[tuple[str, list[str]]] = []
    for root_name in ("trade_rl", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            modules = _imported_modules(path)
            invalid = sorted(modules & stale)
            if invalid:
                offenders.append((str(path.relative_to(ROOT)), invalid))
            if path.is_relative_to(SIMULATION):
                upward = sorted(
                    module
                    for module in modules
                    if module.startswith("trade_rl.strategies")
                    or module.startswith("trade_rl.evaluation")
                )
                if upward:
                    offenders.append((str(path.relative_to(ROOT)), upward))
    if offenders:
        raise RuntimeError(f"invalid simulation imports after migration: {offenders}")


def main() -> None:
    _assert_preconditions()
    _move_files()
    _rewrite_owner_references()
    _write_package_markers()
    _assert_final_tree()
    print("Phase 3B simulation ownership migration created successfully")


if __name__ == "__main__":
    main()
