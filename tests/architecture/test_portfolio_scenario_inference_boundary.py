from __future__ import annotations

import ast
from pathlib import Path

import pytest

import trade_rl.strategies as strategies
import trade_rl.strategies.rl as rl
from tools.agent_repo.source_index import ImportCollector

ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = ROOT / "trade_rl" / "strategies" / "rl"
MODULES = (
    RL_ROOT / "portfolio_scenario_runtime.py",
    RL_ROOT / "portfolio_scenario_inference.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "trade_rl.evaluation",
    "trade_rl.research",
    "trade_rl.strategies.rl.portfolio_training",
    "trade_rl.strategies.runs",
)


def _imports(path: Path, *, package: Path = ROOT / "trade_rl") -> set[str]:
    return ImportCollector(package).collect_direct(path)


def _assert_scenario_import_boundary(imports: set[str], *, label: str) -> None:
    forbidden = {
        module
        for module in imports
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    }
    assert not forbidden, f"{label} imports forbidden scenario owners: {forbidden}"


def test_direct_child_import_is_visible_to_the_scenario_boundary_guard(
    tmp_path: Path,
) -> None:
    package = tmp_path / "trade_rl"
    probe = package / "evaluation" / "scenario_import_probe.py"
    child = package / "strategies" / "rl" / "portfolio_training.py"
    probe.parent.mkdir(parents=True)
    child.parent.mkdir(parents=True)
    probe.write_text(
        "from trade_rl.strategies.rl import portfolio_training\n",
        encoding="utf-8",
    )
    child.write_text("VALUE = 1\n", encoding="utf-8")
    imports = _imports(probe, package=package)
    assert "trade_rl.strategies.rl.portfolio_training" in imports
    with pytest.raises(AssertionError, match="forbidden scenario owners"):
        _assert_scenario_import_boundary(imports, label="synthetic probe")


def test_scenario_inference_owners_stay_in_the_rl_capability() -> None:
    for path in MODULES:
        _assert_scenario_import_boundary(_imports(path), label=path.name)


def test_scenario_inference_does_not_expand_strategy_facades() -> None:
    for name in (
        "PortfolioScenarioInferencePolicy",
        "PortfolioScenarioGymObservationWrapperV2",
        "PortfolioScenarioObservationRuntimeBindingV2",
        "load_portfolio_scenario_inference_bundle",
    ):
        assert name not in strategies.__all__
        assert name not in rl.__all__


def test_scenario_admission_precedes_checkpoint_deserialization() -> None:
    path = RL_ROOT / "portfolio_scenario_inference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_portfolio_scenario_inference_bundle"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(loader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    ordered_calls = (
        "_verify_portfolio_inference_bundle_inputs",
        "_scenario_binding",
        "_make_observation_wrapper",
        "_load_verified_bundle_policy",
    )
    assert all(name in call_lines for name in ordered_calls)
    assert [call_lines[name] for name in ordered_calls] == sorted(
        call_lines[name] for name in ordered_calls
    )
