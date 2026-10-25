from __future__ import annotations

import ast
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def collect_imports(path: Path) -> set[str]:
    return ImportCollector(PACKAGE).collect(path)


def collect_trade_rl_imports(path: Path) -> set[str]:
    return {name for name in collect_imports(path) if within_module(name, "trade_rl")}


def _offenders(root: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    collector = ImportCollector(PACKAGE)
    result: list[str] = []
    for path in sorted(root.rglob("*.py")):
        imports = collector.collect(path)
        if any(
            within_module(name, prefix)
            for name in imports
            for prefix in forbidden_prefixes
        ):
            result.append(path.relative_to(ROOT).as_posix())
    return result


def _symbol_offenders(root: Path, forbidden_names: frozenset[str]) -> list[str]:
    result: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name in forbidden_names for alias in node.names
            ):
                found = True
            elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                found = True
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                found = True
        if found:
            result.append(path.relative_to(ROOT).as_posix())
    return result


def test_no_production_imports_retired_domain() -> None:
    assert _offenders(PACKAGE, ("trade_rl.domain",)) == []


def test_validation_module_is_standard_library_only() -> None:
    path = PACKAGE / "_validation.py"
    assert path.is_file()
    external = sorted(
        name
        for name in collect_imports(path)
        if name.split(".", 1)[0] not in sys.stdlib_module_names
    )
    assert external == []


def test_artifacts_do_not_depend_on_upper_layers() -> None:
    forbidden = (
        "trade_rl.data",
        "trade_rl.risk",
        "trade_rl.simulation",
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.integrations",
    )
    assert _offenders(PACKAGE / "artifacts", forbidden) == []


def test_data_does_not_depend_on_strategy_evaluation_or_simulation() -> None:
    forbidden = (
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.simulation",
    )
    assert _offenders(PACKAGE / "data", forbidden) == []


def test_market_order_rule_values_do_not_import_venue_adapters() -> None:
    imports = collect_imports(PACKAGE / "data" / "market_order_rules.py")
    assert not any(within_module(name, "trade_rl.integrations") for name in imports)


def test_integrations_do_not_depend_on_strategy_or_evaluation() -> None:
    forbidden = ("trade_rl.strategies", "trade_rl.evaluation")
    assert _offenders(PACKAGE / "integrations", forbidden) == []


def test_strategies_do_not_depend_on_evaluation() -> None:
    assert _offenders(PACKAGE / "strategies", ("trade_rl.evaluation",)) == []


def test_portfolio_observation_builder_is_strategy_owned_and_reused() -> None:
    builder = PACKAGE / "strategies" / "portfolio_observation.py"
    assert builder.is_file()
    assert not any(
        within_module(imported, "trade_rl.evaluation")
        for imported in collect_trade_rl_imports(builder)
    )
    builder_tree = ast.parse(builder.read_text(encoding="utf-8"))
    builder_function = next(
        node
        for node in ast.walk(builder_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_portfolio_decision_snapshot"
    )
    parameters = {
        argument.arg
        for argument in (*builder_function.args.args, *builder_function.args.kwonlyargs)
    }
    assert {
        "dataset",
        "index",
        "agent_stop_index",
        "book",
        "active_orders",
        "risk_config",
    } <= parameters
    assert not {"executor", "risk"} & parameters

    replay = (PACKAGE / "evaluation" / "replay.py").read_text(encoding="utf-8")
    tree = ast.parse(replay)
    imported_builder = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "trade_rl.strategies.portfolio_observation"
        and any(
            alias.name == "build_portfolio_decision_snapshot" for alias in node.names
        )
        for node in ast.walk(tree)
    )
    called_builder = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_portfolio_decision_snapshot"
        for node in ast.walk(tree)
    )
    assert imported_builder
    assert called_builder


def test_portfolio_policy_contract_is_detached_from_runtime_owners() -> None:
    path = PACKAGE / "strategies" / "portfolio.py"
    forbidden = (
        "trade_rl.evaluation",
        "trade_rl.risk",
        "trade_rl.simulation",
    )
    imports = collect_trade_rl_imports(path)
    assert not any(
        within_module(imported, prefix) for imported in imports for prefix in forbidden
    )


def test_rl_does_not_depend_on_forecast_family() -> None:
    assert (
        _offenders(
            PACKAGE / "strategies" / "rl",
            ("trade_rl.strategies.forecasts",),
        )
        == []
    )


def test_portfolio_gym_is_an_opt_in_adapter_over_existing_owners() -> None:
    path = PACKAGE / "strategies" / "rl" / "portfolio_env.py"
    imports = collect_trade_rl_imports(path)
    assert "trade_rl.strategies.portfolio_observation" in imports
    assert "trade_rl.simulation.stateful.portfolio_session" in imports
    assert "trade_rl.simulation.objectives.contract" in imports
    assert not any(within_module(name, "trade_rl.evaluation") for name in imports)

    facade = (PACKAGE / "strategies" / "rl" / "__init__.py").read_text(encoding="utf-8")
    assert "portfolio_env" not in facade
    assert "SharedPortfolioGymEnv" not in facade


def test_portfolio_episode_journal_stays_an_opt_in_strategy_wrapper() -> None:
    path = PACKAGE / "strategies" / "rl" / "portfolio_episode_journal.py"
    assert path.is_file()
    imports = collect_trade_rl_imports(path)
    assert "trade_rl.strategies.rl.portfolio_env" in imports
    assert not any(within_module(name, "trade_rl.evaluation") for name in imports)

    facade = (PACKAGE / "strategies" / "rl" / "__init__.py").read_text(encoding="utf-8")
    assert "portfolio_episode_journal" not in facade
    assert "PortfolioEpisodeJournalWrapper" not in facade


def test_risk_does_not_depend_on_strategy_or_evaluation() -> None:
    forbidden = ("trade_rl.strategies", "trade_rl.evaluation")
    assert _offenders(PACKAGE / "risk", forbidden) == []


def test_simulation_does_not_depend_on_strategy_or_evaluation() -> None:
    forbidden = ("trade_rl.strategies", "trade_rl.evaluation")
    assert _offenders(PACKAGE / "simulation", forbidden) == []


def test_native_target_transition_stays_in_simulation_without_facade_export() -> None:
    path = PACKAGE / "simulation" / "native_transition.py"
    imports = collect_trade_rl_imports(path)
    assert "trade_rl.risk.pretrade" in imports
    assert "trade_rl.simulation.accounting" in imports
    assert "trade_rl.simulation.execution" in imports
    assert _offenders(path.parent, ("trade_rl.strategies", "trade_rl.evaluation")) == []
    facade = (PACKAGE / "simulation" / "__init__.py").read_text(encoding="utf-8")
    assert "execute_native_target_transition" not in facade


def test_candidate_runs_do_not_depend_on_controlled_experiments() -> None:
    assert (
        _offenders(
            PACKAGE / "evaluation" / "runs",
            ("trade_rl.evaluation.experiments",),
        )
        == []
    )


def test_controlled_experiments_cannot_reach_sealed_final_test() -> None:
    experiments = PACKAGE / "evaluation" / "experiments"
    assert (
        _offenders(
            experiments,
            ("trade_rl.evaluation.robustness.walk_forward.sealed_test",),
        )
        == []
    )


def test_canonical_bootstrap_dependency_is_one_way_and_development_only() -> None:
    bootstrap = PACKAGE / "evaluation" / "experiments" / "bootstrap"
    forbidden_bootstrap = ("trade_rl.evaluation.robustness.walk_forward.sealed_test",)
    assert _offenders(bootstrap, forbidden_bootstrap) == []

    forbidden_lower_to_bootstrap = ("trade_rl.evaluation.experiments.bootstrap",)
    assert _offenders(PACKAGE / "integrations", forbidden_lower_to_bootstrap) == []
    assert (
        _offenders(PACKAGE / "evaluation" / "runs", forbidden_lower_to_bootstrap) == []
    )


def test_canonical_bootstrap_cannot_execute_research_or_final_lifecycle() -> None:
    bootstrap = PACKAGE / "evaluation" / "experiments" / "bootstrap"
    forbidden = frozenset(
        {
            "run_baseline",
            "define_experiment",
            "run_experiment",
            "verify_experiment",
            "compare_experiment",
            "decide_experiment",
            "freeze_study",
            "SealedTestLedger",
            "authorize_once",
        }
    )
    assert _symbol_offenders(bootstrap, forbidden) == []
