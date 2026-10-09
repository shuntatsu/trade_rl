"""Market preparation composes existing owners without trading or research authority."""

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

PACKAGE = Path(__file__).resolve().parents[2] / "trade_rl"
OWNER = PACKAGE / "evaluation" / "allocation_market_inputs.py"


def test_market_preparation_avoids_learner_ledger_and_research_lifecycle():
    imports = ImportCollector(PACKAGE).collect(OWNER)
    assert not any(
        within_module(module, prefix)
        for module in imports
        for prefix in (
            "trade_rl.simulation",
            "trade_rl.risk",
            "trade_rl.strategies.rl",
            "trade_rl.evaluation.rl_allocation",
            "trade_rl.evaluation.runs",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.final_test",
            "trade_rl.integrations",
            "torch",
            "stable_baselines3",
            "requests",
            "subprocess",
        )
    )
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(
        {
            "inspect_published_market_dataset_artifact",
            "read_text",
            "read_bytes",
            "load",
            "execute_forecast_proposal",
            "step",
            "learn",
        }
    )
    assert {
        "load_market_dataset_artifact",
        "estimate_declared_horizon_costs",
        "validated_training_scope",
        "fit_prequential_simple_ridge",
        "publish_simple_return_stream_artifact",
    } <= calls


def test_market_preparation_is_not_a_tier_one_facade_or_persisted_schema():
    from trade_rl import evaluation
    from trade_rl.evaluation import allocation_market_inputs

    assert set(allocation_market_inputs.__all__) == {
        "PreparedAllocationMarketInputs",
        "prepare_allocation_market_inputs",
    }
    for name in allocation_market_inputs.__all__:
        assert not hasattr(evaluation, name)
    assert not hasattr(
        allocation_market_inputs.PreparedAllocationMarketInputs, "payload"
    )


def test_declared_cost_owner_reuses_native_config_without_execution_or_new_schema():
    from trade_rl import evaluation
    from trade_rl.evaluation import allocation_costs

    owner = PACKAGE / "evaluation" / "allocation_costs.py"
    imports = ImportCollector(PACKAGE).collect(owner)
    assert "trade_rl.simulation.execution" in imports
    assert "trade_rl.data.build.economics" in imports
    assert not any(
        within_module(module, prefix)
        for module in imports
        for prefix in (
            "trade_rl.evaluation.rl_allocation",
            "trade_rl.strategies.rl",
            "trade_rl.evaluation.runs",
            "trade_rl.evaluation.final_test",
            "trade_rl.integrations",
            "torch",
            "stable_baselines3",
            "subprocess",
        )
    )
    tree = ast.parse(owner.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(
        {
            "MarketExecutor",
            "load_market_dataset_artifact",
            "read_bytes",
            "read_text",
            "fit",
            "learn",
            "step",
            "default_rng",
            "normal",
            "random",
        }
    )
    assert set(allocation_costs.__all__) == {
        "DeclaredAllocationCostRecipe",
        "estimate_declared_horizon_costs",
    }
    assert not hasattr(allocation_costs.DeclaredAllocationCostRecipe, "payload")
    for name in allocation_costs.__all__:
        assert not hasattr(evaluation, name)
