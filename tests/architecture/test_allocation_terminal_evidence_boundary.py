"""Closed evidence/report must validate returned data without execution authority."""

import ast
from pathlib import Path


def test_closed_terminal_owners_are_pure_and_have_no_research_gateway():
    root = Path(__file__).resolve().parents[2]
    for owner in (
        "allocation_terminal_evidence",
        "allocation_settled_forecast_comparison",
    ):
        path = root / f"trade_rl/evaluation/{owner}.py"
        tree = ast.parse(path.read_text("utf-8"))
        imports = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module
        }
        forbidden = (
            "stable_baselines3",
            "torch",
            "subprocess",
            "trade_rl.data",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.final_test",
            "trade_rl.evaluation.rl_allocation.training",
            "trade_rl.simulation.execution",
            "trade_rl.simulation.stateful",
        )
        assert not any(
            p == name or name.startswith(p + ".") for p in forbidden for name in imports
        )
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not calls & {
            "MarketExecutor",
            "run_terminal_allocation_execution",
            "run_declared_global_allocation_execution",
            "env.reset",
            "env.step",
            "book.apply_fills",
            "book.advance_base_bar",
        }


def test_actor_free_validation_reuses_leaves_without_fabricating_an_actor():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "trade_rl/evaluation/rl_allocation/transition_facts.py").read_text(
            "utf-8"
        )
    )
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in (
        "validate_terminal_allocation_facts",
        "_returned_book",
        "_returned_execution",
    ):
        calls = {
            ast.unparse(n.func)
            for n in ast.walk(functions[name])
            if isinstance(n, ast.Call)
        }
        assert not calls & {
            "AllocationDecision",
            "AllocationInputs",
            "AfterCostTargetAllocator",
            "expected.propose",
            "decision.propose",
            "MarketExecutor",
        }
    actor = functions["_validate_execution_facts"]
    calls = [
        ast.unparse(n.func)
        for statement in actor.body
        for n in ast.walk(statement)
        if isinstance(n, ast.Call)
    ]
    assert (
        calls.index("_returned_book")
        < calls.index("PreTradeRisk(config).constrain")
        < calls.index("_returned_execution")
    )


def test_closed_terminal_capabilities_and_limits_are_current_docs():
    root = Path(__file__).resolve().parents[2]
    for doc in (
        "architecture/lean-core.md",
        "architecture/package-boundaries.md",
        "architecture/research-assurance.md",
        "research/current-status.md",
    ):
        text = (root / "docs" / doc).read_text("utf-8")
        assert "allocation_terminal_evidence" in text
        assert "allocation_settled_forecast_comparison" in text
        assert "settled" in text and "intrabar" in text
