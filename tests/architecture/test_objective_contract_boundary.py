from __future__ import annotations

from pathlib import Path

from tests.architecture.test_lean_dependency_boundaries import collect_trade_rl_imports
from trade_rl.evaluation import objectives
from trade_rl.evaluation.objectives import clock, contract

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl" / "evaluation" / "objectives"


def test_objective_declarations_depend_only_on_validation_and_artifact_identity() -> (
    None
):
    allowed = (
        "trade_rl._validation",
        "trade_rl.artifacts",
        "trade_rl.evaluation.objectives",
    )
    for path in PACKAGE.glob("*.py"):
        imports = collect_trade_rl_imports(path)
        assert not {
            name
            for name in imports
            if not any(
                name == prefix or name.startswith(prefix + ".") for prefix in allowed
            )
        }


def test_objective_facade_reexports_owners_without_expanding_evaluation_root() -> None:
    import trade_rl.evaluation as evaluation

    assert objectives.__all__ == [
        "CapitalContract",
        "FinancialClockContract",
        "ObjectiveContract",
        "net_equity_increment",
    ]
    assert objectives.CapitalContract is contract.CapitalContract
    assert objectives.ObjectiveContract is contract.ObjectiveContract
    assert objectives.net_equity_increment is contract.net_equity_increment
    assert objectives.FinancialClockContract is clock.FinancialClockContract
    assert set(objectives.__all__).isdisjoint(evaluation.__all__)
