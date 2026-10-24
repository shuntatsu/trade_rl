from __future__ import annotations

from pathlib import Path

import trade_rl.simulation as simulation
from tests.architecture.test_lean_dependency_boundaries import collect_trade_rl_imports
from trade_rl.simulation import objectives
from trade_rl.simulation.objectives import binding, clock, contract

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl" / "simulation" / "objectives"


def test_objective_declarations_use_only_lower_validation_and_identity_owners() -> None:
    allowed = {"trade_rl._validation", "trade_rl.artifacts.hashing"}
    own_package = "trade_rl.simulation.objectives"
    for path in PACKAGE.glob("*.py"):
        imports = collect_trade_rl_imports(path)
        assert not {
            name
            for name in imports
            if name not in allowed
            and name != own_package
            and not name.startswith(own_package + ".")
        }


def test_objective_subpackage_owns_contracts_without_expanding_simulation_root() -> (
    None
):
    assert objectives.__all__ == [
        "CapitalContract",
        "FinancialClockContract",
        "ObjectiveContract",
        "BoundObjectiveClock",
        "net_equity_increment",
    ]
    assert objectives.BoundObjectiveClock is binding.BoundObjectiveClock
    assert objectives.CapitalContract is contract.CapitalContract
    assert objectives.ObjectiveContract is contract.ObjectiveContract
    assert objectives.net_equity_increment is contract.net_equity_increment
    assert objectives.FinancialClockContract is clock.FinancialClockContract
    assert set(objectives.__all__).isdisjoint(simulation.__all__)
