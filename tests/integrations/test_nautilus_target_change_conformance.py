from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.execution_probe import (
    run_same_side_target_change_execution_probe,
)
from trade_rl.simulation.execution_canonicalization import compare_dual_shadow_execution
from trade_rl.simulation.legacy_execution_probe import (
    run_legacy_same_side_target_change_probe,
)


@pytest.mark.nautilus
def test_same_side_target_changes_have_exact_dual_shadow_parity() -> None:
    legacy = run_legacy_same_side_target_change_probe()
    candidate = run_same_side_target_change_execution_probe(
        starting_balance=Decimal("1000"),
    )

    assert [fill.quantity_lots for fill in candidate.fills] == [1000, 1000, -1500, -500]
    assert [fill.position_lots for fill in candidate.fills] == [1000, 2000, 500, 0]

    report = compare_dual_shadow_execution(
        legacy_fills=legacy.fills,
        candidate_fills=candidate.fills,
        legacy_economics=legacy.economics,
        candidate_economics=candidate.economics,
    )
    assert report.fill_parity is True, (
        report.mismatches,
        legacy.fills,
        candidate.fills,
    )
    economics = (
        f"fee={legacy.economics.fee_minor}/{candidate.economics.fee_minor} "
        f"pnl={legacy.economics.realized_pnl_minor}/"
        f"{candidate.economics.realized_pnl_minor} "
        f"equity={legacy.economics.final_equity_minor}/"
        f"{candidate.economics.final_equity_minor}"
    )
    assert report.economic_parity is True, economics
    assert report.exact_parity is True, report.mismatches
