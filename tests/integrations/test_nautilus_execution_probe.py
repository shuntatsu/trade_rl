from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.execution_probe import (
    run_flat_long_flat_execution_probe,
)


@pytest.mark.nautilus
def test_execution_probe_closes_position_and_records_accounting() -> None:
    result = run_flat_long_flat_execution_probe()

    assert result.runtime_version == "1.230.0"
    assert result.orders_closed == 2
    assert result.positions_closed == 1
    assert result.open_positions == 0
    assert result.avg_px_open == "100.1"
    assert result.avg_px_close == "104.9"
    assert result.realized_pnl
    assert result.commissions
    assert result.final_balance
    assert len(result.digest()) == 64
