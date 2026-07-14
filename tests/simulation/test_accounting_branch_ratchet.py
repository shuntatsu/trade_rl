from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def test_book_rejects_invalid_state_contracts() -> None:
    with pytest.raises(ValueError, match="contract_multipliers"):
        BookState(
            quantities=np.array([0.0]),
            cash=1.0,
            mark_prices=np.array([100.0]),
            peak_value=1.0,
            contract_multipliers=np.array([0.0]),
        )
    with pytest.raises(ValueError, match="insolvent"):
        BookState(
            quantities=np.array([0.0]),
            cash=1.0,
            mark_prices=np.array([100.0]),
            peak_value=1.0,
            insolvent=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="termination_reason"):
        BookState(
            quantities=np.array([0.0]),
            cash=1.0,
            mark_prices=np.array([100.0]),
            peak_value=1.0,
            termination_reason="unsupported",
        )
    with pytest.raises(ValueError, match="peak_value"):
        BookState(
            quantities=np.array([1.0]),
            cash=0.0,
            mark_prices=np.array([100.0]),
            peak_value=50.0,
        )
    with pytest.raises(ValueError, match="n_symbols"):
        BookState.zero(0, 1_000.0)


def test_nonpositive_equity_uses_fail_closed_weights_and_margin() -> None:
    book = BookState(
        quantities=np.array([0.0]),
        cash=0.0,
        mark_prices=np.array([100.0]),
        peak_value=1.0,
    )
    assert book.insolvent
    assert book.termination_reason is EconomicTerminationReason.INSOLVENCY
    assert book.weights.tolist() == [0.0]
    assert book.cash_weight == 0.0
    assert book.margin_utilization == 1.0
