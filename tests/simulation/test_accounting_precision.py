from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


def test_from_weights_accepts_subcent_floating_point_reconstruction_error() -> None:
    weights = np.array([-0.14307002, -0.32557599, -0.12499748])
    prices = np.array([95_432.17, 3_456.78, 612.34])

    book = BookState.from_weights(
        weights=weights,
        capital=100_000.0,
        prices=prices,
        max_gross=1.0,
    )

    assert book.portfolio_value == pytest.approx(100_000.0, abs=1e-8)
    assert book.peak_value == 100_000.0


def test_book_state_still_rejects_economically_lower_peak() -> None:
    with pytest.raises(ValueError, match="peak_value cannot be below portfolio_value"):
        BookState(
            quantities=np.array([1.0]),
            cash=1.0,
            mark_prices=np.array([100.0]),
            peak_value=100.0,
        )
