from __future__ import annotations

import numpy as np

from trade_rl.data.builder import _calculate_one_bar_returns


def test_one_bar_returns_are_vectorized_and_require_contiguous_rows() -> None:
    """Only adjacent observable rows contribute to the market breadth channels."""

    close = np.asarray(
        [
            [100.0, 200.0],
            [110.0, 220.0],
            [121.0, 242.0],
            [133.1, 266.2],
        ],
        dtype=np.float64,
    )
    row_present = np.asarray(
        [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=np.bool_,
    )
    active = np.ones_like(row_present)

    returns, available = _calculate_one_bar_returns(
        close=close,
        causal_row_present=row_present,
        symbol_active=active,
    )

    expected = np.zeros_like(close)
    expected[1, 0] = np.log(1.1)
    expected[3, 1] = np.log(1.1)
    np.testing.assert_allclose(returns, expected)
    np.testing.assert_array_equal(
        available,
        np.asarray(
            [
                [False, False],
                [True, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
