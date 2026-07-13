from __future__ import annotations

import numpy as np
import pytest

from mars_lite.eval.strategy_metrics import reannualize_strategy_result
from mars_lite.learning.baselines import StrategyResult


@pytest.mark.parametrize("bars_per_year", [2_190, 365])
def test_baseline_sharpe_uses_effective_annualization(bars_per_year: int) -> None:
    returns = np.array([0.01, -0.005, 0.002, 0.004], dtype=np.float64)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    original = StrategyResult(
        name="diagnostic",
        equity_curve=equity,
        total_return=float(equity[-1] - 1.0),
        sharpe=999.0,
        max_drawdown=0.0,
        turnover_total=1.0,
        n_bars=len(returns),
    )

    result = reannualize_strategy_result(original, bars_per_year=bars_per_year)
    expected = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

    assert result.sharpe == pytest.approx(expected)
    assert original.sharpe == 999.0
