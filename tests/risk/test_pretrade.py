from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def test_soft_turnover_limit_applies_to_valid_rebalance() -> None:
    risk = PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=0.3))
    result = risk.constrain(np.array([0.8, -0.2]), current=np.array([0.2, -0.2]), drawdown=0.0)
    assert result.constrained_turnover == pytest.approx(0.3)
    assert "max_turnover" in result.reasons


def test_hard_concentration_limit_overrides_turnover_for_invalid_current_book() -> None:
    risk = PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=0.1))
    result = risk.constrain(np.array([0.4, 0.0]), current=np.array([0.7, 0.0]), drawdown=0.0)
    assert result.weights[0] <= 0.4 + 1e-12
    assert result.turnover_overridden is True
    assert "hard_risk_turnover_override" in result.reasons


def test_drawdown_stop_flattens_even_when_turnover_limit_is_zero() -> None:
    risk = PreTradeRisk(PreTradeRiskConfig(max_turnover=0.0, drawdown_start=0.1, drawdown_stop=0.2))
    result = risk.constrain(np.array([0.5, -0.5]), current=np.array([0.5, -0.5]), drawdown=0.25)
    np.testing.assert_array_equal(result.weights, np.zeros(2))
    assert result.turnover_overridden is True
