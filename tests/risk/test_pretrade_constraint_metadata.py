from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def test_pretrade_result_carries_constraint_limits_for_causal_cost_derivation() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.25,
            max_abs_weight=0.75,
            max_turnover=2.0,
            drawdown_start=0.12,
            drawdown_stop=0.20,
        )
    )

    result = risk.constrain(
        np.array([0.4, -0.3]),
        current=np.zeros(2),
        drawdown=0.0,
    )

    assert result.max_gross == pytest.approx(1.25)
    assert result.drawdown_budget == pytest.approx(0.12)
