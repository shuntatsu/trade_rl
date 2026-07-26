from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import (
    PreTradeRisk,
    PreTradeRiskConfig,
    RiskConstrainedTarget,
)


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


@pytest.mark.parametrize(
    "metadata",
    [
        {"max_gross": 0.0, "drawdown_budget": 0.1},
        {"max_gross": 1.0, "drawdown_budget": 1.1},
    ],
)
def test_constraint_metadata_fails_closed(metadata: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RiskConstrainedTarget(
            weights=np.array([0.1]),
            requested_turnover=0.1,
            constrained_turnover=0.1,
            was_constrained=False,
            reasons=(),
            risk_scale=1.0,
            **metadata,
        )
