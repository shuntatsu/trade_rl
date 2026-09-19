from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import (
    PreTradeRisk,
    PreTradeRiskConfig,
    RiskConstrainedTarget,
    should_rebind_strategy_proposal,
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


def test_projection_l1_measures_proposal_to_final_emergency_projection() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
        )
    )

    result = risk.constrain(
        np.array([0.4, 0.2]),
        current=np.array([0.4, 0.2]),
        drawdown=0.0,
        emergency_flatten_mask=np.array([True, False]),
    )

    np.testing.assert_allclose(result.proposal_weights, [0.4, 0.2])
    np.testing.assert_allclose(result.pretrade_weights, [0.0, 0.2])
    assert result.projection_l1 == pytest.approx(0.4)
    assert result.projection_l1 == pytest.approx(
        float(np.abs(result.proposal_weights - result.pretrade_weights).sum())
    )


@pytest.mark.parametrize(
    ("reasons", "was_constrained", "expected"),
    [
        (("max_abs_weight",), True, True),
        (("max_gross",), True, True),
        (("max_turnover",), True, False),
        (("max_abs_weight", "max_turnover"), True, False),
        (("drawdown_deleveraging",), True, False),
        ((), False, False),
    ],
)
def test_strategy_proposal_rebinding_distinguishes_persistent_and_transient_risk(
    reasons: tuple[str, ...],
    was_constrained: bool,
    expected: bool,
) -> None:
    constrained = RiskConstrainedTarget(
        weights=np.array([0.1]),
        requested_turnover=0.1,
        constrained_turnover=0.1,
        was_constrained=was_constrained,
        reasons=reasons,
        risk_scale=1.0,
    )

    assert should_rebind_strategy_proposal(constrained) is expected


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
