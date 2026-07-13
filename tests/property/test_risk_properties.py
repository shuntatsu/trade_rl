from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


@dataclass(frozen=True)
class RiskCase:
    target: np.ndarray
    current: np.ndarray
    max_gross: float
    max_abs_weight: float
    max_turnover: float
    drawdown: float


@st.composite
def risk_cases(draw: st.DrawFn) -> RiskCase:
    size = draw(st.integers(min_value=1, max_value=12))
    max_gross = draw(
        st.floats(
            min_value=0.1,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    max_abs_weight = min(max_gross, 0.5)
    max_turnover = draw(
        st.floats(
            min_value=0.0,
            max_value=2.0 * max_gross,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    target = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=-2.0,
                    max_value=2.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=size,
                max_size=size,
            )
        ),
        dtype=np.float64,
    )
    current = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=-0.25,
                    max_value=0.25,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=size,
                max_size=size,
            )
        ),
        dtype=np.float64,
    )
    current = np.clip(current, -max_abs_weight, max_abs_weight)
    gross = float(np.abs(current).sum())
    if gross > max_gross:
        current *= max_gross / gross
    drawdown = draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return RiskCase(
        target=target,
        current=current,
        max_gross=max_gross,
        max_abs_weight=max_abs_weight,
        max_turnover=max_turnover,
        drawdown=drawdown,
    )


@settings(max_examples=120, deadline=None)
@given(case=risk_cases())
def test_risk_output_always_satisfies_hard_limits(case: RiskCase) -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=case.max_gross,
            max_abs_weight=case.max_abs_weight,
            max_turnover=case.max_turnover,
            drawdown_start=0.4,
            drawdown_stop=0.8,
        )
    )
    result = risk.constrain(
        case.target,
        current=case.current,
        drawdown=case.drawdown,
    )

    assert np.isfinite(result.weights).all()
    assert np.max(np.abs(result.weights)) <= case.max_abs_weight + 1e-10
    assert np.abs(result.weights).sum() <= case.max_gross + 1e-10
    realized_turnover = float(np.abs(result.weights - case.current).sum())
    if result.turnover_overridden:
        assert "hard_risk_turnover_override" in result.reasons
    else:
        assert realized_turnover <= case.max_turnover + 1e-10
