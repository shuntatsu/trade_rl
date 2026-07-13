from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


@settings(max_examples=120, deadline=None)
@given(
    target=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    ),
    current=st.data(),
    max_gross=st.floats(
        min_value=0.1,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_turnover=st.floats(
        min_value=0.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    drawdown=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_risk_output_always_satisfies_hard_limits(
    target: list[float],
    current: st.DataObject,
    max_gross: float,
    max_turnover: float,
    drawdown: float,
) -> None:
    current_values = current.draw(
        st.lists(
            st.floats(
                min_value=-0.25,
                max_value=0.25,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=len(target),
            max_size=len(target),
        )
    )
    current_vector = np.asarray(current_values, dtype=np.float64)
    current_gross = float(np.abs(current_vector).sum())
    if current_gross > 1.0:
        current_vector /= current_gross

    max_abs_weight = min(max_gross, 0.5)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=max_gross,
            max_abs_weight=max_abs_weight,
            max_turnover=max_turnover,
            drawdown_start=0.4,
            drawdown_stop=0.8,
        )
    )
    result = risk.constrain(
        np.asarray(target, dtype=np.float64),
        current=current_vector,
        drawdown=drawdown,
    )

    assert np.isfinite(result.weights).all()
    assert np.max(np.abs(result.weights)) <= max_abs_weight + 1e-10
    assert np.abs(result.weights).sum() <= max_gross + 1e-10
    assert np.abs(result.weights - current_vector).sum() <= max_turnover + 1e-10
