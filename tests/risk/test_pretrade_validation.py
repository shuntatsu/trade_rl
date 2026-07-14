from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gross": math.inf},
        {"max_gross": 0.0},
        {"max_gross": 11.0},
        {"max_abs_weight": 0.0},
        {"max_abs_weight": 2.0},
        {"max_turnover": -1.0},
        {"max_turnover": 3.0},
        {"drawdown_start": -0.1},
        {"drawdown_stop": 1.1},
        {"drawdown_start": 0.5, "drawdown_stop": 0.4},
        {"fail_closed_tolerance": -1.0},
        {"emergency_turnover_override": 1},
    ],
)
def test_pretrade_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PreTradeRiskConfig(**kwargs)  # type: ignore[arg-type]


def test_risk_scale_covers_linear_and_equal_threshold_modes() -> None:
    risk = PreTradeRisk(PreTradeRiskConfig(drawdown_start=0.1, drawdown_stop=0.2))
    assert risk.risk_scale(0.1) == pytest.approx(1.0)
    assert risk.risk_scale(0.15) == pytest.approx(0.5)
    assert risk.risk_scale(0.2) == pytest.approx(0.0)
    equal = PreTradeRisk(PreTradeRiskConfig(drawdown_start=0.2, drawdown_stop=0.2))
    assert equal.risk_scale(0.21) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        risk.risk_scale(math.nan)


def test_hard_gross_and_concentration_limits_record_reasons() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=0.6, max_abs_weight=0.4, max_turnover=1.2)
    )
    result = risk.constrain(np.array([0.8, 0.8]), current=np.zeros(2), drawdown=0.15)
    assert "max_abs_weight" in result.reasons
    assert "max_gross" in result.reasons
    assert "drawdown_deleveraging" in result.reasons
    assert result.risk_scale == pytest.approx(0.5)


def test_invalid_targets_and_disabled_emergency_override_fail_closed() -> None:
    risk = PreTradeRisk()
    with pytest.raises(ValueError, match="same shape"):
        risk.constrain(np.array([]), current=np.array([]), drawdown=0.0)
    with pytest.raises(ValueError, match="finite"):
        risk.constrain(np.array([math.nan]), current=np.array([0.0]), drawdown=0.0)

    strict = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=0.4,
            max_turnover=0.1,
            emergency_turnover_override=False,
        )
    )
    with pytest.raises(RuntimeError, match="requires turnover"):
        strict.constrain(np.array([0.4]), current=np.array([0.7]), drawdown=0.0)
