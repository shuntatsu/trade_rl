from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def test_valid_target_is_unchanged() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.7,
            max_turnover=1.0,
        )
    )
    current = np.array([0.2, -0.2])
    target = np.array([0.5, -0.5])

    result = risk.constrain(target, current=current, drawdown=0.0)

    np.testing.assert_array_equal(result.weights, target)
    assert result.was_constrained is False
    assert result.reasons == ()


def test_asset_and_gross_limits_are_applied_deterministically() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.8,
            max_abs_weight=0.5,
            max_turnover=2.0,
        )
    )

    result = risk.constrain(
        np.array([0.9, -0.7, 0.2]),
        current=np.zeros(3),
        drawdown=0.0,
    )

    assert np.max(np.abs(result.weights)) <= 0.5 + 1e-12
    assert np.abs(result.weights).sum() == pytest.approx(0.8)
    assert result.was_constrained is True
    assert "max_abs_weight" in result.reasons
    assert "max_gross" in result.reasons


def test_turnover_limit_projects_along_requested_delta() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=0.3,
        )
    )
    current = np.array([0.2, -0.2])
    target = np.array([0.8, -0.2])

    result = risk.constrain(target, current=current, drawdown=0.0)

    assert np.abs(result.weights - current).sum() == pytest.approx(0.3)
    np.testing.assert_allclose(result.weights, np.array([0.5, -0.2]))
    assert "max_turnover" in result.reasons


def test_drawdown_linearly_deleverages_between_thresholds() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=2.0,
            drawdown_start=0.10,
            drawdown_stop=0.30,
        )
    )

    result = risk.constrain(
        np.array([0.5, -0.5]),
        current=np.zeros(2),
        drawdown=0.20,
    )

    assert np.abs(result.weights).sum() == pytest.approx(0.5)
    assert "drawdown_deleveraging" in result.reasons


def test_drawdown_stop_forces_flat_target() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(drawdown_start=0.10, drawdown_stop=0.20)
    )

    result = risk.constrain(
        np.array([0.5, -0.5]),
        current=np.array([0.1, -0.1]),
        drawdown=0.25,
    )

    np.testing.assert_array_equal(result.weights, np.zeros(2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_gross", 0.0),
        ("max_abs_weight", 0.0),
        ("max_turnover", -0.1),
        ("drawdown_start", -0.1),
        ("drawdown_stop", 1.1),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: float) -> None:
    values = {
        "max_gross": 1.0,
        "max_abs_weight": 1.0,
        "max_turnover": 2.0,
        "drawdown_start": 0.10,
        "drawdown_stop": 0.30,
    }
    values[field] = value

    with pytest.raises(ValueError):
        PreTradeRiskConfig(**values)
