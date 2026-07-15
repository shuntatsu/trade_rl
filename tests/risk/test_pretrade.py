from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def test_soft_turnover_limit_applies_to_valid_rebalance() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=0.3)
    )
    result = risk.constrain(
        np.array([0.8, -0.2]), current=np.array([0.2, -0.2]), drawdown=0.0
    )
    assert result.constrained_turnover == pytest.approx(0.3)
    assert "max_turnover" in result.reasons


def test_hard_concentration_limit_overrides_turnover_for_invalid_current_book() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.4, max_turnover=0.1)
    )
    result = risk.constrain(
        np.array([0.4, 0.0]), current=np.array([0.7, 0.0]), drawdown=0.0
    )
    assert result.weights[0] <= 0.4 + 1e-12
    assert result.turnover_overridden is True
    assert "hard_risk_turnover_override" in result.reasons


def test_drawdown_stop_flattens_even_when_turnover_limit_is_zero() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_turnover=0.0, drawdown_start=0.1, drawdown_stop=0.2)
    )
    result = risk.constrain(
        np.array([0.5, -0.5]), current=np.array([0.5, -0.5]), drawdown=0.25
    )
    np.testing.assert_array_equal(result.weights, np.zeros(2))
    assert result.turnover_overridden is True


def test_target_hysteresis_requires_stronger_entry_than_exit() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=2.0,
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        )
    )

    no_entry = risk.constrain(
        np.array([0.08]), current=np.array([0.0]), drawdown=0.0
    )
    hold = risk.constrain(
        np.array([0.04]), current=np.array([0.20]), drawdown=0.0
    )
    exit_position = risk.constrain(
        np.array([0.02]), current=np.array([0.20]), drawdown=0.0
    )

    np.testing.assert_array_equal(no_entry.weights, np.array([0.0]))
    np.testing.assert_array_equal(hold.weights, np.array([0.20]))
    np.testing.assert_array_equal(exit_position.weights, np.array([0.0]))
    assert "entry_hysteresis" in no_entry.reasons
    assert "hold_hysteresis" in hold.reasons
    assert "exit_hysteresis" in exit_position.reasons


def test_no_trade_band_suppresses_small_target_rebalances() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=2.0,
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        )
    )

    result = risk.constrain(
        np.array([0.23, -0.31]),
        current=np.array([0.20, -0.30]),
        drawdown=0.0,
    )

    np.testing.assert_array_equal(result.weights, np.array([0.20, -0.30]))
    assert result.constrained_turnover == 0.0
    assert "no_trade_band" in result.reasons


def test_hysteresis_flattens_weak_reversal_until_new_entry_is_confirmed() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=2.0,
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        )
    )

    result = risk.constrain(
        np.array([-0.08]), current=np.array([0.20]), drawdown=0.0
    )

    np.testing.assert_array_equal(result.weights, np.array([0.0]))
    assert "reversal_hysteresis" in result.reasons


def test_emergency_flatten_bypasses_turnover_and_no_trade_limits() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=0.0,
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.50,
        )
    )

    result = risk.constrain(
        np.array([0.40, 0.20]),
        current=np.array([0.40, 0.20]),
        drawdown=0.0,
        emergency_flatten_mask=np.array([True, False]),
    )

    np.testing.assert_array_equal(result.weights, np.array([0.0, 0.20]))
    assert result.turnover_overridden is True
    assert "emergency_flatten" in result.reasons
