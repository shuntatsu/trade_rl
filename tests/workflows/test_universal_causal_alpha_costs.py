from __future__ import annotations

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_costs as costs
from trade_rl.simulation.execution import ExecutionCostConfig


class _Market:
    def __init__(self, notionals: list[float]) -> None:
        self.n_bars = len(notionals)
        self._notionals = np.asarray(notionals, dtype=np.float64)

    def market_notional(self, index: int) -> np.ndarray:
        return self._notionals[index : index + 1]


class _CostMarket:
    def __init__(
        self,
        *,
        spread_rate: list[float],
        max_participation_rate: list[float],
    ) -> None:
        self.n_bars = len(spread_rate)
        self._arrays = {
            "fee_rate": np.zeros((self.n_bars, 1), dtype=np.float64),
            "maker_fee_rate": np.zeros((self.n_bars, 1), dtype=np.float64),
            "taker_fee_rate": np.zeros((self.n_bars, 1), dtype=np.float64),
            "spread_rate": np.asarray(spread_rate, dtype=np.float64).reshape(-1, 1),
            "max_participation_rate": np.asarray(
                max_participation_rate, dtype=np.float64
            ).reshape(-1, 1),
        }

    def resolved_array(self, name: str) -> np.ndarray:
        return self._arrays[name]


def test_one_way_cost_rates_do_not_read_future_execution_rows() -> None:
    baseline = _CostMarket(
        spread_rate=[0.001, 0.001, 0.002, 0.003, 0.004, 0.005],
        max_participation_rate=[0.10, 0.10, 0.08, 0.07, 0.06, 0.05],
    )
    changed_future = _CostMarket(
        spread_rate=[0.001, 0.001, 0.002, 0.003, 0.40, 0.50],
        max_participation_rate=[0.10, 0.10, 0.08, 0.07, 0.90, 0.95],
    )
    kwargs = {
        "execution_cost": ExecutionCostConfig(
            spread_rate=0.0,
            impact_rate=0.01,
            max_participation_rate=1.0,
        ),
        "decision_indices": np.asarray([2]),
        "signal_delay_decisions": 1,
        "decision_bars": 1,
    }

    assert costs.causal_alpha_one_way_cost_rates(baseline, **kwargs) == pytest.approx(
        costs.causal_alpha_one_way_cost_rates(changed_future, **kwargs)
    )


def test_liquidity_weight_caps_use_only_prior_lower_tail_volume() -> None:
    market = _Market([100.0, 200.0, 50.0, 400.0, 1_000_000.0])

    caps = costs.causal_alpha_liquidity_weight_caps(
        market,
        decision_indices=np.asarray([2, 3]),
        reference_portfolio_value=100.0,
        max_position_to_market_notional=0.02,
        lookback_decisions=2,
        lower_quantile=0.10,
        safety_multiplier=0.80,
    )

    assert caps.tolist() == pytest.approx([0.0176, 0.0104])


def test_liquidity_weight_caps_are_invariant_to_current_and_future_volume() -> None:
    baseline = _Market([100.0, 200.0, 50.0, 400.0])
    changed_future = _Market([100.0, 200.0, 9_000_000.0, 8_000_000.0])
    kwargs = {
        "decision_indices": np.asarray([2]),
        "reference_portfolio_value": 100.0,
        "max_position_to_market_notional": 0.02,
        "lookback_decisions": 2,
        "lower_quantile": 0.10,
        "safety_multiplier": 0.80,
    }

    assert costs.causal_alpha_liquidity_weight_caps(
        baseline, **kwargs
    ) == pytest.approx(
        costs.causal_alpha_liquidity_weight_caps(changed_future, **kwargs)
    )


def test_liquidity_weight_caps_fail_closed_without_complete_history() -> None:
    with pytest.raises(ValueError, match="history"):
        costs.causal_alpha_liquidity_weight_caps(
            _Market([100.0, 200.0]),
            decision_indices=np.asarray([1]),
            reference_portfolio_value=100.0,
            max_position_to_market_notional=0.02,
            lookback_decisions=2,
            lower_quantile=0.10,
            safety_multiplier=0.80,
        )
