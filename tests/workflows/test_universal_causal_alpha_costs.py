from __future__ import annotations

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_costs as costs


class _Market:
    def __init__(self, notionals: list[float]) -> None:
        self.n_bars = len(notionals)
        self._notionals = np.asarray(notionals, dtype=np.float64)

    def market_notional(self, index: int) -> np.ndarray:
        return self._notionals[index : index + 1]


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
