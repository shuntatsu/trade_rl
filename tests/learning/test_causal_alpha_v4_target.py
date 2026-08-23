from __future__ import annotations

import math

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4TargetConfig,
    _v4_direct_objective,
    _v4_staged_objective,
    causal_alpha_v4_target_path,
)


def _array(value: float, rows: int = 1) -> np.ndarray:
    return np.full(rows, value, dtype=np.float64)


def _path(
    *,
    p4: float = 0.0,
    p24: float = 0.0,
    p72: float = 0.0,
    d4: float = 1.0,
    u4: float = 0.0,
    u24: float = 0.0,
    u72: float = 0.0,
    cost: float = 0.0,
    cap: float = 1.0,
    initial_weight: float = 0.0,
    rows: int = 1,
):
    return causal_alpha_v4_target_path(
        _array(p4, rows),
        _array(p24, rows),
        _array(p72, rows),
        direction_score_4h=_array(d4, rows),
        uncertainty_4h=_array(u4, rows),
        uncertainty_24h=_array(u24, rows),
        uncertainty_72h=_array(u72, rows),
        one_way_cost_rates=_array(cost, rows),
        liquidity_weight_caps=_array(cap, rows),
        config=CausalAlphaV4TargetConfig(),
        initial_weight=initial_weight,
    )


def test_v4_target_config_is_frozen_to_first_hypothesis() -> None:
    config = CausalAlphaV4TargetConfig()
    assert config.slow_target_magnitudes == (0.0, 0.025, 0.05, 0.1, 0.25)
    assert config.fast_deviation_magnitudes == (0.0, 0.025, 0.05)
    assert config.uncertainty_multiplier == 1.0
    assert config.execution_cost_multiplier == 1.5
    assert config.edge_margin == 0.001
    assert config.slow_rebalance_decisions == 16
    assert config.fast_rebalance_decisions == 4
    assert config.maximum_final_target_delta == 0.125
    assert config.maximum_fast_absolute_deviation == 0.05


def test_v4_slow_fusion_and_uncertainty_match_v3_semantics() -> None:
    path = _path(
        p24=0.06,
        p72=0.12,
        u24=0.02,
        u72=0.03,
        d4=0.0,
    )
    expected_mu = 0.5 * (0.06 + 0.12 / 3.0)
    disagreement = 0.5 * abs(0.06 - 0.12 / 3.0)
    expected_sigma = math.sqrt(0.25 * (0.02**2 + (0.03 / 3.0) ** 2) + disagreement**2)

    assert np.isclose(path.slow_expected_returns[0], expected_mu)
    assert np.isclose(path.slow_uncertainties[0], expected_sigma)


def test_v4_staged_objective_charges_final_turnover_hurdle_once() -> None:
    config = CausalAlphaV4TargetConfig()
    previous = 0.10
    anchor = 0.20
    final = 0.15
    cost = 0.002
    slow_mu = 0.04
    fast_mu = -0.01
    slow_sigma = 0.01
    fast_sigma = 0.02

    slow, fast_improvement, staged = _v4_staged_objective(
        previous=previous,
        anchor=anchor,
        final=final,
        slow_expected_return=slow_mu,
        slow_uncertainty=slow_sigma,
        fast_expected_return=fast_mu,
        fast_uncertainty=fast_sigma,
        one_way_cost_rate=cost,
        config=config,
    )
    direct_fast_final = _v4_direct_objective(
        target=final,
        previous=previous,
        expected_return=fast_mu,
        uncertainty=fast_sigma,
        one_way_cost_rate=cost,
        config=config,
    )
    direct_fast_anchor = _v4_direct_objective(
        target=anchor,
        previous=previous,
        expected_return=fast_mu,
        uncertainty=fast_sigma,
        one_way_cost_rate=cost,
        config=config,
    )

    assert np.isclose(fast_improvement, direct_fast_final - direct_fast_anchor)
    expected_cost_hurdle = abs(final - previous) * (
        cost * config.execution_cost_multiplier + config.edge_margin
    )
    slow_alpha_risk = (anchor - previous) * slow_mu - abs(anchor - previous) * slow_sigma
    fast_alpha_risk_delta = (
        (final - previous) * fast_mu
        - abs(final - previous) * fast_sigma
        - (
            (anchor - previous) * fast_mu
            - abs(anchor - previous) * fast_sigma
        )
    )
    assert np.isclose(staged, slow_alpha_risk + fast_alpha_risk_delta - expected_cost_hurdle)
    assert np.isclose(staged, slow + fast_improvement)


def test_direction_disagreement_blocks_risk_increase_but_not_flattening() -> None:
    blocked = _path(
        p4=0.05,
        p24=0.08,
        p72=0.24,
        d4=-1.0,
        initial_weight=0.0,
    )
    assert blocked.targets[0] == 0.0
    assert blocked.reasons[0] == "direction_disagreement_hold"

    reducing = _path(
        p4=0.05,
        p24=-0.08,
        p72=-0.24,
        d4=-1.0,
        initial_weight=0.10,
    )
    assert 0.0 <= reducing.targets[0] < 0.10
    assert reducing.reasons[0] != "direction_disagreement_hold"


def test_fast_forecast_below_uncertainty_cost_and_margin_cannot_trade() -> None:
    path = _path(
        p4=0.002,
        p24=0.0,
        p72=0.0,
        d4=1.0,
        u4=0.0015,
        cost=0.0005,
        initial_weight=0.0,
    )

    assert path.targets[0] == 0.0
    assert path.fast_impulse_change_count == 0
    assert path.submitted_change_count == 0


def test_fast_impulse_is_bounded_around_slow_anchor_and_final_delta() -> None:
    path = _path(
        p4=0.20,
        p24=0.20,
        p72=0.60,
        d4=1.0,
        initial_weight=0.0,
    )

    assert abs(path.fast_deviations[0]) <= 0.05
    assert abs(path.targets[0] - path.slow_anchors[0]) <= 0.05
    assert abs(path.targets[0]) <= 1.0
    assert abs(path.targets[0] - 0.0) <= 0.125


def test_liquidity_deleveraging_overrides_cadence_direction_and_delta_limit() -> None:
    rows = 2
    path = causal_alpha_v4_target_path(
        _array(0.20, rows),
        _array(0.20, rows),
        _array(0.60, rows),
        direction_score_4h=_array(-1.0, rows),
        uncertainty_4h=_array(0.0, rows),
        uncertainty_24h=_array(0.0, rows),
        uncertainty_72h=_array(0.0, rows),
        one_way_cost_rates=_array(0.0, rows),
        liquidity_weight_caps=np.asarray([1.0, 0.20], dtype=np.float64),
        config=CausalAlphaV4TargetConfig(),
        initial_weight=0.80,
    )

    assert path.targets[1] == 0.20
    assert path.reasons[1] == "liquidity_deleverage"
    assert path.liquidity_deleveraging_count == 1
    assert abs(path.targets[1] - path.targets[0]) > 0.125
