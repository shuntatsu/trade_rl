from dataclasses import replace
from fractions import Fraction

import numpy as np

from tests.evaluation.test_shared_cash_replay import _market
from tests.integrations.test_binance_market_order_profile import _profile
from trade_rl.evaluation.directional import CloseAtEndStrategy, evaluate_directional_arm
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def _data():
    return _market(
        np.array([[100], [100], [1], [1], [1], [1]], dtype=float), symbols=("BTCUSDT",)
    ).with_content_identity()


def _factory():
    return ConstantIntentStrategy(PositionIntent.LONG)


def test_shared_replay_passes_profile_to_actual_close_execution():
    data = _data()
    results = []
    for enabled in (False, True):
        results.append(
            run_shared_cash_replay(
                data,
                (CloseAtEndStrategy(_factory(), close_index=4),),
                start_index=0,
                stop_index=5,
                gross_budget=0.1,
                initial_capital=10000,
                execution_cost=ExecutionCostConfig.zero(),
                market_order_profile=_profile(data, reduce_only_exits=enabled),
            )
        )
    ordinary, closing = results
    assert ordinary.book.exact_quantities == (Fraction(10),)
    assert closing.book.exact_quantities == (Fraction(0),)
    assert closing.diagnostics.n_trades == 2
    assert len(closing.returns.values) == 5


def test_directional_profile_result_records_actual_policy_and_exact_flatness():
    data = _data()
    profile = _profile(data)
    result = evaluate_directional_arm(
        data, _factory, start_index=0, stop_index=5, market_order_profile=profile
    )
    actual_policy = MarketExecutor(
        data,
        replace(
            ExecutionCostConfig.zero(),
            max_leverage=1.0,
            processing_bar_volume_capacity=False,
            borrow_rate_multiplier=1.0,
        ),
        market_order_profile=profile,
    ).execution_policy_digest
    assert result["schema"] == "directional_market_profile_arm_v1"
    assert result["execution_policy_digest"] == actual_policy
    assert result["market_order_profile_digest"] == profile.digest
    assert result["market_order_profile"] == profile.canonical_payload()
    assert result["terminal_exact_quantities"] == ["0"]
    assert result["terminal_flat"]
    assert result["metrics"]["n_trades"] == 2
    assert not result["qualified"]  # This synthetic price collapse is still a loss.
    ordinary = evaluate_directional_arm(
        data,
        _factory,
        start_index=0,
        stop_index=5,
        market_order_profile=_profile(data, reduce_only_exits=False),
    )
    assert ordinary["terminal_exact_quantities"] == ["10"]
    assert not ordinary["terminal_flat"]
    assert not ordinary["qualified"]


def test_omitted_profile_retains_legacy_schema_and_policy_digest():
    data = _data()
    omitted = evaluate_directional_arm(data, _factory, start_index=0, stop_index=5)
    explicit_none = evaluate_directional_arm(
        data, _factory, start_index=0, stop_index=5, market_order_profile=None
    )
    assert omitted == explicit_none
    assert omitted["schema"] == "directional_arm_v1"
    assert "market_order_profile_digest" not in omitted
    assert "terminal_exact_quantities" not in omitted


def test_profile_gate_does_not_call_a_tiny_exact_residue_flat():
    # A synthetic last-bar unit change exercises exact residual qualification.
    # This is an accounting boundary test, not a claim about USD-M corporate actions.
    splits = np.ones((6, 1))
    splits[-1] = 1e-12
    data = replace(
        _data(), split_factor=splits, identity_payload_json=None
    ).with_content_identity()
    result = evaluate_directional_arm(
        data, _factory, start_index=0, stop_index=5, market_order_profile=_profile(data)
    )
    residue = Fraction(result["terminal_exact_quantities"][0])
    assert 0 < residue < Fraction("1e-10")
    assert not result["terminal_flat"]
    assert not result["qualified"]
