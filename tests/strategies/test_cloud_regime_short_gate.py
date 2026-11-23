"""Short eligibility policy on top of C2: long exposure is untouched."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.strategies.test_divergence_cloud_follow import _signals, _snapshot
from trade_rl.data.features.cloud_regime_context import CloudRegimeContext
from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.strategies.cloud_regime_short_gate import (
    CloudRegimeShortGateConfig,
    CloudRegimeShortGatePolicy,
)


def _context(
    signals: DivergenceCloudMtfPack, *, allowed: tuple[int, ...], slope: bool = True
) -> CloudRegimeContext:
    n = len(signals.timestamps)
    momentum = np.ones(n, dtype=float)
    momentum[list(allowed)] = -0.1
    cloud_slope = np.full(n, -0.01 if slope else +0.01)
    return CloudRegimeContext(
        timestamps=signals.timestamps,
        momentum_30d=momentum,
        cloud_mid_slope_7d=cloud_slope,
        distance_to_cloud_mid_pct=np.zeros(n),
        realized_vol_30d=np.ones(n),
        ready=np.ones(n, dtype=bool),
    )


def test_short_entry_rejected_until_negative_30d_trend() -> None:
    signal = _signals([-1] * 35)
    context = _context(signal, allowed=tuple(range(12, 35)))
    policy = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signal,
        context=context,
        config=CloudRegimeShortGateConfig(mode="momentum"),
    )
    assert policy.decide_portfolio(_snapshot(signal, 6)).target_weights == (0.0,)
    assert policy.decide_portfolio(_snapshot(signal, 12)).target_weights == (-0.25,)


def test_short_flattens_on_lost_macro_eligibility_and_holds_exact_quantity() -> None:
    signal = _signals([-1] * 30)
    context = _context(signal, allowed=tuple(range(12, 18)))
    policy = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signal,
        context=context,
        config=CloudRegimeShortGateConfig(mode="momentum"),
    )
    assert policy.decide_portfolio(_snapshot(signal, 12)).target_weights == (-0.25,)
    keep = policy.decide_portfolio(_snapshot(signal, 13, quantity=-3, weight=-0.20))
    assert keep.hold_current_quantity_symbols == (0,)
    assert keep.target_weights == (-0.20,)
    assert policy.decide_portfolio(
        _snapshot(signal, 18, quantity=-3, weight=-0.20)
    ).target_weights == (0.0,)


def test_momentum_plus_cloud_slope_mode_uses_both_gates() -> None:
    signal = _signals([-1] * 35)
    pos_slope = _context(signal, allowed=tuple(range(35)), slope=False)
    pol = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signal,
        context=pos_slope,
        config=CloudRegimeShortGateConfig(mode="momentum_and_cloud_slope"),
    )
    assert pol.decide_portfolio(_snapshot(signal, 10)).target_weights == (0.0,)
    neg_slope = _context(signal, allowed=tuple(range(35)), slope=True)
    pol2 = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signal,
        context=neg_slope,
        config=CloudRegimeShortGateConfig(mode="momentum_and_cloud_slope"),
    )
    assert pol2.decide_portfolio(_snapshot(signal, 10)).target_weights == (-0.25,)


def test_long_entry_does_not_require_short_filters_to_be_ready() -> None:
    signal = _signals([1] * 35)
    context = _context(signal, allowed=())
    context = dataclasses.replace(context, ready=np.zeros(35, dtype=bool))
    pol = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signal,
        context=context,
        config=CloudRegimeShortGateConfig(mode="momentum"),
    )
    assert pol.decide_portfolio(_snapshot(signal, 6)).target_weights == (0.25,)


@pytest.mark.parametrize("mode", ["other", "", None])
def test_invalid_mode_rejected(mode: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        CloudRegimeShortGateConfig(mode=mode)


def test_native_replay_filters_short_and_holds_exact_quantity() -> None:
    from tests.strategies.test_divergence_cloud_follow import _native_market
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.simulation import ExecutionCostConfig

    market = _native_market()
    pack = _signals([-1] * 55)
    context = _context(pack, allowed=tuple(range(12, 25)))
    policy = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=pack,
        context=context,
        config=CloudRegimeShortGateConfig(mode="momentum"),
    )
    result = run_shared_portfolio_replay(
        market,
        policy,
        start_index=0,
        stop_index=54,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig(),
    )
    assert result.book.fill_count == 2
    assert result.book.total_cost > 0.0
    assert result.decisions[12].proposal_weights == (-0.25,)
    assert result.decisions[13].requested_hold_symbols == (0,)
    assert result.decisions[25].proposal_weights == (0.0,)
    assert policy.blocked_short_proposals > 0
    assert policy.ineligible_short_flat_requests >= 1


def test_long_decisions_identical_to_unfiltered_native_C2() -> None:
    from tests.strategies.test_divergence_cloud_follow import _native_market
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.simulation import ExecutionCostConfig
    from trade_rl.strategies.divergence_cloud_follow import (
        CloudFollowConfig,
        CloudFollowPolicy,
    )

    market = _native_market()
    pack = _signals([1] * 55)
    ctx = _context(pack, allowed=())
    ctx = dataclasses.replace(ctx, ready=np.zeros(55, dtype=bool))
    cost = ExecutionCostConfig()
    unfiltered = run_shared_portfolio_replay(
        market,
        CloudFollowPolicy(
            symbol="BTCUSDT",
            signals=pack,
            config=CloudFollowConfig(mode="confirmed_cloud"),
        ),
        start_index=0,
        stop_index=54,
        initial_capital=10_000,
        execution_cost=cost,
    )
    filtered = run_shared_portfolio_replay(
        market,
        CloudRegimeShortGatePolicy(
            symbol="BTCUSDT",
            signals=pack,
            context=ctx,
            config=CloudRegimeShortGateConfig(mode="momentum_and_cloud_slope"),
        ),
        start_index=0,
        stop_index=54,
        initial_capital=10_000,
        execution_cost=cost,
    )
    assert unfiltered.decisions == filtered.decisions
    assert unfiltered.book.portfolio_value == filtered.book.portfolio_value
    assert unfiltered.book.total_cost == filtered.book.total_cost


def test_context_cannot_be_shorter_than_signal_clock() -> None:
    pack = _signals([-1] * 35)
    context = _context(pack, allowed=())
    context = dataclasses.replace(context, momentum_30d=context.momentum_30d[:-1])
    with pytest.raises(ValueError, match="one row"):
        CloudRegimeShortGatePolicy(
            symbol="BTCUSDT",
            signals=pack,
            context=context,
            config=CloudRegimeShortGateConfig(mode="momentum"),
        )


def test_terminated_account_keeps_native_fail_closed_exact_hold() -> None:
    signals = _signals([-1] * 35)
    context = _context(signals, allowed=())
    policy = CloudRegimeShortGatePolicy(
        symbol="BTCUSDT",
        signals=signals,
        context=context,
        config=CloudRegimeShortGateConfig(mode="momentum"),
    )
    state = _snapshot(signals, 12, quantity=-3, weight=-0.2)
    ended_account = dataclasses.replace(
        state.account, insolvent=True, termination_reason="insolvency"
    )
    proposal = policy.decide_portfolio(
        dataclasses.replace(state, account=ended_account)
    )
    assert proposal.target_weights == (-0.2,)
    assert proposal.hold_current_quantity_symbols == (0,)
    assert policy.ineligible_short_flat_requests == 0
