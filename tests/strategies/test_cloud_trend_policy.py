"""Mechanism-only tests for role-separated trend/volatility decisions."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.test_divergence_cloud_follow import _native_market, _signals
from tests.strategies.test_divergence_cloud_follow import _snapshot as _base_snapshot
from trade_rl.data.features.cloud_trend_context import CloudTrendContext
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.cloud_trend_policy import CloudTrendConfig, CloudTrendPolicy


def _snapshot(pack, index, **kwargs):
    snap = _base_snapshot(pack, index, **kwargs)
    return replace(
        snap,
        account=replace(
            snap.account,
            exact_quantities=tuple(
                str(Fraction(q)) for q in snap.account.exact_quantities
            ),
        ),
    )


def context(pack, *, adx=0.4, atr=0.02, ready=None):
    n = len(pack.timestamps)
    mask = np.ones(n, dtype=bool) if ready is None else ready
    return CloudTrendContext(
        pack.timestamps, np.where(mask, adx, np.nan), np.where(mask, atr, np.nan), mask
    )


def policy(pack, ctx, mode):
    return CloudTrendPolicy(
        symbol="BTCUSDT", signals=pack, context=ctx, config=CloudTrendConfig(mode=mode)
    )


@pytest.mark.parametrize("short", [False, True])
def test_strong_trend_vetoes_exit_weak_trend_allows_exit(short):
    pack = _signals([1] * 50, bear_4h=(12,))
    direction = -1 if short else 1
    if short:
        pack = replace(
            pack,
            cloud_state=-pack.cloud_state,
            bull_4h=pack.bear_4h,
            bear_4h=pack.bull_4h,
        )
    state = _snapshot(pack, 12, quantity=direction * 10, weight=direction * 0.25)
    strong = policy(pack, context(pack, adx=0.25), "weak_trend_exits")
    assert strong.decide_portfolio(state).hold_current_quantity_symbols == (0,)
    weak = policy(pack, context(pack, adx=0.249), "weak_trend_exits")
    assert weak.decide_portfolio(state).target_weights == (0.0,)
    control = policy(pack, context(pack, adx=0.9), "all_exits")
    assert control.decide_portfolio(state).target_weights == (0.0,)


def test_partial_exit_is_latched_until_confirmed_flat_then_cooldown():
    pack = _signals([1] * 60, bear_4h=(12,))
    p = policy(pack, context(pack, adx=0.2), "weak_trend_exits")
    assert p.decide_portfolio(
        _snapshot(pack, 12, quantity=10, weight=0.25)
    ).target_weights == (0.0,)
    assert p.decide_portfolio(
        _snapshot(pack, 13, quantity=2, weight=0.05)
    ).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 14)).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 37)).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 38)).target_weights == (0.25,)


def test_atr_size_at_entry_only_and_cap():
    pack = _signals([1] * 40)
    p = policy(pack, context(pack, atr=0.02), "weak_trend_atr_entry")
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.125,)
    held = p.decide_portfolio(_snapshot(pack, 6, quantity=10, weight=0.3))
    assert held.target_weights == (0.3,)
    assert held.hold_current_quantity_symbols == (0,)
    capped = policy(pack, context(pack, atr=0.001), "weak_trend_atr_entry")
    assert capped.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.25,)
    snap = _snapshot(pack, 5)
    capped_risk = replace(snap, risk=replace(snap.risk, max_gross=0.03))
    assert policy(pack, context(pack), "weak_trend_atr_entry").decide_portfolio(
        capped_risk
    ).target_weights == (0.03,)


def test_unready_context_blocks_entry_and_latches_existing_exit():
    pack = _signals([1] * 40)
    ready = np.ones(40, dtype=bool)
    ready[5:7] = False
    p = policy(pack, context(pack, ready=ready), "weak_trend_exits")
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.0,)
    assert p.decide_portfolio(
        _snapshot(pack, 6, quantity=10, weight=0.25)
    ).target_weights == (0.0,)
    assert p.decide_portfolio(
        _snapshot(pack, 7, quantity=2, weight=0.05)
    ).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 8)).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 9)).target_weights == (0.25,)


def test_terminated_account_keeps_native_hold_and_clock_mismatch_fails():
    pack = _signals([1] * 30, bear_4h=(12,))
    p = policy(pack, context(pack, adx=0.1), "all_exits")
    snap = _snapshot(pack, 12, quantity=10, weight=0.3)
    dead = replace(snap, account=replace(snap.account, insolvent=True))
    assert p.decide_portfolio(dead).hold_current_quantity_symbols == (0,)
    with pytest.raises(ValueError, match="timestamp"):
        p.decide_portfolio(replace(_snapshot(pack, 13), timestamp_ns=0))
    with pytest.raises(ValueError, match="clock"):
        policy(
            pack,
            replace(context(pack), timestamps=pack.timestamps + np.timedelta64(1, "h")),
            "all_exits",
        )


@pytest.mark.parametrize(
    "mode", ["all_exits", "weak_trend_exits", "weak_trend_atr_entry"]
)
def test_synthetic_native_replay(mode):
    market = _native_market()
    pack = _signals([1] * 25 + [-1] * 30, bear_4h=(12,))
    p = policy(pack, context(pack, adx=0.4, atr=0.02), mode)
    result = run_shared_portfolio_replay(
        market,
        p,
        start_index=0,
        stop_index=54,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert result.book.fill_count > 0
    assert result.book.total_cost > 0
    assert result.ledger_evidence is not None
    assert np.isfinite(result.book.portfolio_value)
    assert result.book.termination_reason is None


def test_invalid_mode_and_bad_context_rejected():
    pack = _signals([1] * 30)
    with pytest.raises(ValueError):
        CloudTrendConfig(mode="unknown")
    with pytest.raises(ValueError):
        context(pack, adx=25)
    with pytest.raises(ValueError):
        context(pack, atr=0)


def test_native_partial_exit_keeps_flat_target_until_residual_is_gone() -> None:
    market = _native_market()
    volume = market.volume.copy()
    volume[13, 0] = 1.0
    market = replace(market, volume=volume)
    pack = _signals([1] * 55, bear_4h=(12,))
    result = run_shared_portfolio_replay(
        market,
        policy(pack, context(pack, adx=0.2), "weak_trend_exits"),
        start_index=0,
        stop_index=54,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(max_participation_rate=0.05),
        capture_ledger_evidence=True,
        settle_terminal_position=True,
    )
    assert result.ledger_evidence is not None
    intervals = {x.start_index: x for x in result.ledger_evidence.intervals.intervals}
    before = Fraction(intervals[12].exact_quantities_before[0])
    residual = Fraction(intervals[12].exact_quantities_after[0])
    assert 0 < residual < before
    assert Fraction(intervals[13].exact_quantities_after[0]) == 0
    decisions = {x.index: x for x in result.decisions}
    assert decisions[13].proposal_weights == (0.0,)
    assert not decisions[13].requested_hold_symbols
    assert decisions[14].proposal_weights == (0.0,)
    assert decisions[37].proposal_weights == (0.0,)
    assert decisions[38].proposal_weights == (0.25,)


def test_strong_adx_does_not_override_cloud_failure_and_input_mutation() -> None:
    pack = _signals([1] * 15 + [-1] * 20)
    p = policy(pack, context(pack, adx=0.9), "weak_trend_exits")
    pack.cloud_state[15] = 1
    assert p.decide_portfolio(
        _snapshot(pack, 15, quantity=10, weight=0.25)
    ).target_weights == (0.0,)


@pytest.mark.parametrize("price_move", [10.0, -10.0])
def test_strong_trend_continuation_and_reversal_are_symmetric_counterexamples(
    price_move: float,
) -> None:
    """Literal synthetic prices isolate both the benefit and the risk of the gate."""
    from trade_rl.data.market import MarketDataset

    n = 55
    c = np.full(n, 100.0)
    c[14:] += price_move
    op = np.r_[c[0], c[:-1]]
    shape = (n, 1)
    pack = _signals([1] * n, bear_4h=(12,))
    data = MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=pack.timestamps,
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros(shape, dtype=np.float32),
        open=op[:, None],
        high=(np.maximum(op, c) + 0.5)[:, None],
        low=(np.minimum(op, c) - 0.5)[:, None],
        close=c[:, None],
        volume=np.full(shape, 100000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((n, 1, 1), dtype=bool),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    expected = {
        "all_exits": 0.0,
        "weak_trend_exits": 25 * price_move,
        "weak_trend_atr_entry": 12.5 * price_move,
    }
    for mode, expected_delta in expected.items():
        out = run_shared_portfolio_replay(
            data,
            policy(pack, context(pack, adx=0.4, atr=0.02), mode),
            start_index=0,
            stop_index=n - 1,
            initial_capital=10000.0,
            execution_cost=ExecutionCostConfig.zero(),
            settle_terminal_position=True,
        )
        assert out.book.portfolio_value == pytest.approx(10000.0 + expected_delta)


def test_actual_synthetic_indicator_pipeline_feeds_native_policy() -> None:
    from tests.data.test_divergence_cloud_mtf import _bars
    from trade_rl.data.features.cloud_trend_context import build_cloud_trend_context
    from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
    from trade_rl.data.market import MarketDataset

    raw = _bars(24 * 90)
    n = len(raw.close)
    # The existing smooth fixture never closes beyond its previous highs.
    # A literal late ramp ensures this is a genuine breakout-path oracle.
    ramp = np.maximum(np.arange(n, dtype=float) - 24 * 80, 0.0)
    raw = replace(
        raw,
        open=raw.open + ramp,
        high=raw.high + ramp,
        low=raw.low + ramp,
        close=raw.close + ramp,
    )
    pack = build_divergence_cloud_mtf(raw)
    ctx = build_cloud_trend_context(raw)
    shape = (n, 1)
    data = MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=raw.timestamps,
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros(shape, dtype=np.float32),
        open=raw.open[:, None],
        high=raw.high[:, None],
        low=raw.low[:, None],
        close=raw.close[:, None],
        volume=raw.volume[:, None],
        funding_rate=raw.funding_rate[:, None],
        tradable=raw.tradable[:, None],
        feature_available=np.ones((n, 1, 1), dtype=bool),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    from trade_rl.strategies.rules.trend_stop import (
        TrendStopConfig,
        TrendStopPortfolioPolicy,
    )

    # Existing Donchian/ATR policy is a separate synthetic mechanism control;
    # no historical trend-stop study is opened or rerun.
    comparator = TrendStopPortfolioPolicy(
        data, TrendStopConfig(entry_bars=20, exit_bars=10)
    )
    reference = run_shared_portfolio_replay(
        data,
        comparator,
        start_index=0,
        stop_index=n - 1,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert reference.book.fill_count >= 2
    assert reference.book.total_cost > 0
    assert reference.ledger_evidence is not None
    assert (
        Fraction(
            reference.ledger_evidence.intervals.intervals[-1].exact_quantities_after[0]
        )
        == 0
    )

    p = policy(pack, ctx, "weak_trend_atr_entry")
    out = run_shared_portfolio_replay(
        data,
        p,
        start_index=0,
        stop_index=n - 1,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert ctx.ready.any()
    assert pack.cloud_ready.any()
    assert out.book.fill_count >= 2
    assert out.book.total_cost > 0
    assert out.ledger_evidence is not None
    assert (
        Fraction(out.ledger_evidence.intervals.intervals[-1].exact_quantities_after[0])
        == 0
    )
