"""Independent decision-table and native transition oracles; no market data."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.test_cloud_trend_policy import _snapshot
from tests.strategies.test_divergence_cloud_follow import _native_market, _signals
from trade_rl.data.features.cloud_regime_switch_context import (
    CloudRegimeSwitchContext,
    build_cloud_regime_switch_context,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.cloud_regime_switch import CloudRegimeSwitchPolicy


def inputs(
    pack,
    *,
    adx=0.3,
    rsi=-0.6,
    bb=-1.2,
    breakout=1,
    exit_breakout=0,
    bull=1,
    bear=0,
    ready=True,
    atr=0.02,
):
    n = len(pack.timestamps)

    def arr(x, dtype=float):
        return np.broadcast_to(np.asarray(x, dtype=dtype), (n,)).copy()

    valid = arr(ready, bool)
    return CloudRegimeSwitchContext(
        symbol="BTCUSDT",
        timestamps=pack.timestamps,
        ready=valid,
        adx_fraction=np.where(valid, arr(adx), np.nan),
        atr_fraction=np.where(valid, arr(atr), np.nan),
        rsi_centered=np.where(valid, arr(rsi), np.nan),
        bollinger_position=np.where(valid, arr(bb), np.nan),
        cloud_state=arr(pack.cloud_state, np.int8),
        entry_breakout=arr(breakout, np.int8),
        exit_breakout=arr(exit_breakout, np.int8),
        bull_1h=arr(bull, np.int16),
        bear_1h=arr(bear, np.int16),
    )


def make(pack, mode="hybrid_divergence", **kwargs):
    return CloudRegimeSwitchPolicy(context=inputs(pack, **kwargs), mode=mode)


@pytest.mark.parametrize("side", [1, -1])
def test_trend_entry_requires_adx_cloud_and_channel(side):
    pack = _signals([side] * 40)
    for mode in ("trend_only", "hybrid", "hybrid_divergence"):
        assert make(pack, mode, breakout=side).decide_portfolio(
            _snapshot(pack, 5)
        ).target_weights == (side * 0.125,)
        assert make(pack, mode, adx=0.249, breakout=side).decide_portfolio(
            _snapshot(pack, 5)
        ).target_weights == (0.0,)
        assert make(pack, mode, breakout=-side).decide_portfolio(
            _snapshot(pack, 5)
        ).target_weights == (0.0,)
        assert make(pack, mode, breakout=side).decide_portfolio(
            _snapshot(pack, 4)
        ).target_weights == (0.0,)


@pytest.mark.parametrize("side", [1, -1])
def test_range_entry_divergence_ablation(side):
    pack = _signals([0] * 40)
    kw = dict(
        adx=0.2,
        rsi=-0.4 * side,
        bb=-1.0 * side,
        breakout=0,
        bull=int(side == 1),
        bear=int(side == -1),
    )
    assert make(pack, "trend_only", **kw).decide_portfolio(
        _snapshot(pack, 5)
    ).target_weights == (0.0,)
    for mode in ("hybrid", "hybrid_divergence"):
        assert make(pack, mode, **kw).decide_portfolio(
            _snapshot(pack, 5)
        ).target_weights == (side * 0.125,)
    assert make(
        pack, "hybrid_divergence", **(kw | {"bull": 0, "bear": 0})
    ).decide_portfolio(_snapshot(pack, 5)).target_weights == (0.0,)
    assert make(pack, "hybrid", **(kw | {"bull": 0, "bear": 0})).decide_portfolio(
        _snapshot(pack, 5)
    ).target_weights == (side * 0.125,)
    assert make(
        pack, "hybrid_divergence", **(kw | {"bull": 1, "bear": 1})
    ).decide_portfolio(_snapshot(pack, 5)).target_weights == (0.0,)
    assert make(pack, "hybrid", **(kw | {"adx": 0.201})).decide_portfolio(
        _snapshot(pack, 5)
    ).target_weights == (0.0,)


def test_hysteresis_and_partial_exit_do_not_change_position_owner():
    pack = _signals([1] * 10 + [0] * 30)
    a = np.r_[np.full(6, 0.3), np.full(4, 0.23), np.full(30, 0.18)]
    p = make(pack, adx=a)
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.125,)
    for i in range(6, 10):
        assert p.decide_portfolio(
            _snapshot(pack, i, quantity=10, weight=0.15)
        ).hold_current_quantity_symbols == (0,)
    for i in (10, 11):
        assert p.decide_portfolio(
            _snapshot(pack, i, quantity=2, weight=0.03)
        ).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 12)).hold_current_quantity_symbols == (0,)
    for i in range(13, 16):
        assert p.decide_portfolio(_snapshot(pack, i)).target_weights == (0.0,)
    assert p.decide_portfolio(_snapshot(pack, 16)).target_weights == (0.125,)


def test_range_exits_at_center_or_maximum_age():
    pack = _signals([0] * 40)
    bb = np.full(40, -1.2)
    bb[9] = 0.0
    p = make(pack, adx=0.2, bb=bb)
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.125,)
    for i in range(6, 9):
        assert p.decide_portfolio(
            _snapshot(pack, i, quantity=10, weight=0.15)
        ).hold_current_quantity_symbols == (0,)
    assert p.decide_portfolio(
        _snapshot(pack, 9, quantity=10, weight=0.15)
    ).target_weights == (0.0,)
    p = make(pack, adx=0.2)
    p.decide_portfolio(_snapshot(pack, 5))
    for i in range(6, 30):
        assert p.decide_portfolio(
            _snapshot(pack, i, quantity=10, weight=0.15)
        ).hold_current_quantity_symbols == (0,)
    assert p.decide_portfolio(
        _snapshot(pack, 30, quantity=10, weight=0.15)
    ).target_weights == (0.0,)


def test_unready_unknown_position_termination_and_invalid_clock():
    pack = _signals([1] * 40)
    p = make(pack)
    s = _snapshot(pack, 5, quantity=10, weight=0.25)
    assert p.decide_portfolio(s).target_weights == (0.0,)
    p = make(pack, ready=False)
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.0,)
    dead = replace(
        _snapshot(pack, 6, quantity=10, weight=0.25),
        account=replace(s.account, insolvent=True),
    )
    assert p.decide_portfolio(dead).hold_current_quantity_symbols == (0,)
    with pytest.raises(ValueError, match="clock"):
        p.decide_portfolio(replace(_snapshot(pack, 7), timestamp_ns=0))
    with pytest.raises(ValueError, match="mode"):
        make(pack, "bad")


def test_native_partial_exit_and_cooldown():
    market = _native_market()
    volume = market.volume.copy()
    volume[11, 0] = 1.0
    market = replace(market, volume=volume)
    pack = _signals([1] * 10 + [0] * 45)
    adx = np.r_[np.full(10, 0.3), np.full(45, 0.2)]
    result = run_shared_portfolio_replay(
        market,
        make(pack, "hybrid", adx=adx),
        start_index=0,
        stop_index=54,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(max_participation_rate=0.05),
        capture_ledger_evidence=True,
        settle_terminal_position=True,
    )
    assert result.ledger_evidence is not None
    ints = {x.start_index: x for x in result.ledger_evidence.intervals.intervals}
    assert (
        0
        < Fraction(ints[10].exact_quantities_after[0])
        < Fraction(ints[10].exact_quantities_before[0])
    )
    assert Fraction(ints[11].exact_quantities_after[0]) == 0
    dec = {x.index: x for x in result.decisions}
    assert dec[11].proposal_weights == (0.0,)
    assert all(dec[i].proposal_weights == (0.0,) for i in range(12, 16))
    assert dec[16].proposal_weights == (0.125,)
    assert result.book.total_cost > 0
    assert (
        Fraction(
            result.ledger_evidence.intervals.intervals[-1].exact_quantities_after[0]
        )
        == 0
    )


def test_full_source_adapter_and_prefix():
    # Real indicator calculations and existing channel logic, generated prices.
    from tests.data.test_divergence_cloud_mtf import _bars
    from trade_rl.data.market import MarketDataset

    raw = _bars(24 * 85)
    n = len(raw.close)
    ramp = np.maximum(np.arange(n, dtype=float) - 24 * 80, 0.0)
    ds = MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=raw.timestamps,
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=(raw.open + ramp)[:, None],
        high=(raw.high + ramp)[:, None],
        low=(raw.low + ramp)[:, None],
        close=(raw.close + ramp)[:, None],
        volume=raw.volume[:, None],
        funding_rate=raw.funding_rate[:, None],
        tradable=raw.tradable[:, None],
        feature_available=np.ones((n, 1, 1), dtype=bool),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    ctx = build_cloud_regime_switch_context(ds)
    assert ctx.ready.any()
    assert np.any(ctx.ready & (ctx.entry_breakout == 1))
    for i in np.flatnonzero(ctx.ready & (ctx.entry_breakout == 1)):
        assert ds.close[i, 0] > np.max(ds.high[i - 20 : i, 0])
    # Independent Wilder RSI recurrence on this uninterrupted generated path.
    price = ds.close[:, 0]
    diffs = np.diff(price)
    gain = np.maximum(diffs, 0)
    loss = np.maximum(-diffs, 0)
    avg_gain = float(np.mean(gain[:14]))
    avg_loss = float(np.mean(loss[:14]))
    for i in range(14, n):
        if i > 14:
            avg_gain = (13 * avg_gain + gain[i - 1]) / 14
            avg_loss = (13 * avg_loss + loss[i - 1]) / 14
        if ctx.ready[i]:
            expected = (
                (avg_gain - avg_loss) / (avg_gain + avg_loss)
                if avg_gain + avg_loss > 1e-12
                else 0.0
            )
            assert ctx.rsi_centered[i] == pytest.approx(expected, abs=1e-12)
    # Independent current-window Bollinger arithmetic.
    for i in np.flatnonzero(ctx.ready)[::13]:
        sample = ds.close[i - 19 : i + 1, 0]
        expected = (sample[-1] - sample.mean()) / (2 * sample.std())
        assert ctx.bollinger_position[i] == pytest.approx(expected, abs=1e-12)
    # Changing only the future OHLC must not rewrite any earlier context.
    cutoff = 24 * 82 + 1
    changed = ds.close.copy()
    changed[cutoff:] *= 1.7
    mutated = replace(ds, close=changed, high=np.maximum(ds.high, changed))
    later = build_cloud_regime_switch_context(mutated)
    for name in (
        "ready",
        "adx_fraction",
        "atr_fraction",
        "rsi_centered",
        "bollinger_position",
        "cloud_state",
        "entry_breakout",
        "exit_breakout",
        "bull_1h",
        "bear_1h",
    ):
        np.testing.assert_array_equal(
            getattr(ctx, name)[:cutoff], getattr(later, name)[:cutoff]
        )
    avail = ds.resolved_array("available_at").copy()
    avail[cutoff, 0] += np.timedelta64(2, "h")
    observed = ds.resolved_array("information_available").copy()
    observed[cutoff, 0] = False
    delayed = build_cloud_regime_switch_context(
        replace(ds, available_at=avail, information_available=observed)
    )
    assert not delayed.ready[cutoff : cutoff + 2].any()
    np.testing.assert_array_equal(ctx.ready[:cutoff], delayed.ready[:cutoff])
    # Full synthetic native replay from the very same source.
    for mode in ("trend_only", "hybrid", "hybrid_divergence"):
        result = run_shared_portfolio_replay(
            ds,
            CloudRegimeSwitchPolicy(context=ctx, mode=mode),
            start_index=0,
            stop_index=n - 1,
            initial_capital=10000.0,
            execution_cost=ExecutionCostConfig(),
            settle_terminal_position=True,
        )
        assert result.book.fill_count >= 2 and result.book.total_cost > 0
        assert result.book.termination_reason is None


@pytest.mark.parametrize("move", [10.0, -10.0])
def test_range_reversion_and_breakout_native_arithmetic(move):
    # Literal supplied state isolates execution and switching mechanics.
    # It is not a claim that the indicators would predict this price move.
    market = _native_market()
    n = market.n_bars
    close = np.full((n, 1), 100.0)
    close[7:, 0] += move
    op = np.vstack((close[0], close[:-1]))
    market = replace(
        market,
        close=close,
        open=op,
        high=np.maximum(close, op) + 0.5,
        low=np.minimum(close, op) - 0.5,
        mark_price=close,
        index_price=close,
    )
    cloud = np.zeros(n, dtype=np.int8)
    if move < 0:
        cloud[8:] = -1
    pack = _signals(cloud.tolist())
    bb = np.full(n, -0.5)
    bb[5] = -1.2
    bb[8:] = 0.0 if move > 0 else -1.2
    rsi = np.zeros(n)
    rsi[5] = -0.6
    signal = np.zeros(n, dtype=np.int16)
    signal[5] = 1
    for mode, bull, delta in (
        ("trend_only", signal, 0.0),
        ("hybrid", np.zeros(n, dtype=np.int16), 12.5 * move),
        ("hybrid_divergence", np.zeros(n, dtype=np.int16), 0.0),
        ("hybrid_divergence", signal, 12.5 * move),
    ):
        p = make(pack, mode, adx=0.2, bb=bb, rsi=rsi, bull=bull, breakout=0)
        out = run_shared_portfolio_replay(
            market,
            p,
            start_index=0,
            stop_index=n - 1,
            initial_capital=10000.0,
            execution_cost=ExecutionCostConfig.zero(),
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        assert out.book.portfolio_value == pytest.approx(10000.0 + delta)
        assert out.ledger_evidence is not None
        assert (
            Fraction(
                out.ledger_evidence.intervals.intervals[-1].exact_quantities_after[0]
            )
            == 0
        )


def test_context_is_immutable_and_invalid_values_refused():
    pack = _signals([1] * 40)
    ctx = inputs(pack)
    with pytest.raises(ValueError):
        ctx.cloud_state.setflags(write=True)
    with pytest.raises(ValueError, match="ADX"):
        inputs(pack, adx=25)
    with pytest.raises(ValueError, match="ATR"):
        inputs(pack, atr=0)
    with pytest.raises(ValueError, match="RSI"):
        inputs(pack, rsi=70)
    with pytest.raises(ValueError, match="directional"):
        inputs(pack, breakout=2)
    with pytest.raises(ValueError, match="nonnegative"):
        inputs(pack, bull=-1)


def test_held_position_exits_on_opposite_channel_and_missing_context():
    pack = _signals([1] * 40)
    ex = np.zeros(40, dtype=np.int8)
    ex[7] = -1
    p = make(pack, exit_breakout=ex)
    p.decide_portfolio(_snapshot(pack, 5))
    assert p.decide_portfolio(
        _snapshot(pack, 6, quantity=10, weight=0.125)
    ).hold_current_quantity_symbols == (0,)
    assert p.decide_portfolio(
        _snapshot(pack, 7, quantity=10, weight=0.125)
    ).target_weights == (0.0,)
    ready = np.ones(40, dtype=bool)
    ready[7] = False
    p = make(pack, ready=ready)
    p.decide_portfolio(_snapshot(pack, 5))
    p.decide_portfolio(_snapshot(pack, 6, quantity=10, weight=0.125))
    assert p.decide_portfolio(
        _snapshot(pack, 7, quantity=10, weight=0.125)
    ).target_weights == (0.0,)


def test_entry_risk_cap_and_range_hysteresis():
    pack = _signals([0] * 40)
    a = np.full(40, 0.20)
    a[6:8] = 0.23
    a[8:] = 0.25
    p = make(pack, adx=a)
    snap = _snapshot(pack, 5)
    capped = replace(snap, risk=replace(snap.risk, max_abs_weight=0.03))
    assert p.decide_portfolio(capped).target_weights == (0.03,)
    for i in (6, 7):
        assert p.decide_portfolio(
            _snapshot(pack, i, quantity=3, weight=0.03)
        ).hold_current_quantity_symbols == (0,)
    assert p.decide_portfolio(
        _snapshot(pack, 8, quantity=3, weight=0.03)
    ).target_weights == (0.0,)


def test_unfilled_entry_is_not_a_filled_position():
    pack = _signals([1] * 40)
    p = make(pack, adx=0.25)
    assert p.decide_portfolio(_snapshot(pack, 5)).target_weights == (0.125,)
    # No fill occurred. The next decision cancels the pending target
    # rather than assuming ownership of a nonexistent position.
    assert p.decide_portfolio(_snapshot(pack, 6)).hold_current_quantity_symbols == (0,)
    assert p.decide_portfolio(_snapshot(pack, 7)).target_weights == (0.125,)
