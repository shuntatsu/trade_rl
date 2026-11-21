"""Synthetic tests of ``AgyAnchorPolicy`` (lag, stride, deadband, HOLD, risk scale)."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.agy_anchor.policy import AgyAnchorPolicy, risk_scale_of
from trade_rl.strategies.agy_anchor.signals import AnchorSignals
from trade_rl.strategies.agy_anchor.targets import interleave_legs, synthesize_row
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)

PAIRS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
LEGS = tuple(f"{pair}:{kind}" for pair in PAIRS for kind in ("spot", "perp"))
ROWS = 120


def _signals(
    *,
    rows: int = ROWS,
    vary: bool = True,
    spot_valid: np.ndarray | None = None,
) -> AnchorSignals:
    base = np.asarray([0.25, 0.25, -0.25, -0.25])
    w = np.stack(
        [np.roll(base, (row // 3) % 4) if vary else base for row in range(rows)]
    )
    ones = np.ones((rows, 4), dtype=bool)
    return AnchorSignals(
        symbols=PAIRS,
        close_ns=np.arange(1, rows + 1, dtype=np.int64) * 14_400_000_000_000,
        w_aw_bidir_75_25=w,
        c_weekly_iv=np.tile([0.25, 0.25, 0.25, 0.25], (rows, 1)),
        vol_cushion_tail=np.full(rows, 0.9),
        w_aw_regime=w,
        w_perp_bidir_carry=w,
        carry_mask=ones,
        perp_mask=ones,
        spot_valid=ones if spot_valid is None else spot_valid,
        perp_valid=ones,
    )


def _snapshot(
    index: int,
    *,
    weights: tuple[float, ...] = (0.0,) * 8,
    value: float = 1.0,
    peak: float = 1.0,
    max_dd: float = 0.0,
    stop: int = ROWS,
) -> PortfolioDecisionSnapshot:
    n = len(LEGS)
    account = PortfolioAccountSnapshot(
        cash=value,
        portfolio_value=value,
        peak_value=peak,
        max_drawdown=max_dd,
        exact_quantities=tuple("0" for _ in range(n)),
        quantities=tuple(w * value for w in weights),
        weights=weights,
        mark_prices=(1.0,) * n,
        contract_multipliers=(1.0,) * n,
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        margin_used=0.0,
        maintenance_margin=0.0,
        maintenance_requirement=0.0,
        margin_deficit=0.0,
        insolvent=False,
        termination_reason=None,
    )
    return PortfolioDecisionSnapshot(
        index=index,
        timestamp_ns=index,
        agent_stop_index=stop,
        remaining_decisions=stop - index,
        symbols=LEGS,
        feature_names=(),
        global_feature_names=(),
        features=((),) * n,
        feature_available=((),) * n,
        feature_staleness=((),) * n,
        global_features=(),
        global_feature_available=(),
        account=account,
        active_orders=(),
        risk=PortfolioRiskLimits(
            max_gross=2.5,
            max_abs_weight=0.55,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
            emergency_turnover_override=True,
            fail_closed_tolerance=1e-10,
        ),
    )


def _target(signals: AnchorSignals, row: int) -> np.ndarray:
    return interleave_legs(*synthesize_row(signals, row))


def test_the_first_decision_proposes_the_row_lagged_by_one_bar() -> None:
    signals = _signals()
    policy = AgyAnchorPolicy(signals=signals, start_index=12)
    proposal = policy.decide_portfolio(_snapshot(12))

    np.testing.assert_array_equal(proposal.target_weights, _target(signals, 11))
    assert not np.array_equal(_target(signals, 11), _target(signals, 12))
    assert proposal.hold_current_quantity_symbols == ()

    unlagged = AgyAnchorPolicy(signals=signals, start_index=12, signal_lag_bars=0)
    np.testing.assert_array_equal(
        unlagged.decide_portfolio(_snapshot(12)).target_weights, _target(signals, 12)
    )


def test_between_rebalance_bars_every_leg_is_held_exactly() -> None:
    signals = _signals()
    policy = AgyAnchorPolicy(signals=signals, start_index=12)
    first = policy.decide_portfolio(_snapshot(12))
    current = first.target_weights
    for index in range(13, 18):  # five non-rebalance bars (stride 6)
        held = policy.decide_portfolio(_snapshot(index, weights=current))
        assert held.hold_current_quantity_symbols == tuple(range(8))
        np.testing.assert_array_equal(held.target_weights, current)
    nxt = policy.decide_portfolio(_snapshot(18, weights=current))
    assert nxt.hold_current_quantity_symbols == ()
    np.testing.assert_array_equal(nxt.target_weights, _target(signals, 17))


def test_turnover_below_the_deadband_holds_the_whole_book() -> None:
    signals = _signals(vary=False)
    policy = AgyAnchorPolicy(signals=signals, start_index=0)
    target = tuple(float(w) for w in _target(signals, 0))
    policy.decide_portfolio(_snapshot(0))
    for index in range(1, 6):
        policy.decide_portfolio(_snapshot(index, weights=target))
    drifted = tuple(w * 1.001 for w in target)  # turnover far below 0.04
    held = policy.decide_portfolio(_snapshot(6, weights=drifted))
    assert held.hold_current_quantity_symbols == tuple(range(8))

    for index in range(7, 12):
        policy.decide_portfolio(_snapshot(index, weights=drifted))
    away = tuple(w * 0.5 for w in target)  # turnover well above 0.04
    moved = policy.decide_portfolio(_snapshot(12, weights=away))
    assert moved.hold_current_quantity_symbols == ()
    np.testing.assert_array_equal(moved.target_weights, target)


def test_a_zero_deadband_always_trades_and_a_huge_one_never_does() -> None:
    signals = _signals(vary=False)
    target = tuple(float(w) for w in _target(signals, 0))
    nearly = tuple(w * 1.0001 for w in target)
    trade = AgyAnchorPolicy(signals=signals, start_index=0, deadband=0.0)
    trade.decide_portfolio(_snapshot(0))
    assert (
        trade.decide_portfolio(
            _snapshot(6, weights=nearly)
        ).hold_current_quantity_symbols
        == ()
    )
    never = AgyAnchorPolicy(signals=signals, start_index=0, deadband=10.0)
    assert never.decide_portfolio(_snapshot(0)).hold_current_quantity_symbols == tuple(
        range(8)
    )


def test_legs_that_are_not_valid_on_the_current_row_keep_their_quantity() -> None:
    spot_valid = np.ones((ROWS, 4), dtype=bool)
    spot_valid[12, 1] = False  # pair 1 spot (leg 2) halted on the decision row
    signals = _signals(spot_valid=spot_valid)
    policy = AgyAnchorPolicy(signals=signals, start_index=12)
    current = (0.0, 0.0, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0)
    proposal = policy.decide_portfolio(_snapshot(12, weights=current))
    assert proposal.hold_current_quantity_symbols == (2,)
    assert proposal.target_weights[2] == current[2]
    expected = _target(signals, 11)
    others = [i for i in range(8) if i != 2]
    np.testing.assert_array_equal(
        np.asarray(proposal.target_weights)[others], expected[others]
    )


def test_risk_scale_ramp_matches_the_pre_trade_risk_definition() -> None:
    assert risk_scale_of(_snapshot(0)) == 1.0
    assert risk_scale_of(_snapshot(0, value=0.85, peak=1.0)) == pytest.approx(0.5)
    assert risk_scale_of(_snapshot(0, value=0.7, peak=1.0)) == 0.0
    # the running maximum drawdown never rises back
    assert risk_scale_of(_snapshot(0, value=1.0, peak=1.0, max_dd=0.15)) == (
        pytest.approx(0.5)
    )


def test_a_falling_risk_scale_retargets_once_and_a_flat_one_uses_the_pre_image() -> (
    None
):
    signals = _signals(vary=False)
    policy = AgyAnchorPolicy(signals=signals, start_index=0)
    target = _target(signals, 0)
    policy.decide_portfolio(_snapshot(0))
    full = tuple(float(w) for w in target)
    policy.decide_portfolio(_snapshot(1, weights=full))

    # drawdown of 15% appears: scale 0.5 < previous 1.0 -> re-propose the intended book
    scale = 0.5
    scaled = tuple(w * scale for w in full)
    stage = _snapshot(2, weights=scaled, value=0.85, peak=1.0, max_dd=0.15)
    retarget = policy.decide_portfolio(stage)
    assert retarget.hold_current_quantity_symbols == ()
    np.testing.assert_array_equal(retarget.target_weights, target)

    # same scale next bar: HOLD would be scaled again, so propose current / scale
    same = _snapshot(3, weights=scaled, value=0.85, peak=1.0, max_dd=0.15)
    pre_image = policy.decide_portfolio(same)
    assert pre_image.hold_current_quantity_symbols == ()
    np.testing.assert_allclose(pre_image.target_weights, np.asarray(scaled) / scale)

    # scale hits zero (drawdown >= 20%): the intended book is proposed (risk zeroes it)
    flat = _snapshot(4, weights=scaled, value=0.7, peak=1.0, max_dd=0.30)
    zeroed = policy.decide_portfolio(flat)
    np.testing.assert_array_equal(zeroed.target_weights, target)


def test_snapshot_and_construction_validation() -> None:
    signals = _signals()
    policy = AgyAnchorPolicy(signals=signals, start_index=12)
    with pytest.raises(ValueError, match="precedes the segment start"):
        policy.decide_portfolio(_snapshot(5))
    policy.decide_portfolio(_snapshot(12))
    with pytest.raises(ValueError, match="advance monotonically"):
        policy.decide_portfolio(_snapshot(12))
    with pytest.raises(ValueError, match="first decision must be a rebalance bar"):
        AgyAnchorPolicy(signals=signals, start_index=0).decide_portfolio(_snapshot(3))
    with pytest.raises(ValueError, match="outside the signal rows"):
        AgyAnchorPolicy(signals=signals, start_index=0).decide_portfolio(
            _snapshot(ROWS + 5, stop=ROWS + 9)
        )
    with pytest.raises(TypeError):
        AgyAnchorPolicy(signals=object(), start_index=0)  # type: ignore[arg-type]
    for kwargs in (
        {"start_index": -1},
        {"start_index": 0, "signal_lag_bars": -1},
        {"start_index": 0, "rebalance_every_bars": 0},
        {"start_index": 0, "deadband": float("nan")},
        {"start_index": 0, "deadband": -0.1},
    ):
        with pytest.raises(ValueError):
            AgyAnchorPolicy(signals=signals, **kwargs)  # type: ignore[arg-type]
    other = AgyAnchorPolicy(signals=_signals(), start_index=0)
    mismatched = _snapshot(0)
    object.__setattr__(mismatched, "symbols", tuple(reversed(LEGS)))
    with pytest.raises(ValueError, match="do not match the alternating anchor legs"):
        other.decide_portfolio(mismatched)


# --- integration with the shared replay engine ----------------------------------------


def _dataset(rows: int) -> MarketDataset:
    n = len(LEGS)
    price = 100.0 + np.cumsum(
        np.random.default_rng(3).normal(0.0, 0.002, size=(rows, n)), axis=0
    )
    open_price = np.vstack((price[0], price[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=LEGS,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(rows) * np.timedelta64(4, "h"),
        features=np.zeros((rows, n, 1), dtype=np.float32),
        global_features=np.zeros((rows, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, price) * 1.001,
        low=np.minimum(open_price, price) * 0.999,
        close=price,
        volume=np.full((rows, n), 1e12),
        funding_rate=np.zeros((rows, n)),
        tradable=np.ones((rows, n), dtype=np.bool_),
        feature_available=np.ones((rows, n, 1), dtype=np.bool_),
        feature_names=("f",),
        global_feature_names=("g",),
        periods_per_year=2190,
        borrow_available=np.broadcast_to(
            np.tile(np.asarray([False, True]), len(PAIRS)), (rows, n)
        ),
    )


def test_the_shared_replay_runs_the_policy_with_daily_holds() -> None:
    signals = _signals()
    dataset = _dataset(ROWS + 1)
    policy = AgyAnchorPolicy(signals=signals, start_index=12)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=2.5,
            max_abs_weight=0.55,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=12,
        stop_index=ROWS,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig(max_leverage=3.0),
        risk=risk,
    )
    decisions = result.decisions
    assert [d.index for d in decisions] == list(range(12, ROWS))
    for decision in decisions:
        if (decision.index - 12) % 6:
            assert decision.requested_hold_symbols == tuple(range(8))
    rebalances = [d for d in decisions if (d.index - 12) % 6 == 0]
    first = rebalances[0]
    np.testing.assert_array_equal(first.proposal_weights, _target(signals, 11))
    assert result.book.fill_count > 0
    assert np.isfinite(np.asarray(result.book.quantities)).all()
