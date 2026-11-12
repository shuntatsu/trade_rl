"""Trend-with-stops policy: causal inputs, stop semantics and an independent oracle.

The replay tests drive the real shared-portfolio executor on tiny synthetic bars with
zero costs, then compare the ledger with hand arithmetic done in ``Fraction``. They
use no market data of any kind.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.rules.trend_stop import (
    TrendStopConfig,
    TrendStopPortfolioPolicy,
    donchian_extrema,
    wilder_atr,
)

Bar = tuple[float, float, float, float]

# Entry breakout at row 4 (close 103 above the prior three highs, all 101), ATR(2) = 3.
FLAT_BARS: list[Bar] = [(100.0, 101.0, 99.0, 100.0)] * 4
BREAKOUT: Bar = (100.0, 104.0, 100.0, 103.0)
RUN_UP = [(103.0, 108.0, 102.0, 107.0), (107.0, 110.0, 106.0, 109.0)]
GAP_DOWN = [
    (109.0, 109.0, 95.0, 101.0),
    (94.0, 98.0, 94.0, 97.0),
    (97.0, 98.0, 96.0, 97.0),
]


def _config(**overrides: float | int | bool) -> TrendStopConfig:
    values: dict[str, float | int | bool] = {
        "entry_bars": 3,
        "exit_bars": 2,
        "atr_period": 2,
        "initial_stop_atr": 2.0,
        "trailing_stop_atr": 3.0,
        "risk_per_trade": 0.005,
        "max_symbol_weight": 0.25,
        "gross_cap": 0.95,
    }
    values.update(overrides)
    return TrendStopConfig(**values)  # type: ignore[arg-type]


def _dataset(
    bars: dict[str, list[Bar]],
    *,
    usable: dict[str, list[bool]] | None = None,
    step_hours: int = 1,
) -> MarketDataset:
    symbols = tuple(bars)
    n_rows = len(next(iter(bars.values())))
    shape = (n_rows, len(symbols))
    columns = [np.asarray(bars[name], dtype=np.float64) for name in symbols]
    open_, high, low, close = (
        np.stack([column[:, k] for column in columns], axis=1) for k in range(4)
    )
    tradable = np.ones(shape, dtype=np.bool_)
    if usable is not None:
        tradable = np.stack(
            [np.asarray(usable[name], dtype=np.bool_) for name in symbols], axis=1
        )
    features = np.zeros((n_rows, len(symbols), 1), dtype=np.float64)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_rows) * np.timedelta64(step_hours, "h"),
        features=features,
        global_features=np.zeros((n_rows, 1), dtype=np.float64),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1.0e12, dtype=np.float64),
        funding_rate=np.zeros(shape, dtype=np.float64),
        tradable=tradable,
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=("unused",),
        global_feature_names=("unused_global",),
        periods_per_year=8_760 // step_hours,
    )


def _replay(
    dataset: MarketDataset,
    config: TrendStopConfig | None = None,
    *,
    stop_index: int | None = None,
) -> tuple[SharedCashPortfolioReplayResult, TrendStopPortfolioPolicy]:
    policy = TrendStopPortfolioPolicy(dataset, config or _config())
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=dataset.n_bars - 1 if stop_index is None else stop_index,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
    )
    return result, policy


def _fills(result: SharedCashPortfolioReplayResult) -> list[tuple[int, int, float]]:
    """(interval start row, symbol, signed quantity) of every nonzero fill."""

    assert result.ledger_evidence is not None
    out: list[tuple[int, int, float]] = []
    for interval in result.ledger_evidence.intervals.intervals:
        for event in interval.order_events:
            if event.filled_quantity != 0.0:
                out.append(
                    (
                        interval.start_index,
                        event.symbol_index,
                        float(event.filled_quantity),
                    )
                )
    return out


def _assert_fills(
    result: SharedCashPortfolioReplayResult,
    expected: list[tuple[int, int, Fraction | float]],
) -> None:
    actual = _fills(result)
    assert [(start, symbol) for start, symbol, _ in actual] == [
        (start, symbol) for start, symbol, _ in expected
    ]
    assert [quantity for _, _, quantity in actual] == pytest.approx(
        [float(quantity) for _, _, quantity in expected], rel=1e-12
    )


def _equity_by_next_row(result: SharedCashPortfolioReplayResult) -> dict[int, float]:
    assert result.ledger_evidence is not None
    return {
        interval.next_index: interval.portfolio_value_after
        for interval in result.ledger_evidence.intervals.intervals
    }


# -- indicators ---------------------------------------------------------------------


def test_donchian_extrema_exclude_the_current_row_and_need_a_usable_window() -> None:
    high = np.array([[1.0], [2.0], [3.0], [9.0], [5.0]])
    low = np.array([[0.5], [1.5], [2.5], [8.0], [4.0]])
    usable = np.ones((5, 1), dtype=np.bool_)

    upper, lower, valid = donchian_extrema(high, low, usable, 3)

    assert list(valid[:, 0]) == [False, False, False, True, True]
    assert upper[3, 0] == 3.0 and lower[3, 0] == 0.5
    assert upper[4, 0] == 9.0 and lower[4, 0] == 1.5

    changed = high.copy()
    changed[3, 0] = 1.0e6  # the current row must not influence its own channel
    assert donchian_extrema(changed, low, usable, 3)[0][3, 0] == upper[3, 0]

    gap = usable.copy()
    gap[1, 0] = False
    assert not donchian_extrema(high, low, gap, 3)[2][3, 0]


def test_wilder_atr_matches_hand_values_and_restarts_after_a_gap() -> None:
    bars = FLAT_BARS + [BREAKOUT]
    dataset = _dataset({"AAA": bars})
    high = dataset.high
    low = dataset.low
    close = dataset.close
    usable = np.ones((5, 1), dtype=np.bool_)

    atr, valid = wilder_atr(high, low, close, usable, 2)

    assert list(valid[:, 0]) == [False, False, True, True, True]
    # TR: 2, 2, 2, 4  ->  seed (2+2)/2 = 2 ; (2*1+2)/2 = 2 ; (2*1+4)/2 = 3
    assert atr[2, 0] == 2.0 and atr[3, 0] == 2.0 and atr[4, 0] == 3.0

    gap = usable.copy()
    gap[3, 0] = False
    _, valid_gap = wilder_atr(high, low, close, gap, 2)
    assert not valid_gap[3, 0] and not valid_gap[4, 0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"exit_bars": 3},
        {"entry_bars": 0},
        {"initial_stop_atr": 0.0},
        {"risk_per_trade": 0.0},
        {"risk_per_trade": 1.0},
        {"max_symbol_weight": 0.99, "gross_cap": 0.5},
        {"gross_cap": 1.5},
    ],
)
def test_config_rejects_inconsistent_parameters(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


# -- stop oracle: hand-computed replay ------------------------------------------------


def test_gap_through_the_stop_is_taken_in_full_and_matches_the_hand_ledger() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    result, policy = _replay(_dataset({"AAA": bars}), stop_index=9)

    # Hand arithmetic (no engine code): ATR(2) at the signal row 4 is 3, so the
    # initial stop distance is 2*3 = 6 and the 0.5% risk budget of 10000 buys
    # 50/6 units. The entry fills at the next open (103). The highest high is 110,
    # so the trailing level is 110 - 3*3 = 101 (above the initial 103 - 6 = 97).
    # Row 7 trades down to 95 (close 101): the stop triggers, and the exit fills at
    # the next open (94), seven points through the stop level.
    quantity = Fraction(50, 6)
    _assert_fills(result, [(4, 0, quantity), (7, 0, -quantity)])
    expected = {
        5: Fraction(10000) + quantity * (107 - 103),
        6: Fraction(10000) + quantity * (109 - 103),
        7: Fraction(10000) + quantity * (101 - 103),
        8: Fraction(10000) + quantity * (94 - 103),
    }
    equity = _equity_by_next_row(result)
    for row, value in expected.items():
        assert equity[row] == pytest.approx(float(value), abs=1e-6), row
    assert float(result.book.portfolio_value) == pytest.approx(9925.0, abs=1e-6)
    assert all(Fraction(value) == 0 for value in result.book.exact_quantities)

    (event,) = policy.stop_events
    assert (event.decision_index, event.symbol_index, event.side) == (7, 0, 1)
    assert event.stop_level == 101.0 and event.trailing
    # The realized loss (-75) exceeds the loss a resting stop at 101 would take (-16.67).
    assert quantity * (94 - 103) < quantity * (101 - 103)
    assert policy.summary()["stop_exits"] == 1


def test_a_wick_through_the_stop_that_recovers_still_exits_at_the_next_open() -> None:
    wick = [(109.0, 109.0, 100.0, 107.0), (108.0, 109.0, 106.0, 108.0)]
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + wick + [(108.0, 109.0, 106.0, 108.0)]
    result, policy = _replay(_dataset({"AAA": bars}), stop_index=9)

    quantity = Fraction(50, 6)
    _assert_fills(result, [(4, 0, quantity), (7, 0, -quantity)])
    (event,) = policy.stop_events
    assert event.stop_level == 101.0
    # Exit price 108 is *better* than the stop level 101: the engine books +41.67,
    # which is the optimism the runner's stop haircut removes.
    assert float(result.book.portfolio_value) == pytest.approx(
        float(Fraction(10000) + quantity * (108 - 103)), abs=1e-6
    )


def test_short_side_mirrors_the_long_side_exactly() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    mirrored = [(200.0 - o, 200.0 - low, 200.0 - h, 200.0 - c) for o, h, low, c in bars]
    long_result, _ = _replay(_dataset({"AAA": bars}), stop_index=9)
    short_result, short_policy = _replay(_dataset({"AAA": mirrored}), stop_index=9)

    _assert_fills(
        short_result,
        [(start, symbol, -quantity) for start, symbol, quantity in _fills(long_result)],
    )
    assert float(short_result.book.portfolio_value) == pytest.approx(9925.0, abs=1e-6)
    (event,) = short_policy.stop_events
    assert event.side == -1 and event.stop_level == 200.0 - 101.0


def test_no_same_direction_reentry_until_the_signal_has_been_absent() -> None:
    recover = [
        (109.0, 109.0, 100.0, 107.0),  # row 7: wick through the stop, close 107
        (108.0, 112.0, 107.0, 111.0),  # row 8: flat again, but close is a new breakout
        (111.0, 111.5, 108.0, 109.0),  # row 9: inside the channel -> block cleared
        (109.0, 118.0, 108.0, 117.0),  # row 10: fresh breakout -> entry allowed
        (117.0, 118.0, 116.0, 117.0),  # row 11: entry fills at this open
        (117.0, 118.0, 116.0, 117.0),
    ]
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + recover
    result, policy = _replay(_dataset({"AAA": bars}), stop_index=11)

    starts = [start for start, _, _ in _fills(result)]
    # entry (decided row 4), stop exit (row 7); row 8 has the signal but is blocked;
    # the next entry is decided at row 10.
    assert starts == [4, 7, 10]
    assert policy.summary()["entries"] == 2


def test_open_positions_are_held_without_churn_between_entry_and_exit() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    result, _ = _replay(_dataset({"AAA": bars}), stop_index=9)

    assert len(_fills(result)) == 2
    assert result.decisions[5].effective_hold_symbols == (0,)
    assert result.decisions[6].effective_hold_symbols == (0,)


def test_gross_cap_scales_simultaneous_entries_pro_rata() -> None:
    calm: list[Bar] = [(100.0, 100.5, 99.5, 100.0)] * 4 + [(100.0, 101.0, 100.0, 101.0)]
    tail: list[Bar] = [(101.0, 101.5, 100.5, 101.0)] * 3
    bars = {name: calm + tail for name in ("AAA", "BBB", "CCC", "DDD")}
    result, _ = _replay(_dataset(bars), stop_index=7)

    proposal = result.decisions[4].proposal_weights
    # ATR = 1 -> stop distance 2, risk weight 0.005*101/2 = 0.2525 > 0.25 cap; four
    # names request 1.0 > 0.95, so each is scaled to 0.95/4.
    assert sum(abs(weight) for weight in proposal) == pytest.approx(0.95)
    assert proposal == pytest.approx((0.2375,) * 4)


def test_unusable_rows_never_open_a_position() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    usable = [True] * 4 + [False] + [True] * 5
    result, policy = _replay(
        _dataset({"AAA": bars}, usable={"AAA": usable}), stop_index=9
    )

    assert _fills(result) == []
    assert policy.summary()["entries"] == 0


# -- a stop exit that fills only in part -------------------------------------------------

# Volume from row 8 on caps every fill: 50% participation of 8 units is 4 units, so the
# 50/6 = 8.33 unit position needs three exit fills (4, 4 and 1/3).
PARTIAL_EXIT_COST = replace(ExecutionCostConfig.zero(), max_participation_rate=0.5)


def partial_exit_dataset(bars: list[Bar]) -> MarketDataset:
    dataset = _dataset({"AAA": bars})
    volume = np.full(dataset.volume.shape, 1.0e12)
    volume[8:] = 8.0
    return replace(dataset, identity_payload_json=None, volume=volume)


def replay_partial_exit(
    bars: list[Bar],
) -> tuple[SharedCashPortfolioReplayResult, TrendStopPortfolioPolicy]:
    dataset = partial_exit_dataset(bars)
    policy = TrendStopPortfolioPolicy(dataset, _config())
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=dataset.n_bars - 1,
        initial_capital=10_000.0,
        execution_cost=PARTIAL_EXIT_COST,
        capture_ledger_evidence=True,
    )
    return result, policy


def test_a_partial_stop_exit_is_one_event_and_finishes_through_a_gap() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN + [(97.0, 98.0, 96.0, 97.0)] * 4
    result, policy = replay_partial_exit(bars)

    # Hand arithmetic: 50/6 units bought at 103; the stop (101) fires on row 7 and the
    # exit fills 4 units at the row-8 open 94, 4 at 97 and the last 1/3 at 97.
    quantity = Fraction(50, 6)
    _assert_fills(
        result,
        [
            (4, 0, quantity),
            (7, 0, Fraction(-4)),
            (8, 0, Fraction(-4)),
            (9, 0, -(quantity - 8)),
        ],
    )
    realized = 4 * (94 - 103) + 4 * (97 - 103) + (quantity - 8) * (97 - 103)
    assert realized == -62
    assert float(result.book.portfolio_value) == pytest.approx(9938.0, abs=1e-6)
    assert all(Fraction(value) == 0 for value in result.book.exact_quantities)
    # One stop event for the whole position, however many exit fills it took; the
    # partly filled position is re-proposed as an exit, never held.
    assert len(policy.stop_events) == 1 and policy.summary()["stop_exits"] == 1
    assert [d.proposal_weights for d in result.decisions[7:10]] == [(0.0,)] * 3
    assert all(not d.effective_hold_symbols for d in result.decisions[7:10])


def test_a_partial_stop_exit_continues_after_the_price_recovers_above_the_stop() -> (
    None
):
    wick = [
        (109.0, 109.0, 100.0, 107.0),  # row 7: wick through 101, close 107
        (108.0, 109.0, 106.0, 108.0),  # row 8: recovered, no longer below the level
        (109.0, 110.0, 107.0, 109.0),
        (109.0, 110.0, 107.0, 109.0),
        (109.0, 110.0, 107.0, 109.0),
    ]
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + wick
    result, policy = replay_partial_exit(bars)

    quantity = Fraction(50, 6)
    _assert_fills(
        result,
        [
            (4, 0, quantity),
            (7, 0, Fraction(-4)),
            (8, 0, Fraction(-4)),
            (9, 0, -(quantity - 8)),
        ],
    )
    # The stop fired once, so the remainder is exited although row 8 is above the stop.
    assert all(Fraction(value) == 0 for value in result.book.exact_quantities)
    assert len(policy.stop_events) == 1
    assert [d.proposal_weights for d in result.decisions[7:10]] == [(0.0,)] * 3


# -- a fired stop takes precedence over a reversal ---------------------------------------


def test_a_stop_and_an_opposite_signal_on_one_bar_exit_without_reversing() -> None:
    # Row 7 trades through the stop (low 95 <= 101) and closes at 99, below the prior
    # three lows (100, 102, 106): a short breakout on the same bar.
    bar7: Bar = (109.0, 109.0, 95.0, 99.0)
    tail: list[Bar] = [(99.0, 100.0, 98.0, 99.0)] * 3
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + [bar7] + tail
    result, policy = _replay(_dataset({"AAA": bars}), stop_index=len(bars) - 1)

    quantity = Fraction(50, 6)
    # The stop exit is the only order: no reversal into a short position.
    _assert_fills(result, [(4, 0, quantity), (7, 0, -quantity)])
    assert result.decisions[7].proposal_weights == (0.0,)
    assert policy.summary()["reversals"] == 0 and policy.summary()["stop_exits"] == 1


def test_no_reversal_while_a_partly_filled_stop_exit_is_pending() -> None:
    gap = [
        (109.0, 109.0, 95.0, 101.0),  # row 7: the stop fires (low 95 <= 101)
        (94.0, 96.0, 90.0, 92.0),  # row 8: still pending, and a short breakout
        (92.0, 93.0, 91.0, 92.0),
        (92.0, 93.0, 91.0, 92.0),
        (92.0, 93.0, 91.0, 92.0),
    ]
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + gap
    result, policy = replay_partial_exit(bars)

    quantity = Fraction(50, 6)
    # Rows 7 to 9 only reduce the long position (4, 4 and 1/3 units); the short
    # breakout of row 8 opens nothing until the position is flat.
    _assert_fills(
        result,
        [
            (4, 0, quantity),
            (7, 0, Fraction(-4)),
            (8, 0, Fraction(-4)),
            (9, 0, -(quantity - 8)),
        ],
    )
    assert [d.proposal_weights for d in result.decisions[7:10]] == [(0.0,)] * 3
    assert all(Fraction(value) == 0 for value in result.book.exact_quantities)
    assert policy.summary()["reversals"] == 0 and len(policy.stop_events) == 1


# -- causality ----------------------------------------------------------------------


def test_future_bars_do_not_change_earlier_decisions() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN + [(94.0, 95.0, 93.0, 94.0)] * 4
    dataset = _dataset({"AAA": bars})
    cut = 8
    scale = np.ones((dataset.n_bars, 1))
    scale[cut + 1 :] = 1.37
    perturbed = replace(
        dataset,
        identity_payload_json=None,
        open=dataset.open * scale,
        high=dataset.high * scale,
        low=dataset.low * scale,
        close=dataset.close * scale,
    )
    base, _ = _replay(dataset, stop_index=cut + 1)
    other, _ = _replay(perturbed, stop_index=cut + 1)

    assert [d.proposal_weights for d in base.decisions] == [
        d.proposal_weights for d in other.decisions
    ]


def test_policy_rejects_an_unbound_snapshot_sequence() -> None:
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    dataset = _dataset({"AAA": bars})
    other = _dataset({"AAA": bars, "BBB": bars})
    policy = TrendStopPortfolioPolicy(other, _config())
    with pytest.raises(ValueError, match="symbols do not match"):
        run_shared_portfolio_replay(
            dataset,
            policy,
            start_index=0,
            stop_index=3,
            execution_cost=ExecutionCostConfig.zero(),
        )
