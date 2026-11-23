"""Read-only C2/C3 attribution and post-event diagnostic contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.evaluation.divergence_cloud_diagnostics import (
    attribute_single_symbol_ledger,
    inspect_opposing_divergence_exits,
)


@dataclass(frozen=True)
class _Interval:
    start_index: int
    next_index: int
    exact_quantities_before: tuple[str, ...]
    exact_quantities_after: tuple[str, ...]
    portfolio_value_before: float
    portfolio_value_after: float
    interval_cost: float = 0.0
    interval_funding: float = 0.0
    interval_borrow_cost: float = 0.0


@dataclass(frozen=True)
class _Decision:
    index: int
    proposal_weights: tuple[float, ...]


def _e(
    start: int,
    before: str,
    after: str,
    pv_before: float,
    pv_after: float,
    *,
    cost: float = 0.0,
    funding: float = 0.0,
) -> _Interval:
    return _Interval(
        start_index=start,
        next_index=start + 1,
        exact_quantities_before=(before,),
        exact_quantities_after=(after,),
        portfolio_value_before=pv_before,
        portfolio_value_after=pv_after,
        interval_cost=cost,
        interval_funding=funding,
    )


def test_attribution_is_cash_equity_conserving_with_inclusive_fill_cost() -> None:
    trace = (
        _e(0, "0", "2", 1000, 995, cost=5),
        _e(1, "2", "2", 995, 1015, funding=2),
        _e(2, "2", "0", 1015, 1007, cost=3),
        _e(3, "0", "-1", 1007, 1003, cost=4),
        _e(4, "-1", "-1", 1003, 1020),
        _e(5, "-1", "0", 1020, 1022, cost=1),
        _e(6, "0", "0", 1022, 1022),
    )
    result = attribute_single_symbol_ledger(trace)
    assert result.net_long_usdt == pytest.approx(7)
    assert result.net_short_usdt == pytest.approx(15)
    assert result.net_flat_usdt == 0
    assert result.net_reversal_usdt == 0
    assert result.net_total_usdt == pytest.approx(22)
    assert result.long_cost_usdt == pytest.approx(8)
    assert result.short_cost_usdt == pytest.approx(5)
    assert result.long_funding_usdt == pytest.approx(2)
    assert result.long_intervals == 3
    assert result.short_intervals == 3
    assert result.flat_intervals == 1
    assert result.reversal_intervals == 0


def test_reversal_is_not_misattributed_to_either_direction() -> None:
    ledger = (
        _e(0, "2", "-3", 1000, 990, cost=6),
        _e(1, "-3", "-3", 990, 1005),
    )
    result = attribute_single_symbol_ledger(ledger)
    assert result.net_reversal_usdt == -10
    assert result.net_short_usdt == 15
    assert result.net_total_usdt == 5
    assert result.reversal_intervals == 1


@pytest.mark.parametrize(
    "bad",
    [
        (_e(0, "0", "1", 1000, 999), _e(2, "1", "1", 999, 1000)),
        (_e(0, "0", "1", 1000, 999), _e(1, "1", "1", 998, 1000)),
        (_e(0, "0", "1", 1000, 999), _e(1, "0", "1", 999, 1000)),
    ],
)
def test_attribution_rejects_gaps_in_identity_or_equity(
    bad: tuple[_Interval, ...],
) -> None:
    with pytest.raises(ValueError):
        attribute_single_symbol_ledger(bad)


def test_exit_event_study_never_reads_beyond_window_and_is_directional() -> None:
    # At t=2, a C3 long is closed by bearish 4h event while the cloud stays up.
    # C2 stays long through hour 4; a short-side event at t=6 sees the reverse.
    n = 10
    price = np.asarray([100, 101, 102, 105, 106, 107, 105, 102, 99, 98], dtype=float)
    cloud = np.asarray([1] * 5 + [-1] * 5, dtype=np.int8)
    bull = np.zeros(n, dtype=np.int16)
    bear = np.zeros(n, dtype=np.int16)
    bear[2] = 1
    bull[6] = 1
    # Exact native-style quantity continuity across each interval.
    c3_positions = ("0", "2", "2", "0", "0", "0", "-2", "0", "0", "0")
    c2_positions = ("0", "2", "2", "2", "2", "0", "-2", "-2", "-2", "-2")
    intervals_c3 = tuple(
        _e(i, c3_positions[i], c3_positions[i + 1], 1000, 1000) for i in range(n - 1)
    )
    intervals_c2 = tuple(
        _e(i, c2_positions[i], c2_positions[i + 1], 1000, 1000) for i in range(n - 1)
    )
    decisions = tuple(_Decision(i, (0.0,)) for i in range(n - 1))
    results = inspect_opposing_divergence_exits(
        c3_intervals=intervals_c3,
        c3_decisions=decisions,
        c2_intervals=intervals_c2,
        close=price,
        cloud_state=cloud,
        bull_4h=bull,
        bear_4h=bear,
        horizons=(2, 4),
        start_index=0,
        stop_index=n - 1,
    )
    assert len(results) == 2
    assert results[0].index == 2
    assert results[0].direction == "long"
    assert results[0].future_signed_returns_pct[2] == pytest.approx(
        100 * (106 / 102 - 1)
    )
    assert results[0].c2_same_side_at_event
    assert results[1].index == 6
    assert results[1].direction == "short"
    assert results[1].future_signed_returns_pct[2] == pytest.approx(
        100 * (1 - 99 / 105)
    )
    assert 4 not in results[1].future_signed_returns_pct


def test_no_exit_on_incompatible_cloud_or_missing_proposal() -> None:
    n = 6
    positions = ("1", "0", "0", "0", "0", "0")
    trace = tuple(
        _e(i, positions[i], positions[i + 1], 1000, 1000) for i in range(n - 1)
    )
    decision = tuple(_Decision(i, (0.25,)) for i in range(n - 1))
    out = inspect_opposing_divergence_exits(
        c3_intervals=trace,
        c3_decisions=decision,
        c2_intervals=trace,
        close=np.full(n, 100.0),
        cloud_state=np.ones(n, dtype=np.int8),
        bull_4h=np.zeros(n, dtype=np.int16),
        bear_4h=np.ones(n, dtype=np.int16),
        horizons=(2,),
        start_index=0,
        stop_index=n - 1,
    )
    assert out == ()


def test_native_portfolio_ledger_attribution_reconciles_with_book() -> None:
    """Conservation uses actual native accounting, not a fabricated P&L oracle."""
    from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
    from trade_rl.data.market import MarketDataset
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.simulation import ExecutionCostConfig
    from trade_rl.strategies.divergence_cloud_follow import (
        CloudFollowConfig,
        CloudFollowPolicy,
    )

    n = 60
    price = 100.0 + 0.13 * np.arange(n) + 3 * np.sin(np.arange(n) / 8)
    op = np.r_[price[0], price[:-1]]
    stamps = np.datetime64("2022-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    data = MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=stamps,
        open=op[:, None],
        close=price[:, None],
        high=(np.maximum(op, price) + 0.1)[:, None],
        low=(np.minimum(op, price) - 0.1)[:, None],
        volume=np.full((n, 1), 1e5),
        funding_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=np.bool_),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_features=np.zeros((n, 1), dtype=np.float32),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    zero = np.zeros(n, dtype=np.int16)
    signals = DivergenceCloudMtfPack(
        timestamps=stamps,
        bull_1h=zero,
        bear_1h=zero,
        bull_4h=zero,
        bear_4h=zero,
        cloud_state=np.array([1] * 25 + [-1] * (n - 25), dtype=np.int8),
        cloud_ready=np.ones(n, dtype=np.bool_),
        events_1h=(),
        events_4h=(),
    )
    result = run_shared_portfolio_replay(
        data,
        CloudFollowPolicy(
            symbol="BTCUSDT",
            signals=signals,
            config=CloudFollowConfig(mode="confirmed_cloud"),
        ),
        start_index=0,
        stop_index=n - 1,
        initial_capital=10000.0,
        execution_cost=ExecutionCostConfig(),
        capture_ledger_evidence=True,
    )
    assert result.ledger_evidence is not None
    attribution = attribute_single_symbol_ledger(
        result.ledger_evidence.intervals.intervals
    )
    assert attribution.net_total_usdt == pytest.approx(
        result.book.portfolio_value - 10000.0, abs=1e-8
    )
    assert attribution.long_intervals > 0
    assert attribution.short_intervals > 0
    assert attribution.long_cost_usdt + attribution.short_cost_usdt == pytest.approx(
        result.book.total_cost
    )
