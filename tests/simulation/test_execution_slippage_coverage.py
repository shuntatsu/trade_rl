from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _market(**overrides: object) -> MarketDataset:
    n_bars = 5
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    values: dict[str, object] = {
        "dataset_id": "b" * 64,
        "symbols": ("S0",),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        "features": np.zeros((n_bars, 1, 1), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(shape, 1_000_000.0),
        "funding_rate": np.zeros(shape),
        "tradable": np.ones(shape, dtype=np.bool_),
        "feature_available": np.ones((n_bars, 1, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }
    values.update(overrides)
    return MarketDataset(**values)


def test_slippage_rounding_capacity_and_borrow_paths() -> None:
    executor = MarketExecutor(
        _market(
            lot_size=np.full((5, 1), 0.25),
            tick_size=np.full((5, 1), 0.5),
            borrow_available=np.zeros((5, 1), dtype=np.bool_),
        ),
        ExecutionCostConfig(
            slippage_std=0.01,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=2.0,
            random_seed=7,
        ),
    )
    assert executor._slippage_rates(0).size == 0
    assert np.isfinite(executor._slippage_rates(3)).all()
    assert executor._round_prices(
        np.array([100.24]), index=1
    ).tolist() == pytest.approx([100.0])
    assert executor._round_quantities(
        np.array([1.13]), index=1
    ).tolist() == pytest.approx([1.0])
    assert executor._capacity_notional(
        np.array([100.0]), np.array([2.0])
    ).tolist() == pytest.approx([200.0])
    constrained = executor._constrain_borrow(
        np.array([-2.0]), current=np.array([-1.0]), index=1
    )
    assert constrained.tolist() == pytest.approx([-1.0])


def test_limit_buy_sell_and_untouched_paths() -> None:
    touched = MarketExecutor(
        _market(
            low=np.full((5, 1), 90.0),
            high=np.full((5, 1), 110.0),
        ),
        ExecutionCostConfig(
            maker_fee_rate=0.001,
            spread_rate=0.002,
            max_participation_rate=1.0,
            order_type="limit",
            limit_offset_rate=0.01,
            maintenance_margin_rate=0.0,
        ),
    )
    buy_book = BookState.zero(1, 1_000.0, touched.dataset.close[0])
    buy = touched._fill_toward_quantities(
        buy_book,
        np.array([1.0]),
        prices=touched.dataset.open[1],
        capacity_volume=touched.dataset.volume[0],
        tradable=touched.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert buy.filled_notional > 0.0
    assert buy.cost_amount > 0.0

    sell_book = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=touched.dataset.close[0],
    )
    sell = touched._fill_toward_quantities(
        sell_book,
        np.array([0.0]),
        prices=touched.dataset.open[1],
        capacity_volume=touched.dataset.volume[0],
        tradable=touched.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert sell.filled_notional > 0.0

    untouched = MarketExecutor(
        _market(
            low=np.full((5, 1), 100.0),
            high=np.full((5, 1), 100.0),
        ),
        replace(touched.cost, maker_fee_rate=0.0, spread_rate=0.0),
    )
    no_fill = untouched._fill_toward_quantities(
        BookState.zero(1, 1_000.0, untouched.dataset.close[0]),
        np.array([1.0]),
        prices=untouched.dataset.open[1],
        capacity_volume=untouched.dataset.volume[0],
        tradable=untouched.dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert no_fill.filled_notional == 0.0


def test_direction_minimum_notional_and_zero_capacity_paths() -> None:
    shape = (5, 1)
    blocked_dataset = _market(
        buy_allowed=np.zeros(shape, dtype=np.bool_),
        sell_allowed=np.zeros(shape, dtype=np.bool_),
    )
    blocked = MarketExecutor(
        blocked_dataset,
        ExecutionCostConfig.zero(),
    )._fill_toward_quantities(
        BookState.zero(1, 1_000.0, blocked_dataset.close[0]),
        np.array([1.0]),
        prices=blocked_dataset.open[1],
        capacity_volume=np.array([0.0]),
        tradable=np.array([True]),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert blocked.filled_notional == 0.0
    assert blocked.max_participation == 0.0

    minimum_dataset = _market(minimum_notional=np.full(shape, 50.0))
    too_small = MarketExecutor(
        minimum_dataset,
        ExecutionCostConfig.zero(),
    )._fill_toward_quantities(
        BookState.zero(1, 1_000.0, minimum_dataset.close[0]),
        np.array([0.1]),
        prices=minimum_dataset.open[1],
        capacity_volume=minimum_dataset.volume[0],
        tradable=minimum_dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert too_small.requested_notional == 0.0


def test_latency_income_carry_and_liquidation_paths() -> None:
    cash_rate = np.zeros(5)
    cash_rate[1] = 0.05
    delayed_dataset = _market(cash_rate=cash_rate)
    delayed = MarketExecutor(
        delayed_dataset,
        replace(ExecutionCostConfig.zero(), order_latency_bars=2),
    ).execute_interval(
        BookState.zero(1, 1_000.0, delayed_dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert delayed.requested_turnover == 0.0
    assert delayed.fill_ratio == 1.0
    assert delayed.interval_cash_interest >= 0.0

    dividend = np.zeros((5, 1))
    dividend[1, 0] = 1.0
    funding = np.zeros((5, 1))
    funding[1, 0] = 0.01
    funding_due = np.zeros((5, 1), dtype=np.bool_)
    funding_due[1, 0] = True
    income_dataset = _market(
        dividend=dividend,
        funding_rate=funding,
        funding_due=funding_due,
    )
    invested = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=income_dataset.close[0],
    )
    income = MarketExecutor(
        income_dataset,
        ExecutionCostConfig.zero(),
    ).execute_interval(
        invested,
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert income.interval_dividend >= 0.0
    assert income.interval_funding <= 0.0

    liquidation_dataset = _market()
    liquidation_executor = MarketExecutor(
        liquidation_dataset,
        ExecutionCostConfig.zero(),
    )
    full = liquidation_executor.liquidate_at_close(
        BookState.from_weights(
            weights=np.array([1.0]),
            capital=1_000.0,
            prices=liquidation_dataset.close[0],
        ),
        index=1,
    )
    assert full.termination_reason == EconomicTerminationReason.LIQUIDATION.value
    assert full.fill_ratio == 1.0

    zero = liquidation_executor.liquidate_at_close(
        BookState.zero(1, 1_000.0, liquidation_dataset.close[0]),
        index=1,
    )
    assert zero.fill_ratio == 1.0

    partial_dataset = _market(volume=np.full((5, 1), 0.01))
    partial = MarketExecutor(
        partial_dataset,
        replace(ExecutionCostConfig.zero(), max_participation_rate=0.1),
    ).liquidate_at_close(
        BookState.from_weights(
            weights=np.array([1.0]),
            capital=1_000.0,
            prices=partial_dataset.close[0],
        ),
        index=1,
    )
    assert 0.0 <= partial.fill_ratio < 1.0
    assert partial.termination_reason is None
