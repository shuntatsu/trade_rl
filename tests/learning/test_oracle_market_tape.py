from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_market_tape import (
    ORACLE_MARKET_TAPE_SCHEMA,
    OracleMarketTape,
    build_oracle_market_tape,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(n_bars: int = 9) -> MarketDataset:
    symbols = ("BTCUSDT", "ETHUSDT")
    n_symbols = len(symbols)
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(n_bars) * 0.01),
            50.0 * np.exp(np.arange(n_bars) * -0.005),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    rows = np.arange(n_bars, dtype=np.float64)[:, None]
    columns = np.arange(n_symbols, dtype=np.float64)[None, :]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=1_000.0 + 10.0 * rows + columns,
        funding_rate=0.0001 * (rows + columns),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
        fee_rate=np.full((n_bars, n_symbols), 0.0002),
        taker_fee_rate=np.full((n_bars, n_symbols), 0.0003),
        spread_rate=np.full((n_bars, n_symbols), 0.0004),
        max_participation_rate=np.full((n_bars, n_symbols), 0.2),
        minimum_notional=5.0 + rows + columns,
        borrow_available=np.ones((n_bars, n_symbols), dtype=np.bool_),
        borrow_rate=0.01 + 0.001 * rows + 0.0001 * columns,
        funding_due=(np.arange(n_bars)[:, None] % 2 == 0)
        & np.ones((1, n_symbols), dtype=np.bool_),
        asset_active=np.ones((n_bars, n_symbols), dtype=np.bool_),
        buy_allowed=np.ones((n_bars, n_symbols), dtype=np.bool_),
        sell_allowed=np.ones((n_bars, n_symbols), dtype=np.bool_),
        mark_price=close * 1.0005,
        dividend=0.01 * rows + 0.001 * columns,
        split_factor=np.ones((n_bars, n_symbols), dtype=np.float64),
        delisting_recovery=np.ones((n_bars, n_symbols), dtype=np.float64),
        cash_rate=np.linspace(0.01, 0.02, n_bars),
    )


def _config() -> OracleTeacherConfig:
    return OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0005,
            taker_fee_rate=0.0006,
            spread_rate=0.0007,
            impact_rate=0.001,
            max_participation_rate=0.15,
            minimum_notional=7.0,
        )
    )


def test_market_tape_matches_direct_step_calculations() -> None:
    market = _market()
    config = _config()
    start, stop = 1, 8
    tape = build_oracle_market_tape(market, (start, stop), config.bellman_parameters)

    for step, close_index in enumerate(range(start, stop - 1)):
        execution_index = close_index + 1
        previous_mark = market.resolved_array("mark_price")[close_index]
        split = market.resolved_array("split_factor")[execution_index]
        raw_factor = market.open[execution_index] * split / previous_mark
        active = market.resolved_array("asset_active")[execution_index]
        recovery = market.resolved_array("delisting_recovery")[execution_index]
        equity_factor = np.where(active, raw_factor, raw_factor * recovery)
        market_notional = market.market_notional(
            execution_index,
            market.open[execution_index],
            volume=market.volume[close_index],
        )
        participation_limit = np.minimum(
            market.resolved_array("max_participation_rate")[execution_index],
            config.execution_cost.max_participation_rate,
        )
        venue_fee = (
            config.execution_cost.taker_fee_rate
            + market.resolved_array("taker_fee_rate")[execution_index]
        )
        base_cost = config.execution_cost.multiplier * (
            config.execution_cost.fee_rate
            + market.resolved_array("fee_rate")[execution_index]
            + venue_fee
            + config.execution_cost.spread_rate
            + market.resolved_array("spread_rate")[execution_index]
        )

        np.testing.assert_allclose(tape.raw_position_factor[step], raw_factor)
        np.testing.assert_allclose(tape.equity_position_factor[step], equity_factor)
        np.testing.assert_allclose(
            tape.mark_open_ratio[step],
            market.resolved_array("mark_price")[execution_index]
            / market.open[execution_index],
        )
        np.testing.assert_array_equal(tape.active[step], active)
        np.testing.assert_array_equal(
            tape.tradable[step], market.tradable[execution_index]
        )
        np.testing.assert_array_equal(
            tape.buy_allowed[step],
            market.resolved_array("buy_allowed")[execution_index],
        )
        np.testing.assert_array_equal(
            tape.sell_allowed[step],
            market.resolved_array("sell_allowed")[execution_index],
        )
        np.testing.assert_array_equal(
            tape.borrow_available[step],
            market.resolved_array("borrow_available")[execution_index],
        )
        np.testing.assert_allclose(tape.market_notional[step], market_notional)
        np.testing.assert_allclose(
            tape.participation_capacity[step],
            participation_limit * market_notional,
        )
        np.testing.assert_allclose(
            tape.minimum_notional[step],
            np.maximum(
                market.resolved_array("minimum_notional")[execution_index],
                config.execution_cost.minimum_notional,
            ),
        )
        np.testing.assert_allclose(tape.base_unit_cost[step], base_cost)
        np.testing.assert_allclose(
            tape.funding_due_rate[step],
            market.funding_rate[execution_index]
            * market.resolved_array("funding_due")[execution_index],
        )
        np.testing.assert_allclose(
            tape.borrow_rate[step],
            market.resolved_array("borrow_rate")[execution_index],
        )
        np.testing.assert_allclose(
            tape.dividend_open_ratio[step],
            market.resolved_array("dividend")[execution_index]
            / market.open[execution_index],
        )
        assert tape.cash_rate[step] == pytest.approx(
            float(market.resolved_array("cash_rate")[execution_index])
        )
        assert tape.elapsed_year_fraction[step] == pytest.approx(
            market.elapsed_year_fraction(close_index, execution_index)
        )


def test_market_tape_is_readonly_contiguous_and_identified() -> None:
    tape = build_oracle_market_tape(_market(), (1, 8), _config().bellman_parameters)

    assert tape.schema_version == ORACLE_MARKET_TAPE_SCHEMA
    assert tape.start == 1
    assert tape.stop == 8
    assert tape.steps == 6
    assert tape.symbol_count == 2
    assert len(tape.digest) == 64
    for array in tape.arrays.values():
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_market_tape_does_not_read_at_or_after_stop() -> None:
    market = _market(10)
    config = _config()
    start, stop = 1, 8
    original = build_oracle_market_tape(market, (start, stop), config.bellman_parameters)

    fee = market.resolved_array("fee_rate").copy()
    funding = market.funding_rate.copy()
    volume = market.volume.copy()
    fee[stop:] *= 100.0
    funding[stop:] += 1.0
    volume[stop:] *= 1000.0
    changed = replace(market, fee_rate=fee, funding_rate=funding, volume=volume)
    rebuilt = build_oracle_market_tape(changed, (start, stop), config.bellman_parameters)

    assert rebuilt.digest == original.digest
    for name, value in original.arrays.items():
        np.testing.assert_array_equal(rebuilt.arrays[name], value)


def test_market_tape_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="range"):
        build_oracle_market_tape(_market(), (0, 1), _config().bellman_parameters)


def test_market_tape_validates_manual_shape_drift() -> None:
    tape = build_oracle_market_tape(_market(), (1, 8), _config().bellman_parameters)
    payload = {name: value for name, value in tape.arrays.items()}
    payload["market_notional"] = np.ones((tape.steps + 1, tape.symbol_count))

    with pytest.raises(ValueError, match="market_notional"):
        OracleMarketTape(
            **payload,
            start=tape.start,
            stop=tape.stop,
            dataset_id=tape.dataset_id,
        )
