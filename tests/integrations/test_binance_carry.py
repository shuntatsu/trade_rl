from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance import carry


def halt_rows():
    start = datetime(2023, 3, 24, 10, tzinfo=UTC)
    rows = []
    for hour in (10, 11, 12, 14, 15):
        timestamp = int(start.replace(hour=hour).timestamp() * 1000)
        price = 100 + hour
        rows.append(
            [timestamp, price, price, price, price, 10, timestamp + 3599999, 1000]
        )
    return rows, start, start.replace(hour=16)


def declared_halt():
    return (
        (
            datetime(2023, 3, 24, 11, 27, tzinfo=UTC),
            datetime(2023, 3, 24, 14, tzinfo=UTC),
        ),
    )


def test_declared_halt_uses_only_previous_mark_and_disables_partial_bins() -> None:
    rows, start, end = halt_rows()
    data = carry.parse_carry_spot_rows(
        rows, start_time=start, end_time=end, halt_intervals=declared_halt()
    )
    np.testing.assert_array_equal(data.close, [110, 111, 112, 112, 114, 115])
    np.testing.assert_array_equal(
        data.tradable, [True, False, False, False, True, True]
    )
    np.testing.assert_array_equal(data.volume, [1000, 0, 0, 0, 1000, 1000])
    assert data.timestamps[3] == np.datetime64("2023-03-24T14:00")
    rows[-2][1:5] = [10000] * 4
    changed = carry.parse_carry_spot_rows(
        rows, start_time=start, end_time=end, halt_intervals=declared_halt()
    )
    np.testing.assert_array_equal(data.close[:4], changed.close[:4])


@pytest.mark.parametrize(
    "defect", ["unknown", "partial", "leading", "duplicate", "unordered", "trailing"]
)
def test_halt_adapter_never_fills_an_unexplained_gap(defect) -> None:
    rows, start, end = halt_rows()
    halts = declared_halt()
    if defect == "unknown":
        halts = ()
    elif defect == "partial":
        rows.pop(1)
    elif defect == "leading":
        rows = rows[3:]
        start = start.replace(hour=13)
    elif defect == "duplicate":
        rows.insert(1, rows[0])
    elif defect == "unordered":
        rows = rows[::-1]
    elif defect == "trailing":
        rows.pop()
    with pytest.raises(ValueError):
        carry.parse_carry_spot_rows(
            rows, start_time=start, end_time=end, halt_intervals=halts
        )


def test_halted_volume_is_validated_before_zero_capacity_masking() -> None:
    rows, start, end = halt_rows()
    rows[1][7] = -1000
    with pytest.raises(ValueError, match="volume"):
        carry.parse_carry_spot_rows(
            rows, start_time=start, end_time=end, halt_intervals=declared_halt()
        )


def series(*, perpetual: bool) -> RawMarketSeries:
    values = np.full(48, 100.0)
    counts = np.zeros(48, dtype=np.int32)
    if perpetual:
        counts[7::8] = 1
    return RawMarketSeries(
        timestamps=np.datetime64("2023-01-01T00", "ns")
        + np.arange(48) * np.timedelta64(1, "h"),
        open=values,
        high=values,
        low=values,
        close=values,
        volume=values * 1000,
        funding_rate=counts * 0.0001,
        funding_available=counts > 0,
        funding_event_count=counts,
        tradable=np.ones(48, dtype=bool),
    )


def test_pairs_keep_separate_costs_funding_and_content_identity() -> None:
    assert hasattr(carry, "assemble_carry_dataset")
    dataset = carry.assemble_carry_dataset(
        {"BTCUSDT": (series(perpetual=False), series(perpetual=True))}
    )
    assert dataset.symbols == ("BTCUSDT:spot", "BTCUSDT:perp")
    assert dataset.identity_verified
    np.testing.assert_array_equal(dataset.funding_rate[:, 0], 0)
    assert dataset.funding_rate[:, 1].sum() == pytest.approx(0.0006)
    np.testing.assert_allclose(
        dataset.resolved_array("taker_fee_rate")[0], [0.001, 0.0005]
    )
    np.testing.assert_array_equal(
        dataset.resolved_array("borrow_available")[0], [False, True]
    )


def test_mismatched_clock_and_missing_funding_fail_closed() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    delayed = replace(
        perp,
        timestamps=perp.timestamps + np.timedelta64(1, "h"),
        available_at=perp.available_at + np.timedelta64(1, "h"),
    )
    with pytest.raises(ValueError, match="clock"):
        carry.assemble_carry_dataset({"BTCUSDT": (spot, delayed)})
    with pytest.raises(ValueError, match="funding"):
        carry.assemble_carry_dataset({"BTCUSDT": (spot, spot)})


def test_missing_internal_funding_and_spot_funding_are_rejected() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    counts = perp.funding_event_count.copy()
    counts[23] = 0
    missing = replace(
        perp,
        funding_event_count=counts,
        funding_available=counts > 0,
        funding_rate=counts * 0.0001,
    )
    with pytest.raises(ValueError, match="funding"):
        carry.assemble_carry_dataset({"BTCUSDT": (spot, missing)})
    with pytest.raises(ValueError, match="spot"):
        carry.assemble_carry_dataset({"BTCUSDT": (perp, perp)})


def test_cost_capacity_and_raw_prices_are_identity_bound() -> None:
    sources = {"BTCUSDT": (series(perpetual=False), series(perpetual=True))}
    base = carry.assemble_carry_dataset(sources)
    expensive = carry.assemble_carry_dataset(sources, cost_multiplier=2.0)
    thin = carry.assemble_carry_dataset(sources, capacity_multiplier=0.1)
    assert len({base.dataset_id, expensive.dataset_id, thin.dataset_id}) == 3
    np.testing.assert_allclose(
        expensive.resolved_array("taker_fee_rate"),
        base.resolved_array("taker_fee_rate") * 2,
    )
    np.testing.assert_allclose(thin.resolved_array("max_participation_rate"), 0.001)
    from trade_rl.data.contracts import VolumeUnit

    assert base.volume_units == (VolumeUnit.QUOTE_NOTIONAL, VolumeUnit.QUOTE_NOTIONAL)


def test_delayed_information_is_not_relabelled_as_observed() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    delayed = replace(spot, available_at=spot.available_at + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="available"):
        carry.assemble_carry_dataset({"BTCUSDT": (delayed, perp)})


def test_canonical_executor_charges_each_leg_once_and_funds_only_perpetual() -> None:
    from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
    from trade_rl.simulation.orders.model import OrderBookState
    from trade_rl.simulation.targets.execution import execute_target_statefully

    spot, perp = series(perpetual=False), series(perpetual=True)
    perp = replace(perp, funding_rate=perp.funding_rate * 10)
    dataset = carry.assemble_carry_dataset({"BTCUSDT": (spot, perp)})
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), processing_bar_volume_capacity=False),
    )
    entry = execute_target_statefully(
        executor,
        BookState.zero(2, 4000, dataset.close[5]),
        OrderBookState.empty(),
        np.array([0.25, -0.25]),
        start_index=5,
        bars=1,
        target_identity="entry",
    )
    assert entry.interval_cost == pytest.approx(2.5)
    funded = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=6, bars=1
    )
    assert funded.interval_funding == pytest.approx(1.0)
    exit_result = execute_target_statefully(
        executor,
        funded.book,
        funded.order_book,
        np.zeros(2),
        start_index=7,
        bars=1,
        target_identity="exit",
    )
    assert exit_result.interval_cost == pytest.approx(2.5)
    assert exit_result.book.cash == pytest.approx(3996.0)
    np.testing.assert_array_equal(exit_result.book.quantities, 0)


def test_malformed_pair_lengths_cannot_relabel_another_symbols_source() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    with pytest.raises(ValueError, match="two legs"):
        carry.assemble_carry_dataset(
            {"BTCUSDT": (spot, perp, spot), "ETHUSDT": (perp,)}
        )
