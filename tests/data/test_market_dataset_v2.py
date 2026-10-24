from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset


def kwargs(n_bars: int = 8, n_symbols: int = 2) -> dict[str, object]:
    timestamps = np.datetime64("2026-01-01T01:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = 100.0 + np.arange(n_bars, dtype=np.float64)
    close = np.column_stack([base + 10.0 * i for i in range(n_symbols)])
    open_price = np.vstack([close[0], close[:-1]])
    return {
        "dataset_id": "a" * 64,
        "symbols": tuple(f"S{i}" for i in range(n_symbols)),
        "timestamps": timestamps,
        "features": np.zeros((n_bars, n_symbols, 2), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": open_price,
        "high": np.maximum(open_price, close) + 1.0,
        "low": np.minimum(open_price, close) - 1.0,
        "close": close,
        "volume": np.full((n_bars, n_symbols), 1_000.0),
        "funding_rate": np.zeros_like(close),
        "tradable": np.ones((n_bars, n_symbols), dtype=np.bool_),
        "feature_available": np.ones((n_bars, n_symbols, 2), dtype=np.bool_),
        "feature_names": ("ret", "rsi"),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }


def test_continuous_market_requires_regular_cadence() -> None:
    values = kwargs()
    timestamps = np.asarray(values["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(16, "h")
    values["timestamps"] = timestamps
    with pytest.raises(ValueError, match="regular"):
        MarketDataset(**values)


def test_market_dataset_uses_exact_timestamp_order_at_nanosecond_extremes() -> None:
    limits = np.iinfo(np.int64)
    ascending = np.array([limits.min + 1, 0, limits.max], dtype=np.int64).astype(
        "datetime64[ns]"
    )
    descending = np.array([limits.max, limits.min + 1, 0], dtype=np.int64).astype(
        "datetime64[ns]"
    )

    ascending_values = kwargs(n_bars=3, n_symbols=1)
    ascending_values.update(
        timestamps=ascending,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )
    market = MarketDataset(**ascending_values)
    np.testing.assert_array_equal(market._timestamp_ns, ascending.astype(np.int64))

    descending_values = {
        **ascending_values,
        "timestamps": descending,
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketDataset(**descending_values)


def test_market_dataset_rejects_overflowed_adjacent_timestamp_interval() -> None:
    limits = np.iinfo(np.int64)
    timestamps = np.array([limits.min + 1, 1, 2], dtype=np.int64).astype(
        "datetime64[ns]"
    )
    values = kwargs(n_bars=3, n_symbols=1)
    values.update(
        timestamps=timestamps,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )

    with pytest.raises(ValueError, match="representable as positive nanoseconds"):
        MarketDataset(**values)


def test_session_calendar_accepts_gaps_and_uses_wall_clock_helpers() -> None:
    values = kwargs()
    timestamps = np.asarray(values["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(16, "h")
    values.update(
        timestamps=timestamps,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )
    market = MarketDataset(**values)
    assert market.regular_cadence is False
    assert market.elapsed_hours(3, 4) == pytest.approx(17.0)
    assert market.lookback_index(4, 8.0) == 3
    assert market.forward_index(3, 8.0) == 4


@pytest.mark.parametrize(
    "periods_per_year",
    [True, 252.0, 1.5, math.nan, math.inf],
)
def test_session_calendar_requires_integer_annualization_clock(
    periods_per_year: object,
) -> None:
    values = kwargs()
    values.update(
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=periods_per_year,
    )

    with pytest.raises(ValueError, match="periods_per_year must be a positive integer"):
        MarketDataset(**values)


def test_market_dataset_accepts_numpy_integer_annualization_clock() -> None:
    values = kwargs()
    values["periods_per_year"] = np.int64(8_760)

    market = MarketDataset(**values)

    assert market.periods_per_year == 8_760
    assert type(market.periods_per_year) is int


def test_numpy_integer_clock_has_same_canonical_dataset_identity() -> None:
    numpy_values = kwargs()
    numpy_values["periods_per_year"] = np.int64(8_760)
    numpy_market = MarketDataset(**numpy_values)
    native_market = MarketDataset(**kwargs())

    numpy_identified = numpy_market.with_content_identity()
    native_identified = native_market.with_content_identity()

    assert numpy_identified.periods_per_year == 8_760
    assert (
        numpy_identified.identity_payload_json
        == native_identified.identity_payload_json
    )
    assert numpy_identified.dataset_id == native_identified.dataset_id


def test_session_calendar_rejects_nat_timestamp_with_valid_availability() -> None:
    values = kwargs()
    valid_timestamps = np.asarray(values["timestamps"]).copy()
    available_at = np.broadcast_to(
        valid_timestamps[:, None], (valid_timestamps.size, 2)
    ).copy()
    timestamps = valid_timestamps.copy()
    timestamps[-1] = np.datetime64("NaT", "ns")
    values.update(
        timestamps=timestamps,
        available_at=available_at,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )

    with pytest.raises(ValueError, match="timestamps must not contain NaT"):
        MarketDataset(**values)


@pytest.mark.parametrize(
    "start",
    ["1677-01-01T00:00:00", "2263-01-01T00:00:00"],
)
def test_market_dataset_rejects_timestamps_outside_nanosecond_range(
    start: str,
) -> None:
    values = kwargs(n_bars=3, n_symbols=1)
    values["timestamps"] = np.datetime64(start, "s") + np.arange(3) * np.timedelta64(
        1, "h"
    )

    with pytest.raises(
        ValueError,
        match=r"timestamps must be exactly representable as datetime64\[ns\]",
    ):
        MarketDataset(**values)


def test_market_dataset_rejects_available_at_outside_nanosecond_range() -> None:
    values = kwargs(n_bars=3, n_symbols=1)
    timestamps = np.datetime64("1677-10-01T00:00:00", "ns") + np.arange(
        3
    ) * np.timedelta64(1, "h")
    values["timestamps"] = timestamps
    values["available_at"] = np.broadcast_to(
        np.datetime64("2263-01-01T00:00:00", "s"), (3, 1)
    )

    with pytest.raises(
        ValueError,
        match=r"available_at must be exactly representable as datetime64\[ns\]",
    ):
        MarketDataset(**values)


@pytest.mark.parametrize(
    "timestamps",
    [
        np.datetime64("1677-09-21T00:12:43.145224193", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        np.datetime64("2262-04-11T23:47:16.854775807", "ns")
        - np.arange(2, -1, -1) * np.timedelta64(1, "h"),
    ],
)
def test_market_dataset_accepts_nanosecond_range_boundaries(
    timestamps: np.ndarray,
) -> None:
    values = kwargs(n_bars=3, n_symbols=1)
    values["timestamps"] = timestamps

    market = MarketDataset(**values)

    assert np.array_equal(market._timestamp_ns, timestamps.astype(np.int64))
    assert not market.timestamps.flags.writeable
    assert not market.available_at.flags.writeable


def test_market_dataset_preserves_coarse_and_exact_subnanosecond_clocks() -> None:
    coarse_values = kwargs(n_bars=3, n_symbols=1)
    coarse_timestamps = np.datetime64("2026-01-01T01:00:00", "s") + np.arange(
        3
    ) * np.timedelta64(1, "h")
    coarse_values["timestamps"] = coarse_timestamps
    coarse_values["available_at"] = np.broadcast_to(coarse_timestamps[:, None], (3, 1))

    exact_values = kwargs(n_bars=3, n_symbols=1)
    exact_timestamps = np.datetime64("1969-12-31T00:00:00", "ns") + np.arange(
        3
    ) * np.timedelta64(1, "h")
    exact_values["timestamps"] = exact_timestamps.astype("datetime64[ps]")

    coarse_market = MarketDataset(**coarse_values)
    exact_market = MarketDataset(**exact_values)
    coarse_identity = coarse_market.with_content_identity()
    coarse_ns_values = {
        **coarse_values,
        "timestamps": coarse_timestamps.astype("datetime64[ns]"),
        "available_at": np.broadcast_to(
            coarse_timestamps.astype("datetime64[ns]")[:, None], (3, 1)
        ),
    }
    coarse_ns_identity = MarketDataset(**coarse_ns_values).with_content_identity()
    exact_ns_values = {**exact_values, "timestamps": exact_timestamps}
    exact_identity = MarketDataset(**exact_ns_values).with_content_identity()

    assert np.array_equal(
        coarse_market._timestamp_ns,
        coarse_market.timestamps.astype("datetime64[ns]").astype(np.int64),
    )
    assert coarse_market.available_at.dtype == np.dtype("datetime64[ns]")
    assert exact_market.timestamps.dtype == np.dtype("datetime64[ps]")
    assert coarse_identity.dataset_id == coarse_ns_identity.dataset_id
    assert exact_identity.dataset_id == exact_market.with_content_identity().dataset_id


@pytest.mark.parametrize("field_name", ["timestamps", "available_at"])
def test_market_dataset_rejects_subnanosecond_precision(
    field_name: str,
) -> None:
    values = kwargs(n_bars=3, n_symbols=1)
    timestamps = np.datetime64("1969-12-31T00:00:00", "ns") + np.arange(
        3
    ) * np.timedelta64(1, "h")
    if field_name == "timestamps":
        subnanosecond_timestamps = timestamps.astype("datetime64[ps]")
        subnanosecond_timestamps[0] += np.timedelta64(1, "ps")
        values["timestamps"] = subnanosecond_timestamps
    else:
        values["timestamps"] = timestamps
        available_at = np.broadcast_to(
            timestamps[:, None].astype("datetime64[ps]"), (3, 1)
        ).copy()
        available_at[0, 0] += np.timedelta64(1, "ps")
        values["available_at"] = available_at

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be exactly representable as datetime64\[ns\]",
    ):
        MarketDataset(**values)


def test_execution_and_missingness_metadata_are_immutable_and_shaped() -> None:
    values = kwargs()
    values["feature_available"] = np.zeros((8, 2, 2), dtype=np.bool_)
    values["feature_staleness_hours"] = np.full((8, 2, 2), 3.0)
    values["borrow_available"] = np.zeros((8, 2), dtype=np.bool_)
    values["mark_price"] = np.asarray(values["close"]) * 1.001
    values["index_price"] = np.asarray(values["close"])
    market = MarketDataset(**values)
    assert market.feature_staleness_hours.shape == (8, 2, 2)
    assert market.borrow_available.sum() == 0
    assert np.all(market.mark_price > market.index_price)
    with pytest.raises(ValueError):
        market.close[0, 0] = 0.0
