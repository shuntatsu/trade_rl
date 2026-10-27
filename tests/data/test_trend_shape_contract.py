"""Synthetic-only regressions for the frozen causal trend-shape feature semantics."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set

_SHAPE_KINDS = (FeatureKind.LINEAR_REGRESSION_SLOPE, FeatureKind.TREND_R2)


def _shape(
    kind: FeatureKind,
    close: np.ndarray,
    *,
    present: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = close.size
    return calculate_feature_events(
        FeatureSpec(name=kind.value, kind=kind, lookback=24),
        open_price=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones(size, dtype=np.float64),
        funding_rate=np.zeros(size, dtype=np.float64),
        funding_available=np.zeros(size, dtype=np.bool_),
        row_present=np.ones(size, dtype=np.bool_) if present is None else present,
        active=np.ones(size, dtype=np.bool_) if active is None else active,
    )


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
def test_24bar_shape_uses_24_closes_not_24_price_intervals(
    kind: FeatureKind,
) -> None:
    close = 100.0 * np.exp(0.003 * np.arange(26, dtype=np.float64))
    values, valid, source_start = _shape(kind, close)
    np.testing.assert_array_equal(np.flatnonzero(valid), [23, 24, 25])
    np.testing.assert_array_equal(source_start[23:], [0, 1, 2])
    np.testing.assert_array_equal(source_start[:23], -np.ones(23, dtype=np.int64))

    # A 24-interval log return at t=24 still needs close[0].
    # Shape at t=24 uses only close[1:25] (23 intervals).
    different_first_close = close.copy()
    different_first_close[0] *= 1.5
    after, after_valid, after_start = _shape(kind, different_first_close)
    assert not math.isclose(
        math.log(close[24] / close[0]),
        math.log(different_first_close[24] / different_first_close[0]),
    )
    np.testing.assert_array_equal(values[24:], after[24:])
    np.testing.assert_array_equal(valid, after_valid)
    np.testing.assert_array_equal(source_start, after_start)


def test_equal_24_interval_returns_can_have_distinct_shape() -> None:
    x = np.arange(25, dtype=np.float64)
    smooth_log = np.linspace(0.0, math.log(1.05), 25)
    choppy_log = smooth_log + 0.04 * np.sin(2.0 * math.pi * 3.0 * x / 24.0)
    smooth = 100.0 * np.exp(smooth_log)
    choppy = 100.0 * np.exp(choppy_log)

    assert math.log(smooth[-1] / smooth[0]) == pytest.approx(
        math.log(choppy[-1] / choppy[0]), abs=1e-14
    )
    smooth_slope, _, _ = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, smooth)
    choppy_slope, _, _ = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, choppy)
    smooth_r2, smooth_valid, _ = _shape(FeatureKind.TREND_R2, smooth)
    choppy_r2, choppy_valid, _ = _shape(FeatureKind.TREND_R2, choppy)

    assert smooth_valid[24] and choppy_valid[24]
    assert smooth_r2[24] == pytest.approx(1.0, abs=1e-10)
    assert choppy_r2[24] < 0.3
    assert smooth_slope[24] != pytest.approx(choppy_slope[24])


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
@pytest.mark.parametrize("missing_field", ("present", "active"))
def test_missing_row_invalidates_every_affected_shape_window(
    kind: FeatureKind, missing_field: str
) -> None:
    close = 100.0 * np.exp(0.002 * np.arange(60, dtype=np.float64))
    mask = np.ones(60, dtype=np.bool_)
    mask[17] = False
    kwargs = {missing_field: mask}
    values, valid, start = _shape(kind, close, **kwargs)
    np.testing.assert_array_equal(np.flatnonzero(valid), np.arange(41, 60))
    np.testing.assert_array_equal(start[:41], -np.ones(41, dtype=np.int64))
    assert start[41] == 18
    assert values[41] == pytest.approx(
        0.002 if kind is FeatureKind.LINEAR_REGRESSION_SLOPE else 1.0,
        abs=1e-10,
    )


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
def test_poisoning_future_rows_cannot_change_past_shape(kind: FeatureKind) -> None:
    x = np.arange(60, dtype=np.float64)
    close = 100.0 * np.exp(0.002 * x + 0.02 * np.sin(x / 4.0))
    poisoned = close.copy()
    poisoned[36:] *= 7.0
    base = _shape(kind, close)
    changed = _shape(kind, poisoned)
    for previous, following in zip(base, changed, strict=True):
        np.testing.assert_array_equal(previous[:36], following[:36])


def test_flat_and_signed_linear_trends_preserve_documented_convention() -> None:
    x = np.arange(30, dtype=np.float64)
    flat = np.full(30, 100.0)
    positive = 100.0 * np.exp(0.004 * x)
    negative = 100.0 * np.exp(-0.004 * x)

    for kind in _SHAPE_KINDS:
        flat_values, valid, start = _shape(kind, flat)
        assert valid[23:].all()
        assert start[23] == 0
        np.testing.assert_array_equal(flat_values[23:], np.zeros(7))

    up_slope = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, positive)[0][-1]
    down_slope = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, negative)[0][-1]
    assert up_slope == pytest.approx(0.004, abs=1e-12)
    assert down_slope == pytest.approx(-0.004, abs=1e-12)
    assert _shape(FeatureKind.TREND_R2, positive)[0][-1] == pytest.approx(1.0)
    assert _shape(FeatureKind.TREND_R2, negative)[0][-1] == pytest.approx(1.0)


def _built_shape(
    *,
    delayed_index: int | None = None,
    delayed_close_multiplier: float = 1.0,
    delayed_lag_ns: int = 3_600_000_000_000,
    delisted_at: datetime | None = None,
) -> MarketDataset:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        65
    ) * np.timedelta64(1, "h")
    phase = np.arange(65, dtype=np.float64)
    close = 100.0 * np.exp(0.003 * phase + 0.018 * np.sin(phase / 5.0))
    close[35] *= delayed_close_multiplier
    available_at = timestamps.copy()
    if delayed_index is not None:
        available_at[delayed_index] += np.timedelta64(delayed_lag_ns, "ns")
    raw = RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=close.copy(),
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=np.full(65, 1000.0),
        funding_rate=np.zeros(65),
        tradable=np.ones(65, dtype=np.bool_),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="slope24",
                    kind=FeatureKind.LINEAR_REGRESSION_SLOPE,
                    lookback=24,
                    max_staleness_hours=2.0,
                ),
                FeatureSpec(
                    name="r2_24",
                    kind=FeatureKind.TREND_R2,
                    lookback=24,
                    max_staleness_hours=2.0,
                ),
            ),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2026, 1, 1, tzinfo=UTC),
                delisted_at=delisted_at,
            ),
        ),
    )


@pytest.mark.parametrize("lag_ns", (1, 3_600_000_000_000))
def test_delayed_native_close_is_bounded_carry_not_fresh_shape(lag_ns: int) -> None:
    original = _built_shape(delayed_index=35, delayed_lag_ns=lag_ns)
    poisoned = _built_shape(
        delayed_index=35,
        delayed_close_multiplier=9.0,
        delayed_lag_ns=lag_ns,
    )
    assert original.feature_names == ("slope24", "r2_24")
    assert original.dataset_id != poisoned.dataset_id
    assert original.information_available is not None
    assert not original.information_available[35, 0]
    assert original.available_at is not None
    assert original.available_at[35, 0] == original.timestamps[35] + np.timedelta64(
        lag_ns, "ns"
    )

    # The delayed close is censored from new shape events, including the
    # 24-bar windows spanning it. No later bar can launder it into a feature.
    np.testing.assert_array_equal(original.features, poisoned.features)
    np.testing.assert_array_equal(
        original.feature_available, poisoned.feature_available
    )
    np.testing.assert_array_equal(
        original.resolved_array("feature_staleness"),
        poisoned.resolved_array("feature_staleness"),
    )
    np.testing.assert_array_equal(
        original.resolved_array("feature_staleness_hours"),
        poisoned.resolved_array("feature_staleness_hours"),
    )

    # The previous valid event stays observable at ages 1h and 2h.
    np.testing.assert_array_equal(
        original.features[35:37, 0],
        np.broadcast_to(original.features[34, 0], (2, 2)),
    )
    assert original.feature_available[35:37, 0].all()
    np.testing.assert_array_equal(
        original.resolved_array("feature_staleness")[35:38, 0],
        [[0.5, 0.5], [1.0, 1.0], [1.0, 1.0]],
    )
    assert not original.feature_available[37:59, 0].any()
    np.testing.assert_array_equal(
        original.features[37:59, 0], np.zeros((22, 2), dtype=np.float32)
    )

    # A genuinely new event requires 24 contiguous usable closes, starting
    # with row 36 after the unavailable row 35.
    assert original.feature_available[59, 0].all()
    np.testing.assert_array_equal(
        original.resolved_array("feature_staleness")[59, 0], [0.0, 0.0]
    )


def test_delisting_clears_stale_shape_carry_at_the_exact_row() -> None:
    delisted = _built_shape(
        delisted_at=datetime(2026, 1, 2, 11, tzinfo=UTC),
    )
    assert delisted.symbol_active[34, 0]
    assert not delisted.symbol_active[35:, 0].any()
    assert delisted.feature_available[34, 0].all()
    assert not delisted.feature_available[35:, 0].any()
    np.testing.assert_array_equal(
        delisted.features[35:, 0], np.zeros((30, 2), dtype=np.float32)
    )


def test_shape_dataset_is_invariant_to_symbol_order() -> None:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        65
    ) * np.timedelta64(1, "h")
    phase = np.arange(65, dtype=np.float64)
    closes = {
        "BTCUSDT": 100.0 * np.exp(0.003 * phase + 0.018 * np.sin(phase / 5.0)),
        "ETHUSDT": 210.0 * np.exp(-0.001 * phase + 0.025 * np.cos(phase / 7.0)),
    }
    source = InMemoryMarketDataSource(
        {
            symbol: RawMarketSeries(
                timestamps=timestamps,
                open=close.copy(),
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=np.full(65, 1000.0),
                funding_rate=np.zeros(65),
                tradable=np.ones(65, dtype=np.bool_),
            )
            for symbol, close in closes.items()
        }
    )
    builder = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="slope24",
                    kind=FeatureKind.LINEAR_REGRESSION_SLOPE,
                    lookback=24,
                ),
                FeatureSpec(
                    name="r2_24",
                    kind=FeatureKind.TREND_R2,
                    lookback=24,
                ),
            ),
        )
    )

    def build(symbols: tuple[str, ...]) -> MarketDataset:
        return builder.build(
            source,
            tuple(
                InstrumentContract(
                    symbol=symbol,
                    listed_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
                for symbol in symbols
            ),
        )

    forward = build(("BTCUSDT", "ETHUSDT"))
    reverse = build(("ETHUSDT", "BTCUSDT"))
    assert forward.symbols == tuple(reversed(reverse.symbols))
    for first, symbol in enumerate(forward.symbols):
        second = reverse.symbols.index(symbol)
        np.testing.assert_array_equal(
            forward.features[:, first], reverse.features[:, second]
        )
        np.testing.assert_array_equal(
            forward.feature_available[:, first],
            reverse.feature_available[:, second],
        )
        np.testing.assert_array_equal(
            forward.resolved_array("feature_staleness")[:, first],
            reverse.resolved_array("feature_staleness")[:, second],
        )


def test_selected_feature_masks_can_change_forecast_training_rows() -> None:
    """A one-factor comparison needs an explicit matched training-row oracle."""
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        65
    ) * np.timedelta64(1, "h")
    phase = np.arange(65, dtype=np.float64)
    close = 100.0 * np.exp(0.002 * phase + 0.01 * np.sin(phase / 5.0))
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=close.copy(),
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=np.full(65, 1000.0),
        funding_rate=np.zeros(65),
        tradable=np.ones(65, dtype=np.bool_),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(name="return_1bar", kind=FeatureKind.LOG_RETURN),
                FeatureSpec(
                    name="slope24",
                    kind=FeatureKind.LINEAR_REGRESSION_SLOPE,
                    lookback=24,
                ),
                FeatureSpec(name="r2_24", kind=FeatureKind.TREND_R2, lookback=24),
            ),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
    )
    cutoff = timestamps[-1]
    baseline = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_cutoff=cutoff,
        horizon_hours=24,
    )
    candidate = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1, 2),
        fit_symbol_indices=(0,),
        fit_cutoff=cutoff,
        horizon_hours=24,
    )

    # The current fitter properly uses only selected feature masks. Thus
    # separately fitted arms can differ in training rows even with the same
    # Dataset, symbols, label horizon, cutoff, and hyperparameters.
    assert baseline.n_samples == 39
    assert candidate.n_samples == 17
    assert not np.array_equal(baseline.label_end_times, candidate.label_end_times)
    np.testing.assert_array_equal(
        baseline.label_end_times[-candidate.n_samples :],
        candidate.label_end_times,
    )
    np.testing.assert_array_equal(
        baseline.features[-candidate.n_samples :, 0],
        candidate.features[:, 0],
    )
    assert np.all(candidate.label_end_times < cutoff)
