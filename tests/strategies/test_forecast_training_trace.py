from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace


def market() -> MarketDataset:
    phase = np.arange(12, dtype=np.float64)
    close = np.stack((100 * np.exp(0.01 * phase), 200 * np.exp(-0.02 * phase)), axis=1)
    features = np.stack(
        (phase[:, None] + [0, 100], phase[:, None] ** 2 + [0, 500]), axis=-1
    )
    available = np.ones(features.shape, dtype=np.bool_)
    available[:3, 1] = False
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + phase.astype(int) * np.timedelta64(1, "h"),
        features=features.astype(np.float32),
        global_features=np.zeros((12, 1), dtype=np.float32),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full((12, 2), 1e6),
        funding_rate=np.zeros((12, 2)),
        tradable=np.ones((12, 2), dtype=np.bool_),
        feature_available=available,
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8760,
    )


def training(dataset: MarketDataset):
    return build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1),
        fit_cutoff=dataset.timestamps[8],
        horizon_hours=2,
    )


def test_trace_proves_each_pooled_row_symbol_and_real_endpoints() -> None:
    dataset = market()
    fitted = training(dataset)
    trace = fitted.trace
    assert isinstance(trace, ForecastTrainingTrace)
    pairs = [(0, row) for row in range(6)] + [(1, row) for row in range(3, 6)]
    assert trace.row_symbols == tuple(dataset.symbols[symbol] for symbol, _ in pairs)
    for field, offset in (
        ("start_times", 0),
        ("end_times", 2),
        ("start_available_at", 0),
        ("end_available_at", 2),
    ):
        np.testing.assert_array_equal(
            getattr(trace, field),
            [dataset.timestamps[row + offset] for _, row in pairs],
        )
    np.testing.assert_array_equal(
        trace.start_close, [dataset.close[row, symbol] for symbol, row in pairs]
    )
    np.testing.assert_array_equal(
        trace.end_close, [dataset.close[row + 2, symbol] for symbol, row in pairs]
    )
    np.testing.assert_array_equal(
        fitted.features, [dataset.features[row, symbol] for symbol, row in pairs]
    )
    np.testing.assert_allclose(
        fitted.labels, np.log(trace.end_close / trace.start_close)
    )
    np.testing.assert_array_equal(trace.label_available_times, trace.end_times)


def test_actual_source_maturity_and_trace_arrays_cannot_be_changed() -> None:
    trace = training(market()).trace
    assert trace is not None
    delayed = replace(trace, end_available_at=trace.end_times + np.timedelta64(15, "m"))
    np.testing.assert_array_equal(
        delayed.label_available_times, delayed.end_available_at
    )
    late_start = replace(
        trace, start_available_at=trace.end_times + np.timedelta64(1, "m")
    )
    np.testing.assert_array_equal(
        late_start.label_available_times, late_start.start_available_at
    )
    for field in (
        "start_times",
        "end_times",
        "start_available_at",
        "end_available_at",
        "start_close",
        "end_close",
        "label_available_times",
    ):
        value = getattr(delayed, field)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert delayed.digest != trace.digest


def test_scope_ignores_unused_future_but_binds_used_features() -> None:
    dataset = market()
    features, close = dataset.features.copy(), dataset.close.copy()
    features[8:] += 123
    close[8:] *= 7
    mutated = replace(
        dataset,
        dataset_id="8" * 64,
        features=features,
        open=close,
        high=close,
        low=close,
        close=close,
    )
    baseline, future = training(dataset), training(mutated)
    assert baseline.trace is not None and future.trace is not None
    assert baseline.trace.digest == future.trace.digest
    assert baseline.scope_digest == future.scope_digest
    assert (
        baseline.scope_digest
        != replace(baseline, features=baseline.features + 1).scope_digest
    )


@pytest.mark.parametrize("missing", [False, True])
def test_excluded_source_endpoint_has_no_phantom_trace_row(missing: bool) -> None:
    dataset = market()
    available_at = np.broadcast_to(dataset.timestamps[:, None], (12, 2)).copy()
    information = np.ones((12, 2), dtype=np.bool_)
    information[5, 0] = False
    if not missing:
        available_at[5, 0] = dataset.timestamps[8]
    fitted = training(
        replace(dataset, available_at=available_at, information_available=information)
    )
    trace = fitted.trace
    assert trace is not None
    btc_starts = trace.start_times[np.array(trace.row_symbols) == "BTCUSDT"]
    np.testing.assert_array_equal(btc_starts, dataset.timestamps[[0, 1, 2, 4]])
    assert fitted.n_samples == len(trace.row_symbols) == 7


def test_legacy_rows_without_trace_cannot_claim_scoped_identity() -> None:
    legacy = replace(training(market()), trace=None)
    assert legacy.n_samples == 9
    with pytest.raises(ValueError):
        _ = legacy.scope_digest


@pytest.mark.parametrize(
    "invalid", ["shape", "backdated", "reversed", "nat", "zero_price"]
)
def test_trace_rejects_invalid_endpoint_evidence(invalid: str) -> None:
    trace = training(market()).trace
    assert trace is not None
    changes = {
        "shape": {"start_close": trace.start_close[:-1]},
        "backdated": {"end_available_at": trace.end_times - np.timedelta64(1, "ns")},
        "reversed": {"end_times": trace.start_times},
        "nat": {
            "start_times": np.full(trace.start_times.shape, np.datetime64("NaT", "ns"))
        },
        "zero_price": {"start_close": np.zeros(trace.start_close.shape)},
    }
    with pytest.raises(ValueError):
        replace(trace, **changes[invalid])


@pytest.mark.parametrize("invalid", ["labels", "ends", "horizon"])
def test_training_rejects_trace_that_disagrees_with_labels(invalid: str) -> None:
    fitted = training(market())
    changes = {
        "labels": {"labels": fitted.labels + 0.25},
        "ends": {"label_end_times": fitted.label_end_times - np.timedelta64(1, "h")},
        "horizon": {"horizon_hours": 3},
    }
    with pytest.raises(ValueError):
        replace(fitted, **changes[invalid])
