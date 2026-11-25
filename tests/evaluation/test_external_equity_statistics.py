"""Generated-path parity with the frozen historical metric method and independent oracles."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.evaluation.comparison.overlapping_blocks import (
    bootstrap_excess,
    moving_block_indices,
)
from trade_rl.evaluation.dev_realdata_study import metrics as historical
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.equity_path import (
    daily_equity,
    daily_returns,
    performance_summary,
)

DAY = 86_400_000_000_000


def test_daily_path_exact_historical_parity_and_known_drawdown() -> None:
    stamps = np.arange(24 * 40 + 1, dtype=np.int64) * (DAY // 24)
    values = 10_000.0 * (1.01 ** (np.arange(stamps.size) / 24))
    values[24 * 20 : 24 * 21] *= 0.5
    times, daily = daily_equity(stamps, values)
    expected_times, expected_daily = historical.daily_equity(stamps, values)
    assert np.array_equal(times, expected_times)
    assert np.array_equal(daily, expected_daily)
    assert np.array_equal(
        daily_returns(daily), historical.daily_returns(expected_daily)
    )
    summary = performance_summary(
        stamps,
        values,
        cagr_days_per_year=PROTOCOL.cagr_days_per_year,
        trading_days_per_year=PROTOCOL.trading_days_per_year,
    )
    assert summary == historical.performance_summary(stamps, values, PROTOCOL)
    assert summary["max_drawdown_pct"] > 48.0
    assert daily.shape == (41,)


@pytest.mark.parametrize(
    "n,block,draws,seed",
    [(60, 20, 8, 0), (13, 5, 31, 8), (3, 20, 7, 2), (181, 20, 2000, 0)],
)
def test_non_circular_indices_and_paired_interval_match_frozen_method(
    n, block, draws, seed
):
    indices = moving_block_indices(n, block=block, resamples=draws, seed=seed)
    assert np.array_equal(
        indices,
        historical.moving_block_indices(n, block=block, resamples=draws, seed=seed),
    )
    width = min(block, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - width + 1, size=(draws, -(-n // width)))
    independent = np.asarray(
        [
            [i for start in row for i in range(start, start + width)][:n]
            for row in starts
        ]
    )
    assert np.array_equal(indices, independent)
    excess = np.random.default_rng(42).normal(0.0005, 0.01, n)
    result = bootstrap_excess(
        excess,
        indices,
        trading_days_per_year=PROTOCOL.trading_days_per_year,
        alpha=PROTOCOL.bootstrap_alpha,
        comparisons=PROTOCOL.bonferroni_comparisons,
    )
    old = historical.bootstrap_excess(excess, indices, PROTOCOL)
    assert result.to_mapping() == old.to_mapping()
    assert (
        result.bonferroni_ci_low
        <= result.ci_low
        <= result.ci_high
        <= result.bonferroni_ci_high
    )


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_block_counts_refused(bad):
    with pytest.raises(ValueError):
        moving_block_indices(20, block=bad, resamples=3, seed=0)
    with pytest.raises(ValueError):
        moving_block_indices(20, block=3, resamples=bad, seed=0)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0, 1, -1]]),
        np.array([[0, 1, 3]]),
        np.array([[0.0, 1.0, 2.0]]),
        np.empty((0, 3), dtype=np.int64),
    ],
)
def test_invalid_resample_matrix_refused(bad):
    with pytest.raises(ValueError):
        bootstrap_excess(
            np.array([0.01, 0.02, 0.03]),
            bad,
            trading_days_per_year=365,
            alpha=0.05,
            comparisons=2,
        )


def test_invalid_paths_and_statistical_parameters_refused():
    with pytest.raises(ValueError):
        daily_equity(np.array([0, DAY, DAY, 2 * DAY], dtype=np.int64), np.ones(4))
    with pytest.raises(ValueError):
        daily_returns(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        daily_equity(
            np.array([0, DAY, 2 * DAY], dtype=np.int64), np.array([1.0, np.nan, 2.0])
        )
    idx = moving_block_indices(3, block=2, resamples=3, seed=0)
    for kwargs in (
        dict(alpha=0.0),
        dict(alpha=1.0),
        dict(comparisons=True),
        dict(comparisons=0),
        dict(trading_days_per_year=0),
    ):
        params = dict(trading_days_per_year=365, alpha=0.05, comparisons=2)
        params.update(kwargs)
        with pytest.raises(ValueError):
            bootstrap_excess(np.array([0.01, 0.02, 0.03]), idx, **params)


def test_nonstandard_confidence_never_emits_misnamed_legacy_fields() -> None:
    idx = moving_block_indices(3, block=2, resamples=10, seed=0)
    result = bootstrap_excess(
        np.array([0.001, 0.002, 0.003]),
        idx,
        trading_days_per_year=365,
        alpha=0.1,
        comparisons=3,
    )
    assert result.alpha == 0.1 and result.comparisons == 3
    with pytest.raises(ValueError, match="ci95"):
        result.to_mapping()


def test_daily_path_crosses_years_without_interpolating() -> None:
    stamps = np.array(
        ["2023-12-29", "2023-12-31", "2024-01-01", "2024-01-03"], dtype="datetime64[ns]"
    ).astype(np.int64)
    path = np.array([100.0, 105.0, 110.0, 120.0])
    ts, values = daily_equity(stamps, path)
    assert np.array_equal(ts, stamps)
    assert np.array_equal(values, path)
    summary = performance_summary(
        stamps,
        path,
        cagr_days_per_year=PROTOCOL.cagr_days_per_year,
        trading_days_per_year=PROTOCOL.trading_days_per_year,
    )
    assert summary == historical.performance_summary(stamps, path, PROTOCOL)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(3, dtype=bool),
        np.array([1 + 2j, 2 + 0j, 3 + 1j]),
        np.array(["1", "2", "3"]),
    ],
)
def test_nonreal_arrays_are_not_silently_coerced(bad) -> None:
    with pytest.raises(ValueError, match="real numeric"):
        daily_equity(np.arange(3, dtype=np.int64) * DAY, bad)
    with pytest.raises(ValueError, match="real numeric"):
        bootstrap_excess(
            bad,
            moving_block_indices(3, block=2, resamples=3, seed=0),
            trading_days_per_year=365,
            alpha=0.05,
            comparisons=2,
        )


@pytest.mark.parametrize("kind", ["masked", "buffer", "opaque"])
@pytest.mark.parametrize("field", ["timestamp", "equity", "excess", "indices"])
def test_public_statistics_refuse_unverifiable_array_owners(kind, field):
    def damaged(value):
        value = np.ascontiguousarray(value)
        mask = np.zeros(value.shape, dtype=bool)
        mask.flat[0] = True
        masked = np.ma.array(value, mask=mask)
        if kind == "masked":
            return masked
        if kind == "buffer":
            return np.frombuffer(memoryview(masked), dtype=value.dtype).reshape(
                value.shape
            )
        return np.lib.stride_tricks.as_strided(
            masked, shape=value.shape, strides=value.strides
        )

    ts = np.arange(3, dtype=np.int64) * DAY
    equity = np.array([100.0, 110.0, 105.0])
    excess = np.array([0.01, -0.02, 0.03])
    idx = moving_block_indices(3, block=2, resamples=3, seed=0)
    with pytest.raises(ValueError, match="native array"):
        if field == "timestamp":
            daily_equity(damaged(ts), equity)
        elif field == "equity":
            daily_equity(ts, damaged(equity))
        else:
            bootstrap_excess(
                damaged(excess) if field == "excess" else excess,
                damaged(idx) if field == "indices" else idx,
                trading_days_per_year=365,
                alpha=0.05,
                comparisons=2,
            )


def test_public_statistics_preserve_plain_slices_and_raw_buffer_arrays():
    ts = np.arange(6, dtype=np.int64)[::2] * DAY
    values = np.array([100.0, 110.0, 105.0])
    for path in (values.copy(), np.frombuffer(values.tobytes(), dtype=np.float64)):
        assert np.array_equal(daily_equity(ts, path)[1], values)


@pytest.mark.parametrize("dtype", ["int64", "uint64", ">i8", ">u8"])
@pytest.mark.parametrize("order", ["fall", "rise"])
def test_public_equity_rejects_precision_losing_integer_path(dtype, order):
    from trade_rl.evaluation.equity_path import max_drawdown, per_year_returns

    # These two distinct integers collapse to the same binary64 value.
    marks = [2**53 + 1, 2**53] if order == "fall" else [2**53, 2**53 + 1]
    values = np.array([*marks, marks[-1]], dtype=dtype)
    ts = np.arange(3, dtype=np.int64) * DAY
    operations = (
        lambda: max_drawdown(values),
        lambda: daily_returns(values),
        lambda: daily_equity(ts, values),
        lambda: per_year_returns(ts, values),
        lambda: performance_summary(
            ts, values, cagr_days_per_year=365, trading_days_per_year=365
        ),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="binary64-compatible"):
            operation()


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        ">i8",
        ">u8",
    ],
)
def test_public_equity_accepts_positive_integer_dtype_boundaries(dtype):
    info = np.iinfo(dtype)
    peak = min(int(info.max), 2**53)
    path = np.array([peak, max(1, peak // 2), peak], dtype=dtype)
    ts = np.arange(3, dtype=np.int64) * DAY
    expected = np.array([peak, max(1, peak // 2), peak], dtype=np.float64)
    assert np.array_equal(daily_equity(ts, path)[1], expected)
    assert np.array_equal(daily_returns(path), daily_returns(expected))


def test_public_equity_refuses_wider_float_before_conversion():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform longdouble has no wider representation")
    values = np.array([1, 2, 3], dtype=np.longdouble)
    with pytest.raises(ValueError, match="binary64-compatible"):
        daily_returns(values)
