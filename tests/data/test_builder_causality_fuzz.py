"""Builder-level causality fuzz over late rows, lifecycle and cross-asset kinds.

The universe has a reference symbol, a gapped symbol with late-published rows
(including one published many bars late), a later-listed symbol and a delisted
symbol. For each cut, decision rows ``t < cut`` of the full build must equal

* a prefix build whose raw series keep only rows with ``timestamp < cut``; and
* builds in which, for one symbol at a time, every row with
  ``available_at >= cut`` was mutated. This covers rows after the cut *and*
  late rows before it that are still unpublished at the cut, so no decision
  may observe an unpublished row directly, through a feature window or
  through ``available_at`` propagation.

Synthetic data only; no economic claim.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MACDPeriods,
    MarketBuildConfig,
    NormalizationMode,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries

_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_N = 160
_HOUR = np.timedelta64(1, "h")


def _symbol(
    seed: int,
    *,
    start: int = 0,
    stop: int = _N,
    gaps: tuple[int, ...] = (),
    delays: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    index = np.asarray([i for i in range(start, stop) if i not in gaps])
    count = len(index)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, count)))
    open_price = close * np.exp(rng.normal(0.0, 0.002, count))
    timestamps = np.datetime64(_T0.replace(tzinfo=None), "ns") + index * _HOUR
    available_at = timestamps.copy()
    for row, hours in (delays or {}).items():
        available_at[np.flatnonzero(index == row)[0]] += hours * _HOUR
    funding_available = index % 8 == 0
    return {
        "timestamps": timestamps,
        "available_at": available_at,
        "open": open_price,
        "high": np.maximum(open_price, close) * 1.003,
        "low": np.minimum(open_price, close) * 0.997,
        "close": close,
        "volume": rng.lognormal(5.0, 0.5, count),
        "funding_rate": np.where(funding_available, rng.normal(0.0, 1e-4, count), 0.0),
        "funding_available": funding_available,
    }


_SYMBOLS = {
    "BTC": _symbol(1),
    "ETH": _symbol(2, gaps=(40, 41, 90), delays={70: 2, 71: 2, 100: 15}),
    "NEW": _symbol(3, start=60, delays={80: 1}),
    "OLD": _symbol(4, stop=110),
}
_CONTRACTS = (
    InstrumentContract("BTC", listed_at=_T0),
    InstrumentContract("ETH", listed_at=_T0),
    InstrumentContract("NEW", listed_at=_T0 + timedelta(hours=60)),
    InstrumentContract("OLD", listed_at=_T0, delisted_at=_T0 + timedelta(hours=110)),
)
_FEATURES = (
    FeatureSpec(
        "ret1_z",
        FeatureKind.LOG_RETURN,
        normalization=NormalizationMode.ROLLING_ZSCORE,
        normalization_window=20,
        min_periods=5,
    ),
    FeatureSpec("rel", FeatureKind.RELATIVE_RETURN_TO_BTC),
    FeatureSpec(
        "corr", FeatureKind.ROLLING_CORRELATION_TO_BTC, lookback=12, min_periods=6
    ),
    FeatureSpec("beta", FeatureKind.ROLLING_BETA_TO_BTC, lookback=12, min_periods=6),
    FeatureSpec(
        "rank", FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK, lookback=6, min_periods=3
    ),
    FeatureSpec("disp", FeatureKind.CROSS_ASSET_DISPERSION),
    FeatureSpec("short_rel", FeatureKind.RELATIVE_RETURN_TO_BTC, max_staleness_hours=1),
    FeatureSpec(
        "rsi_z",
        FeatureKind.RSI,
        lookback=5,
        normalization=NormalizationMode.ROLLING_ZSCORE,
        normalization_window=20,
        min_periods=5,
    ),
    FeatureSpec("vol5", FeatureKind.REALIZED_VOLATILITY, lookback=5),
    FeatureSpec("ema", FeatureKind.EMA_SLOPE, lookback=6),
    FeatureSpec("vz", FeatureKind.VOLUME_ZSCORE, lookback=10, min_periods=3),
    FeatureSpec("obv", FeatureKind.OBV_SLOPE, lookback=9),
    FeatureSpec("obv_acc", FeatureKind.OBV_ACCELERATION, lookback=4),
    FeatureSpec("fz", FeatureKind.FUNDING_ZSCORE, lookback=5),
    FeatureSpec("fc", FeatureKind.FUNDING_CHANGE, lookback=5),
    FeatureSpec("fb", FeatureKind.FUNDING_BPS),
    FeatureSpec("macd", FeatureKind.MACD_HISTOGRAM, macd_periods=MACDPeriods(3, 7, 4)),
    FeatureSpec(
        "stoch", FeatureKind.STOCHASTIC_D, lookback=5, stochastic_smoothing_period=4
    ),
    FeatureSpec("atrc", FeatureKind.ATR_CHANGE, lookback=5),
)
_CONFIG = MarketBuildConfig(
    base_timeframe="1h", features=_FEATURES, cross_asset_reference_symbol="BTC"
)
_ARRAYS = (
    "features",
    "feature_available",
    "feature_staleness_hours",
    "feature_staleness",
    "global_features",
    "global_feature_available",
    "information_available",
    "tradable",
    "symbol_active",
)
_CUTS = (65, 71, 72, 73, 74, 81, 85, 101, 112, 116, 125)


def _raw(
    values: dict[str, np.ndarray],
    *,
    stop: np.datetime64 | None = None,
    mutate_from: np.datetime64 | None = None,
    seed: int = 0,
) -> RawMarketSeries:
    keep = (
        np.ones(len(values["timestamps"]), dtype=np.bool_)
        if stop is None
        else values["timestamps"] < stop
    )
    arrays = {key: np.asarray(value)[keep].copy() for key, value in values.items()}
    if mutate_from is not None:
        mask = arrays["available_at"] >= mutate_from
        scale = np.random.default_rng(seed).uniform(0.3, 3.0)
        for key in ("open", "high", "low", "close"):
            arrays[key][mask] *= scale
        arrays["high"][mask] *= 1.05
        arrays["low"][mask] *= 0.95
        arrays["volume"][mask] *= 7.0
        arrays["funding_rate"][mask] = np.where(
            arrays["funding_available"][mask], 1e-3, 0.0
        )
    return RawMarketSeries(
        timestamps=arrays["timestamps"],
        available_at=arrays["available_at"],
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        volume=arrays["volume"],
        funding_rate=arrays["funding_rate"],
        funding_available=arrays["funding_available"],
        tradable=np.ones(len(arrays["timestamps"]), dtype=np.bool_),
    )


def _build(
    *,
    stop: np.datetime64 | None = None,
    mutate: str | None = None,
    mutate_from: np.datetime64 | None = None,
) -> MarketDataset:
    source = InMemoryMarketDataSource(
        {
            symbol: _raw(
                values,
                stop=stop,
                mutate_from=mutate_from if symbol == mutate else None,
                seed=index,
            )
            for index, (symbol, values) in enumerate(_SYMBOLS.items())
        }
    )
    return MarketDatasetBuilder(_CONFIG).build(source, _CONTRACTS)


def _cut_time(cut: int) -> np.datetime64:
    return np.datetime64(_T0.replace(tzinfo=None), "ns") + cut * _HOUR


def _differences(full: MarketDataset, other: MarketDataset, cut: int) -> list[str]:
    problems = []
    for name in _ARRAYS:
        left = np.asarray(full.resolved_array(name))[:cut]
        right = np.asarray(other.resolved_array(name))[:cut]
        if left.shape != right.shape or not np.array_equal(left, right):
            location = (
                tuple(int(x) for x in np.argwhere(left != right)[0])
                if left.shape == right.shape
                else "shape"
            )
            problems.append(f"{name} first differs at {location}")
    return problems


@pytest.fixture(scope="module")
def full_build() -> MarketDataset:
    return _build()


@pytest.mark.parametrize("cut", _CUTS)
def test_builder_prefix_truncation_invariance(full_build: MarketDataset, cut: int):
    prefix = _build(stop=_cut_time(cut))
    assert _differences(full_build, prefix, cut) == []


@pytest.mark.parametrize("cut", _CUTS)
@pytest.mark.parametrize("symbol", tuple(_SYMBOLS))
def test_builder_unpublished_row_mutation_invariance(
    full_build: MarketDataset, cut: int, symbol: str
):
    mutated = _build(mutate=symbol, mutate_from=_cut_time(cut))
    assert _differences(full_build, mutated, cut) == []


def test_fuzz_exercises_published_late_rows(full_build: MarketDataset) -> None:
    # The fuzz is only meaningful if late rows are hidden at their own row
    # and later feed fresh feature events (no permanent hole).
    eth = full_build.symbols.index("ETH")
    assert full_build.information_available is not None
    assert full_build.feature_staleness_hours is not None
    assert not full_build.information_available[[70, 71, 100], eth].any()
    vol5 = full_build.feature_names.index("vol5")
    # The five-bar window ending at row 75 contains rows 70 and 71.
    assert full_build.feature_staleness_hours[75, eth, vol5] == 0.0
    # Row 100 is published at 115. Every EMA event after the row-90 gap
    # depends on it, so the last pre-row-100 event is carried until then.
    ema = full_build.feature_names.index("ema")
    assert full_build.feature_staleness_hours[114, eth, ema] == 15.0
    assert full_build.feature_staleness_hours[115, eth, ema] == 0.0
