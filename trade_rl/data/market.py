"""Validated in-memory market dataset used by research and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.domain.common import require_sha256, require_unique_non_empty

_HOURS_PER_YEAR = 365.0 * 24.0
_NS_PER_HOUR = 3_600_000_000_000


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Shape-checked regular-time market arrays bound to one content identity.

    Timestamps represent bar close times. A decision made from row ``t`` may first
    execute at the open stored in row ``t + 1``.
    """

    dataset_id: str
    symbols: tuple[str, ...]
    timestamps: np.ndarray
    features: np.ndarray
    global_features: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding_rate: np.ndarray
    tradable: np.ndarray
    feature_available: np.ndarray
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    periods_per_year: int
    _bar_duration_ns: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        symbols = require_unique_non_empty(self.symbols, field="symbols")
        feature_names = require_unique_non_empty(
            self.feature_names,
            field="feature_names",
        )
        global_names = require_unique_non_empty(
            self.global_feature_names,
            field="global_feature_names",
        )
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

        timestamps = _readonly_array(self.timestamps)
        features = _readonly_array(self.features, dtype=np.dtype(np.float32))
        global_features = _readonly_array(
            self.global_features,
            dtype=np.dtype(np.float32),
        )
        open_price = _readonly_array(self.open, dtype=np.dtype(np.float64))
        high = _readonly_array(self.high, dtype=np.dtype(np.float64))
        low = _readonly_array(self.low, dtype=np.dtype(np.float64))
        close = _readonly_array(self.close, dtype=np.dtype(np.float64))
        volume = _readonly_array(self.volume, dtype=np.dtype(np.float64))
        funding = _readonly_array(self.funding_rate, dtype=np.dtype(np.float64))
        tradable = _readonly_array(self.tradable, dtype=np.dtype(np.bool_))
        feature_available = _readonly_array(
            self.feature_available,
            dtype=np.dtype(np.bool_),
        )

        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if not np.issubdtype(timestamps.dtype, np.datetime64):
            raise ValueError("timestamps must use a datetime64 dtype")
        timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
        n_bars = timestamp_ns.shape[0]
        n_symbols = len(symbols)
        if n_bars < 3:
            raise ValueError("market dataset requires at least three bars")
        deltas = np.diff(timestamp_ns)
        if np.any(deltas <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not np.all(deltas == deltas[0]):
            raise ValueError("timestamps must use an exactly regular cadence")
        bar_duration_ns = int(deltas[0])
        bar_hours = bar_duration_ns / _NS_PER_HOUR
        expected_periods = _HOURS_PER_YEAR / bar_hours
        if not math.isclose(expected_periods, round(expected_periods), abs_tol=1e-9):
            raise ValueError("timestamp cadence does not resolve to annual periods")
        if self.periods_per_year != int(round(expected_periods)):
            raise ValueError("periods_per_year does not match timestamp cadence")

        if features.shape != (n_bars, n_symbols, len(feature_names)):
            raise ValueError("features shape does not match bars, symbols, and names")
        if global_features.shape != (n_bars, len(global_names)):
            raise ValueError("global_features shape does not match bars and names")
        price_shape = (n_bars, n_symbols)
        for field_name, array in (
            ("open", open_price),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
            ("funding_rate", funding),
        ):
            if array.shape != price_shape:
                raise ValueError(f"{field_name} shape does not match bars and symbols")
        if tradable.shape != price_shape:
            raise ValueError("tradable shape does not match bars and symbols")
        if feature_available.shape != features.shape:
            raise ValueError("feature_available shape does not match features")

        for field_name, array in (
            ("features", features),
            ("global_features", global_features),
            ("open", open_price),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
            ("funding_rate", funding),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{field_name} must contain only finite values")
        if any(np.any(price <= 0.0) for price in (open_price, high, low, close)):
            raise ValueError("OHLC prices must be strictly positive")
        if np.any(volume < 0.0):
            raise ValueError("volume must be non-negative")
        if (
            np.any(low > high)
            or np.any(low > open_price)
            or np.any(low > close)
            or np.any(high < open_price)
            or np.any(high < close)
        ):
            raise ValueError("OHLC values violate bar price invariants")

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "global_feature_names", global_names)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "open", open_price)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "funding_rate", funding)
        object.__setattr__(self, "tradable", tradable)
        object.__setattr__(self, "feature_available", feature_available)
        object.__setattr__(self, "_bar_duration_ns", bar_duration_ns)

    @property
    def n_bars(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def bar_hours(self) -> float:
        return self._bar_duration_ns / _NS_PER_HOUR

    def bars_for_hours(self, hours: float) -> int:
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        raw = hours / self.bar_hours
        resolved = int(round(raw))
        if resolved <= 0 or not math.isclose(raw, resolved, abs_tol=1e-9):
            raise ValueError("hours must resolve to an integral number of bars")
        return resolved
