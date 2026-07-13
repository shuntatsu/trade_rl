"""Validated in-memory market dataset used by research and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.domain.common import require_sha256, require_unique_non_empty

_HOURS_PER_YEAR = 365.0 * 24.0
_NS_PER_HOUR = 3_600_000_000_000
_ZERO_DIGEST = "0" * 64


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
    execute at the open stored in row ``t + 1``. Policy observations must never read
    row ``t + 1``; execution is allowed to use it as realized simulation state.
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
    symbol_active: np.ndarray | None = None
    feature_staleness: np.ndarray | None = None
    information_available: np.ndarray | None = None
    volume_units: tuple[VolumeUnit, ...] = ()
    contract_multipliers: np.ndarray | None = None
    feature_config_digest: str = _ZERO_DIGEST
    normalization_digest: str = _ZERO_DIGEST
    _bar_duration_ns: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.feature_config_digest, field="feature_config_digest")
        require_sha256(self.normalization_digest, field="normalization_digest")
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
        if np.any(timestamp_ns == np.iinfo(np.int64).min):
            raise ValueError("timestamps must not contain NaT")
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

        price_shape = (n_bars, n_symbols)
        symbol_active_value = (
            np.ones(price_shape, dtype=np.bool_)
            if self.symbol_active is None
            else self.symbol_active
        )
        feature_staleness_value = (
            np.where(feature_available, 0.0, 1.0)
            if self.feature_staleness is None
            else self.feature_staleness
        )
        volume_units_value = (
            tuple(VolumeUnit.BASE_ASSET for _ in symbols)
            if not self.volume_units
            else self.volume_units
        )
        contract_multipliers_value = (
            np.ones(n_symbols, dtype=np.float64)
            if self.contract_multipliers is None
            else self.contract_multipliers
        )
        symbol_active = _readonly_array(
            symbol_active_value,
            dtype=np.dtype(np.bool_),
        )
        information_available_value = (
            symbol_active & tradable
            if self.information_available is None
            else self.information_available
        )
        information_available = _readonly_array(
            information_available_value,
            dtype=np.dtype(np.bool_),
        )
        feature_staleness = _readonly_array(
            feature_staleness_value,
            dtype=np.dtype(np.float32),
        )
        contract_multipliers = _readonly_array(
            contract_multipliers_value,
            dtype=np.dtype(np.float64),
        )
        volume_units = tuple(VolumeUnit(value) for value in volume_units_value)

        if features.shape != (n_bars, n_symbols, len(feature_names)):
            raise ValueError("features shape does not match bars, symbols, and names")
        if global_features.shape != (n_bars, len(global_names)):
            raise ValueError("global_features shape does not match bars and names")
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
        if symbol_active.shape != price_shape:
            raise ValueError("symbol_active shape does not match bars and symbols")
        if information_available.shape != price_shape:
            raise ValueError(
                "information_available shape does not match bars and symbols"
            )
        if feature_available.shape != features.shape:
            raise ValueError("feature_available shape does not match features")
        if feature_staleness.shape != features.shape:
            raise ValueError("feature_staleness shape does not match features")
        if len(volume_units) != n_symbols:
            raise ValueError("volume_units must match dataset symbols")
        if contract_multipliers.shape != (n_symbols,):
            raise ValueError("contract_multipliers must match dataset symbols")

        for field_name, array in (
            ("features", features),
            ("global_features", global_features),
            ("open", open_price),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
            ("funding_rate", funding),
            ("feature_staleness", feature_staleness),
            ("contract_multipliers", contract_multipliers),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{field_name} must contain only finite values")
        if any(np.any(price <= 0.0) for price in (open_price, high, low, close)):
            raise ValueError("OHLC prices must be strictly positive")
        if np.any(volume < 0.0):
            raise ValueError("volume must be non-negative")
        if np.any(contract_multipliers <= 0.0):
            raise ValueError("contract_multipliers must be strictly positive")
        if np.any(feature_staleness < 0.0) or np.any(feature_staleness > 1.0):
            raise ValueError("feature_staleness must be within [0, 1]")
        if np.any(tradable & ~symbol_active):
            raise ValueError("tradable cannot be true for an inactive symbol")
        if np.any(information_available & ~symbol_active):
            raise ValueError("information cannot be available for an inactive symbol")
        if np.any(feature_available & ~symbol_active[:, :, None]):
            raise ValueError("features cannot be available for an inactive symbol")
        if np.any((~feature_available) & (feature_staleness < 1.0)):
            raise ValueError("unavailable features must have maximum staleness")
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
        object.__setattr__(self, "volume_units", volume_units)
        object.__setattr__(self, "timestamps", timestamps.astype("datetime64[ns]"))
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "open", open_price)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "funding_rate", funding)
        object.__setattr__(self, "tradable", tradable)
        object.__setattr__(self, "symbol_active", symbol_active)
        object.__setattr__(self, "information_available", information_available)
        object.__setattr__(self, "feature_available", feature_available)
        object.__setattr__(self, "feature_staleness", feature_staleness)
        object.__setattr__(self, "contract_multipliers", contract_multipliers)
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

    def eligibility_mask(
        self,
        index: int,
        *,
        lookback: int = 0,
        require_features: bool = False,
    ) -> np.ndarray:
        """Return symbols causally eligible at ``index`` over a trailing window."""

        if not 0 <= index < self.n_bars:
            raise IndexError("eligibility index is outside the dataset")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 0:
            raise ValueError("lookback must be a non-negative integer")
        start = index - lookback
        if start < 0:
            raise ValueError("eligibility lookback exceeds available history")
        symbol_active = self.symbol_active
        information_available = self.information_available
        assert symbol_active is not None
        assert information_available is not None
        eligible = np.all(
            symbol_active[start : index + 1]
            & self.tradable[start : index + 1]
            & information_available[start : index + 1],
            axis=0,
        )
        if require_features:
            eligible &= np.all(
                self.feature_available[start : index + 1],
                axis=(0, 2),
            )
        return eligible

    def market_notional(
        self,
        index: int,
        prices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return executable quote-notional volume under explicit unit semantics."""

        if not 0 <= index < self.n_bars:
            raise IndexError("market notional index is outside the dataset")
        price_vector = (
            self.open[index] if prices is None else np.asarray(prices, dtype=np.float64)
        )
        price_vector = np.asarray(price_vector, dtype=np.float64).reshape(-1)
        if (
            price_vector.shape != (self.n_symbols,)
            or not np.isfinite(price_vector).all()
        ):
            raise ValueError("market notional prices do not match dataset symbols")
        if np.any(price_vector <= 0.0):
            raise ValueError("market notional prices must be strictly positive")

        contract_multipliers = self.contract_multipliers
        assert contract_multipliers is not None
        result = np.empty(self.n_symbols, dtype=np.float64)
        for symbol_index, unit in enumerate(self.volume_units):
            raw_volume = self.volume[index, symbol_index]
            if unit is VolumeUnit.QUOTE_NOTIONAL:
                result[symbol_index] = raw_volume
            elif unit is VolumeUnit.BASE_ASSET:
                result[symbol_index] = raw_volume * price_vector[symbol_index]
            else:
                result[symbol_index] = (
                    raw_volume
                    * contract_multipliers[symbol_index]
                    * price_vector[symbol_index]
                )
        return result
