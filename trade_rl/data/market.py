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


def _matrix_or_default(
    value: np.ndarray | None,
    *,
    default: np.ndarray,
    dtype: np.dtype[np.generic],
) -> np.ndarray:
    return _readonly_array(default if value is None else value, dtype=dtype)


def _vector_or_default(
    value: np.ndarray | None,
    *,
    default: np.ndarray,
) -> np.ndarray:
    return _readonly_array(
        default if value is None else value, dtype=np.dtype(np.float64)
    )


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
    mark_price: np.ndarray | None = None
    index_price: np.ndarray | None = None
    taker_fee_rate: np.ndarray | None = None
    spread_rate: np.ndarray | None = None
    quantity_step: np.ndarray | None = None
    min_notional: np.ndarray | None = None
    maintenance_margin_rate: np.ndarray | None = None
    episode_sampling_weight: np.ndarray | None = None
    feature_age_hours: np.ndarray | None = None
    warmup_complete: np.ndarray | None = None
    market_data_age_hours: np.ndarray | None = None
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

        mark_price = _matrix_or_default(
            self.mark_price,
            default=close,
            dtype=np.dtype(np.float64),
        )
        index_price = _matrix_or_default(
            self.index_price,
            default=close,
            dtype=np.dtype(np.float64),
        )
        taker_fee_rate = (
            None
            if self.taker_fee_rate is None
            else _readonly_array(self.taker_fee_rate, dtype=np.dtype(np.float64))
        )
        spread_rate = (
            None
            if self.spread_rate is None
            else _readonly_array(self.spread_rate, dtype=np.dtype(np.float64))
        )
        quantity_step = _vector_or_default(
            self.quantity_step,
            default=np.zeros(n_symbols, dtype=np.float64),
        )
        min_notional = _vector_or_default(
            self.min_notional,
            default=np.zeros(n_symbols, dtype=np.float64),
        )
        maintenance_margin_rate = _vector_or_default(
            self.maintenance_margin_rate,
            default=np.full(n_symbols, 0.005, dtype=np.float64),
        )
        episode_sampling_weight = _vector_or_default(
            self.episode_sampling_weight,
            default=np.ones(n_bars, dtype=np.float64),
        )
        default_age = np.where(feature_available, 0.0, bar_hours)
        feature_age_hours = _matrix_or_default(
            self.feature_age_hours,
            default=default_age,
            dtype=np.dtype(np.float64),
        )
        warmup_complete = _readonly_array(
            np.ones(n_bars, dtype=np.bool_)
            if self.warmup_complete is None
            else self.warmup_complete,
            dtype=np.dtype(np.bool_),
        )
        market_data_age_hours = _vector_or_default(
            self.market_data_age_hours,
            default=np.zeros(n_bars, dtype=np.float64),
        )

        for field_name, array, expected_shape in (
            ("mark_price", mark_price, price_shape),
            ("index_price", index_price, price_shape),
            *(
                (("taker_fee_rate", taker_fee_rate, price_shape),)
                if taker_fee_rate is not None
                else ()
            ),
            *(
                (("spread_rate", spread_rate, price_shape),)
                if spread_rate is not None
                else ()
            ),
            ("quantity_step", quantity_step, (n_symbols,)),
            ("min_notional", min_notional, (n_symbols,)),
            ("maintenance_margin_rate", maintenance_margin_rate, (n_symbols,)),
            ("episode_sampling_weight", episode_sampling_weight, (n_bars,)),
            ("feature_age_hours", feature_age_hours, features.shape),
            ("warmup_complete", warmup_complete, (n_bars,)),
            ("market_data_age_hours", market_data_age_hours, (n_bars,)),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} has an invalid shape")

        for field_name, array in (
            ("features", features),
            ("global_features", global_features),
            ("open", open_price),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
            ("funding_rate", funding),
            ("mark_price", mark_price),
            ("index_price", index_price),
            *(
                (("taker_fee_rate", taker_fee_rate),)
                if taker_fee_rate is not None
                else ()
            ),
            *(("spread_rate", spread_rate),) if spread_rate is not None else (),
            ("quantity_step", quantity_step),
            ("min_notional", min_notional),
            ("maintenance_margin_rate", maintenance_margin_rate),
            ("episode_sampling_weight", episode_sampling_weight),
            ("feature_age_hours", feature_age_hours),
            ("market_data_age_hours", market_data_age_hours),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{field_name} must contain only finite values")
        if any(
            np.any(price <= 0.0)
            for price in (open_price, high, low, close, mark_price, index_price)
        ):
            raise ValueError("OHLC, mark, and index prices must be strictly positive")
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
        for field_name, rates in (
            *(
                (("taker_fee_rate", taker_fee_rate),)
                if taker_fee_rate is not None
                else ()
            ),
            *(("spread_rate", spread_rate),) if spread_rate is not None else (),
            ("maintenance_margin_rate", maintenance_margin_rate),
        ):
            if np.any(rates < 0.0) or np.any(rates >= 1.0):
                raise ValueError(f"{field_name} must be within [0, 1)")
        if np.any(quantity_step < 0.0):
            raise ValueError("quantity_step must be non-negative")
        if np.any(min_notional < 0.0):
            raise ValueError("min_notional must be non-negative")
        if np.any(episode_sampling_weight < 0.0) or not np.any(
            episode_sampling_weight > 0.0
        ):
            raise ValueError(
                "episode_sampling_weight must be non-negative with positive mass"
            )
        if np.any(feature_age_hours < 0.0):
            raise ValueError("feature_age_hours must be non-negative")
        if np.any(market_data_age_hours < 0.0):
            raise ValueError("market_data_age_hours must be non-negative")

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
        object.__setattr__(self, "mark_price", mark_price)
        object.__setattr__(self, "index_price", index_price)
        object.__setattr__(self, "taker_fee_rate", taker_fee_rate)
        object.__setattr__(self, "spread_rate", spread_rate)
        object.__setattr__(self, "quantity_step", quantity_step)
        object.__setattr__(self, "min_notional", min_notional)
        object.__setattr__(self, "maintenance_margin_rate", maintenance_margin_rate)
        object.__setattr__(self, "episode_sampling_weight", episode_sampling_weight)
        object.__setattr__(self, "feature_age_hours", feature_age_hours)
        object.__setattr__(self, "warmup_complete", warmup_complete)
        object.__setattr__(self, "market_data_age_hours", market_data_age_hours)
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

    @property
    def mark_prices(self) -> np.ndarray:
        assert self.mark_price is not None
        return self.mark_price

    @property
    def index_prices(self) -> np.ndarray:
        assert self.index_price is not None
        return self.index_price

    @property
    def quantity_steps(self) -> np.ndarray:
        assert self.quantity_step is not None
        return self.quantity_step

    @property
    def minimum_notionals(self) -> np.ndarray:
        assert self.min_notional is not None
        return self.min_notional

    @property
    def maintenance_margin_rates(self) -> np.ndarray:
        assert self.maintenance_margin_rate is not None
        return self.maintenance_margin_rate

    @property
    def sampling_weights(self) -> np.ndarray:
        assert self.episode_sampling_weight is not None
        return self.episode_sampling_weight

    @property
    def feature_ages(self) -> np.ndarray:
        assert self.feature_age_hours is not None
        return self.feature_age_hours

    @property
    def warmup_mask(self) -> np.ndarray:
        assert self.warmup_complete is not None
        return self.warmup_complete

    @property
    def market_data_ages(self) -> np.ndarray:
        assert self.market_data_age_hours is not None
        return self.market_data_age_hours

    def bars_for_hours(self, hours: float) -> int:
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        raw = hours / self.bar_hours
        resolved = int(round(raw))
        if resolved <= 0 or not math.isclose(raw, resolved, abs_tol=1e-9):
            raise ValueError("hours must resolve to an integral number of bars")
        return resolved
