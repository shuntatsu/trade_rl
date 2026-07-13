"""Validated in-memory market dataset used by research and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trade_rl.domain.common import require_sha256, require_unique_non_empty

_HOURS_PER_YEAR = 365.0 * 24.0
_NS_PER_HOUR = 3_600_000_000_000


class MarketCalendarKind(str, Enum):
    """Timestamp contract used by one market dataset."""

    CONTINUOUS = "continuous_24_7"
    SESSION = "session_calendar"


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


def _optional_array(
    value: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    default: float | bool,
    field_name: str,
) -> np.ndarray:
    if value is None:
        array = np.full(shape, default, dtype=dtype)
    else:
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"{field_name} shape does not match the dataset")
    return _readonly_array(array, dtype=dtype)


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Shape-checked market arrays bound to one content identity.

    Timestamps represent bar close times. A decision made from row ``t`` may first
    execute at the open stored in row ``t + 1``. Continuous datasets require an
    exactly regular cadence. Session datasets may contain overnight, weekend and
    holiday gaps while preserving strictly increasing timestamps.
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
    calendar_kind: MarketCalendarKind | str = MarketCalendarKind.CONTINUOUS
    nominal_bar_hours: float | None = None
    feature_staleness_hours: np.ndarray | None = None
    feature_missing_reason: np.ndarray | None = None
    global_feature_available: np.ndarray | None = None
    global_feature_staleness_hours: np.ndarray | None = None
    global_feature_missing_reason: np.ndarray | None = None
    fee_rate: np.ndarray | None = None
    maker_fee_rate: np.ndarray | None = None
    taker_fee_rate: np.ndarray | None = None
    spread_rate: np.ndarray | None = None
    max_participation_rate: np.ndarray | None = None
    minimum_notional: np.ndarray | None = None
    lot_size: np.ndarray | None = None
    tick_size: np.ndarray | None = None
    borrow_available: np.ndarray | None = None
    borrow_rate: np.ndarray | None = None
    funding_due: np.ndarray | None = None
    asset_active: np.ndarray | None = None
    buy_allowed: np.ndarray | None = None
    sell_allowed: np.ndarray | None = None
    mark_price: np.ndarray | None = None
    index_price: np.ndarray | None = None
    dividend: np.ndarray | None = None
    split_factor: np.ndarray | None = None
    delisting_recovery: np.ndarray | None = None
    cash_rate: np.ndarray | None = None
    _timestamp_ns: np.ndarray = field(init=False, repr=False)
    _bar_duration_ns: int | None = field(init=False, repr=False)
    _nominal_bar_hours: float = field(init=False, repr=False)

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
        try:
            calendar_kind = MarketCalendarKind(self.calendar_kind)
        except ValueError as error:
            raise ValueError("calendar_kind is not supported") from error

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

        bar_duration_ns: int | None
        if calendar_kind is MarketCalendarKind.CONTINUOUS:
            if not np.all(deltas == deltas[0]):
                raise ValueError(
                    "continuous timestamps must use an exactly regular cadence"
                )
            bar_duration_ns = int(deltas[0])
            nominal_bar_hours = bar_duration_ns / _NS_PER_HOUR
            expected_periods = _HOURS_PER_YEAR / nominal_bar_hours
            if not math.isclose(
                expected_periods,
                round(expected_periods),
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "timestamp cadence does not resolve to annual periods"
                )
            if self.periods_per_year != int(round(expected_periods)):
                raise ValueError(
                    "periods_per_year does not match timestamp cadence"
                )
        else:
            bar_duration_ns = None
            if self.nominal_bar_hours is None:
                nominal_bar_hours = float(np.min(deltas)) / _NS_PER_HOUR
            else:
                nominal_bar_hours = float(self.nominal_bar_hours)
            if not math.isfinite(nominal_bar_hours) or nominal_bar_hours <= 0.0:
                raise ValueError("nominal_bar_hours must be finite and positive")

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

        staleness = _optional_array(
            self.feature_staleness_hours,
            shape=features.shape,
            dtype=np.dtype(np.float32),
            default=0.0,
            field_name="feature_staleness_hours",
        )
        if not np.isfinite(staleness).all() or np.any(staleness < 0.0):
            raise ValueError(
                "feature_staleness_hours must be finite and non-negative"
            )
        feature_missing_reason = _optional_array(
            self.feature_missing_reason,
            shape=features.shape,
            dtype=np.dtype(np.int16),
            default=0,
            field_name="feature_missing_reason",
        )
        if np.any(feature_missing_reason < 0):
            raise ValueError("feature_missing_reason must be non-negative")
        global_available = _optional_array(
            self.global_feature_available,
            shape=global_features.shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="global_feature_available",
        )
        global_staleness = _optional_array(
            self.global_feature_staleness_hours,
            shape=global_features.shape,
            dtype=np.dtype(np.float32),
            default=0.0,
            field_name="global_feature_staleness_hours",
        )
        if not np.isfinite(global_staleness).all() or np.any(global_staleness < 0.0):
            raise ValueError(
                "global_feature_staleness_hours must be finite and non-negative"
            )
        global_missing_reason = _optional_array(
            self.global_feature_missing_reason,
            shape=global_features.shape,
            dtype=np.dtype(np.int16),
            default=0,
            field_name="global_feature_missing_reason",
        )
        if np.any(global_missing_reason < 0):
            raise ValueError("global_feature_missing_reason must be non-negative")
        fee_rate = _optional_array(
            self.fee_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="fee_rate",
        )
        maker_fee_rate = _optional_array(
            self.maker_fee_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="maker_fee_rate",
        )
        taker_fee_rate = _optional_array(
            self.taker_fee_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="taker_fee_rate",
        )
        spread_rate = _optional_array(
            self.spread_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="spread_rate",
        )
        participation = _optional_array(
            self.max_participation_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=1.0,
            field_name="max_participation_rate",
        )
        minimum_notional = _optional_array(
            self.minimum_notional,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="minimum_notional",
        )
        lot_size = _optional_array(
            self.lot_size,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="lot_size",
        )
        tick_size = _optional_array(
            self.tick_size,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="tick_size",
        )
        borrow_available = _optional_array(
            self.borrow_available,
            shape=price_shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="borrow_available",
        )
        borrow_rate = _optional_array(
            self.borrow_rate,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="borrow_rate",
        )
        funding_due = _optional_array(
            self.funding_due,
            shape=price_shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="funding_due",
        )
        asset_active = _optional_array(
            self.asset_active,
            shape=price_shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="asset_active",
        )
        if np.any(tradable & ~asset_active):
            raise ValueError("tradable symbols must be active")
        buy_allowed = _optional_array(
            self.buy_allowed,
            shape=price_shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="buy_allowed",
        )
        sell_allowed = _optional_array(
            self.sell_allowed,
            shape=price_shape,
            dtype=np.dtype(np.bool_),
            default=True,
            field_name="sell_allowed",
        )
        mark_price = _optional_array(
            self.mark_price,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=1.0,
            field_name="mark_price",
        ) if self.mark_price is not None else close
        index_price = _optional_array(
            self.index_price,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=1.0,
            field_name="index_price",
        ) if self.index_price is not None else close
        dividend = _optional_array(
            self.dividend,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="dividend",
        )
        split_factor = _optional_array(
            self.split_factor,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=1.0,
            field_name="split_factor",
        )
        delisting_recovery = _optional_array(
            self.delisting_recovery,
            shape=price_shape,
            dtype=np.dtype(np.float64),
            default=1.0,
            field_name="delisting_recovery",
        )
        cash_rate = _optional_array(
            self.cash_rate,
            shape=(n_bars,),
            dtype=np.dtype(np.float64),
            default=0.0,
            field_name="cash_rate",
        )
        for field_name, array in (
            ("fee_rate", fee_rate),
            ("maker_fee_rate", maker_fee_rate),
            ("taker_fee_rate", taker_fee_rate),
            ("spread_rate", spread_rate),
            ("minimum_notional", minimum_notional),
            ("lot_size", lot_size),
            ("tick_size", tick_size),
            ("borrow_rate", borrow_rate),
        ):
            if not np.isfinite(array).all() or np.any(array < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            not np.isfinite(participation).all()
            or np.any(participation <= 0.0)
            or np.any(participation > 1.0)
        ):
            raise ValueError("max_participation_rate must be within (0, 1]")
        if any(np.any(price <= 0.0) for price in (mark_price, index_price)):
            raise ValueError("mark_price and index_price must be strictly positive")
        if not np.isfinite(dividend).all():
            raise ValueError("dividend must be finite")
        if not np.isfinite(split_factor).all() or np.any(split_factor <= 0.0):
            raise ValueError("split_factor must be finite and positive")
        if (
            not np.isfinite(delisting_recovery).all()
            or np.any(delisting_recovery < 0.0)
            or np.any(delisting_recovery > 1.0)
        ):
            raise ValueError("delisting_recovery must be within [0, 1]")
        if not np.isfinite(cash_rate).all():
            raise ValueError("cash_rate must be finite")

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "global_feature_names", global_names)
        object.__setattr__(self, "calendar_kind", calendar_kind)
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
        object.__setattr__(self, "feature_staleness_hours", staleness)
        object.__setattr__(self, "feature_missing_reason", feature_missing_reason)
        object.__setattr__(self, "global_feature_available", global_available)
        object.__setattr__(self, "global_feature_staleness_hours", global_staleness)
        object.__setattr__(self, "global_feature_missing_reason", global_missing_reason)
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "maker_fee_rate", maker_fee_rate)
        object.__setattr__(self, "taker_fee_rate", taker_fee_rate)
        object.__setattr__(self, "spread_rate", spread_rate)
        object.__setattr__(self, "max_participation_rate", participation)
        object.__setattr__(self, "minimum_notional", minimum_notional)
        object.__setattr__(self, "lot_size", lot_size)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "borrow_available", borrow_available)
        object.__setattr__(self, "borrow_rate", borrow_rate)
        object.__setattr__(self, "funding_due", funding_due)
        object.__setattr__(self, "asset_active", asset_active)
        object.__setattr__(self, "buy_allowed", buy_allowed)
        object.__setattr__(self, "sell_allowed", sell_allowed)
        object.__setattr__(self, "mark_price", mark_price)
        object.__setattr__(self, "index_price", index_price)
        object.__setattr__(self, "dividend", dividend)
        object.__setattr__(self, "split_factor", split_factor)
        object.__setattr__(self, "delisting_recovery", delisting_recovery)
        object.__setattr__(self, "cash_rate", cash_rate)
        object.__setattr__(self, "_timestamp_ns", timestamp_ns)
        object.__setattr__(self, "_bar_duration_ns", bar_duration_ns)
        object.__setattr__(self, "_nominal_bar_hours", nominal_bar_hours)

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
    def regular_cadence(self) -> bool:
        return self._bar_duration_ns is not None

    @property
    def bar_hours(self) -> float:
        return self._nominal_bar_hours

    def elapsed_hours(self, start_index: int, end_index: int) -> float:
        if not 0 <= start_index < end_index < self.n_bars:
            raise ValueError("elapsed-hour indices are outside the dataset")
        delta_ns = self._timestamp_ns[end_index] - self._timestamp_ns[start_index]
        return float(delta_ns / _NS_PER_HOUR)

    def bars_for_hours(self, hours: float) -> int:
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        if not self.regular_cadence:
            raise ValueError("irregular session data does not have a fixed bar count")
        raw = hours / self.bar_hours
        resolved = int(round(raw))
        if resolved <= 0 or not math.isclose(raw, resolved, abs_tol=1e-9):
            raise ValueError("hours must resolve to an integral number of bars")
        return resolved

    def lookback_index(self, index: int, hours: float) -> int:
        if not 0 <= index < self.n_bars:
            raise ValueError("lookback index is outside the dataset")
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        target_ns = self._timestamp_ns[index] - int(round(hours * _NS_PER_HOUR))
        result = int(np.searchsorted(self._timestamp_ns, target_ns, side="right") - 1)
        if result < 0 or result >= index:
            raise ValueError("insufficient history for requested hours")
        return result

    def minimum_index_for_history(self, hours: float) -> int:
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        target_ns = self._timestamp_ns[0] + int(round(hours * _NS_PER_HOUR))
        result = int(np.searchsorted(self._timestamp_ns, target_ns, side="left"))
        if result >= self.n_bars:
            raise ValueError("dataset is too short for requested history")
        return result

    def forward_index(
        self,
        start_index: int,
        hours: float,
        *,
        maximum_index: int | None = None,
    ) -> int:
        if not 0 <= start_index < self.n_bars - 1:
            raise ValueError("forward start index is outside the dataset")
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("hours must be finite and positive")
        maximum = self.n_bars - 1 if maximum_index is None else maximum_index
        if not start_index < maximum < self.n_bars:
            raise ValueError("maximum_index is outside the forward range")
        target_ns = self._timestamp_ns[start_index] + int(round(hours * _NS_PER_HOUR))
        result = int(np.searchsorted(self._timestamp_ns, target_ns, side="left"))
        return min(maximum, max(start_index + 1, result))

    def bars_until(
        self,
        start_index: int,
        hours: float,
        *,
        maximum_index: int | None = None,
    ) -> int:
        return self.forward_index(
            start_index,
            hours,
            maximum_index=maximum_index,
        ) - start_index
