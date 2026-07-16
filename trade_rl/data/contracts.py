"""Immutable contracts for causal market data construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from trade_rl.domain.common import require_aware_datetime, require_non_empty


class VolumeUnit(StrEnum):
    """Meaning of the raw volume column for one instrument."""

    BASE_ASSET = "base_asset"
    QUOTE_NOTIONAL = "quote_notional"
    CONTRACTS = "contracts"


class FeatureKind(StrEnum):
    """Maintained causal feature implementations."""

    LOG_RETURN = "log_return"
    REALIZED_VOLATILITY = "realized_volatility"
    VOLUME_ZSCORE = "volume_zscore"
    FUNDING_BPS = "funding_bps"
    RSI = "rsi"
    MACD_LINE = "macd_line"
    MACD_SIGNAL = "macd_signal"
    MACD_HISTOGRAM = "macd_histogram"
    BOLLINGER_POSITION = "bollinger_position"
    BOLLINGER_BANDWIDTH = "bollinger_bandwidth"
    ATR_PCT = "atr_pct"
    ADX = "adx"
    STOCHASTIC_K = "stochastic_k"
    STOCHASTIC_D = "stochastic_d"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    OBV_SLOPE = "obv_slope"
    ICHIMOKU_TENKAN_DISTANCE = "ichimoku_tenkan_distance"
    ICHIMOKU_KIJUN_DISTANCE = "ichimoku_kijun_distance"
    ICHIMOKU_CLOUD_POSITION = "ichimoku_cloud_position"
    ICHIMOKU_CLOUD_THICKNESS = "ichimoku_cloud_thickness"
    BODY_RETURN = "body_return"
    HIGH_LOW_RANGE = "high_low_range"
    UPPER_WICK_RATIO = "upper_wick_ratio"
    LOWER_WICK_RATIO = "lower_wick_ratio"
    CLOSE_LOCATION_VALUE = "close_location_value"
    GAP_RETURN = "gap_return"
    VOLUME_LOG_CHANGE = "volume_log_change"
    PARKINSON_VOLATILITY = "parkinson_volatility"
    GARMAN_KLASS_VOLATILITY = "garman_klass_volatility"
    DOWNSIDE_VOLATILITY = "downside_volatility"
    UPSIDE_VOLATILITY = "upside_volatility"
    VOLATILITY_OF_VOLATILITY = "volatility_of_volatility"
    RANGE_EXPANSION = "range_expansion"
    ATR_CHANGE = "atr_change"
    PLUS_DI = "plus_di"
    MINUS_DI = "minus_di"
    DI_SPREAD = "di_spread"
    EMA_DISTANCE = "ema_distance"
    EMA_SLOPE = "ema_slope"
    LINEAR_REGRESSION_SLOPE = "linear_regression_slope"
    TREND_R2 = "trend_r2"
    MFI = "mfi"
    CMF = "cmf"
    VWAP_DISTANCE = "vwap_distance"
    PRICE_VOLUME_CORRELATION = "price_volume_correlation"
    OBV_CHANGE = "obv_change"
    OBV_ACCELERATION = "obv_acceleration"
    RELATIVE_VOLUME = "relative_volume"
    FUNDING_CHANGE = "funding_change"
    FUNDING_ZSCORE = "funding_zscore"
    RELATIVE_RETURN_TO_BTC = "relative_return_to_btc"
    ROLLING_CORRELATION_TO_BTC = "rolling_correlation_to_btc"
    ROLLING_BETA_TO_BTC = "rolling_beta_to_btc"
    CROSS_SECTIONAL_MOMENTUM_RANK = "cross_sectional_momentum_rank"
    CROSS_ASSET_DISPERSION = "cross_asset_dispersion"


class NormalizationMode(StrEnum):
    NONE = "none"
    ROLLING_ZSCORE = "rolling_zscore"


class FeatureAlignment(StrEnum):
    """Semantic placement of one feature on the decision-time axis."""

    UNSHIFTED_DECISION_TIME = "unshifted_decision_time"


_TIMEFRAME_HOURS = {
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
    "1d": 24.0,
}


def timeframe_hours(value: str) -> float:
    """Return the maintained duration for one canonical timeframe."""

    try:
        return _TIMEFRAME_HOURS[value]
    except KeyError as error:
        raise ValueError(f"unsupported timeframe: {value}") from error


@dataclass(frozen=True, slots=True)
class InstrumentContract:
    """Point-in-time instrument lifetime and execution-volume semantics."""

    symbol: str
    listed_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delisted_at: datetime | None = None
    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET
    contract_multiplier: float = 1.0
    tick_size: float = 0.0
    lot_size: float = 0.0
    minimum_notional: float = 0.0

    def __post_init__(self) -> None:
        require_non_empty(self.symbol, field="symbol")
        require_aware_datetime(self.listed_at, field="listed_at")
        if self.delisted_at is not None:
            require_aware_datetime(self.delisted_at, field="delisted_at")
            if self.delisted_at <= self.listed_at:
                raise ValueError("delisted_at must be later than listed_at")
        if (
            not math.isfinite(self.contract_multiplier)
            or self.contract_multiplier <= 0.0
        ):
            raise ValueError("contract_multiplier must be finite and positive")
        for field_name, value in (
            ("tick_size", self.tick_size),
            ("lot_size", self.lot_size),
            ("minimum_notional", self.minimum_notional),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "listed_at": self.listed_at.isoformat(),
            "delisted_at": (
                self.delisted_at.isoformat() if self.delisted_at is not None else None
            ),
            "volume_unit": self.volume_unit.value,
            "contract_multiplier": self.contract_multiplier,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "minimum_notional": self.minimum_notional,
        }


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One causal feature and its native-timeframe normalization contract."""

    name: str
    kind: FeatureKind
    lookback: int = 1
    normalization: NormalizationMode = NormalizationMode.NONE
    normalization_window: int = 1
    min_periods: int = 1
    max_staleness_hours: float = 24.0
    timeframe: str | None = None
    alignment: FeatureAlignment | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="feature name")
        if self.timeframe is not None:
            require_non_empty(self.timeframe, field="feature timeframe")
            timeframe_hours(self.timeframe)
        if self.alignment is not None:
            try:
                resolved_alignment = FeatureAlignment(
                    getattr(self.alignment, "value", self.alignment)
                )
            except ValueError as error:
                raise ValueError("feature alignment is unsupported") from error
            object.__setattr__(self, "alignment", resolved_alignment)
        if (
            isinstance(self.lookback, bool)
            or not isinstance(self.lookback, int)
            or self.lookback <= 0
        ):
            raise ValueError("lookback must be a positive integer")
        if (
            isinstance(self.normalization_window, bool)
            or not isinstance(self.normalization_window, int)
            or self.normalization_window <= 0
        ):
            raise ValueError("normalization_window must be a positive integer")
        if (
            isinstance(self.min_periods, bool)
            or not isinstance(self.min_periods, int)
            or self.min_periods <= 0
        ):
            raise ValueError("min_periods must be a positive integer")
        if (
            self.normalization is NormalizationMode.ROLLING_ZSCORE
            and self.min_periods > self.normalization_window
        ):
            raise ValueError("min_periods cannot exceed normalization_window")
        if (
            not math.isfinite(self.max_staleness_hours)
            or self.max_staleness_hours <= 0.0
        ):
            raise ValueError("max_staleness_hours must be finite and positive")

    def resolved_timeframe(self, base_timeframe: str) -> str:
        timeframe_hours(base_timeframe)
        return base_timeframe if self.timeframe is None else self.timeframe

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "kind": self.kind.value,
            "lookback": self.lookback,
            "normalization": self.normalization.value,
            "normalization_window": self.normalization_window,
            "min_periods": self.min_periods,
            "max_staleness_hours": self.max_staleness_hours,
        }
        if self.timeframe is not None:
            payload["timeframe"] = self.timeframe
        if self.alignment is not None:
            payload["alignment"] = self.alignment.value
        return payload


@dataclass(frozen=True, slots=True)
class MarketBuildConfig:
    """Complete deterministic configuration for one market dataset build."""

    base_timeframe: str
    features: tuple[FeatureSpec, ...]
    calendar_kind: str = "continuous_24_7"
    session_periods_per_year: int | None = None
    cross_asset_reference_symbol: str | None = None
    schema_version: str = "market_build_v2"

    def __post_init__(self) -> None:
        require_non_empty(self.base_timeframe, field="base_timeframe")
        timeframe_hours(self.base_timeframe)
        raw_calendar = getattr(self.calendar_kind, "value", self.calendar_kind)
        if not isinstance(raw_calendar, str) or raw_calendar not in {
            "continuous_24_7",
            "session_calendar",
        }:
            raise ValueError("calendar_kind is not supported")
        object.__setattr__(self, "calendar_kind", raw_calendar)
        if raw_calendar == "session_calendar":
            if (
                isinstance(self.session_periods_per_year, bool)
                or not isinstance(self.session_periods_per_year, int)
                or self.session_periods_per_year <= 0
            ):
                raise ValueError(
                    "session_periods_per_year must be a positive integer for session data"
                )
        elif self.session_periods_per_year is not None:
            raise ValueError(
                "session_periods_per_year is valid only for session calendar data"
            )
        if self.cross_asset_reference_symbol is not None:
            require_non_empty(
                self.cross_asset_reference_symbol, field="cross_asset_reference_symbol"
            )
        if not self.features:
            raise ValueError("features must not be empty")
        names = tuple(spec.name for spec in self.features)
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if any(spec.timeframe == self.base_timeframe for spec in self.features):
            raise ValueError(
                "base timeframe features must omit timeframe instead of repeating it"
            )
        require_non_empty(self.schema_version, field="schema_version")

    @property
    def bar_hours(self) -> float:
        return timeframe_hours(self.base_timeframe)

    @property
    def native_timeframes(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for spec in self.features:
            resolved = spec.resolved_timeframe(self.base_timeframe)
            if resolved not in ordered:
                ordered.append(resolved)
        return tuple(sorted(ordered, key=timeframe_hours))

    @property
    def global_feature_names(self) -> tuple[str, ...]:
        return (
            "active_fraction",
            "tradable_fraction",
            "market_return_mean",
            "market_return_dispersion",
        )

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "base_timeframe": self.base_timeframe,
            "bar_hours": self.bar_hours,
            "calendar_kind": self.calendar_kind,
            "session_periods_per_year": self.session_periods_per_year,
            "cross_asset_reference_symbol": self.cross_asset_reference_symbol,
            "features": tuple(spec.canonical_payload() for spec in self.features),
            "global_feature_names": self.global_feature_names,
            "schema_version": self.schema_version,
        }
        if any(spec.timeframe is not None for spec in self.features):
            payload["native_timeframes"] = self.native_timeframes
        return payload
