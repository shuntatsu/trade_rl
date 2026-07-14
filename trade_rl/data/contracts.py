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


class NormalizationMode(StrEnum):
    NONE = "none"
    ROLLING_ZSCORE = "rolling_zscore"


@dataclass(frozen=True, slots=True)
class InstrumentContract:
    """Point-in-time instrument lifetime and execution-volume semantics."""

    symbol: str
    listed_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delisted_at: datetime | None = None
    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET
    contract_multiplier: float = 1.0

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

    def canonical_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "listed_at": self.listed_at.isoformat(),
            "delisted_at": (
                self.delisted_at.isoformat() if self.delisted_at is not None else None
            ),
            "volume_unit": self.volume_unit.value,
            "contract_multiplier": self.contract_multiplier,
        }


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One causal feature and its trailing normalization contract."""

    name: str
    kind: FeatureKind
    lookback: int = 1
    normalization: NormalizationMode = NormalizationMode.NONE
    normalization_window: int = 1
    min_periods: int = 1
    max_staleness_hours: float = 24.0

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="feature name")
        if isinstance(self.lookback, bool) or self.lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if (
            isinstance(self.normalization_window, bool)
            or self.normalization_window <= 0
        ):
            raise ValueError("normalization_window must be a positive integer")
        if isinstance(self.min_periods, bool) or self.min_periods <= 0:
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

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "lookback": self.lookback,
            "normalization": self.normalization.value,
            "normalization_window": self.normalization_window,
            "min_periods": self.min_periods,
            "max_staleness_hours": self.max_staleness_hours,
        }


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


@dataclass(frozen=True, slots=True)
class MarketBuildConfig:
    """Complete deterministic configuration for one market dataset build."""

    base_timeframe: str
    features: tuple[FeatureSpec, ...]
    calendar_kind: str = "continuous_24_7"
    session_periods_per_year: int | None = None
    schema_version: str = "market_build_v2"

    def __post_init__(self) -> None:
        require_non_empty(self.base_timeframe, field="base_timeframe")
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
        if self.base_timeframe not in _TIMEFRAME_HOURS:
            raise ValueError(f"unsupported base_timeframe: {self.base_timeframe}")
        if not self.features:
            raise ValueError("features must not be empty")
        names = tuple(spec.name for spec in self.features)
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        require_non_empty(self.schema_version, field="schema_version")

    @property
    def bar_hours(self) -> float:
        return _TIMEFRAME_HOURS[self.base_timeframe]

    @property
    def global_feature_names(self) -> tuple[str, ...]:
        return (
            "active_fraction",
            "tradable_fraction",
            "market_return_mean",
            "market_return_dispersion",
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "base_timeframe": self.base_timeframe,
            "bar_hours": self.bar_hours,
            "calendar_kind": self.calendar_kind,
            "session_periods_per_year": self.session_periods_per_year,
            "features": tuple(spec.canonical_payload() for spec in self.features),
            "global_feature_names": self.global_feature_names,
            "schema_version": self.schema_version,
        }
