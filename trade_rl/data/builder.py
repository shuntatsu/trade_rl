"""Deterministic causal construction of :class:`MarketDataset`."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import (
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.cross_asset_features import (
    CROSS_ASSET_FEATURE_KINDS,
    calculate_cross_asset_feature_events,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.identity import (
    MARKET_DATASET_IDENTITY_SCHEMA,
    content_and_arrays_digest,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.multitimeframe import align_native_feature
from trade_rl.data.source import (
    MarketDataSource,
    MultiTimeframeMarketDataSource,
    RawMarketSeries,
)

_NS_PER_HOUR = 3_600_000_000_000


def _utc_datetime64(value: datetime) -> np.datetime64:
    resolved = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(resolved, "ns")


def _contiguous_window(mask: np.ndarray, start: int, stop: int) -> bool:
    return bool(np.all(mask[start:stop]))


def _carry_feature(
    event_values: np.ndarray,
    event_valid: np.ndarray,
    active: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(event_values, dtype=np.float64)
    available = np.zeros_like(event_valid, dtype=np.bool_)
    age = np.full_like(event_values, max_staleness_hours, dtype=np.float64)
    staleness = np.ones_like(event_values, dtype=np.float64)
    timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    last_index: int | None = None
    last_value = 0.0
    for index in range(len(event_values)):
        if not active[index]:
            last_index = None
            last_value = 0.0
            continue
        if event_valid[index]:
            last_index = index
            last_value = float(event_values[index])
        if last_index is None:
            continue
        age_hours = float(timestamp_ns[index] - timestamp_ns[last_index]) / _NS_PER_HOUR
        normalized = min(age_hours / max_staleness_hours, 1.0)
        age[index] = age_hours
        staleness[index] = normalized
        if age_hours <= max_staleness_hours + 1e-12:
            values[index] = last_value
            available[index] = True
    return values, available, age, staleness


def _carry_aged_feature(
    event_values: np.ndarray,
    event_valid: np.ndarray,
    event_age_hours: np.ndarray,
    active: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carry derived events while retaining the native source event age."""

    values = np.zeros_like(event_values, dtype=np.float64)
    available = np.zeros_like(event_valid, dtype=np.bool_)
    age = np.full_like(event_values, max_staleness_hours, dtype=np.float64)
    staleness = np.ones_like(event_values, dtype=np.float64)
    timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    last_index: int | None = None
    last_value = 0.0
    initial_age = 0.0
    for index in range(len(event_values)):
        if not active[index]:
            last_index = None
            continue
        if event_valid[index]:
            last_index = index
            last_value = float(event_values[index])
            initial_age = float(event_age_hours[index])
        if last_index is None:
            continue
        elapsed = float(timestamp_ns[index] - timestamp_ns[last_index]) / _NS_PER_HOUR
        age_hours = initial_age + elapsed
        age[index] = age_hours
        staleness[index] = min(age_hours / max_staleness_hours, 1.0)
        if age_hours <= max_staleness_hours + 1e-12:
            values[index] = last_value
            available[index] = True
    return values, available, age, staleness


def _regular_clock(
    series: tuple[RawMarketSeries, ...],
    *,
    step_ns: int,
) -> np.ndarray:
    first = min(
        int(value.timestamps[0].astype("datetime64[ns]").astype(np.int64))
        for value in series
    )
    last = max(
        int(value.timestamps[-1].astype("datetime64[ns]").astype(np.int64))
        for value in series
    )
    if last <= first:
        raise ValueError("market source does not contain a usable time range")
    count, remainder = divmod(last - first, step_ns)
    if remainder != 0:
        raise ValueError("market source endpoints do not align to base timeframe")
    return (first + np.arange(count + 1, dtype=np.int64) * step_ns).astype(
        "datetime64[ns]"
    )


def _session_clock(series: tuple[RawMarketSeries, ...]) -> np.ndarray:
    values = np.concatenate(
        tuple(item.timestamps.astype("datetime64[ns]") for item in series)
    )
    timestamps = np.unique(values)
    if timestamps.size < 3:
        raise ValueError("market source does not contain a usable session range")
    return timestamps


def _align_series(
    raw: RawMarketSeries,
    timestamps: np.ndarray,
    *,
    step_ns: int | None,
) -> dict[str, np.ndarray]:
    n_bars = len(timestamps)
    raw_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    clock_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    if step_ns is None:
        indices = np.searchsorted(clock_ns, raw_ns)
        if np.any(indices >= n_bars) or np.any(clock_ns[indices] != raw_ns):
            raise ValueError("raw timestamps fall outside the resolved session clock")
    else:
        first = int(clock_ns[0])
        offsets = raw_ns - first
        if np.any(offsets < 0) or np.any(offsets % step_ns != 0):
            raise ValueError(
                "raw timestamps do not align to the configured base timeframe"
            )
        indices = offsets // step_ns
        if np.any(indices >= n_bars):
            raise ValueError("raw timestamps fall outside the resolved market clock")

    result = {
        "open": np.ones(n_bars, dtype=np.float64),
        "high": np.ones(n_bars, dtype=np.float64),
        "low": np.ones(n_bars, dtype=np.float64),
        "close": np.ones(n_bars, dtype=np.float64),
        "volume": np.zeros(n_bars, dtype=np.float64),
        "funding_rate": np.zeros(n_bars, dtype=np.float64),
        "tradable": np.zeros(n_bars, dtype=np.bool_),
        "funding_available": np.zeros(n_bars, dtype=np.bool_),
        "funding_event_count": np.zeros(n_bars, dtype=np.int32),
        "row_present": np.zeros(n_bars, dtype=np.bool_),
        "information_available": np.zeros(n_bars, dtype=np.bool_),
        "available_at": timestamps.copy(),
    }
    for field_name in ("open", "high", "low", "close", "volume", "funding_rate"):
        result[field_name][indices] = getattr(raw, field_name)
    result["tradable"][indices] = raw.tradable
    assert raw.funding_available is not None
    assert raw.available_at is not None
    result["funding_available"][indices] = raw.funding_available
    assert raw.funding_event_count is not None
    result["funding_event_count"][indices] = raw.funding_event_count
    result["row_present"][indices] = True
    result["available_at"][indices] = raw.available_at
    result["information_available"][indices] = raw.available_at <= raw.timestamps

    last_close = 1.0
    has_close = False
    for index in range(n_bars):
        if result["row_present"][index]:
            last_close = float(result["close"][index])
            has_close = True
        elif has_close:
            result["open"][index] = last_close
            result["high"][index] = last_close
            result["low"][index] = last_close
            result["close"][index] = last_close
    return result


class MarketDatasetBuilder:
    """Build one causal and content-addressed market dataset."""

    def __init__(self, config: MarketBuildConfig) -> None:
        self.config = config

    def build(
        self,
        source: MarketDataSource,
        instruments: tuple[InstrumentContract, ...],
        *,
        identity_provenance: Mapping[str, object] | None = None,
    ) -> MarketDataset:
        if not instruments:
            raise ValueError("instruments must not be empty")
        symbols = tuple(contract.symbol for contract in instruments)
        if len(set(symbols)) != len(symbols):
            raise ValueError("instrument symbols must be unique")
        raw_series = tuple(source.load(symbol) for symbol in symbols)
        step_ns = int(round(self.config.bar_hours * _NS_PER_HOUR))
        if self.config.calendar_kind == "session_calendar":
            timestamps = _session_clock(raw_series)
            alignment_step: int | None = None
        else:
            timestamps = _regular_clock(raw_series, step_ns=step_ns)
            alignment_step = step_ns
        n_bars = len(timestamps)
        n_symbols = len(symbols)
        n_features = len(self.config.features)

        open_price = np.ones((n_bars, n_symbols), dtype=np.float64)
        high = np.ones_like(open_price)
        low = np.ones_like(open_price)
        close = np.ones_like(open_price)
        volume = np.zeros_like(open_price)
        funding_rate = np.zeros_like(open_price)
        row_present = np.zeros((n_bars, n_symbols), dtype=np.bool_)
        raw_tradable = np.zeros_like(row_present)
        funding_available = np.zeros_like(row_present)
        funding_event_count = np.zeros_like(row_present, dtype=np.int32)
        information_available = np.zeros_like(row_present)
        available_at = np.broadcast_to(timestamps[:, None], row_present.shape).copy()
        symbol_active = np.zeros_like(row_present)
        tick_size = np.zeros_like(open_price)
        lot_size = np.zeros_like(open_price)
        minimum_notional = np.zeros_like(open_price)

        for symbol_index, (contract, raw) in enumerate(zip(instruments, raw_series)):
            aligned = _align_series(raw, timestamps, step_ns=alignment_step)
            open_price[:, symbol_index] = aligned["open"]
            high[:, symbol_index] = aligned["high"]
            low[:, symbol_index] = aligned["low"]
            close[:, symbol_index] = aligned["close"]
            volume[:, symbol_index] = aligned["volume"]
            funding_rate[:, symbol_index] = aligned["funding_rate"]
            row_present[:, symbol_index] = aligned["row_present"]
            raw_tradable[:, symbol_index] = aligned["tradable"]
            funding_available[:, symbol_index] = aligned["funding_available"]
            funding_event_count[:, symbol_index] = aligned["funding_event_count"]
            information_available[:, symbol_index] = aligned["information_available"]
            available_at[:, symbol_index] = aligned["available_at"]

            listed = _utc_datetime64(contract.listed_at)
            active = timestamps >= listed
            if contract.delisted_at is not None:
                active &= timestamps < _utc_datetime64(contract.delisted_at)
            symbol_active[:, symbol_index] = active
            resolved_tick, resolved_lot, resolved_minimum = (
                contract.execution_rule_arrays(timestamps)
            )
            tick_size[:, symbol_index] = resolved_tick
            lot_size[:, symbol_index] = resolved_lot
            minimum_notional[:, symbol_index] = resolved_minimum

        information_available &= symbol_active & row_present
        causal_row_present = row_present & information_available
        tradable = symbol_active & row_present & raw_tradable
        features = np.zeros((n_bars, n_symbols, n_features), dtype=np.float64)
        feature_available = np.zeros_like(features, dtype=np.bool_)
        feature_age_hours = np.ones_like(features, dtype=np.float64)
        feature_staleness = np.ones_like(features, dtype=np.float64)
        native_cache: dict[tuple[str, str], RawMarketSeries] = {}
        for symbol_index, contract in enumerate(instruments):
            for feature_index, spec in enumerate(self.config.features):
                if spec.kind in CROSS_ASSET_FEATURE_KINDS:
                    continue
                native_timeframe = spec.resolved_timeframe(self.config.base_timeframe)
                if native_timeframe == self.config.base_timeframe:
                    event_values, event_valid, _ = calculate_feature_events(
                        spec,
                        open_price=open_price[:, symbol_index],
                        high=high[:, symbol_index],
                        low=low[:, symbol_index],
                        close=close[:, symbol_index],
                        volume=volume[:, symbol_index],
                        funding_rate=funding_rate[:, symbol_index],
                        funding_available=funding_available[:, symbol_index],
                        row_present=causal_row_present[:, symbol_index],
                        active=symbol_active[:, symbol_index],
                    )
                    values, available, age_hours, staleness = _carry_feature(
                        event_values,
                        event_valid,
                        symbol_active[:, symbol_index],
                        timestamps,
                        max_staleness_hours=spec.max_staleness_hours,
                    )
                else:
                    if not isinstance(source, MultiTimeframeMarketDataSource):
                        raise ValueError(
                            "native multi-timeframe features require a "
                            "MultiTimeframeMarketDataSource"
                        )
                    key = (contract.symbol, native_timeframe)
                    native = native_cache.get(key)
                    if native is None:
                        native = source.load_timeframe(
                            contract.symbol, native_timeframe
                        )
                        native_cache[key] = native
                    values, available, age_hours, staleness = align_native_feature(
                        spec,
                        native,
                        contract,
                        timestamps,
                        symbol_active[:, symbol_index],
                        timeframe=native_timeframe,
                    )
                features[:, symbol_index, feature_index] = values
                feature_available[:, symbol_index, feature_index] = available
                feature_age_hours[:, symbol_index, feature_index] = age_hours
                feature_staleness[:, symbol_index, feature_index] = staleness

        # Cross-asset channels are derived only after every symbol's native
        # one-bar return has been causally aligned to the base decision clock.
        # This prevents symbol-order dependence and keeps rolling windows on
        # completed native events rather than repeated carried values.
        for feature_index, spec in enumerate(self.config.features):
            if spec.kind not in CROSS_ASSET_FEATURE_KINDS:
                continue
            native_timeframe = spec.resolved_timeframe(self.config.base_timeframe)
            return_candidates = [
                index
                for index, candidate in enumerate(self.config.features)
                if candidate.kind.value == "log_return"
                and candidate.lookback == 1
                and candidate.resolved_timeframe(self.config.base_timeframe)
                == native_timeframe
            ]
            if len(return_candidates) != 1:
                raise ValueError(
                    f"cross-asset {native_timeframe} features require exactly one "
                    "one-bar log return channel"
                )
            return_index = return_candidates[0]
            reference_symbol = self.config.cross_asset_reference_symbol
            if reference_symbol is None:
                raise ValueError(
                    "cross-asset features require cross_asset_reference_symbol"
                )
            events = calculate_cross_asset_feature_events(
                spec,
                aligned_returns=features[:, :, return_index],
                return_available=feature_available[:, :, return_index],
                return_age_hours=feature_age_hours[:, :, return_index],
                symbols=symbols,
                reference_symbol=reference_symbol,
            )
            for symbol_index in range(n_symbols):
                values, available, age_hours, staleness = _carry_aged_feature(
                    events.values[:, symbol_index],
                    events.valid[:, symbol_index],
                    events.source_age_hours[:, symbol_index],
                    symbol_active[:, symbol_index],
                    timestamps,
                    max_staleness_hours=spec.max_staleness_hours,
                )
                features[:, symbol_index, feature_index] = values
                feature_available[:, symbol_index, feature_index] = available
                feature_age_hours[:, symbol_index, feature_index] = age_hours
                feature_staleness[:, symbol_index, feature_index] = staleness

        one_bar_returns = np.zeros((n_bars, n_symbols), dtype=np.float64)
        one_bar_available = np.zeros((n_bars, n_symbols), dtype=np.bool_)
        for symbol_index in range(n_symbols):
            for index in range(1, n_bars):
                mask = (
                    causal_row_present[:, symbol_index] & symbol_active[:, symbol_index]
                )
                if not _contiguous_window(mask, index - 1, index + 1):
                    continue
                one_bar_returns[index, symbol_index] = math.log(
                    close[index, symbol_index] / close[index - 1, symbol_index]
                )
                one_bar_available[index, symbol_index] = True

        global_features = np.zeros(
            (n_bars, len(self.config.global_feature_names)), dtype=np.float64
        )
        global_feature_available = np.ones_like(global_features, dtype=np.bool_)
        global_feature_staleness = np.zeros_like(global_features, dtype=np.float32)
        global_feature_missing_reason = np.zeros_like(global_features, dtype=np.int16)
        global_features[:, 0] = symbol_active.mean(axis=1)
        observable_tradable = tradable & information_available
        global_features[:, 1] = observable_tradable.mean(axis=1)
        for index in range(n_bars):
            sample = one_bar_returns[index, one_bar_available[index]]
            if sample.size:
                global_features[index, 2] = float(np.mean(sample))
                global_features[index, 3] = float(np.std(sample))
            else:
                global_feature_available[index, 2:4] = False
                global_feature_staleness[index, 2:4] = 1.0
                global_feature_missing_reason[index, 2:4] = 1

        features = features.astype(np.float32)
        global_features = global_features.astype(np.float32)
        feature_age_hours = feature_age_hours.astype(np.float32)
        feature_staleness = feature_staleness.astype(np.float32)
        feature_names = tuple(spec.name for spec in self.config.features)
        feature_config_digest = content_digest(self.config.canonical_payload())
        normalization_digest = content_and_arrays_digest(
            {
                "schema": "normalization_state_v1",
                "feature_config_digest": feature_config_digest,
            },
            (
                ("features", features),
                ("feature_available", feature_available),
                ("feature_age_hours", feature_age_hours),
                ("feature_staleness", feature_staleness),
            ),
        )
        metadata = {
            "schema": MARKET_DATASET_IDENTITY_SCHEMA,
            "config": self.config.canonical_payload(),
            "feature_config_digest": feature_config_digest,
            "normalization_digest": normalization_digest,
            "symbols": symbols,
            "instruments": tuple(
                contract.canonical_payload() for contract in instruments
            ),
            "feature_names": feature_names,
            "global_feature_names": self.config.global_feature_names,
        }
        if identity_provenance is not None:
            metadata["metadata_evidence"] = identity_provenance
        periods_per_year = (
            int(round(365.0 * 24.0 / self.config.bar_hours))
            if self.config.calendar_kind == "continuous_24_7"
            else int(self.config.session_periods_per_year or 0)
        )
        return MarketDataset(
            dataset_id="0" * 64,
            symbols=symbols,
            timestamps=timestamps,
            features=features,
            global_features=global_features,
            global_feature_available=global_feature_available,
            global_feature_staleness_hours=global_feature_staleness,
            global_feature_missing_reason=global_feature_missing_reason,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            funding_rate=funding_rate,
            funding_event_count=funding_event_count,
            tradable=tradable,
            symbol_active=symbol_active,
            information_available=information_available,
            available_at=available_at,
            feature_available=feature_available,
            feature_staleness_hours=feature_age_hours,
            feature_staleness=feature_staleness,
            feature_names=feature_names,
            global_feature_names=self.config.global_feature_names,
            volume_units=tuple(contract.volume_unit for contract in instruments),
            contract_multipliers=np.asarray(
                [contract.contract_multiplier for contract in instruments],
                dtype=np.float64,
            ),
            tick_size=tick_size,
            lot_size=lot_size,
            minimum_notional=minimum_notional,
            feature_config_digest=feature_config_digest,
            normalization_digest=normalization_digest,
            periods_per_year=periods_per_year,
            calendar_kind=self.config.calendar_kind,
            nominal_bar_hours=self.config.bar_hours,
        ).with_content_identity(metadata)
