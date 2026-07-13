"""Deterministic causal construction of :class:`MarketDataset`."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import MarketDataSource, RawMarketSeries

_NS_PER_HOUR = 3_600_000_000_000


def _utc_datetime64(value: datetime) -> np.datetime64:
    resolved = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(resolved, "ns")


def _update_digest_array(
    digest: "hashlib._Hash",
    name: str,
    value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype("datetime64[ns]").astype("<i8")
    elif array.dtype == np.dtype(np.bool_):
        array = array.astype(np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        array = array.astype("<f8")
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype("<i8")
    descriptor = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _content_and_arrays_digest(
    metadata: object,
    arrays: Iterable[tuple[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(metadata))
    for name, array in arrays:
        _update_digest_array(digest, name, array)
    return digest.hexdigest()


def _rolling_zscore(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros_like(values, dtype=np.float64)
    result_valid = np.zeros_like(valid, dtype=np.bool_)
    for index in range(len(values)):
        if not valid[index]:
            continue
        start = max(0, index - window + 1)
        mask = valid[start : index + 1]
        sample = values[start : index + 1][mask]
        if sample.size < min_periods:
            continue
        std = float(np.std(sample))
        result[index] = (
            0.0 if std <= 1e-12 else (values[index] - float(np.mean(sample))) / std
        )
        result_valid[index] = True
    return result, result_valid


def _carry_feature(
    event_values: np.ndarray,
    event_valid: np.ndarray,
    active: np.ndarray,
    *,
    bar_hours: float,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(event_values, dtype=np.float64)
    available = np.zeros_like(event_valid, dtype=np.bool_)
    staleness = np.ones_like(event_values, dtype=np.float64)
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
        age_hours = (index - last_index) * bar_hours
        normalized = min(age_hours / max_staleness_hours, 1.0)
        staleness[index] = normalized
        if age_hours <= max_staleness_hours + 1e-12:
            values[index] = last_value
            available[index] = True
    return values, available, staleness


def _contiguous_window(mask: np.ndarray, start: int, stop: int) -> bool:
    return bool(np.all(mask[start:stop]))


def _feature_events(
    spec: FeatureSpec,
    *,
    close: np.ndarray,
    volume: np.ndarray,
    funding_rate: np.ndarray,
    funding_available: np.ndarray,
    row_present: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bars = len(close)
    values = np.zeros(n_bars, dtype=np.float64)
    valid = np.zeros(n_bars, dtype=np.bool_)

    if spec.kind is FeatureKind.FUNDING_BPS:
        valid = funding_available & row_present & active
        values[valid] = funding_rate[valid] * 10_000.0
    elif spec.kind is FeatureKind.LOG_RETURN:
        for index in range(spec.lookback, n_bars):
            start = index - spec.lookback
            if not _contiguous_window(row_present & active, start, index + 1):
                continue
            values[index] = math.log(close[index] / close[start])
            valid[index] = True
    elif spec.kind is FeatureKind.REALIZED_VOLATILITY:
        for index in range(spec.lookback, n_bars):
            start = index - spec.lookback
            if not _contiguous_window(row_present & active, start, index + 1):
                continue
            returns = np.diff(np.log(close[start : index + 1]))
            values[index] = float(np.sqrt(np.mean(np.square(returns))))
            valid[index] = True
    elif spec.kind is FeatureKind.VOLUME_ZSCORE:
        for index in range(n_bars):
            start = max(0, index - spec.lookback + 1)
            mask = (row_present & active)[start : index + 1]
            sample = volume[start : index + 1][mask]
            if (
                not row_present[index]
                or not active[index]
                or sample.size < spec.min_periods
            ):
                continue
            std = float(np.std(sample))
            values[index] = (
                0.0 if std <= 1e-12 else (volume[index] - float(np.mean(sample))) / std
            )
            valid[index] = True
    else:
        raise ValueError(f"unsupported feature kind: {spec.kind}")

    if spec.normalization is NormalizationMode.ROLLING_ZSCORE:
        return _rolling_zscore(
            values,
            valid,
            window=spec.normalization_window,
            min_periods=spec.min_periods,
        )
    return values, valid


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


def _align_series(
    raw: RawMarketSeries,
    timestamps: np.ndarray,
    *,
    step_ns: int,
) -> dict[str, np.ndarray]:
    n_bars = len(timestamps)
    first = int(timestamps[0].astype("datetime64[ns]").astype(np.int64))
    raw_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    offsets = raw_ns - first
    if np.any(offsets < 0) or np.any(offsets % step_ns != 0):
        raise ValueError("raw timestamps do not align to the configured base timeframe")
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
    ) -> MarketDataset:
        if not instruments:
            raise ValueError("instruments must not be empty")
        symbols = tuple(contract.symbol for contract in instruments)
        if len(set(symbols)) != len(symbols):
            raise ValueError("instrument symbols must be unique")
        raw_series = tuple(source.load(symbol) for symbol in symbols)
        step_ns = int(round(self.config.bar_hours * _NS_PER_HOUR))
        timestamps = _regular_clock(raw_series, step_ns=step_ns)
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
        information_available = np.zeros_like(row_present)
        available_at = np.broadcast_to(timestamps[:, None], row_present.shape).copy()
        symbol_active = np.zeros_like(row_present)

        for symbol_index, (contract, raw) in enumerate(zip(instruments, raw_series)):
            aligned = _align_series(raw, timestamps, step_ns=step_ns)
            open_price[:, symbol_index] = aligned["open"]
            high[:, symbol_index] = aligned["high"]
            low[:, symbol_index] = aligned["low"]
            close[:, symbol_index] = aligned["close"]
            volume[:, symbol_index] = aligned["volume"]
            funding_rate[:, symbol_index] = aligned["funding_rate"]
            row_present[:, symbol_index] = aligned["row_present"]
            raw_tradable[:, symbol_index] = aligned["tradable"]
            funding_available[:, symbol_index] = aligned["funding_available"]
            information_available[:, symbol_index] = aligned["information_available"]
            available_at[:, symbol_index] = aligned["available_at"]

            listed = _utc_datetime64(contract.listed_at)
            active = timestamps >= listed
            if contract.delisted_at is not None:
                active &= timestamps < _utc_datetime64(contract.delisted_at)
            symbol_active[:, symbol_index] = active

        information_available &= symbol_active & row_present
        causal_row_present = row_present & information_available
        tradable = symbol_active & row_present & raw_tradable
        features = np.zeros((n_bars, n_symbols, n_features), dtype=np.float64)
        feature_available = np.zeros_like(features, dtype=np.bool_)
        feature_staleness = np.ones_like(features, dtype=np.float64)
        for symbol_index in range(n_symbols):
            for feature_index, spec in enumerate(self.config.features):
                event_values, event_valid = _feature_events(
                    spec,
                    close=close[:, symbol_index],
                    volume=volume[:, symbol_index],
                    funding_rate=funding_rate[:, symbol_index],
                    funding_available=funding_available[:, symbol_index],
                    row_present=causal_row_present[:, symbol_index],
                    active=symbol_active[:, symbol_index],
                )
                values, available, staleness = _carry_feature(
                    event_values,
                    event_valid,
                    symbol_active[:, symbol_index],
                    bar_hours=self.config.bar_hours,
                    max_staleness_hours=spec.max_staleness_hours,
                )
                features[:, symbol_index, feature_index] = values
                feature_available[:, symbol_index, feature_index] = available
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
        global_features[:, 0] = symbol_active.mean(axis=1)
        global_features[:, 1] = tradable.mean(axis=1)
        for index in range(n_bars):
            sample = one_bar_returns[index, one_bar_available[index]]
            if sample.size:
                global_features[index, 2] = float(np.mean(sample))
                global_features[index, 3] = float(np.std(sample))

        feature_names = tuple(spec.name for spec in self.config.features)
        feature_config_digest = content_digest(self.config.canonical_payload())
        normalization_digest = _content_and_arrays_digest(
            {
                "schema": "normalization_state_v1",
                "feature_config_digest": feature_config_digest,
            },
            (
                ("features", features),
                ("feature_available", feature_available),
                ("feature_staleness", feature_staleness),
            ),
        )
        metadata = {
            "schema": "market_dataset_identity_v4",
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
        dataset_id = _content_and_arrays_digest(
            metadata,
            (
                ("timestamps", timestamps),
                ("available_at", available_at),
                ("information_available", information_available),
                ("features", features),
                ("global_features", global_features),
                ("open", open_price),
                ("high", high),
                ("low", low),
                ("close", close),
                ("volume", volume),
                ("funding_rate", funding_rate),
                ("tradable", tradable),
                ("symbol_active", symbol_active),
                ("feature_available", feature_available),
                ("feature_staleness", feature_staleness),
            ),
        )
        periods_per_year = int(round(365.0 * 24.0 / self.config.bar_hours))
        return MarketDataset(
            dataset_id=dataset_id,
            symbols=symbols,
            timestamps=timestamps,
            features=features,
            global_features=global_features,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            funding_rate=funding_rate,
            tradable=tradable,
            symbol_active=symbol_active,
            information_available=information_available,
            available_at=available_at,
            feature_available=feature_available,
            feature_staleness=feature_staleness,
            feature_names=feature_names,
            global_feature_names=self.config.global_feature_names,
            volume_units=tuple(contract.volume_unit for contract in instruments),
            contract_multipliers=np.asarray(
                [contract.contract_multiplier for contract in instruments],
                dtype=np.float64,
            ),
            feature_config_digest=feature_config_digest,
            normalization_digest=normalization_digest,
            periods_per_year=periods_per_year,
        )
