"""Assemble one causal training dataset from maintained PostgreSQL market data."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import (
    FeatureSpec,
    InstrumentContract,
    InstrumentExecutionRule,
    VolumeUnit,
)
from trade_rl.data.economic_semantics import build_market_economic_semantics
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    IndicatorArtifactConnection,
    NativeIndicatorArtifactBundle,
    load_postgres_indicator_artifacts,
)

KLINE_TABLE: Final = "market_raw.binance_usds_m_klines_202101_202606"
FUNDING_TABLE: Final = "market_raw.binance_usds_m_funding_202101_202606"
BASE_TIMEFRAME: Final = "15m"
NATIVE_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")
POLICY_ASSET_IDENTITY_MODE: Final = "identity_free_v1"
_STEP_MS: Final = 15 * 60 * 1000
_MS_PER_HOUR: Final = 60 * 60 * 1000
_ZERO_DIGEST: Final = "0" * 64


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _ordered_unique(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def _epoch_ms(value: datetime) -> int:
    return int(round(value.timestamp() * 1000.0))


def _metadata_entry(
    metadata: Mapping[str, Mapping[str, object]], symbol: str
) -> Mapping[str, object]:
    entry = metadata.get(symbol)
    if not isinstance(entry, Mapping):
        raise ValueError(f"metadata {symbol} must be an object")
    return entry


def _metadata_number(
    metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
) -> float:
    entry = _metadata_entry(metadata, symbol)
    if field not in entry:
        raise ValueError(f"metadata {symbol}.{field} is required")
    raw = entry[field]
    if isinstance(raw, bool) or not isinstance(raw, str | int | float):
        raise ValueError(f"metadata {symbol}.{field} must be numeric")
    try:
        resolved = float(raw)
    except ValueError as error:
        raise ValueError(f"metadata {symbol}.{field} must be numeric") from error
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"metadata {symbol}.{field} must be finite and positive")
    return resolved


def _metadata_datetime(
    metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
) -> datetime:
    entry = _metadata_entry(metadata, symbol)
    raw = entry.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"metadata {symbol}.{field} is required")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"metadata {symbol}.{field} must be an ISO-8601 timestamp"
        ) from error
    return _aware_utc(parsed, field=f"metadata {symbol}.{field}")


def _metadata_optional_datetime(
    metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
) -> datetime | None:
    entry = _metadata_entry(metadata, symbol)
    raw = entry.get(field)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"metadata {symbol}.{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"metadata {symbol}.{field} must be an ISO-8601 timestamp"
        ) from error
    return _aware_utc(parsed, field=f"metadata {symbol}.{field}")


def _load_base_market(
    connection: IndicatorArtifactConnection,
    *,
    symbols: tuple[str, ...],
    start_ms: int,
    end_ms: int,
) -> dict[str, np.ndarray]:
    expected_rows, remainder = divmod(end_ms - start_ms, _STEP_MS)
    if remainder or expected_rows < 3:
        raise ValueError("dataset range must contain aligned 15-minute bars")
    fields = ("open", "high", "low", "close", "volume")
    matrices = {
        field: np.empty((expected_rows, len(symbols)), dtype=np.float64)
        for field in fields
    }
    timestamps_ms: np.ndarray | None = None
    with connection.cursor() as cursor:
        for symbol_index, symbol in enumerate(symbols):
            cursor.execute(
                f"""
                SELECT open_time_ms,
                       open::double precision, high::double precision,
                       low::double precision, close::double precision,
                       quote_volume::double precision
                FROM {KLINE_TABLE}
                WHERE symbol = %s AND interval = %s
                  AND open_time_ms >= %s AND open_time_ms < %s
                ORDER BY open_time_ms
                """,
                (symbol, BASE_TIMEFRAME, start_ms, end_ms),
            )
            rows = tuple(cursor.fetchall())
            if len(rows) != expected_rows:
                raise ValueError(
                    f"PostgreSQL Kline closure mismatch for {symbol}: "
                    f"expected {expected_rows}, observed {len(rows)}"
                )
            matrix = np.asarray(rows, dtype=np.float64)
            if matrix.shape != (expected_rows, 6) or not np.isfinite(matrix).all():
                raise ValueError(f"PostgreSQL Klines are invalid for {symbol}")
            open_ms = np.asarray(matrix[:, 0], dtype=np.int64)
            expected_clock = (
                start_ms + np.arange(expected_rows, dtype=np.int64) * _STEP_MS
            )
            if not np.array_equal(open_ms, expected_clock):
                raise ValueError(f"PostgreSQL Klines are not contiguous for {symbol}")
            event_ms = open_ms + _STEP_MS
            if timestamps_ms is None:
                timestamps_ms = event_ms
            elif not np.array_equal(timestamps_ms, event_ms):
                raise ValueError("PostgreSQL symbol clocks do not match")
            for column, field in enumerate(fields, start=1):
                matrices[field][:, symbol_index] = matrix[:, column]
    assert timestamps_ms is not None
    matrices["timestamps_ms"] = timestamps_ms
    return matrices


def _load_funding(
    connection: IndicatorArtifactConnection,
    *,
    symbols: tuple[str, ...],
    timestamps_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.zeros((len(timestamps_ms), len(symbols)), dtype=np.float64)
    counts = np.zeros((len(timestamps_ms), len(symbols)), dtype=np.int32)
    lower = int(timestamps_ms[0]) - _STEP_MS
    upper = int(timestamps_ms[-1])
    with connection.cursor() as cursor:
        for symbol_index, symbol in enumerate(symbols):
            cursor.execute(
                f"""
                SELECT calc_time_ms, last_funding_rate::double precision
                FROM {FUNDING_TABLE}
                WHERE symbol = %s AND calc_time_ms > %s AND calc_time_ms <= %s
                ORDER BY calc_time_ms
                """,
                (symbol, lower, upper),
            )
            previous: int | None = None
            for raw_time, raw_rate in cursor.fetchall():
                if (
                    isinstance(raw_time, bool)
                    or not isinstance(raw_time, int)
                    or isinstance(raw_rate, bool)
                    or not isinstance(raw_rate, int | float)
                ):
                    raise ValueError(f"PostgreSQL funding row is invalid for {symbol}")
                event = int(raw_time)
                rate = float(raw_rate)
                if previous is not None and event <= previous:
                    raise ValueError(
                        f"PostgreSQL funding times are not unique for {symbol}"
                    )
                previous = event
                if not math.isfinite(rate):
                    raise ValueError(f"PostgreSQL funding rate is invalid for {symbol}")
                index = int(np.searchsorted(timestamps_ms, event, side="left"))
                if index >= len(timestamps_ms):
                    continue
                if event <= int(timestamps_ms[index]) - _STEP_MS:
                    continue
                rates[index, symbol_index] += rate
                counts[index, symbol_index] += 1
    return rates, counts


def _feature_specs() -> tuple[FeatureSpec, ...]:
    return tuple(
        binance_multitimeframe_feature_specs(
            base_timeframe=BASE_TIMEFRAME,
            feature_timeframes=NATIVE_TIMEFRAMES[1:],
        )
    )


def _align_indicators(
    bundle: NativeIndicatorArtifactBundle,
    *,
    timestamps_ms: np.ndarray,
    symbol_vocabulary: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    str,
]:
    vocabulary = _ordered_unique(symbol_vocabulary, field="symbol_vocabulary")
    if not set(bundle.symbols) <= set(vocabulary):
        raise ValueError("indicator symbols must belong to symbol_vocabulary")
    specs = _feature_specs()
    spec_by_name = {spec.name: spec for spec in specs}
    feature_names = tuple(
        name
        for timeframe in NATIVE_TIMEFRAMES
        for name in bundle.get(bundle.symbols[0], timeframe).feature_names
    )
    if feature_names != tuple(spec.name for spec in specs):
        raise ValueError(
            "PostgreSQL indicator feature order differs from code contract"
        )
    shape = (len(timestamps_ms), len(bundle.symbols), len(feature_names))
    values = np.zeros(shape, dtype=np.float32)
    available = np.zeros(shape, dtype=np.bool_)
    age_hours = np.zeros(shape, dtype=np.float32)
    staleness = np.ones(shape, dtype=np.float32)

    offset = 0
    for timeframe in NATIVE_TIMEFRAMES:
        names = bundle.get(bundle.symbols[0], timeframe).feature_names
        for symbol_index, symbol in enumerate(bundle.symbols):
            artifact = bundle.get(symbol, timeframe)
            if artifact.feature_names != names:
                raise ValueError("indicator feature order differs between symbols")
            for local_index, name in enumerate(names):
                valid_rows = np.flatnonzero(artifact.available[:, local_index])
                if valid_rows.size == 0:
                    continue
                source_times = artifact.event_time_ms[valid_rows]
                positions = (
                    np.searchsorted(source_times, timestamps_ms, side="right") - 1
                )
                present = positions >= 0
                safe_positions = np.maximum(positions, 0)
                source_indices = valid_rows[safe_positions]
                ages = (
                    timestamps_ms.astype(np.float64)
                    - artifact.event_time_ms[source_indices].astype(np.float64)
                ) / _MS_PER_HOUR
                maximum = float(spec_by_name[name].max_staleness_hours)
                present &= ages >= -1e-12
                present &= ages <= maximum + 1e-12
                target = offset + local_index
                values[present, symbol_index, target] = artifact.values[
                    source_indices[present], local_index
                ]
                available[present, symbol_index, target] = True
                age_hours[:, symbol_index, target] = maximum
                age_hours[present, symbol_index, target] = np.asarray(
                    ages[present], dtype=np.float32
                )
                staleness[present, symbol_index, target] = np.asarray(
                    np.clip(ages[present] / maximum, 0.0, 1.0), dtype=np.float32
                )
        offset += len(names)

    feature_digest = content_digest(
        {
            "asset_identity_mode": POLICY_ASSET_IDENTITY_MODE,
            "feature_names": feature_names,
            "indicator_feature_config_digest": bundle.feature_config_digest,
            "schema_version": "postgres_indicator_alignment_v2",
        }
    )
    return values, available, age_hours, staleness, feature_names, feature_digest


def build_postgres_market_dataset(
    connection: IndicatorArtifactConnection,
    *,
    symbols: Sequence[str],
    symbol_vocabulary: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    metadata: Mapping[str, Mapping[str, object]],
    metadata_evidence_digest: str,
    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]]
    | None = None,
    indicator_bundle: NativeIndicatorArtifactBundle | None = None,
    slot_symbols: Sequence[str] | None = None,
    symbol_triplet_provenance: Mapping[str, object] | None = None,
) -> MarketDataset:
    """Build one identity-free three-slot dataset from selected market symbols."""

    selected = _ordered_unique(symbols, field="symbols")
    if len(selected) != 3:
        raise ValueError("dynamic training datasets require exactly three symbols")
    vocabulary = _ordered_unique(symbol_vocabulary, field="symbol_vocabulary")
    if not set(selected) <= set(vocabulary):
        raise ValueError("selected symbols must belong to symbol_vocabulary")
    resolved_slots = (
        selected
        if slot_symbols is None
        else _ordered_unique(slot_symbols, field="slot_symbols")
    )
    if len(resolved_slots) != len(selected):
        raise ValueError("slot_symbols must match the selected symbol count")
    start = _aware_utc(start_time, field="start_time")
    end = _aware_utc(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    start_ms = _epoch_ms(start)
    end_ms = _epoch_ms(end)
    if start_ms % _STEP_MS or end_ms % _STEP_MS:
        raise ValueError("dataset range must align to the 15-minute clock")
    bundle = indicator_bundle or load_postgres_indicator_artifacts(
        connection,
        symbols=selected,
        timeframes=NATIVE_TIMEFRAMES,
    )
    if bundle.symbols != selected or bundle.timeframes != NATIVE_TIMEFRAMES:
        raise ValueError("indicator bundle order does not match requested dataset")
    if start < bundle.start_time or end > bundle.end_time:
        raise ValueError("dataset range falls outside the indicator cache")

    raw = _load_base_market(
        connection,
        symbols=selected,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    timestamps_ms = raw.pop("timestamps_ms")
    funding, funding_counts = _load_funding(
        connection,
        symbols=selected,
        timestamps_ms=timestamps_ms,
    )
    (
        features,
        feature_available,
        feature_age,
        feature_staleness,
        feature_names,
        feature_config_digest,
    ) = _align_indicators(
        bundle,
        timestamps_ms=timestamps_ms,
        symbol_vocabulary=vocabulary,
    )
    n_bars = len(timestamps_ms)
    price_shape = (n_bars, len(selected))
    timestamps = timestamps_ms.astype("datetime64[ms]").astype("datetime64[ns]")
    available_at = np.broadcast_to(timestamps[:, None], price_shape).copy()
    if execution_rule_histories is not None:
        missing_histories = set(selected) - set(execution_rule_histories)
        unknown_histories = set(execution_rule_histories) - set(selected)
        if missing_histories or unknown_histories:
            raise ValueError(
                "PostgreSQL execution-rule histories must match selected symbols"
            )
    instruments = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=_metadata_datetime(metadata, symbol, "listed_at"),
            delisted_at=_metadata_optional_datetime(metadata, symbol, "delisted_at"),
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            tick_size=_metadata_number(metadata, symbol, "tick_size"),
            lot_size=_metadata_number(metadata, symbol, "lot_size"),
            minimum_notional=_metadata_number(metadata, symbol, "minimum_notional"),
            execution_rules=(
                ()
                if execution_rule_histories is None
                else tuple(execution_rule_histories[symbol])
            ),
        )
        for symbol in selected
    )

    economics = build_market_economic_semantics(
        timestamps=timestamps,
        instruments=instruments,
        row_present=np.ones(price_shape, dtype=np.bool_),
        raw_tradable=np.ones(price_shape, dtype=np.bool_),
        source_information_available=np.ones(price_shape, dtype=np.bool_),
        available_at=available_at,
        close=raw["close"],
        funding_event_count=funding_counts,
    )
    observable_features = economics.information_available[:, :, None]
    feature_available &= observable_features
    features = np.where(feature_available, features, np.float32(0.0))
    feature_age = np.where(feature_available, feature_age, np.float32(0.0))
    feature_staleness = np.where(feature_available, feature_staleness, np.float32(1.0))

    log_returns = np.zeros(price_shape, dtype=np.float64)
    return_available = np.zeros(price_shape, dtype=np.bool_)
    contiguous = (
        economics.information_available[1:] & economics.information_available[:-1]
    )
    candidate_returns = np.log(raw["close"][1:] / raw["close"][:-1])
    np.copyto(log_returns[1:], candidate_returns, where=contiguous)
    return_available[1:] = contiguous
    global_features = np.zeros((n_bars, 4), dtype=np.float32)
    global_features[:, 0] = economics.symbol_active.mean(axis=1)
    global_features[:, 1] = (economics.tradable & economics.information_available).mean(
        axis=1
    )
    global_available = np.ones((n_bars, 4), dtype=np.bool_)
    for index in range(n_bars):
        sample = log_returns[index, return_available[index]]
        if sample.size:
            global_features[index, 2] = float(np.mean(sample))
            global_features[index, 3] = float(np.std(sample))
        else:
            global_available[index, 2:] = False

    normalization_digest = content_and_arrays_digest(
        {
            "asset_identity_mode": POLICY_ASSET_IDENTITY_MODE,
            "feature_config_digest": feature_config_digest,
            "schema_version": "postgres_indicator_alignment_v2",
        },
        (
            ("features", features),
            ("feature_available", feature_available),
            ("feature_staleness", feature_staleness),
        ),
    )
    dataset = MarketDataset(
        dataset_id=_ZERO_DIGEST,
        symbols=resolved_slots,
        timestamps=timestamps,
        features=features,
        global_features=global_features,
        open=raw["open"],
        high=raw["high"],
        low=raw["low"],
        close=raw["close"],
        volume=raw["volume"],
        funding_rate=funding,
        funding_event_count=funding_counts,
        symbol_active=economics.symbol_active,
        asset_active=economics.asset_active,
        tradable=economics.tradable,
        information_available=economics.information_available,
        available_at=economics.available_at,
        fee_rate=economics.fee_rate,
        maker_fee_rate=economics.maker_fee_rate,
        taker_fee_rate=economics.taker_fee_rate,
        spread_rate=economics.spread_rate,
        max_participation_rate=economics.max_participation_rate,
        minimum_notional=economics.minimum_notional,
        lot_size=economics.lot_size,
        tick_size=economics.tick_size,
        borrow_available=economics.borrow_available,
        borrow_rate=economics.borrow_rate,
        funding_due=economics.funding_due,
        buy_allowed=economics.buy_allowed,
        sell_allowed=economics.sell_allowed,
        mark_price=economics.mark_price,
        index_price=economics.index_price,
        feature_available=feature_available,
        feature_names=feature_names,
        global_feature_names=(
            "active_fraction",
            "tradable_fraction",
            "market_return_mean",
            "market_return_dispersion",
        ),
        periods_per_year=35_040,
        feature_staleness_hours=feature_age,
        feature_staleness=feature_staleness,
        feature_missing_reason=np.asarray(~feature_available, dtype=np.int16),
        global_feature_available=global_available,
        global_feature_staleness_hours=np.zeros((n_bars, 4), dtype=np.float32),
        global_feature_missing_reason=np.asarray(~global_available, dtype=np.int16),
        volume_units=tuple(VolumeUnit.QUOTE_NOTIONAL for _ in selected),
        contract_multipliers=np.ones(len(selected), dtype=np.float64),
        feature_config_digest=feature_config_digest,
        normalization_digest=normalization_digest,
    )
    return dataset.with_content_identity(
        {
            "indicator_artifact_digests": tuple(
                artifact.payload_sha256 for artifact in bundle.artifacts
            ),
            "indicator_cache_id": bundle.cache_id,
            "kline_table": KLINE_TABLE,
            "funding_table": FUNDING_TABLE,
            "metadata_evidence_digest": metadata_evidence_digest,
            "policy_asset_identity_mode": POLICY_ASSET_IDENTITY_MODE,
            "range": (start.isoformat(), end.isoformat()),
            "schema_version": "postgres_dynamic_triplet_dataset_v2",
            "selected_symbols": selected,
            "slot_symbols": resolved_slots,
            "symbol_triplet": dict(symbol_triplet_provenance or {}),
            "symbol_vocabulary": vocabulary,
        }
    )


__all__ = [
    "BASE_TIMEFRAME",
    "FUNDING_TABLE",
    "KLINE_TABLE",
    "NATIVE_TIMEFRAMES",
    "POLICY_ASSET_IDENTITY_MODE",
    "build_postgres_market_dataset",
]
