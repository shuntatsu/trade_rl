"""Cross-asset rolling statistics on *native* events, before base downsampling."""

from __future__ import annotations

import numpy as np

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    timeframe_hours,
)
from trade_rl.data.features.cross_asset import (
    CROSS_ASSET_FEATURE_KINDS,
    calculate_cross_asset_feature_events,
)
from trade_rl.data.features.multitimeframe import (
    _native_events,
    _validate_regular_native_series,
)
from trade_rl.data.features.native_alignment import align_native_events_asof
from trade_rl.data.source import RawMarketSeries

_NS_PER_HOUR = 3_600_000_000_000


def align_native_cross_asset_feature(
    spec: FeatureSpec,
    raws: tuple[RawMarketSeries, ...],
    contracts: tuple[InstrumentContract, ...],
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
    *,
    timeframe: str,
    reference_symbol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count actual native bars, not repeated or hourly-subsampled observations.

    Eligible returns join adjacent physical rows that are active, tradable and
    published by their own closes. Late rows are excluded, not inserted into a
    history after arrival. Rolling histories retain the last eligible pair
    events; they do not reset at a missing event.
    """
    if spec.kind not in CROSS_ASSET_FEATURE_KINDS:
        raise ValueError("native cross-asset feature kind is unsupported")
    if not raws or len(raws) != len(contracts):
        raise ValueError("native sources must match the symbol roster")
    n_symbols = len(raws)
    base_clock = np.asarray(base_timestamps, dtype="datetime64[ns]")
    active = np.asarray(base_active, dtype=np.bool_)
    if active.shape != (len(base_clock), n_symbols):
        raise ValueError("native cross-asset base active mask has wrong shape")
    step = int(round(timeframe_hours(timeframe) * _NS_PER_HOUR))
    for raw in raws:
        _validate_regular_native_series(raw, timeframe)
    begin = min(int(raw.timestamps[0].astype(np.int64)) for raw in raws)
    end = max(int(raw.timestamps[-1].astype(np.int64)) for raw in raws)
    count, remainder = divmod(end - begin, step)
    if remainder or count > 4_000_000:
        raise ValueError("native cross-asset clocks are inconsistent or unbounded")
    timestamps = (begin + np.arange(count + 1, dtype=np.int64) * step).astype(
        "datetime64[ns]"
    )
    returns = np.zeros((len(timestamps), n_symbols), dtype=np.float64)
    return_valid = np.zeros((len(timestamps), n_symbols), dtype=np.bool_)
    arrival_prefix = np.full(
        (len(timestamps), n_symbols), np.iinfo(np.int64).min, dtype=np.int64
    )
    return_spec = FeatureSpec(
        name="native_return_internal", kind=FeatureKind.LOG_RETURN
    )
    for i, (raw, contract) in enumerate(zip(raws, contracts)):
        native_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
        offsets = native_ns - begin
        if np.any(offsets < 0) or np.any(offsets % step != 0):
            raise ValueError("native source is not aligned to the common clock")
        positions = (offsets // step).astype(np.intp)
        values, valid, arrived_at = _native_events(return_spec, raw, contract)
        assert raw.available_at is not None
        assert raw.tradable is not None
        # _native_events already requires both adjacent rows to be active.
        row_eligible = raw.tradable & (raw.available_at <= raw.timestamps)
        valid[1:] &= row_eligible[1:] & row_eligible[:-1]
        returns[positions, i] = values
        return_valid[positions, i] = valid
        arrivals = np.full(len(timestamps), np.iinfo(np.int64).min, dtype=np.int64)
        arrivals[positions[valid]] = (
            arrived_at[valid].astype("datetime64[ns]").astype(np.int64)
        )
        arrival_prefix[:, i] = np.maximum.accumulate(arrivals)

    events = calculate_cross_asset_feature_events(
        spec,
        aligned_returns=returns,
        return_available=return_valid,
        return_age_hours=np.zeros_like(returns),
        symbols=tuple(contract.symbol for contract in contracts),
        reference_symbol=reference_symbol,
    )
    btc = tuple(contract.symbol for contract in contracts).index(reference_symbol)
    values = np.zeros((len(base_clock), n_symbols), dtype=np.float64)
    available = np.zeros((len(base_clock), n_symbols), dtype=np.bool_)
    ages = np.full_like(values, spec.max_staleness_hours)
    staleness = np.ones_like(values, dtype=np.float32)
    for i in range(n_symbols):
        if spec.kind in {
            FeatureKind.RELATIVE_RETURN_TO_BTC,
            FeatureKind.ROLLING_CORRELATION_TO_BTC,
            FeatureKind.ROLLING_BETA_TO_BTC,
        }:
            arrivals = np.maximum(arrival_prefix[:, btc], arrival_prefix[:, i])
        else:
            # The full contemporaneous cross-section may affect the statistic.
            arrivals = np.max(arrival_prefix, axis=1)
        aligned = align_native_events_asof(
            timestamps,
            events.values[:, i, None],
            events.valid[:, i],
            arrivals.astype("datetime64[ns]"),
            base_clock,
            active[:, i],
            max_staleness_hours=spec.max_staleness_hours,
        )
        values[:, i], available[:, i], ages[:, i], staleness[:, i] = (
            channel[:, 0] for channel in aligned
        )
    return values, available, ages, staleness


__all__ = ["align_native_cross_asset_feature"]
