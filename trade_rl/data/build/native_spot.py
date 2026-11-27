"""Opt-in native spot Dataset construction from explicitly pinned bars.

The maintained resampler and clocked feature alignment own time/indicator
semantics. This builder does not authenticate venue captures or grant access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.build.builder import _calculate_one_bar_returns
from trade_rl.data.contracts import PORTABLE_FEATURE_NUMERICS_SCHEMA, VolumeUnit
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    clocked_native_source_digest,
)
from trade_rl.data.features.multitimeframe import align_clocked_native_feature
from trade_rl.data.features.numerics import portable_mean, portable_std
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, _native_unmasked_array
from trade_rl.data.native_execution_clock import (
    calendar_month_clock_payload,
    calendar_month_execution_frequency,
    execution_clock_payload,
    fixed_execution_frequency,
)
from trade_rl.data.native_resampling import (
    NativeResampledBars,
    resample_complete_native_bars,
)
from trade_rl.data.source import NativeRawMarketSeries

_SCHEMA = "native_fixed_spot_build_v1"
_GLOBAL_NAMES = (
    "active_fraction",
    "tradable_fraction",
    "market_return_mean",
    "market_return_dispersion",
)


def _cutoff(value: np.datetime64) -> np.datetime64:
    if type(value) is not np.datetime64:
        raise ValueError("as_of must be an explicit datetime64 scalar")
    return cast(np.datetime64, _datetime64_values_to_ns_exact(np.asarray(value))[()])


def _lifecycle(value: datetime) -> np.datetime64:
    utc = value.astimezone(timezone.utc).replace(tzinfo=None)
    return cast(
        np.datetime64,
        _datetime64_values_to_ns_exact(np.asarray(np.datetime64(utc, "us")))[()],
    )


def spot_raw_from_resampled(
    bars: NativeResampledBars,
    *,
    clock: NativeBarClock,
    as_of: np.datetime64,
    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET,
) -> NativeRawMarketSeries:
    """Convert complete normalized OHLCV into the existing spot raw contract.

    Reuse the exact-grid resampler to validate the supplied aggregate rows.
    Counts must be positive integers but cannot attest to the original
    members of an arbitrary manually constructed NativeResampledBars object.
    The caller retains independently authenticated source/receipt evidence.
    """
    if type(bars) is not NativeResampledBars:
        raise ValueError("expected complete NativeResampledBars")
    if type(volume_unit) is not VolumeUnit or volume_unit not in (
        VolumeUnit.BASE_ASSET,
        VolumeUnit.QUOTE_NOTIONAL,
    ):
        raise ValueError("spot volume unit must be base asset or quote notional")
    if type(clock) is not NativeBarClock:
        raise ValueError("expected explicit native clock")
    counts = _native_unmasked_array(
        bars.member_counts, field_name="native member counts"
    )
    _, closes = clock.interval_bounds(bars.closes)
    if (
        counts.shape != closes.shape
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
    ):
        raise ValueError(
            "native member counts must be positive integers matching the grid"
        )
    checked = resample_complete_native_bars(
        source_clock=clock,
        source_closes=bars.closes,
        source_available_at=bars.available_at,
        source_ohlcv=bars.ohlcv,
        target_clock=clock,
        target_closes=bars.closes,
        as_of=_cutoff(as_of),
    )
    v = checked.ohlcv
    return NativeRawMarketSeries(
        timestamps=checked.closes,
        available_at=checked.available_at,
        open=v[:, 0],
        high=v[:, 1],
        low=v[:, 2],
        close=v[:, 3],
        volume=v[:, 4 if volume_unit is VolumeUnit.BASE_ASSET else 5],
        funding_rate=np.zeros(len(v)),
        funding_available=np.zeros(len(v), dtype=bool),
        funding_event_count=np.zeros(len(v), dtype=np.int32),
        tradable=np.ones(len(v), dtype=bool),
    )


def _verify_source(
    source: ClockedNativeSource,
    clock: NativeBarClock,
    *,
    cutoff: np.datetime64,
    last_close: np.datetime64,
) -> None:
    if type(source) is not ClockedNativeSource or type(clock) is not NativeBarClock:
        raise ValueError("expected explicit pinned native spot source")
    contract, raw = source.contract, source.raw
    if type(raw) is not NativeRawMarketSeries:
        raise ValueError("native spot builder requires strict native raw sources")
    if (
        type(contract.volume_unit) is not VolumeUnit
        or contract.volume_unit
        not in (VolumeUnit.BASE_ASSET, VolumeUnit.QUOTE_NOTIONAL)
        or type(contract.contract_multiplier) not in (float, int)
        or contract.contract_multiplier != 1.0
        or contract.execution_rules
    ):
        raise ValueError(
            "native spot builder requires unit multiplier and static spot rules"
        )
    starts, ends = clock.interval_bounds(raw.timestamps)
    if starts[0] < _lifecycle(contract.listed_at) or (
        contract.delisted_at is not None
        and ends[-1] >= _lifecycle(contract.delisted_at)
    ):
        raise ValueError("native source crosses instrument lifetime")
    assert raw.available_at is not None and raw.funding_event_count is not None
    if ends[-1] > last_close or np.any(raw.available_at > cutoff):
        raise ValueError("native source exceeds requested range or as_of")
    if np.any(raw.funding_rate != 0) or np.any(raw.funding_event_count != 0):
        raise ValueError("native spot source cannot contain funding events")
    if (
        clocked_native_source_digest(raw, contract=contract, clock=clock)
        != source.expected_source_digest
    ):
        raise ValueError("native source digest mismatch")


def _clock_descriptor(clock: NativeBarClock) -> dict[str, object]:
    if type(clock) is not NativeBarClock:
        raise ValueError("expected explicit native feature clock")
    return {
        "kind": clock.kind.value,
        "fixed_seconds": clock.fixed_seconds,
        "fixed_anchor_ns": None
        if clock.fixed_anchor is None
        else int(clock.fixed_anchor.astype(np.int64)),
    }


def _global_features(
    close: np.ndarray, information: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing causal one-bar return/numerics owners."""
    active = np.ones_like(information, dtype=bool)
    returns, ready = _calculate_one_bar_returns(
        close=close,
        causal_row_present=information,
        symbol_active=active,
    )
    values = np.zeros((len(close), len(_GLOBAL_NAMES)), dtype=np.float64)
    available = np.ones_like(values, dtype=bool)
    values[:, 0] = 1
    values[:, 1] = np.count_nonzero(information, axis=1) / close.shape[1]
    for row in range(len(close)):
        sample = returns[row, ready[row]]
        if sample.size:
            values[row, 2] = portable_mean(sample)
            values[row, 3] = portable_std(sample)
        else:
            available[row, 2:] = False
    return _finite_float32(values), available


def _finite_float32(value: np.ndarray) -> np.ndarray:
    if not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max):
        raise ValueError("native feature value or age exceeds Dataset representation")
    return value.astype(np.float32)


def _build_native_spot_dataset(
    *,
    clock: NativeBarClock,
    closes: np.ndarray,
    sources: tuple[ClockedNativeSource, ...],
    feature_groups: tuple[ClockedNativeFeatureGroup, ...],
    as_of: np.datetime64,
    calendar_month: bool,
) -> MarketDataset:
    """Construct a clock-bound Dataset without changing the legacy MarketBuildConfig.

    Execution sources must share the explicit requested suffix grid and may
    have a complete warm-up prefix. Require NativeRawMarketSeries for every
    source, including feature-only sources. Feature clocks are explicit;
    the base-clock feature group, if supplied, must use the execution pins.
    All sources are validated before feature computation. Preserve receipt
    timestamps; late marks are stored, not made tradable by rewriting time.
    No funding, derivatives, time-varying rules or provider authentication.
    Execution costs beyond the static instrument rules remain caller-owned.
    """
    if calendar_month:
        execution_clock = calendar_month_clock_payload(clock)
        frequency = calendar_month_execution_frequency(clock, closes)
    else:
        execution_clock = execution_clock_payload(clock)
        frequency = fixed_execution_frequency(clock)
    _, grid = clock.interval_bounds(closes)
    if len(grid) < 3:
        raise ValueError("native execution grid requires at least three closes")
    cutoff = _cutoff(as_of)
    if grid[-1] > cutoff:
        raise ValueError("native execution grid exceeds as_of")
    if (
        type(sources) is not tuple
        or not sources
        or any(type(s) is not ClockedNativeSource for s in sources)
    ):
        raise ValueError("expected a nonempty pinned execution source roster")
    contracts = tuple(source.contract for source in sources)
    symbols = tuple(c.symbol for c in contracts)
    if len(set(symbols)) != len(symbols):
        raise ValueError("native execution symbols must be unique")
    n, m = len(grid), len(sources)
    for source in sources:
        _verify_source(source, clock, cutoff=cutoff, last_close=grid[-1])
        if not np.array_equal(source.raw.timestamps[-n:], grid):
            raise ValueError(
                "native execution source must end with the complete requested grid"
            )
    if (
        type(feature_groups) is not tuple
        or not feature_groups
        or any(type(g) is not ClockedNativeFeatureGroup for g in feature_groups)
    ):
        raise ValueError("expected explicit native feature groups")
    descriptors = tuple(_clock_descriptor(g.clock) for g in feature_groups)
    if len({content_digest(p) for p in descriptors}) != len(descriptors):
        raise ValueError("native feature clocks must be distinct")
    specs = tuple(spec for group in feature_groups for spec in group.specs)
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("native feature names must be unique")
    for group, descriptor in zip(feature_groups, descriptors, strict=True):
        if tuple(source.contract for source in group.sources) != contracts:
            raise ValueError(
                "native feature source contracts must match execution roster"
            )
        for i, source in enumerate(group.sources):
            if descriptor == execution_clock:
                if source.expected_source_digest != sources[i].expected_source_digest:
                    raise ValueError(
                        "same-clock feature source must match the execution pin"
                    )
                # Still verify the actual supplied raw source unless it is the
                # exact already-verified immutable source object.
                if source is sources[i]:
                    continue
            _verify_source(source, group.clock, cutoff=cutoff, last_close=grid[-1])

    def column(field: str) -> np.ndarray:
        return np.column_stack(tuple(getattr(s.raw, field)[-n:] for s in sources))

    availability = column("available_at")
    information = availability <= grid[:, None]
    shape = (n, m, len(specs))
    values = np.zeros(shape, dtype=np.float32)
    available = np.zeros(shape, dtype=bool)
    ages = np.zeros(shape, dtype=np.float32)
    stale = np.ones(shape, dtype=np.float32)
    offset = 0
    for group in feature_groups:
        for j, source in enumerate(group.sources):
            for k, spec in enumerate(group.specs, start=offset):
                v, mask, age, staleness = align_clocked_native_feature(
                    spec,
                    source.raw,
                    source.contract,
                    grid,
                    information[:, j],
                    clock=group.clock,
                )
                values[:, j, k] = _finite_float32(v)
                available[:, j, k] = mask
                ages[:, j, k] = _finite_float32(age)
                stale[:, j, k] = _finite_float32(staleness)
        offset += len(group.specs)

    close = column("close")
    globals_, globals_available = _global_features(close, information)
    config = {
        "schema_version": "native_calendar_month_spot_build_v1"
        if calendar_month
        else _SCHEMA,
        "execution_clock": execution_clock,
        "groups": tuple(
            {"clock": p, "specs": tuple(s.canonical_payload() for s in g.specs)}
            for g, p in zip(feature_groups, descriptors, strict=True)
        ),
        "features": tuple(spec.canonical_payload() for spec in specs),
        "global_feature_names": _GLOBAL_NAMES,
        "feature_numerics_schema": PORTABLE_FEATURE_NUMERICS_SCHEMA,
        "cross_asset_reference_symbol": None,
    }
    config_digest = content_digest(config)
    normalization_digest = content_and_arrays_digest(
        {
            "schema": "native_calendar_month_spot_normalization_v1"
            if calendar_month
            else "native_fixed_spot_normalization_v1",
            "feature_config_digest": config_digest,
        },
        (
            ("features", values),
            ("feature_available", available),
            ("feature_age_hours", ages),
            ("feature_staleness", stale),
        ),
    )
    zeros = np.zeros((n, m), dtype=np.float64)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=symbols,
        timestamps=grid,
        features=values,
        feature_available=available,
        feature_names=names,
        feature_staleness_hours=ages,
        feature_staleness=stale,
        feature_missing_reason=(~available).astype(np.int16),
        global_features=globals_,
        global_feature_names=_GLOBAL_NAMES,
        global_feature_available=globals_available,
        global_feature_missing_reason=(~globals_available).astype(np.int16),
        open=column("open"),
        high=column("high"),
        low=column("low"),
        close=close,
        volume=column("volume"),
        available_at=availability,
        information_available=information,
        tradable=np.ones((n, m), dtype=bool),
        funding_rate=zeros,
        funding_price_rate=zeros,
        funding_due=np.zeros((n, m), dtype=bool),
        funding_event_count=np.zeros((n, m), dtype=np.int32),
        volume_units=tuple(c.volume_unit for c in contracts),
        contract_multipliers=np.ones(m),
        tick_size=np.tile(np.array([c.tick_size for c in contracts]), (n, 1)),
        lot_size=np.tile(np.array([c.lot_size for c in contracts]), (n, 1)),
        minimum_notional=np.tile(
            np.array([c.minimum_notional for c in contracts]), (n, 1)
        ),
        feature_config_digest=config_digest,
        normalization_digest=normalization_digest,
        periods_per_year=frequency,
        native_clock=clock,
    ).with_content_identity(
        {
            "config": config,
            "instruments": tuple(c.canonical_payload() for c in contracts),
            "execution_source_digests": tuple(
                s.expected_source_digest for s in sources
            ),
            "feature_source_digests": tuple(
                tuple(s.expected_source_digest for s in g.sources)
                for g in feature_groups
            ),
            "as_of_ns": int(cutoff.astype(np.int64)),
        }
    )


def build_native_fixed_spot_dataset(
    *,
    clock: NativeBarClock,
    closes: np.ndarray,
    sources: tuple[ClockedNativeSource, ...],
    feature_groups: tuple[ClockedNativeFeatureGroup, ...],
    as_of: np.datetime64,
) -> MarketDataset:
    """Build the unchanged v8 fixed 3d/7d/week spot Dataset contract."""
    return _build_native_spot_dataset(
        clock=clock,
        closes=closes,
        sources=sources,
        feature_groups=feature_groups,
        as_of=as_of,
        calendar_month=False,
    )


def build_native_calendar_month_spot_dataset(
    *,
    clock: NativeBarClock,
    closes: np.ndarray,
    sources: tuple[ClockedNativeSource, ...],
    feature_groups: tuple[ClockedNativeFeatureGroup, ...],
    as_of: np.datetime64,
) -> MarketDataset:
    """Build a v9 spot Dataset on actual consecutive UTC calendar months."""
    return _build_native_spot_dataset(
        clock=clock,
        closes=closes,
        sources=sources,
        feature_groups=feature_groups,
        as_of=as_of,
        calendar_month=True,
    )


__all__ = [
    "build_native_calendar_month_spot_dataset",
    "build_native_fixed_spot_dataset",
    "spot_raw_from_resampled",
]
