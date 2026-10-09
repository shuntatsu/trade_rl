"""Opt-in per-native-clock rolling path signatures aligned as-of to base decisions.

This does not replace the single-clock Signature in signature.py, alter the
execution clock, or establish economic improvement. Native windows are built
from actually timestamped completed closes, never inferred OHLC intrabar paths.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from itertools import product

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import MarketCalendarKind, timeframe_hours
from trade_rl.data.features.multitimeframe import _validate_regular_native_series
from trade_rl.data.features.native_alignment import align_native_events_asof
from trade_rl.data.features.signature import _segment_signature
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import MultiTimeframeMarketDataSource, RawMarketSeries


def _source_digest(raw: RawMarketSeries) -> str:
    digest = hashlib.sha256()
    for name, array, dtype in (
        ("timestamps", raw.timestamps, "<i8"),
        ("available_at", raw.available_at, "<i8"),
        ("close", raw.close, "<f8"),
        ("volume", raw.volume, "<f8"),
        ("tradable", raw.tradable, "u1"),
    ):
        digest.update(name.encode("ascii") + b"\0")
        native = np.asarray(array)
        if name in {"timestamps", "available_at"}:
            native = native.astype("datetime64[ns]").astype(np.int64)
        digest.update(np.ascontiguousarray(native, dtype=dtype).tobytes())
    return digest.hexdigest()


def _rolling_signature_events(
    timestamps: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    tradable: np.ndarray,
    available_at: np.ndarray,
    *,
    window_bars: int,
    depth: int,
    include_volume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native signature events and the *maximum* input availability time."""
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    available_at = np.asarray(available_at, dtype="datetime64[ns]")
    usable = np.asarray(tradable, dtype=np.bool_) & np.isfinite(closes) & (closes > 0.0)
    if include_volume:
        usable &= np.isfinite(volumes) & (volumes > 0.0)
    if np.any(available_at < timestamps):
        raise ValueError("native raw publication time precedes bar close")
    dim = 3 if include_volume else 2
    n_features = sum(dim**level for level in range(1, depth + 1))
    results = np.zeros((len(timestamps), n_features), dtype=np.float64)
    valid = np.zeros(len(timestamps), dtype=np.bool_)
    arrivals = np.full(len(timestamps), np.datetime64("NaT", "ns"))
    missing_prefix = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(~usable, dtype=np.int64))
    )
    # Translation invariance follows from taking log-channel increments.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(usable, closes, 1.0))
        volume_logs = np.log(np.where(usable, volumes, 1.0))
    for end in range(window_bars - 1, len(timestamps)):
        start = end - window_bars + 1
        if missing_prefix[end + 1] != missing_prefix[start]:
            continue
        increments = np.empty((window_bars - 1, dim), dtype=np.float64)
        increments[:, 0] = 1.0 / float(window_bars - 1)
        increments[:, 1] = np.diff(logs[start : end + 1])
        if include_volume:
            increments[:, 2] = np.diff(volume_logs[start : end + 1])
        signature = _segment_signature(increments, depth)
        if not np.isfinite(signature).all():
            continue
        if not np.isfinite(signature.astype(np.float32)).all():
            continue
        results[end] = signature
        valid[end] = True
        arrivals[end] = np.max(available_at[start : end + 1])
    return results, valid, arrivals


def with_multitimeframe_path_signatures(
    dataset: MarketDataset,
    source: MultiTimeframeMarketDataSource | None = None,
    *,
    base_timeframe: str,
    windows_by_timeframe: Mapping[str, int],
    depth: int = 2,
    include_volume: bool = False,
) -> MarketDataset:
    """Append per-native-clock Signature features to an immutable base Dataset.

    Every native window is calculated before as-of alignment to one base
    decision clock. An eligible native event may be delivered late, but never
    before its maximum constituent `available_at`. No native bar is subsampled
    before calculating its Signature.
    """
    if dataset.calendar_kind != MarketCalendarKind.CONTINUOUS:
        raise ValueError("native Signature v1 requires continuous market calendar")
    if not math.isclose(
        timeframe_hours(base_timeframe), dataset.bar_hours, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("base_timeframe differs from source Dataset clock")
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 3:
        raise ValueError("Signature depth must be between 1 and 3")
    if not isinstance(include_volume, bool):
        raise ValueError("include_volume must be boolean")
    if not windows_by_timeframe:
        raise ValueError("Signature native timeframe windows cannot be empty")
    if any(
        not isinstance(window, int)
        or isinstance(window, bool)
        or not 3 <= window <= 512
        for window in windows_by_timeframe.values()
    ):
        raise ValueError("Signature native window_bars must be between 3 and 512")
    windows = tuple(
        sorted(windows_by_timeframe.items(), key=lambda item: timeframe_hours(item[0]))
    )
    if any(tf != base_timeframe for tf, _ in windows) and (
        source is None or not isinstance(source, MultiTimeframeMarketDataSource)
    ):
        raise ValueError("native timeframes require MultiTimeframeMarketDataSource")
    letters = ("t", "p", "v") if include_volume else ("t", "p")
    words = tuple(
        "".join(word)
        for level in range(1, depth + 1)
        for word in product(letters, repeat=level)
    )
    all_values: list[np.ndarray] = []
    all_available: list[np.ndarray] = []
    all_age: list[np.ndarray] = []
    all_staleness: list[np.ndarray] = []
    all_names: list[str] = []
    source_digests: list[dict[str, str]] = []
    active = dataset.resolved_array("symbol_active")
    for timeframe, window in windows:
        names = tuple(
            f"mt_path_sig_v1_{timeframe}_w{window}_d{depth}_{''.join(letters)}_{word}"
            for word in words
        )
        if set(names) & (set(dataset.feature_names) | set(all_names)):
            raise ValueError("multitimeframe Signature feature names already exist")
        all_names.extend(names)
        staleness_limit = max(1.0, 2.0 * timeframe_hours(timeframe))
        values = np.zeros(
            (dataset.n_bars, dataset.n_symbols, len(words)), dtype=np.float32
        )
        available = np.zeros_like(values, dtype=np.bool_)
        ages = np.full(values.shape, staleness_limit, dtype=np.float64)
        staleness = np.ones(values.shape, dtype=np.float32)
        for symbol_index, symbol in enumerate(dataset.symbols):
            if timeframe == base_timeframe:
                times = dataset.timestamps
                closes = dataset.close[:, symbol_index]
                volumes = dataset.volume[:, symbol_index]
                tradable = (
                    dataset.tradable[:, symbol_index]
                    & dataset.resolved_array("information_available")[:, symbol_index]
                )
                arrived = dataset.resolved_array("available_at")[:, symbol_index]
            else:
                assert source is not None
                raw = source.load_timeframe(symbol, timeframe)
                _validate_regular_native_series(raw, timeframe)
                times = raw.timestamps
                closes = raw.close
                volumes = raw.volume
                tradable = raw.tradable
                assert raw.available_at is not None
                arrived = raw.available_at
                source_digests.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "raw_source_sha256": _source_digest(raw),
                    }
                )
            if len(times) < window:
                raise ValueError(f"not enough {timeframe} bars for Signature window")
            signatures, event_valid, event_available_at = _rolling_signature_events(
                times,
                closes,
                volumes,
                tradable,
                arrived,
                window_bars=window,
                depth=depth,
                include_volume=include_volume,
            )
            aligned = align_native_events_asof(
                times,
                signatures,
                event_valid,
                event_available_at,
                dataset.timestamps,
                active[:, symbol_index],
                max_staleness_hours=staleness_limit,
            )
            values[:, symbol_index] = aligned[0].astype(np.float32)
            available[:, symbol_index] = aligned[1]
            ages[:, symbol_index] = aligned[2]
            staleness[:, symbol_index] = aligned[3]
        all_values.append(values)
        all_available.append(available)
        all_age.append(ages)
        all_staleness.append(staleness)
    definition: dict[str, object] = {
        "schema": "native_multitimeframe_path_signature_v1",
        "source_dataset_id": dataset.dataset_id,
        "base_timeframe": base_timeframe,
        "windows": [[tf, window] for tf, window in windows],
        "depth": depth,
        "channels": list(letters),
        "path": "piecewise_linear_native_completed_closes",
        "availability": "max_window_source_available_at_asof_decision",
        "source_digests": source_digests,
    }
    added_available = np.concatenate(all_available, axis=2)
    augmented = replace(
        dataset,
        identity_payload_json=None,
        feature_names=dataset.feature_names + tuple(all_names),
        features=np.concatenate((dataset.features, *all_values), axis=2),
        feature_available=np.concatenate(
            (dataset.feature_available, added_available), axis=2
        ),
        feature_staleness_hours=np.concatenate(
            (dataset.resolved_array("feature_staleness_hours"), *all_age), axis=2
        ),
        feature_staleness=np.concatenate(
            (dataset.resolved_array("feature_staleness"), *all_staleness), axis=2
        ),
        feature_missing_reason=np.concatenate(
            (
                dataset.resolved_array("feature_missing_reason"),
                (~added_available).astype(np.int16),
            ),
            axis=2,
        ),
        feature_config_digest=content_digest(definition),
    )
    return augmented.with_content_identity(definition)


__all__ = ["with_multitimeframe_path_signatures"]
