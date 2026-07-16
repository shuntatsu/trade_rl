"""Causal cross-asset features computed from aligned native return events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec

_EPSILON = 1e-12

CROSS_ASSET_FEATURE_KINDS = frozenset(
    {
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    }
)


@dataclass(frozen=True, slots=True)
class CrossAssetFeatureEvents:
    values: np.ndarray
    valid: np.ndarray
    source_age_hours: np.ndarray


def _new_event_mask(available: np.ndarray, age_hours: np.ndarray) -> np.ndarray:
    if available.shape != age_hours.shape or available.ndim != 2:
        raise ValueError("aligned return availability and age must be [time, asset]")
    result = np.zeros_like(available, dtype=np.bool_)
    result[0] = available[0]
    result[1:] = available[1:] & (
        (age_hours[1:] <= _EPSILON)
        | ~available[:-1]
        | (age_hours[1:] < age_hours[:-1] - _EPSILON)
    )
    return result


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    count = values.size
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(count, dtype=np.float64)
    # Average ranks for exact ties to avoid encoding symbol order.
    unique, inverse = np.unique(values, return_inverse=True)
    for group in range(unique.size):
        members = np.flatnonzero(inverse == group)
        if members.size > 1:
            ranks[members] = float(np.mean(ranks[members]))
    return 2.0 * ranks / float(count - 1) - 1.0


def calculate_cross_asset_feature_events(
    spec: FeatureSpec,
    *,
    aligned_returns: np.ndarray,
    return_available: np.ndarray,
    return_age_hours: np.ndarray,
    symbols: tuple[str, ...],
    reference_symbol: str,
) -> CrossAssetFeatureEvents:
    """Calculate one feature using only native return events available by each row."""

    if spec.kind not in CROSS_ASSET_FEATURE_KINDS:
        raise ValueError("feature is not a maintained cross-asset kind")
    returns = np.asarray(aligned_returns, dtype=np.float64)
    available = np.asarray(return_available, dtype=np.bool_)
    ages = np.asarray(return_age_hours, dtype=np.float64)
    if returns.shape != available.shape or returns.shape != ages.shape:
        raise ValueError("aligned cross-asset inputs must share a shape")
    if returns.ndim != 2 or returns.shape[1] != len(symbols):
        raise ValueError("aligned cross-asset inputs must be [time, symbol]")
    if not reference_symbol:
        raise ValueError("cross-asset reference symbol must be explicit")
    reference_matches = [
        index for index, symbol in enumerate(symbols) if symbol == reference_symbol
    ]
    if len(reference_matches) != 1:
        raise ValueError(
            "cross-asset reference symbol must occur exactly once in the universe"
        )
    btc_index = reference_matches[0]
    event_mask = _new_event_mask(available, ages)
    n_bars, n_symbols = returns.shape
    values = np.zeros((n_bars, n_symbols), dtype=np.float64)
    valid = np.zeros((n_bars, n_symbols), dtype=np.bool_)
    source_age_hours = np.zeros((n_bars, n_symbols), dtype=np.float64)

    if spec.kind is FeatureKind.RELATIVE_RETURN_TO_BTC:
        for index in range(n_bars):
            current = event_mask[index]
            if not current[btc_index]:
                continue
            valid[index, current] = True
            values[index, current] = returns[index, current] - returns[index, btc_index]
            source_age_hours[index, current] = np.maximum(
                ages[index, current], ages[index, btc_index]
            )
        return CrossAssetFeatureEvents(
            values=values, valid=valid, source_age_hours=source_age_hours
        )

    if spec.kind is FeatureKind.CROSS_ASSET_DISPERSION:
        for index in range(n_bars):
            current = event_mask[index]
            if not np.any(current):
                continue
            dispersion = float(np.std(returns[index, current]))
            values[index, current] = dispersion
            valid[index, current] = True
            source_age_hours[index, current] = float(np.max(ages[index, current]))
        return CrossAssetFeatureEvents(
            values=values, valid=valid, source_age_hours=source_age_hours
        )

    if spec.kind in {
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
    }:
        pair_histories: list[list[tuple[float, float]]] = [[] for _ in range(n_symbols)]
        for index in range(n_bars):
            if not event_mask[index, btc_index]:
                continue
            reference = float(returns[index, btc_index])
            for symbol_index in range(n_symbols):
                if not event_mask[index, symbol_index]:
                    continue
                pair_history = pair_histories[symbol_index]
                pair_history.append((float(returns[index, symbol_index]), reference))
                if len(pair_history) > spec.lookback:
                    del pair_history[0]
                if len(pair_history) < spec.min_periods:
                    continue
                sample = np.asarray(pair_history, dtype=np.float64)
                asset_sample = sample[:, 0]
                btc_sample = sample[:, 1]
                btc_variance = float(np.var(btc_sample))
                if spec.kind is FeatureKind.ROLLING_BETA_TO_BTC:
                    value = (
                        0.0
                        if btc_variance <= _EPSILON
                        else float(np.cov(asset_sample, btc_sample, ddof=0)[0, 1])
                        / btc_variance
                    )
                else:
                    asset_std = float(np.std(asset_sample))
                    btc_std = float(np.std(btc_sample))
                    value = (
                        0.0
                        if asset_std <= _EPSILON or btc_std <= _EPSILON
                        else float(np.corrcoef(asset_sample, btc_sample)[0, 1])
                    )
                    value = float(np.clip(value, -1.0, 1.0))
                values[index, symbol_index] = value
                valid[index, symbol_index] = True
                source_age_hours[index, symbol_index] = max(
                    float(ages[index, symbol_index]),
                    float(ages[index, btc_index]),
                )
        return CrossAssetFeatureEvents(
            values=values, valid=valid, source_age_hours=source_age_hours
        )

    # Cross-sectional momentum rank. Histories are advanced only on each asset's
    # newly available native event, then ranked at rows with at least one event.
    histories: list[list[float]] = [[] for _ in range(n_symbols)]
    for index in range(n_bars):
        current = event_mask[index]
        for symbol_index in np.flatnonzero(current):
            momentum_history = histories[int(symbol_index)]
            momentum_history.append(float(returns[index, symbol_index]))
            if len(momentum_history) > spec.lookback:
                del momentum_history[0]
        eligible = np.asarray(
            [len(history) >= spec.min_periods for history in histories], dtype=np.bool_
        )
        eligible &= current
        indices = np.flatnonzero(eligible)
        if indices.size == 0:
            continue
        momentum = np.asarray(
            [sum(histories[int(symbol_index)]) for symbol_index in indices],
            dtype=np.float64,
        )
        values[index, indices] = _normalized_ranks(momentum)
        valid[index, indices] = True
        source_age_hours[index, indices] = float(np.max(ages[index, indices]))
    return CrossAssetFeatureEvents(
        values=values, valid=valid, source_age_hours=source_age_hours
    )


__all__ = [
    "CROSS_ASSET_FEATURE_KINDS",
    "CrossAssetFeatureEvents",
    "calculate_cross_asset_feature_events",
]
