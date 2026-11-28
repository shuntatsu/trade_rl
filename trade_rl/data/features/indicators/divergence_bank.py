"""Opt-in oscillator bank reusing the canonical trade_rl indicator numerics.

No second RSI/MACD/MFI/CCI formula is introduced. Invalid warm-up, gaps and
non-tradable rows are represented by NaN rather than valid zero observations.
A row published after its own timestamp (``available_at > timestamp``) is not
information at that bar: like the Dataset builder's ``row_present &
information_available`` rule, it is treated as a missing row, so no value reads
it and every calculation restarts after it instead of bridging it.  The bank
cannot see a native timestamp missing from ``raw``; callers pass a complete
native grid or mark such rows non-tradable.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features.core import calculate_feature_events
from trade_rl.data.features.indicators.extra import (
    EXTRA_INDICATOR_NAMES,
    build_extra_indicators,
)
from trade_rl.data.source import RawMarketSeries

DIVERGENCE_INDICATOR_SPECS = MappingProxyType(
    {
        "RSI": (FeatureKind.RSI, 14),
        "MACD": (FeatureKind.MACD_LINE, 26),
        "MACD_HIST": (FeatureKind.MACD_HISTOGRAM, 35),
        "MACD_SIGNAL": (FeatureKind.MACD_SIGNAL, 35),
        "STOCH_K": (FeatureKind.STOCHASTIC_K, 14),
        # 14-period %K smoothed by an explicit 3-period mean (14/3).
        "STOCH_D": (FeatureKind.STOCHASTIC_D, 14),
        "CCI": (FeatureKind.CCI, 20),
        "MFI": (FeatureKind.MFI, 14),
        "CMF": (FeatureKind.CMF, 21),
        "WILLR": (FeatureKind.WILLIAMS_R, 14),
        "OBV_SLOPE": (FeatureKind.OBV_SLOPE, 24),
        "DI_SPREAD": (FeatureKind.DI_SPREAD, 14),
    }
)
_STOCHASTIC_D_SMOOTHING_PERIOD = 3
DEFAULT_DIVERGENCE_INDICATORS = (
    "RSI",
    "MACD_HIST",
    "STOCH_K",
    "CCI",
    "MFI",
    "CMF",
    "OBV_SLOPE",
)


def build_divergence_bank(
    raw: RawMarketSeries,
    *,
    names: Iterable[str] = DEFAULT_DIVERGENCE_INDICATORS,
) -> dict[str, np.ndarray]:
    """Calculate registered indicators at each native, completed bar close.

    Each series is strictly causal and has NaN where the canonical feature is
    unavailable. Named feature values retain trade_rl's established scaling.
    """
    chosen = tuple(names)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("indicator names must be non-empty and unique")
    unknown = set(chosen) - DIVERGENCE_INDICATOR_SPECS.keys() - EXTRA_INDICATOR_NAMES
    if unknown:
        raise ValueError(f"unsupported divergence indicators: {sorted(unknown)}")
    assert raw.funding_available is not None
    assert raw.available_at is not None
    published_on_time = raw.available_at.astype("datetime64[ns]") <= (
        raw.timestamps.astype("datetime64[ns]")
    )
    usable = raw.tradable & published_on_time
    bank: dict[str, np.ndarray] = {}
    for name in chosen:
        if name in EXTRA_INDICATOR_NAMES:
            continue
        kind, lookback = DIVERGENCE_INDICATOR_SPECS[name]
        values, valid, _ = calculate_feature_events(
            FeatureSpec(
                name=f"divergence_{name}",
                kind=kind,
                lookback=lookback,
                stochastic_smoothing_period=(
                    _STOCHASTIC_D_SMOOTHING_PERIOD
                    if kind is FeatureKind.STOCHASTIC_D
                    else None
                ),
            ),
            open_price=raw.open,
            high=raw.high,
            low=raw.low,
            close=raw.close,
            volume=raw.volume,
            funding_rate=raw.funding_rate,
            funding_available=raw.funding_available,
            row_present=published_on_time,
            active=raw.tradable,
        )
        selected = np.where(valid, values, np.nan)
        selected.setflags(write=False)
        bank[name] = selected
    bank.update(
        build_extra_indicators(
            raw,
            names=(name for name in chosen if name in EXTRA_INDICATOR_NAMES),
            usable=usable,
        )
    )
    return {name: bank[name] for name in chosen}


__all__ = [
    "DIVERGENCE_INDICATOR_SPECS",
    "DEFAULT_DIVERGENCE_INDICATORS",
    "build_divergence_bank",
]
