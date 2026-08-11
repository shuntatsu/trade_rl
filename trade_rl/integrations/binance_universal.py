from __future__ import annotations

from collections.abc import Sequence

from trade_rl.data.contracts import FeatureSpec
from trade_rl.data.universal_features import universal_target_local_features
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs


def binance_universal_feature_specs(
    *,
    base_timeframe: str,
    feature_timeframes: Sequence[str],
) -> tuple[FeatureSpec, ...]:
    """Return the identity-free target-local Binance universal feature contract."""

    features = universal_target_local_features(
        binance_multitimeframe_feature_specs(
            base_timeframe=base_timeframe,
            feature_timeframes=feature_timeframes,
        )
    )
    all_maintained_clocks = {base_timeframe, *feature_timeframes} == {
        "15m",
        "1h",
        "4h",
        "1d",
    }
    if all_maintained_clocks and len(features) != 206:
        raise RuntimeError(
            "universal Binance feature contract drifted from the maintained 206-channel schema"
        )
    return features
