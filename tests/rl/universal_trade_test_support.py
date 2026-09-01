from __future__ import annotations

from trade_rl.data.contracts import FeatureKind, FeatureSpec


def make_u1_feature_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="15m__ret", kind=FeatureKind.LOG_RETURN, lookback=1),
        FeatureSpec(
            name="1h__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="1h",
        ),
        FeatureSpec(
            name="4h__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="4h",
        ),
        FeatureSpec(
            name="1d__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="1d",
        ),
    )
