from __future__ import annotations

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events


def _last_feature_value(
    kind: FeatureKind,
    *,
    close: list[float],
    lookback: int,
    volume: list[float] | None = None,
) -> float:
    close_array = np.asarray(close, dtype=np.float64)
    n_bars = close_array.size
    volume_array = np.asarray(
        volume if volume is not None else [1_000.0] * n_bars,
        dtype=np.float64,
    )
    values, valid, _ = calculate_feature_events(
        FeatureSpec(name=kind.value, kind=kind, lookback=lookback),
        open_price=close_array.copy(),
        high=close_array.copy(),
        low=close_array.copy(),
        close=close_array,
        volume=volume_array,
        funding_rate=np.zeros(n_bars, dtype=np.float64),
        funding_available=np.zeros(n_bars, dtype=np.bool_),
        row_present=np.ones(n_bars, dtype=np.bool_),
        active=np.ones(n_bars, dtype=np.bool_),
    )
    assert valid[-1]
    return float(np.float32(values[-1]))


def test_identity_bound_feature_windows_match_portable_reference() -> None:
    trend_r2 = _last_feature_value(
        FeatureKind.TREND_R2,
        close=[
            29015.0,
            29448.4,
            29237.06,
            29302.11,
            29237.07,
            29213.8,
            29197.3,
            29107.71,
            29025.89,
            29229.6,
            29259.29,
            29341.99,
            29257.82,
            29493.66,
            29354.58,
            29210.84,
            29324.21,
            29099.0,
            29086.62,
            29048.47,
            29214.14,
            29177.77,
            29270.89,
            29337.16,
        ],
        lookback=24,
    )
    price_volume_correlation = _last_feature_value(
        FeatureKind.PRICE_VOLUME_CORRELATION,
        close=[
            2843.24,
            2856.95,
            2850.99,
            2933.6,
            2954.63,
            2944.99,
            2935.22,
            2925.0,
            2939.76,
            2942.98,
            2939.95,
            2946.1,
            2941.37,
            2944.51,
            2931.12,
            2927.95,
            2933.63,
            2924.87,
            2923.3,
            2928.07,
            2925.4,
            2929.04,
            2893.79,
            2829.29,
            2740.65,
        ],
        volume=[
            178371479.99161,
            217700240.79219,
            170218251.66315,
            985906310.00321,
            499528188.98515,
            123546521.23818,
            90631407.76651,
            115347509.62008,
            124767163.22539,
            183858682.93128,
            72922996.68414,
            83538939.41229,
            48627626.81203,
            42328264.51671,
            102484093.06324,
            69669619.92084,
            73489528.86574,
            178626929.63437,
            100539933.68714,
            96449855.22685,
            68576812.92353,
            81692701.41202,
            292732021.44597,
            871571360.35913,
            1265830807.72211,
        ],
        lookback=24,
    )
    realized_volatility = _last_feature_value(
        FeatureKind.REALIZED_VOLATILITY,
        close=[63283.1, 63146.6, 63192.1, 63255.1, 63237.4],
        lookback=4,
    )

    assert trend_r2 == 2.619638950207559e-10
    assert price_volume_correlation == -3.698034561239183e-06
    assert realized_volatility == 0.0012502663303166628
