"""Small native fixtures for the corrected decision-value (TD8) tests."""

from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

COST = ExecutionCostConfig(
    fee_rate=0.0005,
    spread_rate=0.0002,
    impact_rate=0.0001,
    max_participation_rate=0.05,
    order_latency_bars=0,
)
ONE_WAY = COST.fee_rate + COST.spread_rate + COST.impact_rate
RISK_CONFIG = PreTradeRiskConfig(
    max_gross=0.50,
    max_abs_weight=0.25,
    max_turnover=1.0,
    drawdown_start=0.15,
    drawdown_stop=0.20,
)
TARGETS = (0.25, 0.25)


def risk() -> PreTradeRisk:
    return PreTradeRisk(RISK_CONFIG)


def signal_market(
    n_bars: int = 400,
    *,
    drift: float = 0.0,
    signal_strength: float = 0.004,
    noise: float = 0.002,
    seed: int = 7,
    n_features: int = 3,
) -> MarketDataset:
    """Two-symbol market whose next-bar log return is driven by feature 0.

    Feature 0 at bar ``t`` is known at ``t`` and predicts the return into
    ``t + 1``; remaining features are inert noise (used by transforms).
    """
    rng = np.random.default_rng(seed)
    timestamps = np.array(
        [
            np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(i, "h")
            for i in range(n_bars)
        ],
        dtype="datetime64[ns]",
    )
    signal = rng.normal(0.0, 1.0, size=(n_bars, 2))
    log_ret = np.zeros((n_bars, 2))
    log_ret[1:] = (
        drift + signal_strength * signal[:-1] + noise * rng.normal(size=(n_bars - 1, 2))
    )
    close = np.exp(np.cumsum(log_ret, axis=0)) * np.array([100.0, 50.0])
    open_ = np.vstack((close[:1], close[:-1]))
    features = np.zeros((n_bars, 2, n_features), dtype=np.float64)
    features[:, :, 0] = signal * signal_strength
    for j in range(1, n_features):
        features[:, :, j] = rng.normal(0.0, 1.0 + j, size=(n_bars, 2))
    return MarketDataset(
        dataset_id="a" * 64,
        feature_config_digest="b" * 64,
        normalization_digest="c" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        open=open_,
        high=np.maximum(open_, close) * 1.001,
        low=np.minimum(open_, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 2), 1.0e9),
        funding_rate=np.zeros((n_bars, 2)),
        features=features,
        available_at=np.repeat(timestamps[:, None], 2, axis=1),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, n_features), dtype=np.bool_),
        feature_names=tuple(f"f_{i}" for i in range(n_features)),
        global_feature_names=("g_0", "g_1"),
        global_features=np.tile(np.array([1.0, 0.0]), (n_bars, 1))
        + 0.01 * rng.normal(size=(n_bars, 2)),
        global_feature_available=np.ones((n_bars, 2), dtype=np.bool_),
        periods_per_year=8_760,
        calendar_kind=MarketCalendarKind.CONTINUOUS,
        nominal_bar_hours=1.0,
    )
