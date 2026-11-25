"""Seeded synthetic stage for the learning lab's software tests and dry runs.

Everything is generated from a seed: ``<pair>:spot`` / ``<pair>:perp`` random-walk legs on a
4h grid and a *stand-in* anchor signal (piecewise-constant random weights). The stand-in
only exercises the mechanics; it is not the ported anchor's signal and carries no economic
meaning. No market data, file or network is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.agy_anchor.signals import AnchorSignals

BAR = np.timedelta64(4, "h")
PERIODS_PER_YEAR = 2190
DEFAULT_PAIRS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")


@dataclass(frozen=True, slots=True)
class SyntheticStage:
    dataset: MarketDataset
    signals: AnchorSignals


def synthetic_stage(
    *,
    n_bars: int,
    pairs: tuple[str, ...] = DEFAULT_PAIRS,
    seed: int = 0,
    first_stamp: str = "2022-01-01T00:00:00",
    volatility: float = 0.01,
    drift: float = 0.0002,
    signal_hold_rows: int = 18,
) -> SyntheticStage:
    """A complete synthetic dataset plus stand-in anchor signals over ``n_bars`` rows."""
    if n_bars < 12:
        raise ValueError("n_bars must be at least 12")
    n_pairs = len(pairs)
    legs = tuple(f"{pair}:{kind}" for pair in pairs for kind in ("spot", "perp"))
    n_legs = len(legs)
    rng = np.random.default_rng(seed)
    stamps = np.datetime64(first_stamp, "ns") + np.arange(n_bars) * BAR

    pair_return = rng.normal(drift, volatility, size=(n_bars, n_pairs))
    basis = rng.normal(0.0, volatility / 20.0, size=(n_bars, n_pairs))
    log_return = np.repeat(pair_return, 2, axis=1)
    log_return[:, 1::2] += basis
    close = 100.0 * np.exp(np.cumsum(log_return, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    wick = np.abs(rng.normal(0.0, volatility / 2.0, size=(2, n_bars, n_legs)))
    high = np.maximum(open_, close) * (1.0 + wick[0])
    low = np.minimum(open_, close) * (1.0 - wick[1])
    shape = (n_bars, n_legs)
    features = np.zeros((n_bars, n_legs, 1), dtype=np.float64)
    dataset = MarketDataset(
        dataset_id=f"{seed:x}".rjust(64, "5"),
        symbols=legs,
        timestamps=stamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float64),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=rng.uniform(1.0e12, 2.0e12, size=shape),
        funding_rate=np.zeros(shape, dtype=np.float64),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=("unused",),
        global_feature_names=("unused_global",),
        periods_per_year=PERIODS_PER_YEAR,
    )

    blocks = -(-n_bars // signal_hold_rows)
    block_w = rng.choice([-0.25, 0.0, 0.25], size=(blocks, n_pairs), p=[0.3, 0.2, 0.5])
    weights = np.repeat(block_w, signal_hold_rows, axis=0)[:n_bars]
    ones = np.ones((n_bars, n_pairs), dtype=bool)
    signals = AnchorSignals(
        symbols=pairs,
        close_ns=np.arange(1, n_bars + 1, dtype=np.int64) * 14_400_000_000_000,
        w_aw_bidir_75_25=weights,
        c_weekly_iv=np.full((n_bars, n_pairs), 0.25),
        vol_cushion_tail=np.full(n_bars, 0.9),
        w_aw_regime=weights,
        w_perp_bidir_carry=weights,
        carry_mask=ones,
        perp_mask=ones,
        spot_valid=ones,
        perp_valid=ones,
    )
    return SyntheticStage(dataset=dataset, signals=signals)


__all__ = [
    "BAR",
    "DEFAULT_PAIRS",
    "PERIODS_PER_YEAR",
    "SyntheticStage",
    "synthetic_stage",
]
