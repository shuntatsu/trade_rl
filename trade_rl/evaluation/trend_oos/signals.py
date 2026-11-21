"""Causal signals of the out-of-sample test: lowturn's, with unlisted rows masked.

``lowturn_screen.signals.build_signals`` computes every signal from the ``close`` array
and refuses any dataset at or after the 2026-08-01 embargo (so it cannot serve the
FORWARD stage). Its helpers are reused here unchanged; the only addition is the mask: a
row whose leg is inactive (the spot market did not exist yet, or a perpetual that is
never loaded) carries a constant sentinel price in the dataset, so it is turned into NaN
*before* any lookback is formed. A signal is therefore NaN, and the unchanged policy
gives the leg weight 0, until every close its lookback needs is a real published
close. A halted bar keeps its preceding published close as the mark, exactly as in
lowturn. The embargo is enforced by the stage guards of :mod:`trade_rl.evaluation.trend_oos.data`.
"""

from __future__ import annotations

import numpy as np

from trade_rl.data.features.mtf6_signals import validate_alternating_spot_perp_dataset
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.lowturn_screen import protocol as lowturn_protocol
from trade_rl.evaluation.lowturn_screen import signals as lowturn_signals
from trade_rl.evaluation.lowturn_screen.signals import LowturnSignals

_readonly = lowturn_signals._readonly
_lagged_ratio = lowturn_signals._lagged_ratio
_annualized_vol = lowturn_signals._annualized_vol
_sma = lowturn_signals._sma


def build_oos_signals(dataset: MarketDataset) -> LowturnSignals:
    """Precompute the lowturn signals of the dataset with inactive rows as missing."""
    pair_names = validate_alternating_spot_perp_dataset(dataset)
    close_raw = np.asarray(dataset.close, dtype=np.float64)
    priced = np.asarray(dataset.resolved_array("asset_active"), dtype=bool)
    clean = np.where(
        priced & np.isfinite(close_raw) & (close_raw > 0.0), close_raw, np.nan
    )
    usable = lowturn_signals.usable_mask(dataset)

    def gated(values: np.ndarray) -> np.ndarray:
        return np.where(usable, values, np.nan)

    p = lowturn_protocol
    return LowturnSignals(
        pair_names=pair_names,
        symbols=tuple(dataset.symbols),
        r30=_readonly(gated(_lagged_ratio(clean, 0, p.R30_BARS))),
        r90=_readonly(gated(_lagged_ratio(clean, 0, p.R90_BARS))),
        r180=_readonly(gated(_lagged_ratio(clean, 0, p.R180_BARS))),
        vol60=_readonly(gated(_annualized_vol(clean, p.VOL_WINDOW_BARS))),
        xs_score=_readonly(
            gated(_lagged_ratio(clean, p.XS_RECENT_BARS, p.XS_PAST_BARS))
        ),
        sma_fast=_readonly(gated(_sma(clean, p.SMA_FAST_BARS))),
        sma_slow=_readonly(gated(_sma(clean, p.SMA_SLOW_BARS))),
        usable=_readonly(usable),
    )


__all__ = ["build_oos_signals"]
