"""Target synthesis of the agy anchor (4H settings): carry leg plus 1.5x cushioned alpha.

Lab ``round4_factors.synthesize_portable_alpha_targets`` as called by
``round6_allweather_features.synthesize_aw_pa`` with the ``ANC_BIDIR_75_25`` 4H settings
(``c_scale=1.0``, ``a_scale=1.5 * vol_cushion_tail``, ``w_max_leg=0.55``,
``gross_cap=2.5``): the carry book is long spot and short perp at half weight each, the
bidirectional alpha lives on the perpetual legs, spot is long-only and the whole book is
scaled down to ``gross_cap``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from trade_rl.strategies.agy_anchor import primitives as pr
from trade_rl.strategies.agy_anchor.signals import AnchorSignals

CARRY_SCALE: Final = 1.0
ALPHA_SCALE: Final = 1.5
W_MAX_LEG: Final = 0.55
GROSS_CAP: Final = 2.5
#: Lab ``r3c.DEADBAND``: total turnover below this keeps the current book.
DEADBAND: Final = 0.04
#: Lab ``r3c.STRIDE``: one decision every six 4h bars (daily).
REBALANCE_EVERY_BARS: Final = 6


def synthesize_row(signals: AnchorSignals, row: int) -> tuple[np.ndarray, np.ndarray]:
    """``(spot_weights, perp_weights)`` the anchor wants from signal row ``row``."""
    if not 0 <= row < signals.n_rows:
        raise ValueError("row outside the signal range")
    carry = np.where(
        signals.carry_mask[row], np.maximum(signals.c_weekly_iv[row], 0.0), 0.0
    )
    alpha = np.where(signals.perp_mask[row], signals.w_aw_bidir_75_25[row], 0.0)
    a_scale = ALPHA_SCALE * float(signals.vol_cushion_tail[row])
    ws_raw = 0.5 * float(CARRY_SCALE) * carry
    wp_raw = -0.5 * float(CARRY_SCALE) * carry + a_scale * alpha
    return pr.project_hybrid_weights(
        ws_raw, wp_raw, w_max_leg=W_MAX_LEG, gross_cap=GROSS_CAP
    )


def interleave_legs(spot: np.ndarray, perp: np.ndarray) -> np.ndarray:
    """Alternating ``(spot_0, perp_0, spot_1, perp_1, ...)`` leg vector."""
    legs = np.empty(2 * spot.size, dtype=np.float64)
    legs[0::2] = spot
    legs[1::2] = perp
    return legs


__all__ = [
    "ALPHA_SCALE",
    "CARRY_SCALE",
    "DEADBAND",
    "GROSS_CAP",
    "REBALANCE_EVERY_BARS",
    "W_MAX_LEG",
    "interleave_legs",
    "synthesize_row",
]
