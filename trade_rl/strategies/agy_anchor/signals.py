"""Klines + funding signal chain of the agy anchor ``ANC_BIDIR_75_25``.

Port of the lab round 4 -> 6 -> 7 -> 8 chain restricted to the factors the anchor uses:

* hybrid alphas ``E1`` (42/84-bar momentum), ``E2`` (spot-vs-perp taker-flow
  divergence), ``E3`` (basis z-score) and ``E4`` (3-event funding) -- lab
  ``hybrid_env.build_hybrid_features`` ``A1..A4`` and ``round4_factors.build_round4_pack``;
* ``N3`` (funding excess; the open-interest leg is neutral so it reduces to the funding
  term) and ``N4`` (term basis + funding surprise) -- ``round4_factors``;
* the trend / revert sleeves, ``theta_trend`` regime blend and weekly IV carry --
  ``round6_allweather_features.build_round6_aw_pack``;
* the bidirectional perp carry (Hyperliquid premium neutral) --
  ``round7_frontier_features._build_perp_bidir_carry``;
* ``w_aw_bidir_75_25`` and the tail volatility cushion --
  ``round8_nonlinear_features``.

Neutralised inputs (identical to the parity fixture variant): Hyperliquid premium
(``hl_valid`` all False, so ``z_hl_safe == 0``), Binance open interest / long-short
metrics (``metrics_valid`` all False, so ``z_doi == 0``) and the 15m jump ratio
(``j_mean_ema == 0``, a term that is zero whenever it is below its 0.25 threshold).
Constants are the lab constants, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.strategies.agy_anchor import primitives as pr
from trade_rl.strategies.agy_anchor.inputs import AnchorInputs, funding_observations

# --- lab constants (do not retune) -------------------------------------------------
BLEND_REGIME: Final = 0.75
BLEND_BIDIR: Final = 0.25
CUSHION_SIGMA_REF: Final = 0.180
CUSHION_EXPONENT: Final = 0.70
CUSHION_FLOOR: Final = 0.76
CUSHION_CAP: Final = 1.00
CUSHION_EMA_KEEP: Final = 0.65
CUSHION_EMA_NEW: Final = 0.35
CUSHION_CLOCK_BARS: Final = 6
THETA_BASE: Final = 0.38
THETA_DISPERSION: Final = 0.12
THETA_FUNDING_DISPERSION: Final = 0.08
THETA_LOW: Final = 0.20
THETA_HIGH: Final = 0.55
CARRY_HALFLIFE_BARS: Final = 126.0
CARRY_REBALANCE_BARS: Final = 42
CARRY_NAME_CAP: Final = 0.25
BIDIR_CLOCK_BARS: Final = 18
SLOW_CLOCK_BARS: Final = 18
PERIODS_PER_YEAR_4H: Final = 2190.0
MIN_VALID: Final = 4


@dataclass(frozen=True, slots=True)
class AnchorSignals:
    """Per-row anchor outputs on the symbol axis of the inputs (all causal)."""

    symbols: tuple[str, ...]
    close_ns: np.ndarray
    w_aw_bidir_75_25: np.ndarray
    c_weekly_iv: np.ndarray
    vol_cushion_tail: np.ndarray
    w_aw_regime: np.ndarray
    w_perp_bidir_carry: np.ndarray
    carry_mask: np.ndarray
    perp_mask: np.ndarray
    spot_valid: np.ndarray
    perp_valid: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.close_ns.size)


def _hybrid_alphas(
    inputs: AnchorInputs, fr3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw ``A1..A4`` of the lab hybrid feature builder, as per-row signed weights."""
    p_close = inputs.perp_close
    s_close = inputs.spot_close
    t, k = p_close.shape

    def lag_ret(h: int) -> np.ndarray:
        r = np.zeros((t, k), dtype=np.float64)
        if t > h:
            r[h:] = np.log(p_close[h:] / p_close[:-h])
        return r

    ret_42 = lag_ret(42)
    ret_84 = lag_ret(84)

    basis = np.log(p_close / s_close)
    basis_168h_ma = pr.rolling_mean_causal(basis, 42)
    basis_168h_std = np.maximum(pr.rolling_std_causal(basis, 42), 1e-5)
    basis_z_168h = (basis - basis_168h_ma) / basis_168h_std

    perp_qv = np.where(inputs.perp_valid, inputs.perp_quote_volume, 0.0)
    perp_tb = np.where(inputs.perp_valid, inputs.perp_taker_buy_quote_volume, 0.0)
    spot_qv = np.where(inputs.spot_valid, inputs.spot_quote_volume, 0.0)
    spot_tb = np.where(inputs.spot_valid, inputs.spot_taker_buy_quote_volume, 0.0)
    perp_tbr = np.where(perp_qv > 1.0, perp_tb / np.maximum(perp_qv, 1.0) - 0.5, 0.0)
    spot_tbr = np.where(spot_qv > 1.0, spot_tb / np.maximum(spot_qv, 1.0) - 0.5, 0.0)
    flow_div_24h = pr.rolling_mean_causal(spot_tbr, 6) - pr.rolling_mean_causal(
        perp_tbr, 6
    )

    all_ones = np.ones((t, k), dtype=np.bool_)
    raw_a1 = pr.ema_causal(
        pr.masked_xsec_demean(0.5 * ret_42 + 0.5 * ret_84, all_ones), 12.0
    )
    raw_a2 = pr.ema_causal(pr.masked_xsec_demean(flow_div_24h, all_ones), 30.0)
    raw_a3 = pr.ema_causal(pr.masked_xsec_demean(-basis_z_168h, all_ones), 12.0)
    raw_a4 = pr.ema_causal(
        -(0.5 * fr3 + 0.5 * pr.masked_xsec_demean(fr3, all_ones)), 12.0
    )
    return raw_a1, raw_a2, raw_a3, raw_a4


def _signed_weight_rows(raw: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw, dtype=np.float64)
    for bar in range(raw.shape[0]):
        out[bar] = pr.normalize_signed_weights(raw[bar], w_max=0.25, gross_max=1.0)
    return out


def _term_basis_z(
    inputs: AnchorInputs,
    perp_mask: np.ndarray,
    carry_mask: np.ndarray,
    f_last: np.ndarray,
    f7: np.ndarray,
) -> np.ndarray:
    """Lab ``round4_factors._compute_term_basis_z`` (N4)."""
    basis = np.where(
        carry_mask & (inputs.perp_close > 0.0) & (inputs.spot_close > 0.0),
        np.clip(
            np.log(inputs.perp_close / np.maximum(inputs.spot_close, 1e-8)), -0.05, 0.05
        ),
        0.0,
    )
    term_diff = pr.ema_causal(basis, 6.0) - pr.ema_causal(basis, 42.0)
    basis_vol = np.maximum(pr.rolling_std_causal(basis, 84), 1e-4)
    term_z = np.where(carry_mask, np.clip(term_diff / basis_vol, -4.0, 4.0), 0.0)
    f_surp = pr.masked_xsec_z(f_last - f7, perp_mask, min_valid=MIN_VALID)
    raw = (
        -0.7 * pr.masked_xsec_z(term_z, carry_mask, min_valid=MIN_VALID) - 0.3 * f_surp
    )
    sig = pr.ema_causal(raw, 6.0)
    return pr.neutral_z(sig, perp_mask)


def _funding_excess_z(
    perp_mask: np.ndarray, f3: np.ndarray, f7: np.ndarray
) -> np.ndarray:
    """Lab ``round4_factors._compute_oi_crowd_delev`` (N3) with the OI leg neutral."""
    fund_exc = (f3 - f7) + 0.5 * f3
    z_fund = np.clip(
        pr.masked_xsec_z(fund_exc, perp_mask, min_valid=MIN_VALID), -2.5, 2.5
    )
    raw = -z_fund
    sig = pr.ema_causal(raw, 12.0)
    return pr.neutral_z(sig, perp_mask)


def _weekly_carry_iv(
    f_last: np.ndarray,
    perp_valid: np.ndarray,
    spot_valid: np.ndarray,
    inv_vol: np.ndarray,
) -> np.ndarray:
    """Lab ``round6._compute_weekly_carry_iv``."""
    f_raw = np.nan_to_num(f_last, nan=0.0, posinf=0.0, neginf=0.0)
    f_ema = pr.ema_causal(f_raw, CARRY_HALFLIFE_BARS)
    carry_mask = perp_valid & spot_valid
    t_len, s_len = f_ema.shape
    out = np.zeros((t_len, s_len), dtype=np.float64)
    w_cur = np.zeros(s_len, dtype=np.float64)
    for t in range(t_len):
        if t == 0 or ((t + 1) % CARRY_REBALANCE_BARS == 0):
            m = carry_mask[t]
            if int(m.sum()) >= 2:
                pos = np.where(m, np.maximum(f_ema[t], 0.0) * inv_vol[t], 0.0)
                w_cur = (
                    pr.cap_nonneg_side(pos, CARRY_NAME_CAP, 1.0)
                    if float(pos.sum()) > 1e-12
                    else np.zeros(s_len)
                )
            else:
                w_cur = np.zeros(s_len, dtype=np.float64)
        out[t] = np.where(carry_mask[t], w_cur, 0.0)
    return out


def _tail_cushion(
    c_weekly_iv: np.ndarray,
    w_dir: np.ndarray,
    r_perp: np.ndarray,
    r_spot: np.ndarray,
) -> np.ndarray:
    """Lab ``round8._compute_two_cushions`` tail cushion (full cushion is unused)."""
    t_len = r_perp.shape[0]
    port_ret = np.zeros(t_len, dtype=np.float64)
    if t_len > 1:
        basis_pnl = np.sum(c_weekly_iv[:-1] * (r_spot[1:] - r_perp[1:]), axis=1)
        alpha_pnl = np.sum(w_dir[:-1] * r_perp[1:], axis=1)
        port_ret[1:] = basis_pnl + alpha_pnl
    ret_sq = port_ret * port_ret
    c7 = pr.causal_window_sum(ret_sq[:, None], 42)[:, 0]
    n7 = np.minimum(np.arange(1, t_len + 1, dtype=np.float64), 42.0)
    sigma_7d = np.sqrt(np.maximum(c7 / np.maximum(n7, 6.0), 0.0) * PERIODS_PER_YEAR_4H)
    c30 = pr.causal_window_sum(ret_sq[:, None], 180)[:, 0]
    n30 = np.minimum(np.arange(1, t_len + 1, dtype=np.float64), 180.0)
    sigma_30d = np.sqrt(
        np.maximum(c30 / np.maximum(n30, 12.0), 0.0) * PERIODS_PER_YEAR_4H
    )
    sigma_eff = np.maximum(sigma_7d, sigma_30d)
    raw_tail = np.clip(
        (CUSHION_SIGMA_REF / np.maximum(sigma_eff, CUSHION_SIGMA_REF))
        ** CUSHION_EXPONENT,
        CUSHION_FLOOR,
        CUSHION_CAP,
    )
    raw_tail[:12] = 1.0
    cush_tail = np.ones(t_len, dtype=np.float64)
    ema_t = 1.0
    for t in range(t_len):
        if t == 0 or ((t + 1) % CUSHION_CLOCK_BARS == 0):
            ema_t = CUSHION_EMA_KEEP * ema_t + CUSHION_EMA_NEW * float(raw_tail[t])
        cush_tail[t] = float(np.clip(ema_t, CUSHION_FLOOR, CUSHION_CAP))
    return cush_tail


def build_anchor_signals(inputs: AnchorInputs) -> AnchorSignals:
    """Build the causal anchor outputs for every row of ``inputs``."""
    perp_valid = np.asarray(inputs.perp_valid, dtype=bool)
    spot_valid = np.asarray(inputs.spot_valid, dtype=bool)
    perp_mask = perp_valid
    carry_mask = perp_valid & spot_valid
    t_len, s_len = inputs.perp_close.shape

    f_last, f3, f7 = funding_observations(inputs)

    # hybrid alphas -> E1..E4 (round 4)
    raw_a = _hybrid_alphas(inputs, f3)
    e1, e2, e3, e4 = (
        pr.neutral_z(_signed_weight_rows(raw), perp_mask, min_valid=MIN_VALID)
        for raw in raw_a
    )
    n3 = _funding_excess_z(perp_mask, f3, f7)
    n4 = _term_basis_z(inputs, perp_mask, carry_mask, f_last, f7)

    # round 6: volatility sizing, sleeves, regime blend, weekly carry
    r4h = pr.causal_log_ret(inputs.perp_close, perp_mask, clip_val=0.35)
    sigma_30d = np.maximum(pr.rolling_std_causal(r4h, 180), 0.005)
    inv_vol = sigma_30d ** (-0.75)

    vol_30d_mom = np.maximum(pr.rolling_std_causal(r4h, 180), 0.005)
    r_30d = pr.causal_window_sum(r4h, 180)
    r_14d = pr.causal_window_sum(r4h, 84)
    r_2d = pr.causal_window_sum(r4h, 12)
    mom_slow = ((r_30d - r_2d) + (r_14d - r_2d)) / (2.0 * vol_30d_mom * np.sqrt(84.0))
    z_slow_mom = pr.freeze_raw_on_clock(
        pr.neutral_z(mom_slow, perp_mask), SLOW_CLOCK_BARS
    )
    f_last0 = np.nan_to_num(f_last, nan=0.0)
    basis = np.where(
        carry_mask & (inputs.perp_close > 0.0) & (inputs.spot_close > 0.0),
        np.clip(
            np.log(inputs.perp_close / np.maximum(inputs.spot_close, 1e-8)), -0.05, 0.05
        ),
        0.0,
    )
    z_slow_rev = pr.freeze_raw_on_clock(
        pr.neutral_z(
            0.5 * (-pr.neutral_z(pr.ema_causal(f_last0, 84.0), perp_mask))
            + 0.5 * (-pr.neutral_z(pr.ema_causal(basis, 84.0), carry_mask)),
            perp_mask,
        ),
        SLOW_CLOCK_BARS,
    )
    s_trend = 0.5 * e1 + 0.5 * z_slow_mom
    s_revert_1d = 0.30 * e4 + 0.30 * n4 + 0.15 * e3 + 0.10 * e2 + 0.15 * n3
    s_revert = 0.60 * s_revert_1d + 0.40 * z_slow_rev
    w_s_trend = pr.normalize_iv_balanced(s_trend, perp_mask, inv_vol)
    w_s_revert = pr.normalize_iv_balanced(s_revert, perp_mask, inv_vol)

    c_weekly_iv = _weekly_carry_iv(f_last, perp_valid, spot_valid, inv_vol)

    # all-weather regime blend, theta in [0.20, 0.55]
    next_ret = np.zeros((t_len, s_len), dtype=np.float64)
    next_ret[:-1] = (inputs.perp_close[1:] / inputs.perp_close[:-1]) - 1.0
    ret_clean = np.nan_to_num(next_ret, nan=0.0, posinf=0.0, neginf=0.0)
    ret_known = np.zeros_like(ret_clean)
    ret_known[1:] = ret_clean[:-1]
    cnt_m = np.maximum(perp_mask.sum(axis=1, keepdims=True), 1.0)
    mkt_ret_known = np.where(perp_mask, ret_known, 0.0).sum(axis=1) / cnt_m[:, 0]
    x_disp = np.sqrt(
        np.maximum(
            ((np.where(perp_mask, ret_known - mkt_ret_known[:, None], 0.0)) ** 2).sum(
                axis=1
            )
            / cnt_m[:, 0],
            0.0,
        )
    )
    disp_14d = pr.causal_window_sum(x_disp[:, None], 84)[:, 0] / 84.0
    disp_60d = pr.causal_window_sum(x_disp[:, None], 360)[:, 0] / 360.0
    disp_std = np.maximum(pr.rolling_std_causal(x_disp[:, None], 360)[:, 0], 1e-6)
    disp_z = np.tanh((disp_14d - disp_60d) / disp_std)

    f3_clean = np.nan_to_num(f3, nan=0.0)
    f3_mean = np.where(perp_mask, f3_clean, 0.0).sum(axis=1) / cnt_m[:, 0]
    f3_std = np.sqrt(
        np.maximum(
            ((np.where(perp_mask, f3_clean - f3_mean[:, None], 0.0)) ** 2).sum(axis=1)
            / cnt_m[:, 0],
            0.0,
        )
    )
    f3_m90 = pr.causal_window_sum(f3_std[:, None], 540)[:, 0] / np.minimum(
        np.arange(1, t_len + 1), 540.0
    )
    f3_s90 = np.maximum(pr.rolling_std_causal(f3_std[:, None], 540)[:, 0], 1e-6)
    f3_std_z = np.clip((f3_std - f3_m90) / f3_s90, -1.0, 2.0)

    theta_trend = np.clip(
        THETA_BASE + THETA_DISPERSION * disp_z - THETA_FUNDING_DISPERSION * f3_std_z,
        THETA_LOW,
        THETA_HIGH,
    )
    w_aw_regime = np.zeros((t_len, s_len), dtype=np.float64)
    for t in range(t_len):
        m = perp_mask[t]
        if int(m.sum()) < MIN_VALID:
            continue
        comb = theta_trend[t] * w_s_trend[t] + (1.0 - theta_trend[t]) * w_s_revert[t]
        w_aw_regime[t] = pr.normalize_balanced_neutral_row(
            np.where(m, comb, 0.0), w_max=0.25, gross_max=1.0
        )

    # round 7: bidirectional perp carry (Hyperliquid premium neutral)
    fr_blend = 0.55 * pr.ema_causal(f3_clean, 6.0) + 0.45 * pr.ema_causal(
        np.nan_to_num(f7, nan=0.0), 12.0
    )
    z_carry_dir = pr.neutral_z(-fr_blend, perp_mask, min_valid=MIN_VALID, clip_z=3.0)
    raw_score = 0.50 * z_carry_dir + 0.40 * n4
    frozen_score = pr.freeze_raw_on_clock(raw_score, BIDIR_CLOCK_BARS)
    frozen_inv_vol = pr.freeze_raw_on_clock(inv_vol, BIDIR_CLOCK_BARS)
    w_bidir = pr.normalize_iv_balanced(frozen_score, perp_mask, frozen_inv_vol)

    # round 8: anchor blend and tail cushion
    w_anchor = pr.normalize_neutral_matrix(
        BLEND_REGIME * w_aw_regime + BLEND_BIDIR * w_bidir, perp_mask
    )
    r_perp = pr.causal_log_ret(inputs.perp_close, perp_valid, clip_val=0.35)
    r_spot = pr.causal_log_ret(inputs.spot_close, spot_valid, clip_val=0.35)
    cushion = _tail_cushion(c_weekly_iv, w_anchor, r_perp, r_spot)

    return AnchorSignals(
        symbols=inputs.symbols,
        close_ns=np.asarray(inputs.close_ns),
        w_aw_bidir_75_25=w_anchor,
        c_weekly_iv=c_weekly_iv,
        vol_cushion_tail=cushion,
        w_aw_regime=w_aw_regime,
        w_perp_bidir_carry=w_bidir,
        carry_mask=carry_mask,
        perp_mask=perp_mask,
        spot_valid=spot_valid,
        perp_valid=perp_valid,
    )


__all__ = ["AnchorSignals", "build_anchor_signals"]
