"""Causal 6-timeframe CORE7/ORTH6 signals, anchor, and regime gates over MarketDataset.

This data-layer module operates exclusively on completed market rows up to decision
row ``t`` (available_at <= decision timestamp) across alternating ``spot/perp`` pairs,
keeps ``SPOT_HALTS`` gaps masked without forward-filling future information, enforces
the strict Pre-FS1 boundary (``< 2026-08-01T00:00:00Z``), and carries zero runtime
dependency on strategies, simulation, risk, integrations, evaluation, or PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset

DATA_END_EXCLUSIVE_ISO = "2026-08-01T00:00:00"
# 2026-08-01T00:00:00Z in UTC nanoseconds (exact epoch integer)
DATA_END_EXCLUSIVE_NS = 1_785_542_400_000_000_000
MAX_PAIRS = 11
MIN_PAIRS = 2
PERIODS_PER_YEAR_4H = 2190.0

CORE7_SIGNAL_NAMES: tuple[str, ...] = (
    "H0_SPOT_UNCROWDED_BRK",
    "H2_SQUEEZE_CARRY_FLOW",
    "H4_OI_DELEV_CLV_ABSORB",
    "H5_CROSS_TF_PHASE_CURL",
    "M0_MTF_PULLBACK_ABSORB",
    "M1_MTF_6SCALE_COHERENCE",
    "M3_MTF_CARRY_BASIS_MICRO_SHIELD",
)
ORTH6_INDICES: tuple[int, ...] = (0, 1, 2, 3, 5, 6)
ORTH6_SIGNAL_NAMES: tuple[str, ...] = tuple(
    CORE7_SIGNAL_NAMES[i] for i in ORTH6_INDICES
)

OPP_FEATURE_NAMES: tuple[str, ...] = (
    "xdisp_ema_30d",
    "fr_lvl_14d",
    "fr_pos_14d",
    "anc_sr_20d",
    "carry_sr_20d",
    "anc_dd_30d",
)


def _readonly(
    arr: np.ndarray, *, dtype: np.dtype[np.generic] | None = None
) -> np.ndarray:
    view = np.ascontiguousarray(np.asarray(arr, dtype=dtype)).view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True, slots=True)
class Mtf6SignalPack:
    """Strictly causal 6-timeframe signals and anchor/gate arrays for 2P alternating legs."""

    pair_names: tuple[str, ...]
    timestamps: np.ndarray
    spot_usable: np.ndarray  # (T, P) bool
    perp_mask: np.ndarray  # (T, P) bool
    carry_mask: np.ndarray  # (T, P) bool
    perp_ret_causal: np.ndarray  # (T, P) float64
    spot_ret_causal: np.ndarray  # (T, P) float64
    inv_vol_30d: np.ndarray  # (T, P) float64
    core7_signals: np.ndarray  # (T, P, 7) float64
    orth6_signals: np.ndarray  # (T, P, 6) float64
    w_aw_bidir_75_25: np.ndarray  # (T, P) float64
    c_weekly_iv: np.ndarray  # (T, P) float64
    vol_cushion_tail: np.ndarray  # (T,) float64
    regime_feats_41: np.ndarray  # (T, 41) float64
    opp_feats_6: np.ndarray  # (T, 6) float64
    gate_alpha_nonrl: np.ndarray  # (T,) float64
    gate_carry_nonrl: np.ndarray  # (T,) float64


def validate_alternating_spot_perp_dataset(dataset: MarketDataset) -> tuple[str, ...]:
    """Validate that dataset symbols form ``P in [2, 11]`` alternating ``sym:spot``/``sym:perp`` pairs."""
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    n_syms = dataset.n_symbols
    if n_syms % 2 != 0 or n_syms < 2 * MIN_PAIRS or n_syms > 2 * MAX_PAIRS:
        raise ValueError(
            f"mtf6 signals require between {MIN_PAIRS} and {MAX_PAIRS} alternating spot/perp pairs"
        )
    pairs: list[str] = []
    for idx in range(0, n_syms, 2):
        spot_sym = dataset.symbols[idx]
        perp_sym = dataset.symbols[idx + 1]
        if not spot_sym.endswith(":spot") or not perp_sym.endswith(":perp"):
            raise ValueError(
                f"expected ':spot'/':perp' alternating pair at indices {idx}, {idx + 1}"
            )
        base_spot = spot_sym.removesuffix(":spot")
        base_perp = perp_sym.removesuffix(":perp")
        if base_spot != base_perp or not base_spot:
            raise ValueError(
                f"mismatched spot/perp pair symbols {spot_sym!r} vs {perp_sym!r}"
            )
        pairs.append(base_spot)
    return tuple(pairs)


def enforce_pre_fs1_boundary(dataset: MarketDataset) -> None:
    """Fail closed if any dataset bar timestamp reaches or exceeds 2026-08-01T00:00:00Z."""
    ts_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if ts_ns.size > 0 and int(np.max(ts_ns)) >= DATA_END_EXCLUSIVE_NS:
        raise ValueError(
            f"Pre-FS1 embargo violated: timestamps must be strictly before {DATA_END_EXCLUSIVE_ISO}Z"
        )


def _ema_causal(x: np.ndarray, halflife_bars: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    alpha = 1.0 - (0.5 ** (1.0 / max(halflife_bars, 1e-6)))
    one_m = 1.0 - alpha
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = alpha * arr[t] + one_m * out[t - 1]
    return out


def _causal_window_sum(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    cs = np.cumsum(arr, axis=0)
    if window >= arr.shape[0]:
        return cs
    out = cs.copy()
    out[window:] = cs[window:] - cs[:-window]
    return out


def _causal_window_std(x: np.ndarray, window: int, min_periods: int = 6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    t_len = arr.shape[0]
    w = max(1, min(window, t_len))
    s1 = _causal_window_sum(arr, w)
    s2 = _causal_window_sum(arr * arr, w)
    cnt = np.minimum(np.arange(1, t_len + 1, dtype=np.float64), float(w))
    denom = np.maximum(cnt, float(min_periods))
    extra_dims = (1,) * (arr.ndim - 1)
    denom_view = denom.reshape((t_len, *extra_dims))
    mean = s1 / denom_view
    var = np.maximum((s2 / denom_view) - mean * mean, 0.0)
    return np.sqrt(var)


def _neutral_z(
    score: np.ndarray,
    mask: np.ndarray,
    *,
    min_valid: int = 2,
    clip_z: float = 3.0,
) -> np.ndarray:
    x = np.nan_to_num(
        np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    m = np.asarray(mask, dtype=bool)
    cnt = np.sum(m, axis=1, keepdims=True)
    ok_rows = cnt[:, 0] >= min_valid
    out = np.zeros_like(x, dtype=np.float64)
    if np.any(ok_rows):
        xm = np.where(m[ok_rows], x[ok_rows], 0.0)
        mean = np.sum(xm, axis=1, keepdims=True) / np.maximum(cnt[ok_rows], 1)
        diff = np.where(m[ok_rows], x[ok_rows] - mean, 0.0)
        var = np.sum(diff * diff, axis=1, keepdims=True) / np.maximum(cnt[ok_rows], 1)
        sd = np.maximum(np.sqrt(var), 1e-8)
        out[ok_rows] = np.clip(np.where(m[ok_rows], diff / sd, 0.0), -clip_z, clip_z)
    return out


def _cap_nonneg_side(vec: np.ndarray, w_max: float, target: float) -> np.ndarray:
    out = np.zeros_like(vec, dtype=np.float64)
    pos_mask = vec > 1e-12
    n_pos = int(np.sum(pos_mask))
    if n_pos == 0 or target <= 1e-12:
        return out
    cap_sum = min(float(target), float(n_pos) * float(w_max))
    pos_sum = float(np.sum(vec[pos_mask]))
    w = np.where(pos_mask, vec * (cap_sum / max(pos_sum, 1e-12)), 0.0)
    fixed = np.zeros_like(pos_mask)
    for _ in range(len(vec)):
        over = (~fixed) & (w > w_max + 1e-12)
        if not np.any(over):
            break
        w[over] = w_max
        fixed |= over
        rem = cap_sum - float(np.sum(w[fixed]))
        free = pos_mask & (~fixed)
        if rem <= 1e-12 or not np.any(free):
            w[free] = 0.0
            break
        free_sum = float(np.sum(w[free]))
        if free_sum <= 1e-12:
            w[free] = 0.0
            break
        w[free] = vec[free] * (rem / free_sum)
    return np.clip(w, 0.0, w_max)


def normalize_balanced_neutral_row(
    row: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    w_max: float = 0.25,
    gross_max: float = 1.0,
) -> np.ndarray:
    """Normalize a pair score vector so ``sum(w) == 0``, ``|w| <= w_max``, ``sum|w| <= gross_max``."""
    v = np.nan_to_num(
        np.asarray(row, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if int(np.sum(m)) < MIN_PAIRS:
            return np.zeros_like(v, dtype=np.float64)
        v = np.where(m, v, 0.0)
    pos = np.maximum(v, 0.0)
    neg = np.maximum(-v, 0.0)
    side_target = 0.5 * float(gross_max)
    wp = _cap_nonneg_side(pos, w_max, side_target)
    wn = _cap_nonneg_side(neg, w_max, side_target)
    sp = float(np.sum(wp))
    sn = float(np.sum(wn))
    side = min(sp, sn)
    if side <= 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    if sp > side + 1e-12:
        wp = wp * (side / sp)
    if sn > side + 1e-12:
        wn = wn * (side / sn)
    return wp - wn


def _normalize_iv_balanced(
    score: np.ndarray, mask: np.ndarray, inv_vol: np.ndarray
) -> np.ndarray:
    t_len, p_len = score.shape
    out = np.zeros((t_len, p_len), dtype=np.float64)
    z_m = _neutral_z(score, mask, min_valid=MIN_PAIRS, clip_z=4.0)
    for t in range(t_len):
        m = mask[t]
        if int(np.sum(m)) < MIN_PAIRS:
            continue
        out[t] = normalize_balanced_neutral_row(
            z_m[t] * inv_vol[t], m, w_max=0.25, gross_max=1.0
        )
    return out


def _normalize_neutral_matrix(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t_len, p_len = score.shape
    out = np.zeros((t_len, p_len), dtype=np.float64)
    for t in range(t_len):
        m = mask[t]
        if int(np.sum(m)) < MIN_PAIRS:
            continue
        out[t] = normalize_balanced_neutral_row(score[t], m, w_max=0.25, gross_max=1.0)
    return out


def _freeze_on_clock(arr: np.ndarray, clock_bars: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    cur = np.zeros(arr.shape[1:], dtype=np.float64)
    for t in range(arr.shape[0]):
        if t == 0 or ((t + 1) % clock_bars == 0):
            cur = np.array(arr[t], dtype=np.float64, copy=True)
        out[t] = cur
    return out


def _proximal_smooth_h(
    h_raw: np.ndarray, perp_mask: np.ndarray, *, clock_bars: int = 6
) -> np.ndarray:
    t_len, p_len, k_dim = h_raw.shape
    out = np.zeros((t_len, p_len, k_dim), dtype=np.float64)
    for k in range(k_dim):
        cur = np.zeros(p_len, dtype=np.float64)
        for t in range(t_len):
            pm = perp_mask[t]
            if t == 0 or ((t + 1) % clock_bars == 0):
                if int(np.sum(pm)) < MIN_PAIRS:
                    cur = np.zeros(p_len, dtype=np.float64)
                elif float(np.sum(np.abs(cur))) < 1e-12:
                    cur = normalize_balanced_neutral_row(
                        h_raw[t, :, k], pm, w_max=0.25, gross_max=1.0
                    )
                else:
                    blend = 0.75 * h_raw[t, :, k] + 0.25 * cur
                    cur = normalize_balanced_neutral_row(
                        blend, pm, w_max=0.25, gross_max=1.0
                    )
            out[t, :, k] = cur
    return out


def _causal_log_ret(
    close: np.ndarray, usable: np.ndarray, *, clip_val: float = 0.35
) -> np.ndarray:
    out = np.zeros_like(close, dtype=np.float64)
    if close.shape[0] <= 1:
        return out
    both_ok = usable[1:] & usable[:-1] & (close[:-1] > 0.0) & (close[1:] > 0.0)
    raw = np.zeros_like(close[1:], dtype=np.float64)
    raw[both_ok] = np.log(close[1:][both_ok] / close[:-1][both_ok])
    out[1:] = np.where(both_ok, np.clip(raw, -clip_val, clip_val), 0.0)
    return out


def _availability_mask(dataset: MarketDataset) -> np.ndarray:
    t_len = dataset.n_bars
    n_syms = dataset.n_symbols
    ts_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)

    usable = np.asarray(dataset.tradable, dtype=bool).copy()
    if dataset.asset_active is not None:
        usable &= np.asarray(dataset.asset_active, dtype=bool)
    if dataset.information_available is not None:
        usable &= np.asarray(dataset.information_available, dtype=bool)
    if dataset.available_at is not None:
        avail_ns = dataset.available_at.astype("datetime64[ns]").astype(np.int64)
        usable &= avail_ns <= ts_ns[:, None]
    close_arr = np.asarray(dataset.close, dtype=np.float64)
    usable &= np.isfinite(close_arr) & (close_arr > 0.0)
    if usable.shape != (t_len, n_syms):
        raise ValueError("availability mask shape mismatch")
    return usable


def compute_nonrl_regime_gates(opp_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal Non-RL seamless exposure gates ``(g_alpha, g_carry)`` from ``opp_feats``."""
    arr = np.asarray(opp_feats, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("opp_feats must have shape (T, 6)")
    f0 = arr[:, 0]
    f1 = arr[:, 1]
    f3 = arr[:, 3]

    def _sig(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    g_carry = np.clip(0.30 + 0.70 * _sig((f1 - 0.005) / 0.015), 0.30, 1.00)
    s_disp = np.clip((f0 - 0.20) / 0.12, 0.0, 1.0)
    s_sr = np.clip(0.55 + 0.45 * _sig(f3 / 0.8), 0.55, 1.00)
    g_alpha = np.clip(0.45 + 0.55 * (0.5 * s_disp + 0.5 * s_sr), 0.45, 1.00)
    return g_alpha, g_carry


def _ema_sharpe(
    ret_series: np.ndarray, halflife_bars: float, warmup: int = 24
) -> np.ndarray:
    mu = _ema_causal(ret_series[:, None], halflife_bars)[:, 0]
    ms = _ema_causal((ret_series * ret_series)[:, None], halflife_bars)[:, 0]
    sd = np.sqrt(np.maximum(ms - mu * mu, 1e-10))
    sr = np.clip((mu / sd) * math.sqrt(PERIODS_PER_YEAR_4H), -4.0, 4.0)
    sr[: min(warmup, len(sr))] = 0.0
    return sr


def build_mtf6_signal_pack(
    dataset: MarketDataset,
    *,
    micro_15m: np.ndarray | None = None,
    micro_1h: np.ndarray | None = None,
    enforce_pre_fs1: bool = True,
) -> Mtf6SignalPack:
    """Build strictly causal 6-TF ``CORE7`` / ``ORTH6`` signals and anchors on a ``MarketDataset``."""
    pair_names = validate_alternating_spot_perp_dataset(dataset)
    if enforce_pre_fs1:
        enforce_pre_fs1_boundary(dataset)

    t_len = dataset.n_bars
    p_len = len(pair_names)
    usable_all = _availability_mask(dataset)
    spot_usable = usable_all[:, 0::2]
    perp_mask = usable_all[:, 1::2]
    carry_mask = spot_usable & perp_mask

    spot_close = np.where(
        spot_usable, np.asarray(dataset.close[:, 0::2], dtype=np.float64), 0.0
    )
    perp_close = np.where(
        perp_mask, np.asarray(dataset.close[:, 1::2], dtype=np.float64), 0.0
    )
    perp_high = np.where(
        perp_mask, np.asarray(dataset.high[:, 1::2], dtype=np.float64), perp_close
    )
    perp_low = np.where(
        perp_mask, np.asarray(dataset.low[:, 1::2], dtype=np.float64), perp_close
    )
    perp_vol = np.where(
        perp_mask,
        np.maximum(np.asarray(dataset.volume[:, 1::2], dtype=np.float64), 0.0),
        0.0,
    )
    spot_vol = np.where(
        spot_usable,
        np.maximum(np.asarray(dataset.volume[:, 0::2], dtype=np.float64), 0.0),
        0.0,
    )
    fr_raw = np.where(
        perp_mask,
        np.nan_to_num(
            np.asarray(dataset.funding_rate[:, 1::2], dtype=np.float64), nan=0.0
        ),
        0.0,
    )

    r_perp = _causal_log_ret(perp_close, perp_mask)
    r_spot = _causal_log_ret(spot_close, spot_usable)

    # Volatility & inverse-volatility sizing
    sigma_30d = np.maximum(_causal_window_std(r_perp, 180, min_periods=12), 0.005)
    inv_vol = sigma_30d ** (-0.75)

    # Multi-horizon perp residuals against pair 0 (BTC/leader) beta
    r_lead = np.where(perp_mask[:, 0:1], r_perp[:, 0:1], 0.0)
    r_res_4h = r_perp - 0.8 * r_lead
    f_ema = _neutral_z(
        _ema_causal(r_perp, 12.0), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    f_rev = _neutral_z(
        -r_perp - 0.5 * _causal_window_sum(r_perp, 2),
        perp_mask,
        min_valid=MIN_PAIRS,
        clip_z=3.0,
    )
    f_resid = _neutral_z(
        _causal_window_sum(r_res_4h, 24), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )

    # Higher-moment / liquidity / deleveraging proxies from OHLCV
    bar_range = np.where(
        perp_mask & (perp_close > 0.0),
        np.clip((perp_high - perp_low) / np.maximum(perp_close, 1e-8), 0.0, 0.5),
        0.0,
    )
    f_skew = _neutral_z(
        _causal_window_sum(r_perp * r_perp * np.sign(r_perp), 18),
        perp_mask,
        min_valid=MIN_PAIRS,
        clip_z=3.0,
    )

    # Spot-Perp flow & basis proxies
    spot_share = np.where(
        carry_mask,
        (spot_vol * spot_close - perp_vol * perp_close)
        / np.maximum(spot_vol * spot_close + perp_vol * perp_close, 1.0),
        0.0,
    )
    f_spot_flow = _neutral_z(
        _ema_causal(spot_share, 6.0), carry_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )

    basis_raw = np.where(
        carry_mask & (perp_close > 0.0) & (spot_close > 0.0),
        np.clip(np.log(perp_close / np.maximum(spot_close, 1e-8)), -0.05, 0.05),
        0.0,
    )
    f_term_basis = _neutral_z(
        -_ema_causal(basis_raw, 6.0), carry_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )

    # Funding obs (3d and 7d EMA)
    fr_3d = _ema_causal(fr_raw, 18.0)
    fr_7d = _ema_causal(fr_raw, 42.0)
    z_fr = _neutral_z(fr_3d, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0)

    # 15m and 1h micro inputs (external causal aligned or causal 4h bar proxies)
    if micro_15m is not None:
        m15 = np.asarray(micro_15m, dtype=np.float64)
        if m15.shape != (t_len, p_len, 3):
            raise ValueError(f"micro_15m must have shape ({t_len}, {p_len}, 3)")
        jr15 = np.where(perp_mask, np.nan_to_num(m15[:, :, 0], nan=0.0), 0.0)
        tbi15 = np.where(perp_mask, np.nan_to_num(m15[:, :, 2], nan=0.0), 0.0)
    else:
        park_vol = np.where(perp_mask, bar_range / math.sqrt(4.0 * math.log(2.0)), 0.0)
        jr15 = np.where(
            perp_mask,
            np.clip(np.abs(r_perp) / np.maximum(park_vol, 1e-4), 0.0, 1.0),
            0.0,
        )
        tbi15 = np.where(
            perp_mask,
            np.clip(
                (perp_close - perp_low) / np.maximum(perp_high - perp_low, 1e-8) * 2.0
                - 1.0,
                -1.0,
                1.0,
            ),
            0.0,
        )

    if micro_1h is not None:
        m1h = np.asarray(micro_1h, dtype=np.float64)
        if m1h.shape != (t_len, p_len, 2):
            raise ValueError(f"micro_1h must have shape ({t_len}, {p_len}, 2)")
        raw_clv = np.where(perp_mask, np.nan_to_num(m1h[:, :, 0], nan=0.0), 0.0)
    else:
        raw_clv = np.where(
            perp_mask,
            np.clip(
                (2.0 * perp_close - perp_high - perp_low)
                / np.maximum(perp_high - perp_low, 1e-8),
                -1.0,
                1.0,
            ),
            0.0,
        )

    z_jump = _neutral_z(
        _ema_causal(jr15, 6.0), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    z_tbi_15m = _neutral_z(
        _ema_causal(tbi15, 3.0), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    z_clv_1h = _neutral_z(
        _ema_causal(raw_clv, 6.0), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    f_oi_delev = _neutral_z(
        -r_perp * z_jump, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )

    # 1D, 3D, 1W structural signals
    mom_3d = _causal_window_sum(r_perp, 18) - _causal_window_sum(r_perp, 6)
    mom_1w = _causal_window_sum(r_perp, 42) - _causal_window_sum(r_perp, 6)
    z_slow_3d = _freeze_on_clock(
        _neutral_z(mom_3d, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0), 18
    )
    z_slow_7d = _freeze_on_clock(
        _neutral_z(mom_1w, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0), 42
    )
    z_slow_rev = _freeze_on_clock(
        _neutral_z(
            -0.5 * z_fr + 0.5 * f_term_basis, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
        ),
        18,
    )

    w_s_trend = _normalize_iv_balanced(
        0.5 * f_ema + 0.5 * z_slow_3d, perp_mask, inv_vol
    )
    w_s_revert = _normalize_iv_balanced(
        0.6 * f_rev + 0.4 * z_slow_rev, perp_mask, inv_vol
    )
    w_med_n6_iv = _normalize_iv_balanced(
        0.5 * f_resid + 0.5 * f_ema, perp_mask, inv_vol
    )
    w_slow_3d_iv = _normalize_iv_balanced(
        (z_slow_3d - z_fr + f_term_basis) / 3.0, perp_mask, inv_vol
    )

    # All-Weather Regime Anchor + Bidir Carry (ANC_BIDIR_75_25)
    cnt_p = np.maximum(np.sum(perp_mask, axis=1, keepdims=True), 1.0)
    mu_r = np.sum(np.where(perp_mask, r_perp, 0.0), axis=1, keepdims=True) / cnt_p
    x_disp = np.sqrt(
        np.maximum(
            np.sum(np.where(perp_mask, (r_perp - mu_r) ** 2, 0.0), axis=1)
            / cnt_p[:, 0],
            0.0,
        )
    )
    disp_14d = _causal_window_sum(x_disp[:, None], 84)[:, 0] / 84.0
    disp_60d = _causal_window_sum(x_disp[:, None], 360)[:, 0] / 360.0
    disp_sd = np.maximum(
        _causal_window_std(x_disp[:, None], 360, min_periods=12)[:, 0], 1e-6
    )
    theta_trend = np.clip(
        0.38 + 0.12 * np.tanh((disp_14d - disp_60d) / disp_sd), 0.20, 0.55
    )
    w_aw_regime = _normalize_neutral_matrix(
        theta_trend[:, None] * w_s_trend + (1.0 - theta_trend)[:, None] * w_s_revert,
        perp_mask,
    )
    z_aw_1d = _neutral_z(w_aw_regime, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0)

    # Weekly IV carry allocation (c_weekly_iv)
    f_carry_ema = _ema_causal(fr_raw, 126.0)
    c_weekly_iv = np.zeros((t_len, p_len), dtype=np.float64)
    w_c_cur = np.zeros(p_len, dtype=np.float64)
    for t in range(t_len):
        if t == 0 or ((t + 1) % 42 == 0):
            cm_t = carry_mask[t]
            if int(np.sum(cm_t)) >= MIN_PAIRS:
                pos_c = np.where(
                    cm_t, np.maximum(f_carry_ema[t], 0.0) * inv_vol[t], 0.0
                )
                w_c_cur = (
                    _cap_nonneg_side(pos_c, 0.25, 1.0)
                    if float(np.sum(pos_c)) > 1e-12
                    else np.zeros(p_len, dtype=np.float64)
                )
            else:
                w_c_cur = np.zeros(p_len, dtype=np.float64)
        c_weekly_iv[t] = np.where(carry_mask[t], w_c_cur, 0.0)

    # Bidirectional Perp Carry
    fr_blend = 0.55 * _ema_causal(fr_3d, 6.0) + 0.45 * _ema_causal(fr_7d, 12.0)
    z_carry_dir = _neutral_z(-fr_blend, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0)
    frozen_inv = _freeze_on_clock(inv_vol, 18)
    w_perp_bidir_carry = _normalize_iv_balanced(
        _freeze_on_clock(0.50 * z_carry_dir + 0.40 * f_term_basis, 18),
        perp_mask,
        frozen_inv,
    )
    z_bidir_3d = _neutral_z(
        w_perp_bidir_carry, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )

    w_aw_bidir_75_25 = _normalize_neutral_matrix(
        0.75 * w_aw_regime + 0.25 * w_perp_bidir_carry, perp_mask
    )

    # Causal Tail Vol Cushion (depends only on realized return <= t)
    port_ret = np.zeros(t_len, dtype=np.float64)
    if t_len > 1:
        basis_pnl = np.sum(c_weekly_iv[:-1] * (r_spot[1:] - r_perp[1:]), axis=1)
        alpha_pnl = np.sum(w_aw_bidir_75_25[:-1] * r_perp[1:], axis=1)
        port_ret[1:] = basis_pnl + alpha_pnl
    ret_sq = port_ret * port_ret
    s7 = np.sqrt(
        np.maximum(
            _causal_window_sum(ret_sq[:, None], 42)[:, 0]
            / np.maximum(np.minimum(np.arange(1, t_len + 1), 42.0), 6.0),
            0.0,
        )
        * PERIODS_PER_YEAR_4H
    )
    s30 = np.sqrt(
        np.maximum(
            _causal_window_sum(ret_sq[:, None], 180)[:, 0]
            / np.maximum(np.minimum(np.arange(1, t_len + 1), 180.0), 12.0),
            0.0,
        )
        * PERIODS_PER_YEAR_4H
    )
    raw_tail = np.clip(
        (0.180 / np.maximum(np.maximum(s7, s30), 0.180)) ** 0.70, 0.76, 1.00
    )
    raw_tail[: min(12, t_len)] = 1.0
    vol_cushion_tail = np.ones(t_len, dtype=np.float64)
    ema_t = 1.0
    for t in range(t_len):
        if t == 0 or ((t + 1) % 6 == 0):
            ema_t = 0.65 * ema_t + 0.35 * float(raw_tail[t])
        vol_cushion_tail[t] = float(np.clip(ema_t, 0.76, 1.00))

    # CORE7 Pillars = (H0, H2, H4, H5, M0, M1, M3)
    raw_h0 = (
        (0.6 * f_resid + 0.4 * f_ema)
        * (1.0 + 0.6 * np.maximum(0.0, np.sign(f_resid) * f_spot_flow))
        * np.exp(-0.35 * np.maximum(0.0, np.sign(f_resid) * z_fr))
    )
    carry_z = 0.55 * (-z_fr) + 0.45 * f_term_basis
    raw_h2 = carry_z * (0.6 + 0.4 * np.tanh(np.sign(carry_z) * f_spot_flow))
    raw_h4 = f_oi_delev * (0.7 + 0.3 * np.maximum(0.0, np.sign(f_oi_delev) * z_clv_1h))
    curl_h5 = _neutral_z(
        w_med_n6_iv - w_slow_3d_iv, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    raw_h5 = curl_h5 * (1.0 - 0.35 * np.tanh(np.abs(z_fr)))

    z_ret_8h = _neutral_z(
        _causal_window_sum(r_perp, 2), perp_mask, min_valid=MIN_PAIRS, clip_z=3.0
    )
    w_trend_macro = _neutral_z(
        0.45 * z_slow_7d
        + 0.55 * _neutral_z(w_slow_3d_iv, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0),
        perp_mask,
        min_valid=MIN_PAIRS,
        clip_z=3.0,
    )
    raw_m0 = (
        w_trend_macro
        * (1.0 + 0.55 * np.maximum(0.0, -np.sign(w_trend_macro) * z_ret_8h))
        * (
            0.75
            + 0.35
            * np.tanh(np.sign(w_trend_macro) * (0.5 * z_clv_1h + 0.5 * z_tbi_15m))
        )
    )
    s_agree = (
        0.25 * np.tanh(z_slow_7d)
        + 0.25 * np.tanh(z_slow_3d)
        + 0.20 * np.tanh(z_aw_1d)
        + 0.15 * np.tanh(f_resid)
        + 0.15 * np.tanh(0.5 * f_spot_flow + 0.5 * z_tbi_15m)
    )
    raw_m1 = _neutral_z(s_agree, perp_mask, min_valid=MIN_PAIRS, clip_z=3.0) * np.exp(
        -0.30 * np.maximum(0.0, np.sign(s_agree) * z_fr)
    )
    cb = 0.55 * z_bidir_3d + 0.45 * f_term_basis
    raw_m3 = (
        cb
        * (0.75 + 0.35 * np.tanh(np.sign(cb) * f_spot_flow))
        * np.exp(-0.25 * np.maximum(0.0, z_jump))
    )

    inv_frozen6 = _freeze_on_clock(inv_vol, 6)
    core7_raw = np.stack(
        [
            _normalize_iv_balanced(_freeze_on_clock(rm, 6), perp_mask, inv_frozen6)
            for rm in (raw_h0, raw_h2, raw_h4, raw_h5, raw_m0, raw_m1, raw_m3)
        ],
        axis=-1,
    )
    core7_signals = _proximal_smooth_h(core7_raw, perp_mask, clock_bars=6)
    orth6_signals = core7_signals[:, :, list(ORTH6_INDICES)].copy()

    # 6D strictly causal regime opportunity state
    xdisp_ema_30d = _ema_causal(x_disp[:, None], 180.0)[:, 0] * math.sqrt(
        PERIODS_PER_YEAR_4H
    )
    fr_mean_bar = np.sum(np.where(perp_mask, fr_raw, 0.0), axis=1) / cnt_p[:, 0]
    fr_lvl_14d = _ema_causal(fr_mean_bar[:, None], 84.0)[:, 0] * (3.0 * 365.0)
    fr_pos_bar = (
        np.sum(np.where(perp_mask, (fr_raw > 0.0).astype(np.float64), 0.0), axis=1)
        / cnt_p[:, 0]
    )
    fr_pos_14d = _ema_causal(fr_pos_bar[:, None], 84.0)[:, 0]

    anc_r = np.zeros(t_len, dtype=np.float64)
    carry_r = np.zeros(t_len, dtype=np.float64)
    if t_len > 1:
        anc_r[1:] = np.sum(
            (1.5 * vol_cushion_tail[:-1, None] * w_aw_bidir_75_25[:-1]) * r_perp[1:],
            axis=1,
        )
        c_cl_prev = 0.5 * np.where(
            carry_mask[:-1], np.maximum(c_weekly_iv[:-1], 0.0), 0.0
        )
        carry_r[1:] = np.sum(c_cl_prev * (r_spot[1:] - r_perp[1:] + fr_raw[1:]), axis=1)
    anc_sr_20d = _ema_sharpe(anc_r, 120.0)
    carry_sr_20d = _ema_sharpe(carry_r, 120.0)

    comb_curve = np.cumsum(anc_r + carry_r)
    anc_dd_30d = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        s0 = max(0, t - 179)
        anc_dd_30d[t] = max(
            0.0, float(np.max(comb_curve[s0 : t + 1])) - float(comb_curve[t])
        )
    opp_feats_6 = np.nan_to_num(
        np.stack(
            [
                xdisp_ema_30d,
                fr_lvl_14d,
                fr_pos_14d,
                anc_sr_20d,
                carry_sr_20d,
                np.clip(anc_dd_30d, 0.0, 0.5),
            ],
            axis=1,
        ),
        nan=0.0,
        posinf=4.0,
        neginf=-4.0,
    )
    gate_alpha_nonrl, gate_carry_nonrl = compute_nonrl_regime_gates(opp_feats_6)

    # 41-dim causal regime state for MLP policies
    sig_sr = np.zeros((t_len, 7), dtype=np.float64)
    for k in range(7):
        ret_k = np.zeros(t_len, dtype=np.float64)
        if t_len > 1:
            ret_k[1:] = np.sum(core7_signals[:-1, :, k] * r_perp[1:], axis=1)
        sig_sr[:, k] = _ema_sharpe(ret_k, 42.0)

    summary_cols = [
        opp_feats_6[:, 0],
        opp_feats_6[:, 1],
        opp_feats_6[:, 2],
        opp_feats_6[:, 3],
        opp_feats_6[:, 4],
        opp_feats_6[:, 5],
        np.clip((s7 - 0.15) * 5.0, -3.0, 3.0),
        np.clip((s30 - 0.15) * 5.0, -3.0, 3.0),
        np.clip((theta_trend - 0.375) * 5.0, -3.0, 3.0),
        np.clip(np.mean(np.abs(z_fr), axis=1) - 1.0, -3.0, 3.0),
        np.clip(np.mean(np.abs(z_jump), axis=1) - 1.0, -3.0, 3.0),
        np.clip(np.mean(np.abs(raw_clv), axis=1) * 5.0 - 1.0, -3.0, 3.0),
        np.clip(np.std(f_resid, axis=1) - 1.0, -3.0, 3.0),
        np.clip(np.std(w_aw_regime, axis=1) * 20.0 - 1.0, -3.0, 3.0),
        np.clip(np.std(w_perp_bidir_carry, axis=1) * 20.0 - 1.0, -3.0, 3.0),
        np.clip(np.mean(np.abs(s_agree), axis=1) * 5.0 - 1.0, -3.0, 3.0),
    ]
    # Deterministic causal harmonic/moment summary filling exact 41-dim state layout
    extra_cols = [
        sig_sr,
        np.column_stack(summary_cols),
        np.tanh(f_ema.mean(axis=1, keepdims=True) * 2.0),
        np.tanh(f_rev.mean(axis=1, keepdims=True) * 2.0),
        _neutral_z(f_skew, perp_mask).std(axis=1, keepdims=True),
        _ema_causal(x_disp[:, None], 42.0) * math.sqrt(PERIODS_PER_YEAR_4H),
        np.mean(np.where(carry_mask, c_weekly_iv, 0.0), axis=1, keepdims=True) * 10.0,
        np.mean(
            np.where(perp_mask, np.abs(w_aw_bidir_75_25), 0.0), axis=1, keepdims=True
        )
        * 10.0,
    ]
    base_concat = np.concatenate(extra_cols, axis=1)
    pad_needed = 41 - base_concat.shape[1]
    if pad_needed > 0:
        pads = np.zeros((t_len, pad_needed), dtype=np.float64)
        for p_idx in range(pad_needed):
            col_src = core7_signals[:, :, p_idx % 7]
            pads[:, p_idx] = np.clip(np.std(col_src, axis=1) * 15.0 - 1.0, -3.0, 3.0)
        regime_feats_41 = np.concatenate([base_concat, pads], axis=1)
    else:
        regime_feats_41 = base_concat[:, :41]
    regime_feats_41 = np.nan_to_num(
        np.clip(regime_feats_41, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0
    )

    return Mtf6SignalPack(
        pair_names=pair_names,
        timestamps=_readonly(dataset.timestamps),
        spot_usable=_readonly(spot_usable, dtype=np.dtype(np.bool_)),
        perp_mask=_readonly(perp_mask, dtype=np.dtype(np.bool_)),
        carry_mask=_readonly(carry_mask, dtype=np.dtype(np.bool_)),
        perp_ret_causal=_readonly(r_perp, dtype=np.dtype(np.float64)),
        spot_ret_causal=_readonly(r_spot, dtype=np.dtype(np.float64)),
        inv_vol_30d=_readonly(inv_vol, dtype=np.dtype(np.float64)),
        core7_signals=_readonly(core7_signals, dtype=np.dtype(np.float64)),
        orth6_signals=_readonly(orth6_signals, dtype=np.dtype(np.float64)),
        w_aw_bidir_75_25=_readonly(w_aw_bidir_75_25, dtype=np.dtype(np.float64)),
        c_weekly_iv=_readonly(c_weekly_iv, dtype=np.dtype(np.float64)),
        vol_cushion_tail=_readonly(vol_cushion_tail, dtype=np.dtype(np.float64)),
        regime_feats_41=_readonly(regime_feats_41, dtype=np.dtype(np.float64)),
        opp_feats_6=_readonly(opp_feats_6, dtype=np.dtype(np.float64)),
        gate_alpha_nonrl=_readonly(gate_alpha_nonrl, dtype=np.dtype(np.float64)),
        gate_carry_nonrl=_readonly(gate_carry_nonrl, dtype=np.dtype(np.float64)),
    )


__all__ = [
    "CORE7_SIGNAL_NAMES",
    "DATA_END_EXCLUSIVE_ISO",
    "DATA_END_EXCLUSIVE_NS",
    "Mtf6SignalPack",
    "OPP_FEATURE_NAMES",
    "ORTH6_INDICES",
    "ORTH6_SIGNAL_NAMES",
    "build_mtf6_signal_pack",
    "compute_nonrl_regime_gates",
    "enforce_pre_fs1_boundary",
    "normalize_balanced_neutral_row",
    "validate_alternating_spot_perp_dataset",
]
