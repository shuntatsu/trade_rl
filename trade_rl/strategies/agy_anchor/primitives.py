"""Numeric primitives of the agy anchor chain, ported operation for operation.

Every function mirrors the agy lab function named in its docstring (lab files
``hybrid_env.py``, ``round4_factors.py``, ``round6_allweather_features.py``,
``round8_nonlinear_features.py``). The order of floating-point operations is kept
identical on purpose: the parity fixture compares the weights bit for bit, so no function
here may be "simplified" or vectorised differently. All of them are causal row by row:
row ``t`` of every output depends only on rows ``<= t`` of the inputs (windows are
trailing, EMAs are recursive).
"""

from __future__ import annotations

import math

import numpy as np

FloatArray = np.ndarray


def rolling_mean_causal(x: FloatArray, window: int) -> FloatArray:
    """Lab ``hybrid_env._rolling_mean_causal`` (expanding head, then a full window)."""
    t, k = x.shape
    out = np.empty((t, k), dtype=np.float64)
    head_limit = min(t, window - 1)
    for idx in range(head_limit):
        out[idx] = np.mean(x[: idx + 1], axis=0)
    if t >= window:
        win = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
        out[window - 1 :] = np.mean(win, axis=2)
    return out


def rolling_std_causal(x: FloatArray, window: int) -> FloatArray:
    """Lab ``hybrid_env._rolling_std_causal`` (population std, zero at row 0)."""
    t, k = x.shape
    out = np.zeros((t, k), dtype=np.float64)
    head_limit = min(t, window - 1)
    for idx in range(1, head_limit):
        out[idx] = np.std(x[: idx + 1], axis=0)
    if t >= window:
        win = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
        out[window - 1 :] = np.std(win, axis=2)
    return out


def ema_causal(x: FloatArray, halflife_bars: float) -> FloatArray:
    """Lab ``hybrid_env._ema_causal`` (recursive, seeded with the first row)."""
    alpha = 1.0 - (0.5 ** (1.0 / halflife_bars))
    one_m = 1.0 - alpha
    t = x.shape[0]
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for bar in range(1, t):
        out[bar] = alpha * x[bar] + one_m * out[bar - 1]
    return out


def masked_xsec_demean(
    x: FloatArray, mask: FloatArray, *, min_valid: int = 4
) -> FloatArray:
    """Lab ``hybrid_env._masked_xsec_demean``."""
    out = np.zeros_like(x, dtype=np.float64)
    cnt = np.sum(mask, axis=1)
    valid_rows = cnt >= min_valid
    if np.any(valid_rows):
        xm = np.where(mask[valid_rows], x[valid_rows], 0.0)
        mean = np.sum(xm, axis=1, keepdims=True) / cnt[valid_rows, None]
        out[valid_rows] = np.where(mask[valid_rows], x[valid_rows] - mean, 0.0)
    return out


def masked_xsec_z(x: FloatArray, mask: FloatArray, *, min_valid: int = 4) -> FloatArray:
    """Lab ``hybrid_env._masked_xsec_z``."""
    out = np.zeros_like(x, dtype=np.float64)
    cnt = np.sum(mask, axis=1)
    valid_rows = cnt >= min_valid
    if np.any(valid_rows):
        m = mask[valid_rows]
        xv = np.where(m, x[valid_rows], 0.0)
        mean = np.sum(xv, axis=1, keepdims=True) / cnt[valid_rows, None]
        diff = np.where(m, x[valid_rows] - mean, 0.0)
        var = np.sum(diff * diff, axis=1, keepdims=True) / cnt[valid_rows, None]
        std = np.sqrt(np.maximum(var, 1e-12))
        out[valid_rows] = np.where(std > 1e-8, diff / std, 0.0)
    return out


def normalize_signed_weights(
    raw: FloatArray, *, w_max: float = 0.25, gross_max: float = 1.0
) -> FloatArray:
    """Lab ``hybrid_env._normalize_signed_weights`` (one row)."""
    out = np.zeros_like(raw, dtype=np.float64)
    abs_sum = float(np.abs(raw).sum())
    if not math.isfinite(abs_sum) or abs_sum <= 1e-12:
        return out
    w = raw / abs_sum * gross_max
    for _ in range(4):
        over = np.abs(w) > w_max
        if not np.any(over):
            break
        capped = np.clip(w, -w_max, w_max)
        used = float(np.abs(capped[over]).sum())
        rem = max(0.0, gross_max - used)
        free = ~over
        free_abs = float(np.abs(w[free]).sum())
        if free_abs <= 1e-12:
            w = capped
            break
        w = capped.copy()
        w[free] = w[free] * (rem / free_abs)
    w = np.clip(w, -w_max, w_max)
    gross = float(np.abs(w).sum())
    if gross > gross_max + 1e-12:
        w = w * (gross_max / gross)
    return w


def project_hybrid_weights(
    w_spot_raw: FloatArray,
    w_perp_raw: FloatArray,
    *,
    w_max_leg: float,
    gross_cap: float,
) -> tuple[FloatArray, FloatArray]:
    """Lab ``hybrid_env.project_hybrid_weights`` (spot long-only, perp signed)."""
    ws = np.clip(w_spot_raw, 0.0, w_max_leg)
    wp = np.clip(w_perp_raw, -w_max_leg, w_max_leg)
    tot = float(np.abs(ws).sum() + np.abs(wp).sum())
    if tot > gross_cap + 1e-12:
        scale = gross_cap / tot
        ws = ws * scale
        wp = wp * scale
    return ws, wp


def causal_window_sum(x: FloatArray, window: int) -> FloatArray:
    """Lab ``round4_factors._causal_window_sum`` (zero-padded trailing sum)."""
    arr = np.nan_to_num(
        np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if window <= 1:
        return arr.copy()
    pad = ((window - 1, 0),) + ((0, 0),) * (arr.ndim - 1)
    padded = np.pad(arr, pad, mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window, axis=0)
    return np.asarray(win.sum(axis=-1))


def causal_log_ret(
    close: FloatArray, valid: FloatArray, clip_val: float = 0.35
) -> FloatArray:
    """Lab ``round4_factors._causal_log_ret``."""
    r = np.zeros_like(close, dtype=np.float64)
    v = valid[1:] & valid[:-1] & (close[1:] > 0.0) & (close[:-1] > 0.0)
    raw = np.zeros_like(close[1:], dtype=np.float64)
    raw[v] = np.log(close[1:][v] / close[:-1][v])
    r[1:] = np.clip(raw, -clip_val, clip_val)
    return r


def neutral_z(
    x: FloatArray, mask: FloatArray, *, min_valid: int = 4, clip_z: float = 3.0
) -> FloatArray:
    """Lab ``round4_factors._neutral_z``."""
    z = masked_xsec_z(np.nan_to_num(x, nan=0.0), mask, min_valid=min_valid)
    z = np.clip(z, -clip_z, clip_z)
    return masked_xsec_demean(z, mask, min_valid=min_valid)


def cap_nonneg_side(vec: FloatArray, w_max: float, target: float) -> FloatArray:
    """Lab ``round4_factors._cap_nonneg_side`` (water-fill under a per-name cap)."""
    out = np.zeros_like(vec, dtype=np.float64)
    pos_mask = vec > 1e-12
    n_pos = int(pos_mask.sum())
    if n_pos == 0 or target <= 1e-12:
        return out
    cap_sum = min(float(target), float(n_pos) * float(w_max))
    w = np.where(pos_mask, vec * (cap_sum / float(vec[pos_mask].sum())), 0.0)
    fixed = np.zeros_like(pos_mask)
    for _ in range(len(vec)):
        over = (~fixed) & (w > w_max + 1e-12)
        if not np.any(over):
            break
        w[over] = w_max
        fixed |= over
        rem = cap_sum - float(w[fixed].sum())
        free = pos_mask & (~fixed)
        if rem <= 1e-12 or not np.any(free):
            w[free] = 0.0
            break
        free_sum = float(vec[free].sum())
        if free_sum <= 1e-12:
            w[free] = 0.0
            break
        w[free] = vec[free] * (rem / free_sum)
    return np.asarray(np.clip(w, 0.0, w_max))


def normalize_balanced_neutral_row(
    row: FloatArray, *, w_max: float = 0.25, gross_max: float = 1.0
) -> FloatArray:
    """Lab ``round4_factors.normalize_balanced_neutral_row`` (sum 0, gross <= max)."""
    pos = np.maximum(row, 0.0)
    neg = np.maximum(-row, 0.0)
    side_target = 0.5 * float(gross_max)
    wp = cap_nonneg_side(pos, w_max, side_target)
    wn = cap_nonneg_side(neg, w_max, side_target)
    sp = float(wp.sum())
    sn = float(wn.sum())
    side = min(sp, sn)
    if side <= 1e-12:
        return np.zeros_like(row, dtype=np.float64)
    if sp > side + 1e-12:
        wp = wp * (side / sp)
    if sn > side + 1e-12:
        wn = wn * (side / sn)
    return np.asarray(wp - wn)


def normalize_iv_balanced(
    score: FloatArray, mask: FloatArray, inv_vol: FloatArray
) -> FloatArray:
    """Lab ``round6_allweather_features._normalize_iv_balanced``."""
    t_len, s_len = score.shape
    out = np.zeros((t_len, s_len), dtype=np.float64)
    dm = masked_xsec_demean(score, mask, min_valid=4)
    for t in range(t_len):
        m = mask[t]
        if int(m.sum()) < 4:
            continue
        row = np.where(m, dm[t] * inv_vol[t], 0.0)
        out[t] = normalize_balanced_neutral_row(row, w_max=0.25, gross_max=1.0)
    return out


def normalize_neutral_matrix(raw: FloatArray, perp_mask: FloatArray) -> FloatArray:
    """Lab ``round8_nonlinear_features._normalize_neutral_matrix``."""
    t_len, s_len = raw.shape
    out = np.zeros((t_len, s_len), dtype=np.float64)
    for t in range(t_len):
        m = perp_mask[t]
        if int(m.sum()) < 4:
            continue
        out[t] = normalize_balanced_neutral_row(
            np.where(m, raw[t], 0.0), w_max=0.25, gross_max=1.0
        )
    return out


def freeze_raw_on_clock(score: FloatArray, clock_bars: int) -> FloatArray:
    """Lab ``round6_allweather_features._freeze_raw_on_clock``."""
    t_len, s_len = score.shape
    out = np.zeros((t_len, s_len), dtype=np.float64)
    cur = np.zeros(s_len, dtype=np.float64)
    for t in range(t_len):
        if t == 0 or ((t + 1) % clock_bars == 0):
            cur = score[t].copy()
        out[t] = cur
    return out
