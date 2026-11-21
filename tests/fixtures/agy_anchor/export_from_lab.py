"""Export the agy anchor parity fixture from the read-only agy lab (not run in CI).

Two reference builds are exported. ``out_*`` is the lab as it is. ``fix_*`` is the same
lab with ONE defect corrected: lab ``round7_frontier_features._build_perp_bidir_carry``
takes ``pack4.factor_scores[:, :, 9]`` (axis 2 is the SYMBOL axis, so this is symbol index
9's eleven factor scores, which only broadcast against the eleven symbols by the
coincidence 11 factors == 11 symbols) where the comment says "N4_TERM_BASIS_Z"
(per-symbol, ``factor_scores[:, 9, :]``). The port implements the intended per-symbol N4;
see README.md.

Usage: ``python export_from_lab.py <rl_lab_dir> <output.npz>``

Rebuilds lab rounds 4 -> 8 on the lab's own cached pre-2026-08 data with the
"klines + funding only, taker kept" variant (Hyperliquid, Binance OI metrics and the 15m
jump neutralised) and writes the window 2021-01-01 .. 2021-12-31 (2190 4h rows). The lab
directory is only read; nothing is written there (``sys.dont_write_bytecode``).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

N_ROWS = (
    2190  # 365 days x 6 four-hour bars (2021-01-01T00:00Z .. 2021-12-31T20:00Z open)
)


def main(lab_dir: str, out_path: str) -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, lab_dir)
    import round10_mtf6_features as r10f  # type: ignore[import-not-found]

    r10f.install_scipy_stats_shim()  # the lab imports scipy.stats lazily; shim first
    import hybrid_env as he  # type: ignore[import-not-found]
    import round4_factors as r4f  # type: ignore[import-not-found]
    import round5_mtf_features as r5f  # type: ignore[import-not-found]
    import round6_allweather_features as r6f  # type: ignore[import-not-found]
    import round7_frontier_features as r7f  # type: ignore[import-not-found]
    import round8_nonlinear_features as r8f  # type: ignore[import-not-found]

    data = he.load_hybrid_data()
    micro_1h = r4f.load_micro_data()
    micro_15m = r5f.load_15m_micro_data()

    # variant: Hyperliquid, OI/metrics and the 15m jump neutralised, taker volume kept
    data = dataclasses.replace(
        data,
        hl_valid=np.zeros_like(data.hl_valid),
        metrics_valid=np.zeros_like(data.metrics_valid),
        ls_valid=np.zeros_like(data.ls_valid),
        oi_value=np.zeros_like(data.oi_value),
    )
    micro_15m = dataclasses.replace(
        micro_15m, jump_15m=np.zeros_like(micro_15m.jump_15m)
    )

    p4 = r4f.build_round4_pack(data, micro_1h)
    p5 = r5f.build_round5_mtf_pack(data, micro_1h, micro_15m, p4)
    p6 = r6f.build_round6_aw_pack(data, micro_1h, micro_15m, p4, p5)
    p7 = r7f.build_round7_frontier_pack(data, micro_1h, micro_15m, p4, p5, p6)
    p8 = r8f.build_round8_nonlinear_pack(data, micro_1h, micro_15m, p4, p5, p6, p7)

    # corrected variant: per-symbol N4 (factor_scores[:, 9, :]) in the bidirectional carry
    perp_mask_all = np.asarray(p6.perp_mask, dtype=bool)
    f3 = np.nan_to_num(data.funding_obs_3d)
    f7 = np.nan_to_num(data.funding_obs_7d)
    fr_blend = 0.55 * he._ema_causal(f3, 6.0) + 0.45 * he._ema_causal(f7, 12.0)
    z_carry_dir = r4f._neutral_z(-fr_blend, perp_mask_all, min_valid=4, clip_z=3.0)
    raw_score = 0.50 * z_carry_dir + 0.40 * p4.factor_scores[:, 9, :]
    w_bidir_fix = r6f._normalize_iv_balanced(
        r6f._freeze_raw_on_clock(raw_score, 18),
        perp_mask_all,
        r6f._freeze_raw_on_clock(p6.inv_vol_30d, 18),
    )
    p7_fix = dataclasses.replace(p7, w_perp_bidir_carry=w_bidir_fix)
    p8_fix = r8f.build_round8_nonlinear_pack(
        data, micro_1h, micro_15m, p4, p5, p6, p7_fix
    )

    n = N_ROWS
    close_ns = np.asarray(data.close_ns[:n], dtype=np.int64)
    window_end = int(close_ns[-1])
    sym = np.asarray(data.raw_fr_sym)
    ns = np.asarray(data.raw_fr_ns, dtype=np.int64)
    rate = np.asarray(data.raw_fr_rate, dtype=np.float64)
    keep = ns <= window_end

    # lab targets (4H pack lag 0): synthesize_aw_pa with the 4H settings
    def targets(pack: object) -> tuple[np.ndarray, np.ndarray]:
        ws = np.zeros((n, len(data.symbols)))
        wp = np.zeros((n, len(data.symbols)))
        for k in range(n):
            ws[k], wp[k] = r6f.synthesize_aw_pa(
                c_w=pack.c_weekly_iv[k],  # type: ignore[attr-defined]
                alpha_w=pack.w_aw_bidir_75_25[k],  # type: ignore[attr-defined]
                c_scale=1.0,
                a_scale=1.5 * float(pack.vol_cushion_tail[k]),  # type: ignore[attr-defined]
                w_max_leg=0.55,
                gross_cap=2.5,
                carry_mask_t=np.asarray(pack.carry_mask[k], dtype=bool),  # type: ignore[attr-defined]
                perp_mask_t=np.asarray(pack.perp_mask[k], dtype=bool),  # type: ignore[attr-defined]
            )
        return ws, wp

    ws, wp = targets(p8)
    ws_fix, wp_fix = targets(p8_fix)
    np.savez_compressed(
        Path(out_path),
        symbols=np.asarray(data.symbols),
        close_ns=close_ns,
        perp_close=data.perp_close[:n],
        spot_close=data.spot_close[:n],
        perp_quote_volume=data.perp_quote_vol[:n],
        spot_quote_volume=data.spot_quote_vol[:n],
        perp_taker_buy_quote_volume=data.perp_taker_buy[:n],
        spot_taker_buy_quote_volume=data.spot_taker_buy[:n],
        perp_valid=np.asarray(data.perp_valid[:n], dtype=bool),
        spot_valid=np.asarray(data.spot_valid[:n], dtype=bool),
        funding_event_symbol_index=sym[keep].astype(np.int16),
        funding_event_ns=ns[keep],
        funding_event_rate=rate[keep],
        out_w_aw_bidir_75_25=p8.w_aw_bidir_75_25[:n],
        out_c_weekly_iv=p6.c_weekly_iv[:n],
        out_vol_cushion_tail=p8.vol_cushion_tail[:n],
        out_carry_mask=np.asarray(p8.carry_mask[:n], dtype=bool),
        out_perp_mask=np.asarray(p8.perp_mask[:n], dtype=bool),
        out_target_spot=ws,
        out_target_perp=wp,
        out_w_perp_bidir_carry=p7.w_perp_bidir_carry[:n],
        fix_w_perp_bidir_carry=w_bidir_fix[:n],
        fix_w_aw_bidir_75_25=p8_fix.w_aw_bidir_75_25[:n],
        fix_vol_cushion_tail=p8_fix.vol_cushion_tail[:n],
        fix_target_spot=ws_fix,
        fix_target_perp=wp_fix,
        out_w_aw_regime=p6.w_aw_regime[:n],
        out_w_s_trend=p6.w_s_trend[:n],
        out_w_s_revert=p6.w_s_revert[:n],
        out_inv_vol_30d=p6.inv_vol_30d[:n],
        out_funding_obs_last=data.funding_obs_last[:n],
        out_funding_obs_3d=data.funding_obs_3d[:n],
        out_funding_obs_7d=data.funding_obs_7d[:n],
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
