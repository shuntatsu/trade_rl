"""Synthetic G2 tests for causal 6-TF signals, SPOT_HALTS, Pre-FS1 boundary, and Non-RL gates."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from trade_rl.data.features.mtf6_signals import (
    CORE7_SIGNAL_NAMES,
    ORTH6_INDICES,
    build_mtf6_signal_pack,
    compute_nonrl_regime_gates,
    enforce_pre_fs1_boundary,
    normalize_balanced_neutral_row,
)
from trade_rl.data.market import MarketDataset


def _make_alternating_dataset(
    *,
    n_bars: int = 96,
    pairs: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    seed: int = 7,
    start_iso: str = "2026-01-01T00:00:00",
    spot_halt_slice: tuple[int, int, int] | None = None,
) -> MarketDataset:
    rng = np.random.default_rng(seed)
    n_pairs = len(pairs)
    n_syms = 2 * n_pairs
    symbols = tuple(sym for pair in pairs for sym in (f"{pair}:spot", f"{pair}:perp"))
    ts = np.datetime64(start_iso, "ns") + np.arange(n_bars) * np.timedelta64(4, "h")

    close = np.empty((n_bars, n_syms), dtype=np.float64)
    open_arr = np.empty((n_bars, n_syms), dtype=np.float64)
    high = np.empty((n_bars, n_syms), dtype=np.float64)
    low = np.empty((n_bars, n_syms), dtype=np.float64)
    volume = np.empty((n_bars, n_syms), dtype=np.float64)
    funding = np.zeros((n_bars, n_syms), dtype=np.float64)
    tradable = np.ones((n_bars, n_syms), dtype=np.bool_)

    for p in range(n_pairs):
        base = 100.0 * (p + 1)
        ret = rng.normal(0.0005, 0.015, size=n_bars)
        p_close = base * np.exp(np.cumsum(ret))
        s_close = p_close * (1.0 + rng.normal(0.0, 0.001, size=n_bars))

        for offset, series in ((0, s_close), (1, p_close)):
            col = 2 * p + offset
            close[:, col] = series
            open_arr[0, col] = series[0]
            open_arr[1:, col] = series[:-1] * (
                1.0 + rng.normal(0.0, 0.001, size=n_bars - 1)
            )
            bar_max = np.maximum(open_arr[:, col], close[:, col])
            bar_min = np.minimum(open_arr[:, col], close[:, col])
            high[:, col] = bar_max * (
                1.0 + np.abs(rng.normal(0.002, 0.002, size=n_bars))
            )
            low[:, col] = bar_min * np.maximum(
                0.5, 1.0 - np.abs(rng.normal(0.002, 0.002, size=n_bars))
            )
            volume[:, col] = rng.uniform(100_000.0, 250_000.0, size=n_bars)

        funding[:, 2 * p + 1] = rng.normal(0.0001, 0.0002, size=n_bars)

    if spot_halt_slice is not None:
        pair_idx, h_start, h_end = spot_halt_slice
        tradable[h_start:h_end, 2 * pair_idx] = False

    zero_rates = np.zeros((n_bars, n_syms), dtype=np.float64)
    dataset_id = hashlib.sha256(
        f"synth_mtf6:{n_bars}:{n_pairs}:{seed}:{start_iso}:{spot_halt_slice}".encode(
            "utf-8"
        )
    ).hexdigest()
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=symbols,
        timestamps=ts,
        features=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_arr,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=funding,
        tradable=tradable,
        feature_available=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        feature_names=("stub",),
        global_feature_names=("stub_global",),
        periods_per_year=2190,
        fee_rate=zero_rates,
        maker_fee_rate=zero_rates,
        taker_fee_rate=zero_rates,
        spread_rate=zero_rates,
        contract_multipliers=np.ones(n_syms, dtype=np.float64),
    )


def test_mtf6_future_perturbation_is_zero_before_cut() -> None:
    ds_base = _make_alternating_dataset(
        n_bars=120, pairs=("BTCUSDT", "ETHUSDT", "BNBUSDT"), seed=11
    )
    cut = 71

    rng = np.random.default_rng(999)
    open_p = np.asarray(ds_base.open, dtype=np.float64).copy()
    high_p = np.asarray(ds_base.high, dtype=np.float64).copy()
    low_p = np.asarray(ds_base.low, dtype=np.float64).copy()
    close_p = np.asarray(ds_base.close, dtype=np.float64).copy()
    vol_p = np.asarray(ds_base.volume, dtype=np.float64).copy()
    fr_p = np.asarray(ds_base.funding_rate, dtype=np.float64).copy()

    mult = np.exp(
        rng.normal(0.0, 0.35, size=(ds_base.n_bars - cut - 1, ds_base.n_symbols))
    )
    open_p[cut + 1 :] *= mult
    close_p[cut + 1 :] *= mult * np.exp(rng.normal(0.0, 0.15, size=mult.shape))
    high_p[cut + 1 :] = np.maximum(open_p[cut + 1 :], close_p[cut + 1 :]) * 1.05
    low_p[cut + 1 :] = np.minimum(open_p[cut + 1 :], close_p[cut + 1 :]) * 0.95
    vol_p[cut + 1 :] *= 3.5
    fr_p[cut + 1 :] += 0.005

    from dataclasses import replace

    ds_pert = replace(
        ds_base,
        dataset_id="f" * 64,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=vol_p,
        funding_rate=fr_p,
    )

    pack_base = build_mtf6_signal_pack(ds_base)
    pack_pert = build_mtf6_signal_pack(ds_pert)

    assert len(CORE7_SIGNAL_NAMES) == 7
    assert pack_base.core7_signals.shape == (120, 3, 7)
    assert pack_base.orth6_signals.shape == (120, 3, 6)
    np.testing.assert_array_equal(
        pack_base.orth6_signals, pack_base.core7_signals[:, :, list(ORTH6_INDICES)]
    )

    sl = slice(0, cut + 1)
    for arr_a, arr_b in (
        (pack_base.core7_signals[sl], pack_pert.core7_signals[sl]),
        (pack_base.orth6_signals[sl], pack_pert.orth6_signals[sl]),
        (pack_base.w_aw_bidir_75_25[sl], pack_pert.w_aw_bidir_75_25[sl]),
        (pack_base.c_weekly_iv[sl], pack_pert.c_weekly_iv[sl]),
        (pack_base.vol_cushion_tail[sl], pack_pert.vol_cushion_tail[sl]),
        (pack_base.regime_feats_41[sl], pack_pert.regime_feats_41[sl]),
        (pack_base.opp_feats_6[sl], pack_pert.opp_feats_6[sl]),
        (pack_base.gate_alpha_nonrl[sl], pack_pert.gate_alpha_nonrl[sl]),
        (pack_base.gate_carry_nonrl[sl], pack_pert.gate_carry_nonrl[sl]),
    ):
        assert float(np.max(np.abs(arr_a - arr_b))) == 0.0


def test_mtf6_spot_halts_mask_without_forward_fill_or_nan() -> None:
    ds_halt = _make_alternating_dataset(
        n_bars=90,
        pairs=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        seed=23,
        spot_halt_slice=(0, 30, 42),
    )
    pack = build_mtf6_signal_pack(ds_halt)

    assert not np.any(pack.spot_usable[30:42, 0])
    assert not np.any(pack.carry_mask[30:42, 0])
    assert np.all(pack.spot_ret_causal[30:43, 0] == 0.0)
    assert np.all(pack.c_weekly_iv[30:42, 0] == 0.0)
    assert np.all(np.isfinite(pack.core7_signals))
    assert np.all(np.isfinite(pack.regime_feats_41))
    assert np.all(np.isfinite(pack.opp_feats_6))


def test_pre_fs1_boundary_refuses_2026_08_01_and_allows_prior() -> None:
    # 4 bars starting 2026-07-31T12:00 -> last bar is 2026-08-01T00:00:00 (must fail)
    ds_violates = _make_alternating_dataset(
        n_bars=4,
        pairs=("BTCUSDT", "ETHUSDT"),
        start_iso="2026-07-31T12:00:00",
    )
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        enforce_pre_fs1_boundary(ds_violates)
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        build_mtf6_signal_pack(ds_violates)

    # 3 bars starting 2026-07-31T12:00 -> last bar is 2026-07-31T20:00:00 (must succeed)
    ds_ok = _make_alternating_dataset(
        n_bars=3,
        pairs=("BTCUSDT", "ETHUSDT"),
        start_iso="2026-07-31T12:00:00",
    )
    enforce_pre_fs1_boundary(ds_ok)
    pack = build_mtf6_signal_pack(ds_ok)
    assert pack.core7_signals.shape == (3, 2, 7)


def test_compute_nonrl_regime_gates_hand_computed_parity_and_throttling() -> None:
    opp = np.array(
        [
            [0.05, -0.05, 0.2, -2.5, -1.0, 0.15],  # cold/unfavorable regime
            [0.45, 0.08, 0.9, 2.5, 2.0, 0.0],  # strong/favorable regime
            [0.26, 0.005, 0.5, 0.0, 0.0, 0.02],  # midpoint
        ],
        dtype=np.float64,
    )
    g_alpha, g_carry = compute_nonrl_regime_gates(opp)

    def sig(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v))))

    expected_carry = [
        min(1.0, max(0.30, 0.30 + 0.70 * sig((row[1] - 0.005) / 0.015))) for row in opp
    ]
    expected_alpha = [
        min(
            1.0,
            max(
                0.45,
                0.45
                + 0.55
                * (
                    0.5 * min(1.0, max(0.0, (row[0] - 0.20) / 0.12))
                    + 0.5 * min(1.0, max(0.55, 0.55 + 0.45 * sig(row[3] / 0.8)))
                ),
            ),
        )
        for row in opp
    ]
    np.testing.assert_allclose(g_alpha, expected_alpha, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_carry, expected_carry, rtol=1e-12, atol=1e-12)

    assert g_alpha[0] < g_alpha[2] < g_alpha[1]
    assert g_carry[0] < g_carry[2] < g_carry[1]
    assert g_alpha[0] >= 0.45 and g_alpha[1] <= 1.00
    assert g_carry[0] >= 0.30 and g_carry[1] <= 1.00


def test_normalize_balanced_neutral_row_exact_balance_and_caps() -> None:
    row = normalize_balanced_neutral_row(
        np.array([3.0, 0.1, -0.5, -2.5], dtype=np.float64),
        np.array([True, True, True, True]),
        w_max=0.25,
        gross_max=1.0,
    )
    assert abs(float(np.sum(row))) < 1e-12
    assert float(np.max(np.abs(row))) <= 0.25 + 1e-12
    assert float(np.sum(np.abs(row))) <= 1.0 + 1e-12
