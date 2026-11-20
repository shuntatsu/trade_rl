# agy anchor parity fixture

Status: Active. Parity oracle for `trade_rl/strategies/agy_anchor/`; it is test data,
not evidence of any economic result.

`agy_anchor_klines_funding_2021.npz` holds klines + funding inputs for
**2021-01-01 .. 2021-12-31** (2190 four-hour rows, 11 symbols) and the outputs of the
agy lab `ANC_BIDIR_75_25` chain on exactly those inputs. All rows are **known-used
pre-2026-08 development data**; nothing from 2026-08-01 or later is included, and no
backtest or return is stored.

## Provenance

The lab is read-only and external (agy "rl_lab", scratch of an Antigravity session):

- inputs come from the lab cache `cache/real_dev_4h_pre20260801.npz`
  (`extract_real_data.py`: Docker PostgreSQL market store, hourly Binance klines
  aggregated to 4h; buckets with fewer than four hours are invalid and carry zero volume)
  loaded by `hybrid_env.load_hybrid_data`;
- the lab chain is `round4_factors.build_round4_pack` -> `round5_mtf_features.build_round5_mtf_pack`
  -> `round6_allweather_features.build_round6_aw_pack` -> `round7_frontier_features.build_round7_frontier_pack`
  -> `round8_nonlinear_features.build_round8_nonlinear_pack`, then
  `round6_allweather_features.synthesize_aw_pa` with the 4H settings
  (`c_scale=1.0`, `a_scale=1.5 * vol_cushion_tail`, `w_max_leg=0.55`, `gross_cap=2.5`);
- `export_from_lab.py` (this directory) regenerates the file:
  `python export_from_lab.py <rl_lab_dir> agy_anchor_klines_funding_2021.npz`. It is not run in
  CI and never writes to the lab directory.

### Neutralised inputs (variant "klines + funding only, taker kept")

| Lab input | Neutralised as | Effect on the anchor |
|---|---|---|
| Hyperliquid premium (`hl_valid`) | all False | `z_hl_safe == 0` in the bidirectional carry |
| Binance open interest / long-short metrics (`metrics_valid`, `ls_valid`, `oi_value`) | all False / zero | `z_doi == 0`, so N3 reduces to its funding term |
| 15m jump (`jump_15m`) | zero | the `theta_trend` jump penalty is 0 (it only fires above 0.25 and never did in the lab window) |

Taker-buy quote volume (spot and perp) is kept: it drives factor E2.

## Arrays

Inputs: `symbols`, `close_ns` (bar-close UTC ns), `{perp,spot}_close`,
`{perp,spot}_quote_volume`, `{perp,spot}_taker_buy_quote_volume`, `{perp,spot}_valid`
(invalid bars have zero volume and a forward-filled close) and the raw funding events
`funding_event_{symbol_index,ns,rate}` (including 2020-12-20 .. 2020-12-31 warm-up events
that precede the first row; the count-based "3d/7d" funding means need them).
Symbols are in lab order: BTC ETH SOL BNB XRP ADA DOGE LINK DOT AVAX LTC (the Tier D roster
order differs; the port is symbol-order invariant, tested to 1e-12).

Outputs (`out_*` = lab as is, `fix_*` = lab with the defect below corrected):
`w_aw_bidir_75_25`, `c_weekly_iv`, `vol_cushion_tail`, `carry_mask`, `perp_mask`,
`w_aw_regime`, `w_s_trend`, `w_s_revert`, `inv_vol_30d`, `w_perp_bidir_carry`,
funding observations, and the final 4H-L0 target spot/perp weights per row.

## The one disclosed deviation from the lab as it is

Lab `round7_frontier_features._build_perp_bidir_carry` reads
`pack4.factor_scores[:, :, 9]` with the comment "N4_TERM_BASIS_Z". `factor_scores` has
shape `(T, 11 factors, 11 symbols)`, so that slice is **symbol index 9's eleven factor
scores** (E0..E4, N0..N5 of the AVAX column in lab order), which only broadcasts against the
eleven symbols by the coincidence 11 factors == 11 symbols. The intended per-symbol term
basis is `factor_scores[:, 9, :]`. Run as is, the sleeve therefore depends on the symbol at
index 9 and on the 1h-microstructure factors N0, N1, N2, N5 and E0.

The port implements the intended per-symbol N4. Parity is exact (max abs difference 0.0 on
every field) against `fix_*` and the lab as is differs only through that sleeve (`out_*`
versus `fix_*`: bidirectional sleeve up to 0.38, anchor weights up to 0.27, cushion up to
0.078, targets up to 0.40; `w_aw_regime`, `c_weekly_iv` and the masks are identical).
`tests/strategies/test_agy_anchor.py` pins both facts. Replicating the defect literally
would need the lab's 1h-micro factors and the AVAX symbol position; that is a separate
decision for the owner.
