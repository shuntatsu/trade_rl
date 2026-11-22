# Forward test of agy's CLEAN_ALL arm at gross ≤ 1 (`agy_clean_all_forward_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol was written before any bar, funding event or metric at or after 2026-08-01 was read for this signal.

Protocol digest: `8d559a2aaa2831001daa0930191e47b9301ae4d7e1e9623a308cd8bfe6fe2b49`

## Question

agy's Round 17 non-RL arm `R17_NONRL_CLEAN_ALL_175` is a cross-sectional spot/perp book over the 11 Tier D pairs. It combines the axis-fixed anchor (`w_clean_anc`) with eight low-turnover factors (`CLEAN8`). In agy's lab it returned +576% at 6 bp over 2023-01 to 2026-07.

A re-evaluation on the same known-used data (4H decisions, one-bar signal delay, `ECON_BRAKE_CAP`) found the following:

| variant | return | max DD | Sharpe | 2025 | 2026 H1 |
|---|---|---|---|---|---|
| as built (gross ≈ 2.15), 6 bp | +519% | 17.8% | 1.94 | +18.5% | +3.1% |
| as built, 12 bp | +342% | 22.2% | 1.58 | +7.5% | −2.3% |
| gross ≤ 1.0, 6 bp | +148% | 9.6% | 2.03 | +8.1% | +2.1% |
| gross ≤ 1.0, 12 bp | +112% | 11.8% | 1.68 | +3.3% | −0.7% |

- Most of the headline return comes from leverage and from 2023–24.
- Capping gross at 1.0 keeps the Sharpe and halves the drawdown.

**Hypothesis:** the gross-capped arm keeps a positive, cost-robust return on data it has never seen.

**Counter-hypothesis:** the edge was a 2023–24 regime effect, and the decay seen in 2025–26 continues.

**Cheapest falsifier:** the first two months after the lab embargo (2026-08-01 to 2026-09-30). They already exist in the local market store and have never been read for this signal. That window is an interim with an early-stop rule, not a decision. The decision is the full forward window.

**Prior exposure, disclosed:**
- agy explored the 2021-01 to 2026-07 data across 17 rounds and hundreds of arms.
- `CLEAN_ALL_175` was selected on that data.
- The gross cap of 1.0 was chosen after seeing the table above, which is known-used data only.
- The agy anchor's FORWARD stage under `agy_anchor_realistic_v1` was never run, so no post-2026-08 result of the anchor signal has been seen. The `trend_long_spot_oos_v1` FORWARD interim reports did read post-2026-08 spot data, for a different signal.
- The planner has public knowledge of 2026 market history.

## Signal (frozen)

- **Code:** agy's lab, session 7e52c53c `scratch/rl_lab`, snapshot taken 2026-11-22 to `C:\trade_rl_runs\agy-lab-snapshot-20261122`. It holds 47 `.py` files, with manifest digest `c4457c86…1683`. The frozen pre-2026-08 `round10`/`round17` pack caches and the Binance metrics cache have manifest digest `d07130ae…749ef7`. Both digests are constants of `tools/agy_clean_all_forward.py` and part of the protocol digest.
- **Arm:** `nonrl_alpha_and_scale_np(…, "R17_NONRL_CLEAN_ALL_175", k)` → `synthesize_r17_np` with the `ECON_BRAKE_CAP` scalars (`run_round15_econ_compact_gpu._econ_scalars`).
  - Decisions are made every 6 bars (24 h) from 2026-08-01T00:00Z.
  - Between decisions the book is held.
  - The lab's deadband applies.
- **Gross cap:** after synthesis, the decision target is scaled proportionally so that spot gross plus perp gross is at most 1.0. The cap applies to decision targets only. Between decisions the held weights drift with prices and a halted leg keeps its weight, so the executed gross may exceed 1.0 without any trade. The executed gross is recorded per cell as `max_executed_gross`, as reported data only.
- **Execution:** the lab's `execute_hybrid_step` (two legs, exact conservation, linear cost plus `IMPACT_QUAD = 0.002` on weight changes, realized funding).
  - The lab has no participation cap.
  - The universe is 11 majors and the intended capital is small (≤ $100k), so the cap is disclosed and not modelled.
  - A decision target with gross ≤ 1.0 needs no financing. Drift between decisions is small and is not modelled as financing.
- No constant of the lab changes.

## Data

- **Sources:** the lab's own sources are used unchanged:
  - the local market store (`trade-rl-market-store-postgres-1`) for 1h/15m klines and funding;
  - Binance Vision daily `metrics` archives;
  - Hyperliquid `fundingHistory`.
- **Window extension:** the only change to the lab is its embargo. For a stage, the harness copies the snapshot and replaces five literal forms of the 2026-08-01 embargo with the stage's data end:
  - the SQL bound;
  - the two `datetime(2026, 8, 1, …)` constants;
  - the metrics last day;
  - the epoch assert;
  - the `pre20260801` cache names.

  The harness refuses if any literal never matches, or if any survives. The snapshot itself is never edited.
- **Frozen metrics splice:** after the four build scripts, `build` replaces the rows of the fresh `binance_metrics_4h_fwd*.npz` whose bar opens before 2026-08-01 with the rows of agy's original `binance_metrics_4h_pre20260801.npz` (kept in the snapshot's `frozen_cache`). It requires equal `symbols`, a frozen row count equal to the number of pre-embargo bars of the base 4h cache, and exactly one matching file. Rows at or after 2026-08-01 stay as fetched. The build receipt records the splice row count and the sha256 of the file before and after. See Amendment 1 for the reason.
- **Parity oracle:** before any return is computed, `pack10`, `pack17` and the econ controls (`compute_r15_econ_controls(compute_r12_opp_feats(pack10))`, every array) built from the new data must equal the frozen ones over every row before 2026-08-01 to within 1e-12. The last frozen row is excluded, because its next-bar return and funding belong to the first out-of-sample bar. A fixed list of required fields must have been compared, or the run refuses:
  - `pack10`: `spot_valid`, `perp_valid`, `perp_mask`, `carry_mask`, `spot_next_ret`, `perp_next_ret`, `funding_realized`, `vol_cushion_tail`, `c_weekly_iv`, `w_aw_bidir_75_25`;
  - `pack17`: `w_clean_anc`, `w_clean_all`, `perp_mask`.

  A mismatch refuses the run and writes `parity-refusal.json`. A refusal is not a result.
- **Loop oracle:** before the out-of-sample replay, both arms at lag 1 and 6 bp are replayed uncapped over the known-used window from 2023-01-01 to the last pre-embargo row by the harness loop (`replay_cell`) and by the lab's own `eval_r17_4h`. The final equities must agree to a relative 1e-9, or the run refuses. Both values are recorded.
- **Unused lab import:** the lab imports `trade_rl/risk/pretrade.py` from `C:\dev\trade_rl-core7-mtf-ppo`. This arm does not use it. `result.json` records its sha256, or `null` if it is absent.
- **Market store coverage:** funding in the market store currently ends at 2026-09-30T16:00Z, so INTERIM_A ends at 2026-09-30T00:00Z. Later stages need the store to be extended first, by its own ingestion and not by this protocol.

## Stages

| Stage | Data end (last bar close) | Runs not before | Role |
|---|---|---|---|
| INTERIM_A | 2026-09-30T00:00Z | 2026-11-22 | interim with early stop; never a decision |
| FINAL | 2027-05-01T00:00Z | 2027-05-02 | decisive, once |

- FINAL requires `--interim-result` pointing to an INTERIM_A `result.json` with the same protocol digest and the label `INTERIM_NOT_A_DECISION`. `STOPPED_EARLY` ends the line. `INVALID` needs an amendment.
- Each stage can be claimed once: before computing anything, `run` creates `C:\trade_rl_runs\agy_clean_all_forward_ledger\<stage>.claim` by exclusive create, holding the digest, commit, G4 comment id and time. It refuses if the file exists. This is local single-use enforcement on this host, not a remote lock. A refusal after the claim also consumes it, and needs an amendment.

- Each stage starts flat at 2026-08-01T00:00Z.
- Warm-up comes from known-used rows before that date.
- Monthly interim reports between INTERIM_A and FINAL may be added only as further `INTERIM_*` stages, with the same cells and early-stop rule, by a result-blind amendment.

## Cells (frozen)

| Cell | Arm | One-way cost | Signal lag | Gates |
|---|---|---|---|---|
| `CANDIDATE__BASE` | CLEAN_ALL_175, gross ≤ 1 | 6 bp | 1 bar | yes |
| `CANDIDATE__HIGH_COST` | CLEAN_ALL_175, gross ≤ 1 | 12 bp | 1 bar | yes |
| `CANDIDATE__STRESS` | CLEAN_ALL_175, gross ≤ 1 | 20 bp | 1 bar | reported |
| `CANDIDATE__NO_DELAY` | CLEAN_ALL_175, gross ≤ 1 | 6 bp | 0 | reported |
| `CONTROL_ANC__BASE` | ANC_BIDIR_75_25, gross ≤ 1 | 6 bp | 1 bar | reported |

CASH returns 0.

A cell is invalid if, at any bar, any of the following holds:
- a target is not finite;
- the conservation error exceeds 1e-9;
- equity is not positive;
- a capped decision target has gross above 1.0 + 1e-9.

Executed gross never invalidates a cell.

## Decision rule (frozen)

**INTERIM stages** label `INTERIM_NOT_A_DECISION`, unless an early-stop condition holds. The early-stop conditions are:
- `CANDIDATE__BASE` total return ≤ −8%;
- a gating cell's max drawdown ≥ 20%.

If either holds, the label is `STOPPED_EARLY`, and the line stops: FINAL is not run.

**FINAL** labels `CLEAN_ALL_FORWARD_SUPPORTED` iff all cells are valid and the following hold:
1. `CANDIDATE__BASE` total return is above 0.
2. `CANDIDATE__BASE` annual Sharpe (4h log returns, √2190) is at least 0.5.
3. `CANDIDATE__HIGH_COST` total return is above 0.
4. The max drawdown of both gating cells is below 20%.

If all cells are valid but a condition fails, the label is `CLEAN_ALL_FORWARD_NOT_SUPPORTED`. Any invalid cell gives `INVALID`.

- Gating is defined by the Cells table (the `Gates` column). The early-stop and final drawdown conditions use only the gating cells, `CANDIDATE__BASE` and `CANDIDATE__HIGH_COST`.
- Reported cells never gate.
- Sharpe uses the population variance (ddof 0) of the 4h log returns, no risk-free rate, and is scaled by √2190.

## Authorization

- Reads of post-2026-08 data are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247).
- Each stage also needs its own G4 relay on Issue #810 quoting this protocol id, the digest and the exact 40-hex code commit.
- The commit is `git rev-parse HEAD` of the harness repository. `build` and `run` refuse if tracked files are modified.
- `build` and `run` first enforce `not_before`, verify the snapshot, the prepare receipt (digest, stage) and the per-file sha256 of the patched `work/lab`, then fetch both comments twice. The comment URL must be exactly an Issue #810 comment, and the fetched comments must belong to Issue #810, be owner-authored and be unedited between the fetches. Only then is data fetched or read. Body hashes and the commit go into `result.json`.
- Outputs are write-once.

## Stop rule

- No change to any constant, cost, cap, cell, window or rule after any stage result.
- No rescue run.
- An INVALID or a refusal for a defect outside the economics needs a result-blind amendment before any rerun.
- Higher leverage is a new protocol.
- No label authorizes orders, paper trading or live trading.

## Amendment 1 (result-blind, before any run)

This protocol was revised after independent review 6011046246 (BLOCKED), before any stage ran and before any data at or after 2026-08-01 was read. Changes:
- Invalidity now tests the capped decision target, not the executed gross, which legitimately drifts. `max_executed_gross` is reported.
- FINAL requires an INTERIM_A `INTERIM_NOT_A_DECISION` result. Each stage is claimed once in a local ledger.
- The G4 relay must quote the exact code commit. `build` is gated like `run`, and the patched lab is hash-verified.
- Parity has a fixed required-field list and covers the econ controls. A loop oracle against the lab's `eval_r17_4h` was added.
- The protocol digest binds this spec text (the `Protocol digest:` line excluded).
- Stated: Sharpe convention, the gating cells for the drawdown conditions, the unused `pretrade.py` import.
- Frozen metrics splice: a result-blind dry run on known-used pre-2026-08 data rebuilt the lab caches from today's sources. The 4h base, Hyperliquid, 1h micro, 1h bars and 15m micro caches were identical to agy's. `binance_metrics_4h` was not: for all 11 symbols, rows 7182 to 7349 (bars 2024-04-12T00Z to 2024-05-09T20Z) differ slightly in `oi_value`, `top_ls_pos`, `top_ls_cnt`, `global_ls_cnt` and `taker_ls_vol`. `metrics_valid` and `ls_valid` and the neighbouring rows are identical, so the differences are row-local. Binance Vision appears to have republished those daily files after agy fetched them on 2026-11-19. Agy's original cache joined the snapshot (frozen cache manifest digest `d07130ae…749ef7`), and `build` splices its pre-embargo rows in. This was found on known-used data only.
