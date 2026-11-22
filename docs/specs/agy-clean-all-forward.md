# Forward test of agy's CLEAN_ALL arm at gross ≤ 1 (`agy_clean_all_forward_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol was written before any bar, funding event or metric at or after 2026-08-01 was read for this signal.

Protocol digest: `9d55aade760c93084b6420ca9a183885e2aa892ab60ce30a49590bee532c67d5`

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

- **Code:** agy's lab, session 7e52c53c `scratch/rl_lab`, snapshot taken 2026-11-22 to `C:\trade_rl_runs\agy-lab-snapshot-20261122`. It holds 47 `.py` files, with manifest digest `c4457c86…1683`. The frozen pre-2026-08 `round10`/`round17` pack caches have manifest digest `60aaea5e…305a`. Both digests are constants of `tools/agy_clean_all_forward.py` and part of the protocol digest.
- **Arm:** `nonrl_alpha_and_scale_np(…, "R17_NONRL_CLEAN_ALL_175", k)` → `synthesize_r17_np` with the `ECON_BRAKE_CAP` scalars (`run_round15_econ_compact_gpu._econ_scalars`).
  - Decisions are made every 6 bars (24 h) from 2026-08-01T00:00Z.
  - Between decisions the book is held.
  - The lab's deadband applies.
- **Gross cap:** after synthesis, the target is scaled proportionally so that spot gross plus perp gross is at most 1.0.
- **Execution:** the lab's `execute_hybrid_step` (two legs, exact conservation, linear cost plus `IMPACT_QUAD = 0.002` on weight changes, realized funding).
  - The lab has no participation cap.
  - The universe is 11 majors and the intended capital is small (≤ $100k), so the cap is disclosed and not modelled.
  - With gross ≤ 1.0, spot long gross plus perp margin never exceeds equity, so no financing arises.
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
- **Parity oracle:** before any return is computed, every `pack10` and `pack17` array row before 2026-08-01 must equal agy's frozen packs to within 1e-12. The last frozen row is excluded, because its next-bar return and funding belong to the first out-of-sample bar. A mismatch refuses the run and writes `parity-refusal.json`. A refusal is not a result.
- **Market store coverage:** funding in the market store currently ends at 2026-09-30T16:00Z, so INTERIM_A ends at 2026-09-30T00:00Z. Later stages need the store to be extended first, by its own ingestion and not by this protocol.

## Stages

| Stage | Data end (last bar close) | Runs not before | Role |
|---|---|---|---|
| INTERIM_A | 2026-09-30T00:00Z | 2026-11-22 | interim with early stop; never a decision |
| FINAL | 2027-05-01T00:00Z | 2027-05-02 | decisive, once |

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
- the executed gross exceeds 1.0.

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

Reported cells never gate.

## Authorization

- Reads of post-2026-08 data are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247).
- Each stage also needs its own G4 relay on Issue #810 quoting this protocol id and digest.
- `run` fetches both comments twice, requires them to be owner-authored and unedited between the fetches, and records their body hashes in `result.json`.
- Outputs are write-once.

## Stop rule

- No change to any constant, cost, cap, cell, window or rule after any stage result.
- No rescue run.
- An INVALID or a refusal for a defect outside the economics needs a result-blind amendment before any rerun.
- Higher leverage is a new protocol.
- No label authorizes orders, paper trading or live trading.
