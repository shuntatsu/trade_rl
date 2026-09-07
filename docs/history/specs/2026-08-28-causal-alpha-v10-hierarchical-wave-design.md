# Causal Alpha V10 Hierarchical Wave Design

## Understanding summary

- Build one symbol-free policy shared by all nine Binance futures symbols.
- Maximize simulator-authoritative after-cost wealth; reward remains exactly
  `100 * net_log_return`.
- Preserve independent per-symbol long and short positions rather than forming
  a same-timestamp cross-sectional portfolio.
- Correct V9's failure mode: a profitable aggregate path still lost on four
  symbols because a 4-hour signal was also responsible for multi-day holding.
- Implement the requested two-stage decomposition: a 72-hour slow regime owns
  continuation, while a 4-hour fast signal times entries and early reversals.
- Keep fixed Signal, Selection, and Admission ordering; do not inspect holdout
  or start BC/RL unless the preceding gate passes.
- Preserve durable replay leaves, immutable provenance, and the 15-minute
  simulator. One-minute execution remains out of scope until profitable gross
  paths fail for execution-resolution reasons.

## Assumptions and non-functional requirements

- The user's prior instructions to use a two-stage design, accept the
  specification, delegate all in-scope choices, and continue are the explicit
  design confirmation and implementation authorization.
- Scale is fixed at 8 Selection episodes x 9 symbols x 3 candidates. The run
  must remain practical on the existing CPU/Docker host.
- Inputs are read-only frozen PostgreSQL/runtime artifacts. No network fetch,
  symbol exclusion, symbol ID, or instrument descriptor is permitted.
- Every replay leaf is written atomically and must validate its contract, fit,
  target, config, symbol, episode, and source digests on resume.
- Maintenance ownership stays in focused V10 modules; V7-V9 artifacts and
  behavior remain immutable controls.

## Evidence and root cause

V9 completed all 216 Selection replays. Its nonlinear candidate achieved
symbol-balanced gross wealth `1.012909` and net wealth `1.006463`, but failed
minimum symbol net wealth (`0.958693`), median symbol net wealth (`0.998146`),
and positive net scope fraction (`0.430556`). Its aggregate long exposure lost
net log return `-0.103633`, while short exposure earned `0.174549`. Neutral
fast-signal periods retained positions and lost `-0.050294` in the low
confidence attribution cell. Liquidity q1/q2 and volatility q1/q4 were also
negative, whereas liquidity q3/q4 and volatility q2/q3 were positive.

The root cause is responsibility collapse: one fast predictor chooses both the
entry and whether a position remains a valid multi-day wave. V10 separates
those decisions and makes entry execution-regime aware.

## Alternatives considered

1. **Recommended: hierarchical 72h regime + 4h trigger.** Preserves multi-day
   waves, permits 8-hour early exits, and addresses stale fast-signal holds
   without using symbol identity.
2. **Neutral-signal immediate exit.** Smaller change, but it recreates churn
   and cannot hold three-day waves through ordinary 4-hour uncertainty.
3. **Meta-label classifier trained on V9 trade success.** Flexible, but adds a
   second learned target whose sample definition is Selection-informed and has
   greater overfitting and maintenance risk.

## Fixed model design

- Pool the existing 62 local/global causal features across all nine symbols.
- Fit a fast 4-hour model over a rolling four-week window and a slow 72-hour
  model over a rolling twelve-week window.
- Use labels whose end index is strictly before the fit cutoff.
- Subsample non-overlapping labels at 16 decisions for fast and 288 decisions
  for slow.
- The fast model uses three deterministic random-ReLU ridge heads over the raw
  62 features plus 128 hidden features. The slow model uses three 32-feature
  random-ReLU-only ridge heads and excludes the raw features from its design.
  This preserves non-overlapping 72-hour labels while keeping the earliest
  72-row pooled fit above two observations per slow coefficient.
- Both horizons use fixed seeds, ridge strength `1.0`, and edge margin `0.001`.
- A horizon is qualified only when all heads agree in direction and
  `abs(mean) > ensemble_std + 0.001`.

## Two-stage exposure state machine

- Evaluate both stages every four hours and hold the emitted target between
  evaluations.
- A fresh entry requires two consecutive fast signals whose qualified direction
  agrees with the qualified slow direction.
- Entry is allowed only when the causal calibration-tail execution state has
  liquidity at or above its median and realized volatility between its 25th
  and 75th percentiles. These boundaries are frozen before each Selection
  episode from calibration data only.
- Enter signed target `0.10`; risk and liquidity caps may reduce it immediately.
- The last qualified slow direction is latched as the regime. A slow signal in
  the owned direction refreshes that latch and authorizes continuation through
  neutral fast observations, allowing multi-day waves; an unqualified slow
  observation does not erase a valid regime.
- Two consecutive qualified fast signals against the position exit to flat,
  supporting waves that last only eight hours.
- Two consecutive qualified slow signals against the position also exit.
- Six consecutive neutral slow observations (24 hours) expire a stale wave.
- Never flip directly. A new opposite position needs a fresh two-observation
  entry after the exit.
- An inherited episode-start position must earn the same coherent fast/slow
  confirmation or is exited after two evaluations.

## Candidates and gates

1. `v8_robust_control`: immutable V8 robust control.
2. `v9_nonlinear_control`: immutable corrected V9 nonlinear wave.
3. `hierarchical_wave`: the only new V10 hypothesis.

Signal requires valid unique fast and slow fit identities and horizon liveness
across all 72 symbol-scopes. Selection reuses every unchanged V7/V8 numerical
gate. Admission remains unopened unless Selection passes. BC/RL remains
unopened unless Admission passes.

## Decision log

- Keep 15-minute execution because V9 gross and net moved together and no
  unexplained execution rejection occurred; one-minute data would not address
  the observed prediction/holding failure.
- Use 72 hours rather than 24 hours for the slow stage to represent the stated
  multi-day wave objective.
- Use twelve weeks for slow fitting so non-overlapping 72-hour labels provide
  every causally available pooled row. The frozen V4 context exposes only 72
  such rows at the earliest cutoff, so the slow design is capped at 32 hidden
  coefficients instead of weakening non-overlap.
- Use the calibration-derived liquidity/volatility boundaries instead of hard
  numeric market thresholds, preserving causal scale invariance across symbols.
- Reject direction-wide short-only filtering despite current short profits;
  it would not be universal across changing market regimes.
- Preserve all V9 rejection artifacts as audit evidence and create a new V10
  output root, image identity, and result chain.
