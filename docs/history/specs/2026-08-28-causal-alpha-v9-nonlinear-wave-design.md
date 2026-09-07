# Causal Alpha V9 Nonlinear Wave Design

## Outcome

V9 tests one new universal hypothesis: a recent-regime nonlinear pooled signal
can identify 4-hour directional edges, while an exposure state machine can hold
multi-day waves without reacting to neutral observations.

The objective remains after-cost wealth. Reward remains exactly
`100 * net_log_return`. Selection and Admission gates remain unchanged.

## Evidence motivating the change

V8 completed 216 independent replays and reduced balanced net loss from
`0.935641` to `0.995010`, but its calibrated candidate was flat for nearly all
decisions and had only `15.28%` positive scopes. Linear expanding-history
forecasts therefore do not provide sufficient actionable alpha.

The only new research hypothesis is nonlinear recent-regime prediction. V9
does not add symbol identity, exclude symbols, relax gates, or use holdout data.

## Signal layer

- Pool the same local and global causal context features across the nine train
  symbols; do not include symbol ID or instrument descriptors.
- Use only labels whose end index is strictly before the fit cutoff.
- Use a four-week rolling fit window, representing four independent seven-day
  wave regimes.
- Subsample one label every four hours to remove overlapping-label inflation.
- Fit three deterministic random-ReLU ridge heads with fixed seeds, 128 hidden
  features per head, and ridge strength `1.0`.
- A signal is qualified only when all three heads agree on direction and
  `abs(mean_prediction) > ensemble_std + 0.001`.
- The `0.001` margin is the existing fixed edge margin, not a new tuned gate.

## Exposure layer

- Evaluate a qualified signal every four hours and hold the target between
  evaluations.
- Enter target `0.10`, the smallest existing discrete target strictly outside
  the fixed runtime `no_trade_band=0.05`, only after two consecutive qualified signals
  in the same direction. The initial `0.025` design was rejected by the first
  simulator episode because every proposed trade was correctly suppressed.
- Once V9 owns a position, neutral or uncertain observations hold it. This is
  the explicit multi-day wave behavior.
- Exit to flat only after two consecutive qualified opposite signals.
- Never flip directly. After exit, require a fresh two-observation entry.
- An inherited episode-start position is not considered V9-owned. It must earn
  two same-direction confirmations or be exited.
- Liquidity and risk caps may reduce exposure immediately.

## Fixed candidates

1. `v7_control`: exact V7 target compiler control.
2. `v8_robust_control`: exact V8 robust calibrated exposure control.
3. `nonlinear_wave`: the single new V9 hypothesis.

## Gate order

1. Signal liveness and fit identity.
2. Selection over 8 episodes x 9 symbols x 3 candidates.
3. Admission on untouched holdout only if Selection passes.
4. BC/RL only if Admission passes.

Every replay leaf is persisted atomically and revalidated by contract, fit,
forecast, target, config, symbol, episode, and nested evidence digests.

## Execution resolution

V8 showed gross-signal failure and very high net/gross retention. V9 therefore
continues on the maintained 15-minute simulator. One-minute data remains
deferred until a profitable gross path is rejected by execution evidence.
