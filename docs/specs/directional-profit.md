# Directional trading under a drawdown budget

Status: Active

The user explicitly prioritizes profit from directional trading and selected
20% peak-to-trough drawdown as the research tolerance on 2026-09-17. The prior
carry prototype is separate unfinished work and is not the primary strategy.
There is no authorization in this experiment to transmit live orders.

## Objective and immutable first experiment

Maximize net development return among predeclared candidates that satisfy the
drawdown budget. Do not optimize trade count or leverage. This new study does
not mutate, rerun, or change the decision of an existing sealed experiment.

Use the calibrated successor Dataset artifact from run 34803217815, Artifact
10331899302, Dataset ID
`6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518`.
The five symbols and execution economics remain unchanged. Fit scope and
baseline feature/threshold configuration come from that artifact's StudyPlan.
Train before 2023-01-01; evaluate 2023-01-01 through 2025-01-01, exclusively
development data. No 2025+ data is consumed or claimed unused.

Compare existing trend, mean reversion, ridge24, lightgbm24, and teacher-free
PPO with cash/constant-long/constant-short controls. PPO uses existing main
semantics, seeds 0,1,2,3,4 and 262144 steps per seed, frozen before training;
no change to the separate pending interleaved-PPO research implementation.

Add exactly one new simple price-channel candidate: enter long above the
previous 20 days' high, short below the previous 20 days' low; exit long below
the previous 10 days' low and exit short above the previous 10 days' high.
Opposite 20-day breakouts can reverse. Use completed hourly candles; exclude
the current candle from historical bounds. No tuning or parameter grid.

Each arm uses one shared account, 10000 USDT initial capital, total gross at
most 0.5 and per-symbol exposure at most 0.1, quantity-preserving holds,
causal previous-bar participation, original dataset costs, and existing
canonical execution/accounting. Reduce exposure from 10% drawdown and request
flat at 20%; a gap may exceed the intended bound and must fail qualification.
Schedule a final flat decision early enough for delayed next-open execution;
unfilled terminal exposure must fail qualification rather than disappear.

Retain raw returns, full/year return, drawdown, cost, turnover, funding,
termination, terminal holdings, code/runtime/data/config identity. Each fixed
candidate is evaluated once per scope; execution failures are recorded and
not turned into tuning opportunities.

Qualification requires positive full return, positive return in both calendar
years, drawdown <=20%, no economic termination, and a flat terminal account.
PPO additionally requires at least 4/5 seeds to pass the base screen and both
stresses, with positive full/year medians over all five seeds. Rank qualified
candidates by full net return (five-seed median for PPO), tie-break by lower
turnover (five-seed median for PPO) and then the frozen simplicity order:
trend, mean reversal, channel breakout, ridge24, lightgbm24, PPO. Controls are benchmarks, never
learned winners. This is development screening, not final winner admission.

Run predeclared double-cost and one-extra-bar-latency stress for every
qualified candidate, retaining all results. A development candidate failing
positive stress return or the 20% bound stops before prospective validation.
Report individual-symbol evaluation for qualified candidates, without
dropping losing symbols or changing allocations after seeing results.

No candidate is called production-ready. Old experiments have used portions
of 2025-2026, so independent final evidence requires a separately frozen
prospective paper stream. Existing final-test authorization stays intact.

## Implementation boundary and checks

The channel strategy consumes causally prepared channels. Data preparation
owns availability and rolling extrema; strategy owns logical intent; the
existing risk/executor own sizing limits and economic accounting. A standalone
study runner composes maintained public capabilities and publishes immutable
source/config/raw-result evidence. Existing canonical candidate identities and
suite membership remain unchanged.

Tests cover shifted historical extrema, future perturbation invariance,
warmup/missing-data behavior, hold/exit/reversal, and terminal close. Final
checks follow the complete permanent CI contract; existing Windows-only
baseline test failures must not be presented as a passing Linux CI.
