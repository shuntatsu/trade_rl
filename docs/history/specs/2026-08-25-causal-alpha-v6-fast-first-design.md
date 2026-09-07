# Causal Alpha V6 Fast-First Retention Design

## Status

Approved for implementation under the operator's delegated design authority.

V6 is a new research-only successor to the completed V5 r7 Signal rejection.
V4 and V5 code, artifacts, schemas, thresholds, and historical results remain
immutable. V6 may reuse their public contracts but may not rewrite their
evidence.

## Understanding summary

- The objective is to maximize each symbol's after-cost wealth through trading,
  not to rank all symbols at one timestamp.
- Long/short position state, execution, PnL, reward, and risk are independent per
  symbol; a shared model may learn cross-symbol structure without symbol-ID
  lookup or symbol-specific intercepts.
- V5 r7 evaluated 72 complete slow Signal scopes and produced zero active rows:
  374 confidence abstentions and 274 direction disagreements across 648 causal
  samples. V5 therefore stopped before economic replay, BC, or RL.
- The unchanged V4 fast 4h lane passed its fixed Signal gate in that same run.
- A wave may end after four or eight hours or persist for three to seven days.
  Holding is signal-state-driven, never enforced by a clock lock.
- Reward remains `scale * log(net_equity_after / net_equity_before)` after all
  execution costs. No proxy reward replaces asset growth.
- One-minute data is not introduced until execution evidence, rather than
  forecast weakness, establishes a need.

## Assumptions and non-functional requirements

- The authored universe remains the same nine train symbols and eight Signal
  episode clusters used by the artifact-bound V4/V5 research path.
- The existing DB-backed Binance runtime, V4 context artifact, execution model,
  risk controls, latency, partial-fill, participation, and cost contracts remain
  authoritative.
- Docker execution remains non-root, deterministic, content-addressed, limited
  to two numerical threads and 7.63 GiB memory.
- All research artifacts are immutable leaves with upstream run, code, config,
  dataset, context, fit, contract, and candidate identities.
- No live routing, private API credential, or production promotion is authorized.
- V6 is maintained in V6-named modules. V4 and V5 must not import V6.

## Evidence basis

The authoritative V5 r7 output is
`/workspace/var/runs/causal-alpha-v5-prod-20260825-r7` in the
`trade-rl-training-data` volume. The Signal diagnostic digest is
`6545d3d26fa2a095fc7c978d602be58361bc7492b35ae62651c1caec4fd21dc9`.

The V5 failure is not permission to lower `minimum_selective_confidence`, omit
symbols or episodes, or bypass Signal. It falsifies the slow-anchor-first
hypothesis and motivates a new versioned fast-first target policy.

## Non-goals

V6 does not:

- reinterpret V5 r7 as profitable or successful;
- lower any V4 or V5 threshold;
- use the V5 slow calibrator for entry;
- tune against Signal, Selection, or Admission outcomes after they are opened;
- require every monthly symbol episode to be profitable;
- introduce a minimum holding duration;
- add one-minute prediction features;
- train BC or RL before V6 Admission passes;
- authorize Production GO or live capital.

## Architecture

```text
DB-backed Binance data + frozen runtime/context identities
                         |
                         v
             unchanged causal V4 fit
                         |
              +----------+----------+
              |                     |
              v                     v
       fast 4h forecast       24h/72h context
              |                     |
              +----------+----------+
                         v
             per-symbol V6 controller
              |                     |
       fast-only baseline   fast + slow retention
              |                     |
              +----------+----------+
                         v
         unchanged execution/risk simulator
                         |
                         v
     Signal -> paired Selection -> untouched Admission
                         |
                  only after pass
                         v
                 BC -> economic gate -> RL
```

The V4 predictor is shared across train symbols. Each controller invocation owns
only one symbol's current position, pending confirmation state, liquidity cap,
risk cap, costs, actions, and wealth path.

## Fixed V6 target configuration

The first V6 hypothesis contains exactly two candidates:

1. `fast_only`: the economic baseline;
2. `fast_slow_retention`: the challenger differing only in slow retention
   filtering.

Both candidates use:

```text
target magnitudes              = (0.0, 0.025, 0.05, 0.10, 0.25)
maximum absolute target        = 0.25
maximum target delta           = 0.125
fast rebalance decisions       = 4
slow context decisions         = 16
uncertainty multiplier         = 1.0
execution cost multiplier      = 1.5
edge margin                    = 0.001
confirmation count             = 2
strong reversal threshold      = 0.02
liquidity lookback decisions   = 96
liquidity lower quantile       = 0.10
liquidity safety multiplier    = 0.80
```

The values are inherited from maintained V3/V4 economic and target contracts;
they are not selected from V5 r7 outcomes.

## Fast proposal

At each authored fast cadence, the compiler scores causal target candidates with
the 4h prediction:

```text
delta = target - previous_target
objective = delta * expected_return_4h
          - abs(delta) * uncertainty_4h
          - abs(delta) * (1.5 * one_way_cost_rate + 0.001)
```

Exposure increase or reversal also requires non-zero agreement between the 4h
return prediction and independent 4h direction score. Hold and same-sign risk
reduction are always admissible. The deterministic winner maximizes objective,
then minimizes distance from the previous target, then absolute exposure, then
the signed target.

A non-risk proposal must win twice on consecutive fast cadences. A reversal with
`abs(expected_return_4h) >= 0.02` may bypass the second confirmation, but not
uncertainty, cost, direction, liquidity, or risk checks.

## Slow retention state

Slow context never initiates a position. For an existing non-zero position:

```text
supportive = sign(prediction_24h) == position_sign
          and sign(prediction_72h) == position_sign

opposed = sign(prediction_24h) == -position_sign
       and sign(prediction_72h) == -position_sign

mixed = otherwise
```

- `supportive`: allow hold and same-sign add. A weak fast reduction is held; a
  confirmed or strong fast reversal may reduce, exit, or flip.
- `mixed` or `opposed`: forbid add. Allow hold, same-sign reduction, exit, and a
  confirmed or strong fast reversal.
- `flat`: ignore slow context and allow a qualified fast entry.
- Slow state alone never changes a target and never flips a position.

The `fast_only` candidate skips this filter. Every other prediction, objective,
candidate, cadence, cap, and execution input is identical, which makes the
paired economic difference attributable to retention.

## Transition and override order

Every decision receives exactly one terminal reason. Precedence is:

1. `liquidity_deleverage`;
2. `risk_projection`;
3. `unactionable_hold`;
4. `cadence_hold`;
5. `direction_disagreement_hold`;
6. `cost_or_uncertainty_hold`;
7. `confirmation_hold`;
8. `slow_support_hold`;
9. `slow_add_suppressed`;
10. transition-derived `hold_flat`, `hold_position`, `entry`, `add`, `reduce`,
    `exit`, or `flip`.

No-volume or zero-liquidity decisions cannot enter or add. Missing forecast,
uncertainty, descriptor, or actionability evidence permits only hold or mandatory
risk reduction. Episode initialization uses the contract's persisted weight.

## Signal gate

Signal remains prediction evidence, not a profitability claim.

- Reuse the unchanged V4 fast 4h Signal evidence and frozen gate.
- Require exactly 72 raw scopes, eight independent episodes, all nine symbols,
  and exact episode coverage.
- Require the unchanged fast Rank IC, top-bottom spread, and direction-accuracy
  excess lower confidence bounds to be non-negative.
- Build both V6 target paths on the exact same scopes and require complete reason,
  actionability, initial-state, fit, forecast, contract, candidate, and cap
  identities.
- Require the fast-only baseline to produce at least one non-flat target globally.
  Economic meaningful execution remains a Selection responsibility.
- Slow predictive accuracy is diagnostic only because slow does not initiate a
  trade.

## Paired Selection

Both fixed candidates replay every economic train scope with identical data,
fit, environment, initial state, cost, liquidity, and risk identities.

A candidate is eligible only when:

```text
symbol_balanced_gross_wealth > 1.0
symbol_balanced_net_wealth   > 1.0
every symbol aggregate net_wealth >= 1.0
median symbol net_wealth >= 1.0
positive net scope fraction >= 0.5
turnover p95 <= 1.0 per day
meaningful execution scope count > 0
hard risk violation count == 0
unexplained execution rejection count == 0
```

Worst symbol-episode return, CVaR10, drawdown, holding duration, flat fraction,
cost, turnover, flips, submitted/executed changes, and net/gross retention are
always persisted.

Selection rules:

- If neither candidate is eligible, reject.
- If exactly one is eligible, select it.
- If both are eligible, select `fast_slow_retention` only when it has strictly
  higher symbol-balanced net wealth, no lower worst-symbol aggregate net wealth,
  no higher turnover p95, no higher execution cost, and no more sign flips.
- Otherwise select `fast_only`.

Selection never reads Admission records.

## Untouched Admission

Admission opens one latest untouched holdout per train symbol only after Signal
and Selection pass. The selected candidate and fast-only baseline are both
evaluated once; this paired baseline is predeclared and cannot select a new
candidate.

Admission requires:

```text
fit_knowledge_cutoff == holdout_start
aggregate gross return > 0.0
aggregate net return   > 0.0
positive net symbols   >= 6 of 9
worst symbol net return >= -0.02
hard risk violation count == 0
unexplained execution rejection count == 0
```

If the selected candidate is `fast_slow_retention`, its aggregate net wealth may
not be lower than the paired fast-only holdout wealth. A failed Admission cannot
publish a package and performs zero BC/RL updates.

## Replay evidence

Each symbol/episode replay persists:

- gross/net wealth and return;
- reward total and equality to scaled net log return;
- maximum drawdown;
- turnover per day and total execution cost;
- submitted/executed changes and rejections;
- closed trades and sign flips;
- reason counts;
- flat-time fraction and time-weighted absolute exposure;
- completed and open holding durations;
- liquidity/risk projections;
- candidate, target, fit, forecast, contract, runtime, context, and code digests.

## BC and RL boundary

An admitted V6 package supplies the selected causal teacher action paths through
the maintained generic episode-aligned teacher interface. All three maintained
algorithm families consume identical teacher actions.

BC must first pass reconstruction and causal economic gates. A failed BC gate
performs zero critic warm-start and zero PPO updates. RL retains the pure net-log
reward, independent per-symbol environments, unchanged execution/risk model, and
multi-seed evaluation. Positive intermediate reward or a profitable baseline is
not learned-policy uplift.

The research Admission implementation and the admitted-teacher training adapter
are separate sub-projects. The latter is implemented or invoked only after a
real Admission package exists.

## One-minute data decision

V6 does not add one-minute prediction inputs. A later execution-only hypothesis
may use one-minute bars only when Selection shows either:

- positive gross wealth but non-positive net wealth with costs consuming at
  least half of positive gross edge; or
- bar-path sensitivity changes at least 10% of scope decisions.

## Error handling and durability

- Scientific rejection, invalid evidence, execution failure, and OOM are distinct
  terminal states.
- Cutoff fits execute sequentially; superseded runtime/context inputs are released
  before stages.
- Progress is emitted per cutoff, candidate, and gate.
- No stage may overwrite an immutable leaf.
- No package is published until Admission passes.
- Failed artifacts and Docker state are retained for diagnosis.

## Testing strategy

1. Hand-computable TDD for flat entry, long/short symmetry, add, hold, reduce,
   exit, confirmed flip, strong flip, slow suppression, no-volume, cost loss,
   missing input, liquidity deleverage, risk projection, and non-zero initial
   weight.
2. Digest and schema tests for target, Signal, replay, Selection, Admission, and
   package artifacts.
3. Paired-oracle tests proving the two candidates differ only by slow retention.
4. Gate falsification for missing symbols/episodes, all-flat paths, losing
   symbols, excessive turnover, risk violations, and unexplained rejections.
5. Holdout-isolation and zero-update tests.
6. Targeted tests, Ruff, Mypy, import boundaries, Docker provenance, non-root
   probe, and real DB-backed run.
7. Only after Admission: BC economic validation, multi-seed PPO, artifact audit,
   and final report.

## Decision log

| Decision | Alternatives | Reason |
|---|---|---|
| Fast 4h initiates trades | slow-first; learned meta-gate | V4 fast passed; V5 slow produced zero active rows |
| Slow controls retention only | omit slow; slow initiates trades | preserves multi-day context without trusting failed slow direction |
| Keep fast-only baseline | challenger only | makes retention value causally attributable |
| No clock holding lock | 1/3/7-day minimum | waves vary from hours to days; risk must exit immediately |
| Shared fit, independent execution | symbol models; cross-sectional portfolio | generalization without symbol memorization or coupled PnL |
| Every symbol aggregate net wealth non-negative in Selection | mean-only gate | prevents a few winners from hiding structurally losing symbols |
| Paired baseline in Admission | selected-only holdout | prevents retention from degrading unseen economics |
| No one-minute input | immediate 1m expansion | current failure is forecast activation, not execution-cost evidence |
| New V6 schemas | mutate V5 | preserves historical evidence and fail-closed identities |

