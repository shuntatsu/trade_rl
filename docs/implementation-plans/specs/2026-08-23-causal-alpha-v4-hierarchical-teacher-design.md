# Causal Alpha V4 Hierarchical Teacher Design

## Status

Design only. This document does not authorize production promotion, live routing,
or reinterpretation of any existing Causal Alpha V3 evidence.

The predeclared V3 market/residual counterfactual in
`2026-08-22-causal-alpha-v3-market-residual-decomposition-design.md` remains
immutable research history. V4 does not rewrite its hypothesis, thresholds, ridge
strengths, evidence, or pass/fail semantics.

## Objective

Build the next research teacher generation around the economic objective already
maintained by Trade RL: cost-inclusive net-equity log growth. Reward remains
unchanged.

The V4 teacher separates responsibilities that the pooled V3 ridge currently mixes:

1. fixed market-proxy return;
2. target-instrument residual alpha;
3. fast cross-market flow impulse;
4. independent direction evidence; and
5. state-dependent forecast uncertainty.

The target outcome is not more trading. The target outcome is a causal teacher
whose action changes are supported by economically meaningful, time-varying edge
and whose current information set can be reproduced by the downstream BC/PPO
student and serving path.

## Non-goals

V4 does not:

- change `reward_t = scale * log(net_equity_after / net_equity_before)`;
- add baseline/excess-growth shaping to the reward;
- change hard-risk, margin, liquidation, execution-cost, latency, partial-fill,
  participation, or target-weight accounting semantics;
- lower existing Signal, economic-selection, Teacher-admission, BC, or RL gates to
  make a weak hypothesis pass;
- mutate or silently retune retained V3 evidence;
- build a simultaneous multi-asset portfolio policy;
- duplicate the complete 206-channel futures technical feature stack for Spot;
- admit non-point-in-time on-chain history into backtest evidence;
- introduce a large nonlinear neural teacher before the deterministic hierarchical
  design is falsified;
- authorize direct exchange routing or Production GO.

A separate prerequisite patch may correct activity-accounting semantics where
closed-trade count is currently used as a proxy for meaningful execution. That
patch must be independently reviewable and must not be bundled into V4 model
performance claims.

## Current Evidence

The current V3 lane uses symbol-balanced, overlap-aware pooled ridge models for
24-hour and 72-hour forward gross log returns. Their predictions are converted to a
24-hour-equivalent expected return and uncertainty bundle before the conservative
cost-aware target compiler.

The evidence preceding this design shows that bounded pooled-ridge variations can
improve relative ranking while absolute direction remains unstable. The existing
2026-08-22 market/residual counterfactual is therefore the correct next direct test
of the common-factor hypothesis and remains unchanged.

V4 adds four architectural conclusions:

- a 15-minute decision clock should not force newly available Spot/derivative flow
  information through only 24h/72h labels;
- a teacher that reads current information unavailable to the student has an
  irreducible BC reconstruction error by construction;
- hidden per-symbol residual models are a poor canonical dependency for a Universal
  student that does not consume a private symbol ID;
- non-zero prediction level is not signal liveness: dynamic contribution must be
  material relative to residual error and execution hurdle.

## Options Considered

### Option A — append Spot/on-chain columns to the pooled V3 ridge

Rejected as the main architecture. It leaves common direction, residual ranking,
fast flow, and slow trend on one loss and one representation. High-frequency flow
can be averaged away by 24h/72h regression and teacher/student information mismatch
remains unresolved.

### Option B — use the existing V3 market/residual design unchanged as canonical V4

Retained only as the already-authored V3 counterfactual. It is a valid falsification
of the common-factor hypothesis. It is not adopted unchanged as V4 because its
nine-symbol current aggregate and per-symbol residual heads do not yet define a
clean student/serving information contract.

### Option C — deterministic hierarchical V4 teacher

Chosen.

Use a fixed BTC market proxy; fixed BTC/ETH public Global Market Context; a shared
residual model conditioned on target-local information and public instrument
descriptors; separate 4h fast and 24h/72h slow responsibilities; an independent
shared direction score; and state-conditioned uncertainty. Reuse deterministic
ridge primitives initially.

### Option D — nonlinear mixture-of-experts teacher immediately

Deferred. It adds optimizer, capacity, calibration, determinism, and overfitting
questions before the hierarchical decomposition itself has been falsified.

## High-level Architecture

```text
                         auxiliary public context
                  BTC/ETH Spot + Perp + derivatives
                         + optional PIT flows
                                   |
                                   v
                   +-------------------------------+
                   | BTC Market Proxy Return Heads |
                   | 4h / 24h / 72h               |
                   +-------------------------------+
                                   |
                         beta_s(t) * proxy
                                   |
                                   v
 target-local futures + local Spot/perp + descriptors + global context
                                   |
                                   v
                   +-------------------------------+
                   | Shared Residual Return Heads  |
                   | 4h / 24h / 72h               |
                   +-------------------------------+
                                   |
                                   +----> mu_s,h
                                   |
 target-local + global context ----+
                                   |
                                   v
                   +-------------------------------+
                   | Shared Direction Heads        |
                   | 4h / 24h / 72h sign score    |
                   +-------------------------------+
                                   |
                                   v
                   +-------------------------------+
                   | State-conditioned uncertainty |
                   +-------------------------------+
                         |                    |
                         v                    v
                 24h/72h slow anchor     4h fast impulse
                         |                    |
                         +---------+----------+
                                   v
                        conservative final target
                                   |
                                   v
                        existing execution/risk
                                   |
                                   v
                       existing pure net-log reward
```

## Information-set Invariant

The central V4 invariant is:

> Every current-time input that can affect a V4 teacher action must be
> reconstructible from the same decision-time public information available to the
> student and serving path.

Future labels may be used only after their label ends become strictly earlier than
the fit knowledge cutoff. Current teacher inputs may not depend on validation/test
outcomes, the Teacher-admission holdout, or a hidden train-symbol-only state.

Consequences:

- V4 does not use the current equal-weight state of the authored nine-symbol train
  universe as an implicit market input;
- V4 uses fixed BTC/ETH auxiliary context with stable semantics across train,
  validation, test, and serving;
- the residual return function is shared across target instruments;
- instrument-specific behavior is conditioned on public continuous descriptors and
  local market state, not `symbol -> hidden model` dispatch;
- causal beta is an explicit observable V4 input and can be reconstructed from
  target and BTC histories;
- V4 BC is inadmissible until every V4 context channel that affects teacher actions
  exists in the student observation contract.

## Runtime and Student Observation Boundary

V4 introduces auxiliary context as a distinct runtime capability rather than
silently expanding the maintained 206 target-local channels.

The existing target-local 206 features retain their V3 meaning. V4 adds two
separate observation/context blocks:

```text
local_cross_market_context
  = target Spot/perpetual/derivative relationships

global_market_context
  = fixed BTC/ETH public market context
```

The Universal observation schema must expose both blocks as explicit immutable
keys/counts. The hierarchical policy encoder may project these blocks separately
before joining the target-local/instrument representation. V4 does not require
copying the global vector into every sequence token.

The runtime manifest must bind auxiliary BTC/ETH Spot/perpetual datasets and any
futures-metrics/PIT-flow artifacts needed to construct the global block. Read-only
serving must use the same constructor and fail closed if required context is
missing, stale, identity-drifted, or unavailable. It must not replace missing
context with zero vectors.

## Data Architecture

V4 adds information by source role, not by multiplying generic indicators.

### Existing target-local futures block

The maintained 206 target-local features and nine instrument descriptors remain
available without semantic reinterpretation.

### Core local cross-market profile

`cross_market_core_v1` contains exactly 24 channels built from target Spot,
target USD-M perpetual, and funding data:

1. `spot_log_return_1h`
2. `spot_log_return_4h`
3. `spot_log_return_24h`
4. `spot_log_quote_volume_robust_z_4h`
5. `spot_log_quote_volume_robust_z_24h`
6. `spot_perp_log_basis`
7. `spot_perp_basis_change_1h`
8. `spot_perp_basis_change_4h`
9. `spot_perp_basis_robust_z_7d`
10. `spot_minus_perp_log_return_1h`
11. `spot_minus_perp_log_return_4h`
12. `spot_to_perp_log_quote_volume_ratio_1h`
13. `spot_to_perp_log_quote_volume_ratio_4h`
14. `spot_to_perp_log_quote_volume_ratio_24h`
15. `spot_taker_quote_imbalance_1h`
16. `spot_taker_quote_imbalance_4h`
17. `perp_taker_quote_imbalance_1h`
18. `perp_taker_quote_imbalance_4h`
19. `spot_minus_perp_taker_imbalance_1h`
20. `spot_minus_perp_taker_imbalance_4h`
21. `funding_rate`
22. `funding_rate_change`
23. `funding_rate_robust_z_7d`
24. `basis_z_x_flow_divergence_4h`

The taker imbalance for an interval is conceptually:

```text
(2 * taker_buy_quote_volume / total_quote_volume) - 1
```

and is unavailable when the source interval has no valid positive total quote
volume. It is never replaced with an informative zero when the source field is
missing.

The basis is:

```text
log(perpetual_mark_price / spot_close_price)
```

using aligned decision-time values only. Return and volume windows are trailing,
fully closed windows. Robust z-scores use a fixed causal trailing window and an
authored minimum support; the exact numerical primitive is frozen in the
implementation plan before any V4 outcome is read.

### Derivatives extension profile

`cross_market_derivatives_v1` adds exactly seven channels to the 24-channel core:

25. `open_interest_log_change_1h`
26. `open_interest_log_change_4h`
27. `open_interest_log_change_24h`
28. `global_long_short_ratio_robust_z_4h`
29. `top_position_long_short_ratio_robust_z_4h`
30. `basis_z_x_open_interest_change_4h`
31. `funding_z_x_open_interest_change_4h`

This profile is enabled only when a pre-outcome data-capability audit proves
immutable historical coverage for the complete authored interval. Binance Vision
USD-M futures `metrics` archives are preferred over short-retention REST history.
The capability decision may inspect source coverage and schema but may not inspect
return labels, model metrics, or trading outcomes.

If full required coverage is absent, the generation is explicitly authored as
`cross_market_core_v1`; it does not silently create sparse OI fields or switch
profiles after seeing performance.

Historical liquidation data is intentionally excluded from the first V4 profiles.
It may be added in a later authored generation only after an immutable source with
adequate interval coverage and timestamp semantics is proven.

### Fixed Global Market Context

`global_market_core_v1` contains 38 channels. For each anchor in
`(BTCUSDT, ETHUSDT)`, include the following 17 channels:

1. Spot return 1h
2. Spot return 4h
3. Spot return 24h
4. perpetual return 1h
5. perpetual return 4h
6. perpetual return 24h
7. Spot/perpetual basis
8. basis change 4h
9. basis robust z 7d
10. Spot taker imbalance 1h
11. Spot taker imbalance 4h
12. perpetual taker imbalance 1h
13. perpetual taker imbalance 4h
14. Spot/perpetual log quote-volume ratio 4h
15. Spot/perpetual log quote-volume ratio 24h
16. funding rate
17. funding robust z 7d

The final four core channels are:

35. `btc_minus_eth_perp_return_4h`
36. `btc_minus_eth_perp_return_24h`
37. `btc_minus_eth_basis`
38. `btc_eth_perp_return_dispersion_4h`

`global_market_derivatives_v1` adds six channels, for a total of 44. For each of
BTC and ETH add:

- open-interest log change 4h;
- open-interest log change 24h;
- global long/short ratio robust z 4h.

The global context constructor, ordered anchor tuple, channel order, staleness
policy, and data identities are part of the V4 artifact digest.

### Optional point-in-time flow profile

`pit_flow_v1` is a lower-frequency optional extension to the Global Market Context.
It contains:

- BTC exchange netflow 6h and 24h;
- ETH exchange netflow 6h and 24h;
- USDT exchange netflow 6h and 24h;
- USDC exchange netflow 6h and 24h when the provider has equivalent PIT coverage;
- BTC exchange-reserve change 24h and 7d;
- ETH exchange-reserve change 24h and 7d.

Only a provider with point-in-time or otherwise revision-frozen historical
semantics is admissible. Artifact identity binds provider, metric identity,
observation time, effective time, retrieval time, PIT/revision mode, raw-content
digest, and staleness contract.

If no admissible source/credential exists, `pit_flow_v1` is disabled explicitly.
V4 does not scrape or substitute current reconstructed histories and call them
historical evidence.

### Feature timing

Every added source must carry or derive:

- event/source-end time;
- first usable decision time;
- availability;
- staleness;
- raw/source identity.

A value may enter decision `t` only when it was observable no later than the
decision close under the existing signal-delay contract. Higher-timeframe and
low-frequency sources use as-of joins and only completed observations. Publication
latency is applied where source event time and usable time differ.

## Market Proxy and Label Decomposition

V4 avoids a current train-universe aggregate as the market factor. The first V4
market proxy is fixed:

```text
market_proxy_symbol = BTCUSDT
market_proxy_instrument = USD-M perpetual
```

The preflight requires the market proxy to belong to training capability for the
V4 generation and not to be a validation/test-only label source. If that invariant
changes in a future split, the V4 generation is invalid rather than silently using
held-out future BTC outcomes for fitting.

For horizon `h` in `{4h, 24h, 72h}`:

```text
market_proxy_label_h(t)
  = BTCUSDT forward gross log return with the same causal timing rule
```

Only labels whose ends are strictly before the knowledge cutoff enter fitting.
BTC/ETH current public context may be observed at inference; no ETH future outcome
is used to fit the market proxy unless a future separately authored design changes
the proxy contract.

### Causal beta

For BTC itself, `beta_BTC(t) = 1.0` by identity.

For every other target, V4 computes beta only from fully observed historical
returns:

```text
beta_s(t)
  = cov_past(r_s_4h, r_BTC_4h) / var_past(r_BTC_4h)
```

The first V4 authored beta contract is:

```text
return_horizon = 4h
lookback = 720h
minimum_complete_samples = 90
clip = [-3.0, 3.0]
```

The window ends at or before decision `t`; it never includes the forward label
being predicted. If support is insufficient or BTC variance is numerically zero,
beta is unavailable and the affected V4 decision is non-actionable; it is not
filled with a learned or guessed value.

Beta is included in the student-observable V4 context and its source range/digest is
bound to forecast evidence.

For each training symbol and horizon:

```text
residual_label_s,h(t)
  = symbol_label_s,h(t)
    - beta_s(t) * market_proxy_label_h(t)
```

The decomposition must reconstruct the original symbol label within explicit
floating-point tolerance.

## Estimator Semantics

### BTC market-proxy return heads

Fit one deterministic overlap-aware weighted ridge model per horizon:

```text
market_proxy_return_4h
market_proxy_return_24h
market_proxy_return_72h
```

Current inputs are Global Market Context only. The first V4 generation reuses the
existing weighted/objective-normalized ridge primitive. A nonlinear model family
requires a separately authored hypothesis.

### Shared residual return heads

Fit one shared residual ridge per horizon across all train symbols:

```text
residual_return_h = f_h(
    existing_target_local_features,
    local_cross_market_context,
    global_market_context,
    instrument_descriptors,
    causal_beta
)
```

Use overlap-aware symbol-balanced weights. There is no canonical
`symbol -> residual model` dispatch.

### Final expected return

For target symbol `s`:

```text
mu_s,h(t)
  = beta_s(t) * market_proxy_prediction_h(t)
    + residual_prediction_s,h(t)
```

Persist proxy prediction, beta, beta-scaled contribution, residual contribution,
and final prediction separately and require exact reconstruction within authored
tolerance.

### Shared direction heads

Fit one shared weighted ridge direction model per horizon on:

```text
direction_label_s,h(t) = sign(symbol_label_s,h(t))
```

so exact zero labels remain zero and do not invent a direction. The direction score
is signed evidence, not a calibrated probability and is never added to expected
return.

Direction is a consensus constraint:

- risk-reducing movement toward zero is always allowed;
- increasing absolute exposure requires final return forecast and direction score
  to have the same non-zero sign;
- a fast sign reversal requires the same agreement;
- disagreement results in HOLD or risk reduction, never a forced opposite trade.

Return regression and direction score are evaluated and persisted separately.

## Fast and Slow Responsibilities

The 4h head is not averaged into the 24h/72h forecast.

### Slow anchor

The first V4 generation deliberately keeps the existing V3 24h/72h
24-hour-equivalent fusion and horizon-disagreement uncertainty **only for the slow
anchor**. This isolates the effect of the new decomposition and fast-flow lane from
a simultaneous slow-fusion redesign.

The slow controller produces `anchor_weight` on an authored slow cadence or when
risk/liquidity forces a reduction.

### Fast impulse

The 4h final return/direction/uncertainty bundle produces a bounded tactical
deviation around the slow anchor. It consumes the new cross-market flow/basis/
derivatives information without requiring that information to move a 24h/72h
regression first.

The implementation plan freezes `fast_max_abs_delta` and its decision cadence
before outcomes are read. The fast lane remains subject to the existing overall
absolute-weight, liquidity, execution, and hard-risk caps.

### Exact cost rule

V4 must not double-charge cost across slow and fast stages.

For current weight `w0`, slow anchor `a`, and final candidate `f`, total execution
hurdle equals the existing one-way cost function for the **actual final change**
`abs(f - w0)` exactly once.

If implementation uses staged objective comparisons, the fast stage may use the
marginal difference:

```text
C(abs(f - w0)) - C(abs(a - w0))
```

relative to the already scored anchor. Tests must prove staged and direct total
cost are identical.

## State-conditioned Uncertainty

V4 retains final composite residual RMSE and slow horizon disagreement, then adds a
deterministic train-only state calibration.

The first design uses four mutually resolved states:

- normal;
- high realized-volatility;
- low-liquidity;
- basis/positioning stress.

Thresholds and severity precedence are fixed from eligible train-prefix quantiles
before the first V4 outcome is read. Each horizon/state estimates weighted RMSE of
the **final reconstructed forecast**, not independent component RMSEs added under an
independence assumption.

A state below the authored effective-sample minimum falls back to the global
horizon RMSE. It never borrows holdout residuals.

Persist state, threshold digest, support/effective sample size, global RMSE,
state RMSE, selected uncertainty, and fallback reason.

## Signal Liveness Evidence

V4 makes intercept-dominated or stale predictions visible before economic replay.
For each fit/symbol/horizon persist non-promotable diagnostics including:

- prediction mean/std/min/max/quantiles;
- unique count under authored numerical tolerance;
- median and maximum near-identical run length;
- intercept;
- `std(prediction - intercept)`;
- weighted final residual RMSE;
- `std(prediction - intercept) / RMSE`;
- constant-feature and available-feature counts;
- contribution variance by existing timeframe;
- contribution variance for local cross-market, global market, beta-scaled proxy,
  and shared residual families;
- direction-score mean/std and sign balance;
- slow-anchor change count;
- fast-impulse change count.

These diagnostics do not lower or replace the canonical Signal Gate. They make
failure interpretable.

## Artifact and Identity Contract

Introduce distinct V4 schemas and never reinterpret V3 payloads. Conceptually:

```text
causal_alpha_v4_local_cross_market_v1
causal_alpha_v4_global_market_context_v1
causal_alpha_v4_beta_v1
causal_alpha_v4_fit_config_v1
causal_alpha_v4_fit_v1
causal_alpha_v4_forecast_v1
causal_alpha_v4_target_path_v1
causal_alpha_v4_signal_diagnostic_v1
```

V4 identity binds at least:

- source/run/runtime/dataset identities;
- market proxy identity;
- local/global feature profile names, channel order, source digests, and coverage;
- explicit derivatives/PIT profile enabled/disabled state;
- target-local feature and descriptor schemas;
- knowledge cutoff;
- beta config/source range/digest;
- all proxy/residual/direction model digests;
- overlap/symbol-balance weight digests;
- uncertainty state config/support/RMSE;
- student-observation schema digest;
- forecast component reconstruction;
- target compiler configuration.

Stale source content, channel-order drift, changed PIT mode, missing auxiliary
context, wrong beta, hidden per-symbol model state, or student/teacher observation
mismatch fails closed.

## Research Sequence

The retained V3 counterfactual is not rewritten by V4. Research order is:

```text
1. execute/preserve the already-authored V3 market/residual counterfactual
2. retain its result as independent evidence
3. audit V4 source capabilities without reading model/trading outcomes
4. freeze one V4 feature profile and model configuration
5. run V4 train-only Signal counterfactual on earlier chronological contracts
6. evaluate unchanged canonical Signal evidence plus V4 component/liveness evidence
7. only after Signal admission, run existing economic selection
8. only after selection, open Teacher-admission holdout once
9. only after Teacher admission, run BC/DAgger/anchored PPO stages
```

V4 Signal failure stops that generation. It does not authorize threshold
relaxation, feature fishing, hidden profile switching, holdout inspection, or silent
model-family replacement.

## Quality Contract

### Objective

Produce a causal, auditable, student-reproducible hierarchical teacher whose
forecast/target behavior is materially state-dependent and whose admitted candidate
has positive after-cost economics under the existing reward/execution contract.

### Acceptance Criteria

1. Reward, risk, execution, action semantics, and V3 historical artifact meanings
   are unchanged.
2. Every V4 current-time teacher input that can affect actions is present in the
   declared student/serving observation capability.
3. `beta * market_proxy + residual` reconstructs every final horizon prediction.
4. Beta uses only past target/BTC returns and obeys the frozen 4h/720h/90-sample
   contract.
5. 4h fast and 24h/72h slow contributions are separately persisted and
   attributable.
6. Direction evidence is independently evaluated and never blocks risk-reducing
   movement toward zero.
7. State uncertainty never reads a residual whose forward label is unavailable at
   the fit cutoff.
8. Slow+fast target construction charges the actual final turnover hurdle once.
9. Spot/derivative/PIT features have explicit availability, staleness, source
   timing, provenance, and immutable identity.
10. The data-capability audit selects core vs derivatives profile without reading
    return labels or trading outcomes.
11. No non-PIT on-chain history enters training/backtest evidence.
12. Signal-liveness evidence directly exposes an intercept-only or nearly static
    predictor.
13. Existing required Signal/economic/Teacher-admission gates all pass before any
    downstream learner update.
14. Passing Teacher V4 is not reported as BC/PPO success; learner contribution is
    evaluated separately.

### Invariants

- no future label crosses a knowledge cutoff;
- validation/test/Teacher-admission outcomes never flow into fit/tuning;
- historical artifacts never change meaning in place;
- final action remains one target weight for one concrete instrument;
- reward remains pure net-equity log growth;
- hard risk remains independent of teacher success;
- identical immutable inputs produce identical teacher outputs;
- train-symbol row count cannot accidentally change symbol mass;
- teacher current information is student/serving observable;
- unavailable source data is not silently converted to meaningful zero;
- BTC proxy future outcomes are training labels only and never taken from a
  validation/test-only capability.

### Failure Modes

- Spot/perpetual timestamp alignment leaks a future close/mark;
- historical OI/ratio API retention truncates early history without failing;
- current reconstructed on-chain wallet labels rewrite old backtest values;
- BTC proxy is accidentally assigned a held-out-only role;
- causal beta includes the forward prediction interval;
- beta has insufficient support or near-zero BTC variance but is silently filled;
- shared residual model leaks private symbol identity;
- Teacher global context is absent from BC/PPO/serving observation;
- auxiliary BTC/ETH context goes stale and serving silently substitutes zero;
- fast head dominates and creates turnover with no after-cost edge;
- slow and fast stages double-charge one execution delta;
- direction disagreement blocks necessary risk reduction;
- stress-state support is too small and overfits RMSE;
- prediction has a non-zero intercept but negligible dynamic contribution;
- overlapping 4h/24h/72h labels inflate apparent independent support;
- feature profile changes after performance is observed;
- closed-trade count is mistaken for executed-change evidence.

### Risk

High. V4 changes teacher data, auxiliary runtime capability, student observation,
labels, model decomposition, uncertainty, target composition, and artifact identity.
A defect can create lookahead, false profitability evidence, unreproducible BC
behavior, serving mismatch, or unnecessary turnover while ordinary unit tests still
pass.

### Test Oracle

Correctness is judged from observable contracts, not training loss alone.

Required synthetic/controlled oracles include:

- market-only path: residual is zero and beta-scaled BTC proxy reconstructs target;
- residual-only path: BTC proxy return is zero and residual reconstructs target;
- two targets with known different beta and identical residual alpha;
- BTC identity beta exactly one;
- insufficient beta support produces non-actionable evidence;
- modifying any future target/BTC return cannot move an earlier beta/forecast;
- return/direction agreement and disagreement;
- direction disagreement still permits flattening;
- known 4h impulse around stable 24h/72h anchor;
- direct-vs-staged cost equality for final target;
- source becoming available immediately before vs after decision boundary;
- stale/missing Spot/OI/global context fails according to contract;
- non-PIT revised on-chain payload changes identity or is rejected;
- state-RMSE fallback with insufficient support;
- intercept-only model is flagged dynamically inactive;
- Teacher and student context constructors produce exactly equal vectors/digests;
- old V3 artifacts still parse and preserve their original meaning.

Economic oracles include gross return, net return, turnover, submitted change,
filled/executed change, closed trades as a separate statistic, maximum drawdown,
execution rejections, hard-risk violations, and target-reason attribution.

### Required Test Layers

- lint / format / type check / static analysis / import boundaries;
- unit tests for source timing, feature formulas, beta, label decomposition,
  direction, uncertainty, target composition, liveness, and identity;
- property/contract tests for reconstruction, causality, source/order drift, and
  deterministic digests;
- integration tests for Spot + USD-M + funding + optional Vision metrics into
  immutable V4 context artifacts;
- integration tests for auxiliary Global Market Context through training and
  read-only serving observation construction;
- integration tests through the real V4 replay environment;
- regression tests proving reward/risk/execution/V3 schemas remain unchanged;
- artifact corruption/resume tests;
- full repository test suite;
- frontend tests only if Studio/report UI is changed;
- build/package/platform checks required by repository CI;
- immutable train-only V4 counterfactual before economic adoption.

### Quality Gate

Do not report V4 complete unless, on the same final HEAD:

- Acceptance Criteria are mapped to concrete evidence;
- targeted and related integration tests pass;
- lint/format/type/static/import checks pass;
- full test suite passes or every unrelated pre-existing failure is independently
  reproduced and documented;
- build/package/platform checks pass;
- changed lines and important failure paths have meaningful assertions;
- source-latency and label-cutoff causality are tested;
- falsification review attempts to break reconstruction, source identity, beta,
  cost, direction, and teacher/student information invariants;
- architecture self-review finds no hidden per-symbol dependency or observation
  mismatch;
- an independent review reconstructs the evaluation from original requirements,
  final diff, tests, and actual outputs;
- CI/required checks are verified for the same final commit when CI is available;
- unverified items and residual risks are explicitly reported.

A green unit suite, successful training process, or passing Signal Gate alone is
not this Quality Gate.

## Falsification Review Questions

Before promotion, explicitly attempt to prove V4 wrong:

- Can Teacher see any current input Student/serving cannot reproduce?
- Can a future Spot/OI/PIT value move an earlier prediction?
- Can a future target/BTC return move an earlier beta?
- Can a validation/test-only BTC label enter market-proxy fit?
- Can symbol order alter shared predictions without identity drift?
- Can return prediction stay non-zero while dynamic contribution is negligible?
- Can the 4h lane create turnover with no incremental after-cost edge?
- Can direction disagreement block flattening?
- Can unavailable OI/PIT/global context be mistaken for zero state?
- Can a provider revision preserve a stale artifact digest?
- Can slow and fast stages charge two costs for one final delta?
- Can overlapping labels inflate confidence?
- Can zero closed trades incorrectly reject a path with real filled turnover?

## External Evidence Informing the Design

External work informs feature families but does not establish Trade RL
profitability.

- `arXiv:2108.09750`, *Fragmentation, Price Formation, and Cross-Impact in Bitcoin
  Markets*, supports modeling cross-market state/leader-lagger effects rather than
  assuming one venue always leads.
- `arXiv:2212.06888`, *Fundamentals of Perpetual Futures*, supports basis/funding as
  structural Spot-perpetual state.
- `arXiv:2411.06327`, *Return and Volatility Forecasting Using On-Chain Flows in
  Cryptocurrency Markets*, motivates bounded exchange-flow context while showing
  effects can differ by asset.
- Binance public-data documentation/issues show USD-M futures `metrics` archives
  contain historical derivative statistics, while short-retention REST endpoints
  alone are not a valid long-history backtest source. V4 therefore gates those
  channels on immutable coverage capability.

## Implementation Boundary

This document is the design contract only. The implementation plan must split work
into independently reviewable stages at minimum:

1. prerequisite execution-activity accounting correction if still required;
2. Spot/taker/futures-metrics ingestion and immutable context artifacts;
3. Global Market Context runtime + student/serving observation contract;
4. causal beta and BTC-proxy label decomposition;
5. proxy/residual/direction model bundle;
6. state-conditioned uncertainty;
7. slow-anchor/fast-impulse target compiler;
8. V4 artifact/reporting/liveness evidence;
9. train-only counterfactual and existing gate execution;
10. downstream BC/PPO only after Teacher admission.

No stage may use later empirical outcomes to rewrite an earlier authored acceptance
condition.
