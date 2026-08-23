# Causal Alpha V4 Hierarchical Teacher Design

## Status

Design only. This document does not authorize production promotion, live routing,
or reinterpretation of any existing Causal Alpha V3 evidence.

The predeclared V3 market/residual counterfactual in
`2026-08-22-causal-alpha-v3-market-residual-decomposition-design.md` remains
immutable research history. V4 does not rewrite its hypothesis, thresholds,
ridge strengths, evidence, or pass/fail semantics.

## Objective

Build the next research teacher generation around the economic objective already
maintained by Trade RL: cost-inclusive net-equity log growth. Reward remains
unchanged.

The teacher must improve the quality and liveness of the causal expected-return
signal by separating four responsibilities that the pooled V3 ridge currently
mixes together:

1. common crypto-market return;
2. target-instrument residual alpha;
3. fast cross-market flow impulse; and
4. state-dependent forecast uncertainty.

The target outcome is not more trading. The target outcome is a causal teacher
whose action changes are supported by economically meaningful, time-varying edge
and whose information set can be reproduced by the downstream BC/PPO student.

## Non-goals

This change does not:

- change `reward_t = scale * log(net_equity_after / net_equity_before)`;
- add baseline/excess-growth shaping to the reward;
- change hard-risk, margin, liquidation, execution-cost, latency, partial-fill,
  participation, or target-weight accounting semantics;
- lower existing Signal, economic-selection, Teacher-admission, BC, or RL gates to
  make a weak hypothesis pass;
- silently tune V3 retained evidence or mutate the existing 2026-08-22
  market/residual counterfactual;
- build a simultaneous multi-asset portfolio policy;
- duplicate the complete 206-channel futures technical feature set for Spot;
- make non-point-in-time on-chain data admissible for backtests;
- introduce a large nonlinear neural teacher before deterministic linear
  decomposition has been falsified;
- authorize exchange order routing or Production GO.

A separate prerequisite patch may correct activity-accounting semantics where
closed-trade count is currently used as a proxy for meaningful execution. That
patch must remain independently reviewable and must not be bundled into V4 model
performance claims.

## Current Evidence

The current V3 lane uses symbol-balanced, overlap-aware pooled ridge models for
24-hour and 72-hour forward gross log returns. The two horizons are converted to a
24-hour-equivalent expected return and uncertainty bundle before a conservative
cost-aware target compiler.

The fresh research evidence preceding this design shows that bounded variations of
the pooled ridge can produce useful relative ranking while absolute direction
remains unstable. The existing 2026-08-22 design therefore correctly isolates a
common market component from symbol residuals as the next falsification step.

V4 extends that diagnosis rather than replacing it. The additional architectural
findings are:

- a 15-minute decision clock should not force all newly available Spot/derivative
  flow information through only 24h/72h labels;
- a teacher that reads cross-symbol current state unavailable to the deployed
  student creates an irreducible teacher/student information mismatch;
- fully per-symbol hidden models also create a transfer problem for a Universal
  policy that does not consume a secret symbol ID;
- static fit RMSE plus horizon disagreement is insufficient to describe forecast
  risk during volatility/liquidity/basis stress;
- non-zero prediction level is not signal liveness: the dynamic contribution must
  be material relative to residual error and execution hurdle.

## Options Considered

### Option A — add Spot/on-chain columns to the existing pooled V3 ridge

Rejected as the main architecture. It is easy to implement but asks the same model
to learn common direction, residual ranking, fast flow, and slow trend on one loss.
New high-frequency channels can be averaged away by 24h/72h regression and the
teacher/student information problem remains unresolved.

### Option B — market/residual only, with one market model and per-symbol residual models

Retained as the already-authored V3 counterfactual. It is the correct next
falsification of the common-factor hypothesis and must run without retroactive V4
changes. It is not adopted unchanged as the final V4 production-teacher design
because per-symbol hidden heads and train-universe aggregate current inputs need a
student-transfer contract.

### Option C — hierarchical deterministic V4 teacher

Chosen.

Use a fixed global market context available to both teacher and student; a common
market return model; a shared residual model conditioned on target-local features
and public instrument descriptors; a shared symbol-direction score; separate 4h
fast and 24h/72h slow responsibilities; and state-conditioned residual uncertainty.
Reuse deterministic ridge primitives where possible and keep reward/execution/risk
unchanged.

### Option D — nonlinear mixture-of-experts teacher immediately

Deferred. A nonlinear teacher may ultimately be justified, but it adds optimizer,
capacity, calibration, determinism, and overfitting questions before the linear
hierarchical contract has been falsified.

## High-level Architecture

```text
                         fixed global market context
                  BTC/ETH Spot + Perp + derivatives + PIT flow
                                      |
                                      v
                    +----------------------------------+
                    | Common Market Return Heads       |
                    | 4h / 24h / 72h                  |
                    +----------------------------------+
                                      |
                         causal beta_s(t) x market
                                      |
                                      v
 target-local futures + Spot + basis + derivatives + descriptors
                                      |
                                      v
                    +----------------------------------+
                    | Shared Residual Return Heads     |
                    | 4h / 24h / 72h                  |
                    +----------------------------------+
                                      |
                                      +----> mu_s,h
                                      |
 target-local + global context -------+
                                      |
                                      v
                    +----------------------------------+
                    | Shared Direction Heads           |
                    | 4h / 24h / 72h sign scores      |
                    +----------------------------------+
                                      |
                                      v
                    +----------------------------------+
                    | State-conditioned uncertainty    |
                    +----------------------------------+
                         |                       |
                         v                       v
                 24h/72h slow anchor        4h fast impulse
                         |                       |
                         +-----------+-----------+
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

The most important new invariant is:

> Every current-time input used to generate a teacher action must be constructible
> from the same decision-time public information available to the downstream
> student and serving path.

Training labels may of course use realized future outcomes after their label ends
become known, subject to the existing knowledge cutoff. Current-time teacher inputs
may not depend on validation/test outcomes or on hidden train-symbol state that the
student cannot observe.

Consequences:

- the market head does not consume an implicit current aggregate over whichever
  nine symbols happened to be authored as the training universe unless that exact
  aggregate is also part of the student observation and serving contract;
- V4 instead uses a fixed Global Market Context whose semantic identity is stable
  across train, validation, test, and serving;
- the residual return function is shared across instruments;
- instrument-specific behavior is conditioned on continuous public descriptors,
  not on a private per-symbol model selector;
- if V4 adds Global Market Context to teacher prediction, the same frozen context
  channels must be added to the Universal student observation schema before V4 BC
  is admissible.

## Data Architecture

V4 adds information by source role, not by duplicating every technical indicator
on every venue.

### Existing target-local futures block

The maintained 206 target-local market features and nine instrument descriptors
remain available. Their exact identity and causality contract are unchanged unless
a later implementation plan explicitly bumps the feature schema.

### Core cross-market feature pack

The first V4 feature generation uses public exchange data and is expected to be
reproducible without an authenticated trading account.

For each target instrument where Spot and USD-M perpetual data both exist, author a
bounded local cross-market pack containing approximately the following channels.
The final ordered list is frozen before the first V4 counterfactual and is not
expanded after observing its result.

#### Local Spot/perpetual state

- Spot log return: 1h, 4h, 24h;
- Spot quote-volume robust z-score: 4h, 24h;
- perpetual/Spot log basis level;
- basis change: 1h, 4h;
- basis robust z-score over a fixed trailing window;
- Spot minus perpetual return: 1h, 4h;
- Spot/perpetual quote-volume ratio: 1h, 4h, 24h;
- Spot taker-flow imbalance: 1h, 4h when source fields are available;
- perpetual taker-flow imbalance: 1h, 4h when source fields are available;
- Spot minus perpetual flow imbalance: 1h, 4h;
- open-interest change: 1h, 4h, 24h when immutable historical coverage exists;
- funding level, funding change, and funding robust z-score;
- basis-z x open-interest-change;
- funding-z x open-interest-change;
- basis-z x Spot/perpetual-flow divergence.

This is intentionally larger than a minimal basis-only experiment but remains
small relative to a duplicated 206-channel Spot technical stack.

### Fixed Global Market Context

Use fixed semantic anchors rather than a changing train-symbol aggregate. BTC and
ETH are the initial public market anchors because they have deep Spot/perpetual
markets and are meaningful across target symbols.

The global context contains an authored subset of:

- BTC/ETH Spot returns: 1h, 4h, 24h;
- BTC/ETH perpetual returns: 1h, 4h, 24h;
- BTC/ETH basis level/z-score/change;
- BTC/ETH Spot and perpetual flow imbalance: 1h, 4h;
- BTC/ETH Spot/perpetual volume ratio;
- BTC/ETH open-interest change where immutable historical coverage exists;
- BTC/ETH funding level/z-score;
- cross-anchor dispersion and BTC-vs-ETH relative return;
- optional PIT BTC/ETH exchange netflow and stablecoin exchange flow as described
  below.

The Global Market Context has one explicit schema/digest and is identical in
meaning regardless of the target instrument.

### Historical derivatives availability

Current repository ingestion already supports Binance public Spot and USD-M market
construction, funding, mark/index price, and multi-timeframe data. V4 may extend
that source contract for Binance Vision futures `metrics` archives or equivalent
immutable public artifacts when their historical coverage is complete enough for
the authored research interval.

Do not backfill a long historical training interval from an endpoint that only
returns a short recent window. Missing historical derivatives coverage fails the
feature family closed or excludes that family from the predeclared generation.
It must not be silently imputed as zero and interpreted as market state.

### On-chain / exchange-flow pack

On-chain flow is a secondary, lower-frequency context source, not the first source
of fast actions.

V4 defines an optional `pit_flow` pack containing only a few high-value families:

- BTC exchange netflow: 6h, 24h;
- ETH exchange netflow: 6h, 24h;
- USDT exchange netflow: 6h, 24h;
- USDC exchange netflow: 6h, 24h when supported;
- optionally BTC/ETH exchange reserve change: 24h and 7d.

This pack is admissible only from a provider that exposes point-in-time or
otherwise revision-frozen historical semantics suitable for systematic backtests.
The artifact identity must bind provider, metric identity, observation timestamp,
effective timestamp, retrieval timestamp, revision/PIT mode, raw-content digest,
and allowed staleness.

A non-PIT current label history must never be presented as historical evidence.
If no admissible provider/credential is available, the first V4 generation runs
with `pit_flow` disabled and records that absence explicitly; it does not substitute
unverified scraped values.

### Feature timing

Every added source must provide or derive:

- event/source end time;
- first usable decision time;
- source availability flag;
- staleness;
- raw/source identity.

A value can enter decision `t` only if it was observable no later than the decision
close under the maintained signal-delay contract. Publication latency is applied
where source timestamps are not identical to information availability.

Higher-timeframe and low-frequency flows use as-of joins and never use a partially
formed future bar.

## Label Decomposition

V4 preserves causal forward gross log-return labels and the existing strict
knowledge-cutoff rule.

For horizon `h` in `{4h, 24h, 72h}` at decision `t`, define the common market label
from the fixed training universe only for model fitting:

```text
market_label_h(t) = equal_weight_mean_s(return_s,h(t))
```

This uses train labels only after all required label ends are strictly before the
knowledge cutoff. The training-universe label is not itself a current-time input.

### Causal beta

For target symbol `s`, estimate `beta_s(t)` only from returns fully observed before
`t`. Use a deterministic rolling covariance/variance estimator over an authored
lookback, with a minimum support requirement and an authored finite clip.

Conceptually:

```text
beta_s(t) = cov_past(r_s, r_market) / var_past(r_market)
```

The implementation must specify exact return clock, lookback, support, clipping,
and zero-variance fallback before the counterfactual. It may not fit beta from the
future label being predicted.

The residual label is:

```text
residual_label_s,h(t)
  = symbol_label_s,h(t) - beta_s(t) * market_label_h(t)
```

The decomposition must reconstruct the symbol label within numerical tolerance.

## Estimator Semantics

### Common market return heads

Fit one deterministic overlap-aware return model per horizon using only Global
Market Context as current-time inputs:

```text
market_return_4h
market_return_24h
market_return_72h
```

The first V4 model family reuses weighted/objective-normalized ridge. A nonlinear
replacement requires a separate authored hypothesis.

### Shared residual return heads

Fit one shared model per horizon across all training symbols:

```text
residual_return_h = f_h(
    target_local_features,
    local_cross_market_features,
    global_market_context,
    instrument_descriptors
)
```

Use symbol-balanced overlap-aware weights so one highly liquid/long-history symbol
does not dominate simply by row count.

There is no hidden `symbol -> model` lookup in the V4 canonical path.

### Final expected return

For each symbol/horizon:

```text
mu_s,h(t)
  = beta_s(t) * market_prediction_h(t)
    + residual_prediction_s,h(t)
```

Persist common, beta, residual, and final contributions separately.

### Shared direction heads

A separate shared direction model per horizon is trained on the sign of the
original symbol forward return using the same causal current-time information set.
The first implementation reuses deterministic weighted ridge on `{-1, +1}` labels
rather than introducing a new probabilistic solver.

The output is a signed direction score, not a calibrated probability.

Direction score is not added to expected return. It is an independent consensus
constraint:

- reducing absolute exposure toward zero is always allowed subject to risk and
  execution rules;
- increasing exposure in a direction requires the final return forecast and
  direction score to have the same non-zero sign;
- a fast sign reversal requires the same consensus;
- disagreement produces HOLD or exposure reduction, never a forced opposite trade.

This keeps return magnitude and directional classification as separately
falsifiable capabilities.

## Fast and Slow Responsibilities

V4 does not average the 4h flow signal into the 24h/72h return forecast.

### Slow anchor

24h and 72h remain the medium/slow investment horizon. Their existing
24-hour-equivalent fusion and disagreement-aware uncertainty may be retained as the
initial slow-anchor primitive to isolate the architectural change.

The slow controller produces a desired `anchor_weight`. It is recomputed only on
an authored slow cadence or a risk/liquidity emergency.

### Fast impulse

The 4h head produces a bounded tactical deviation around the slow anchor. It is
intended to consume Spot/perpetual flow, basis, open-interest, and other faster
cross-market state.

The fast component must not independently create unlimited target turnover. It has
its own maximum deviation and uses the existing overall liquidity and absolute
weight caps.

### Cost accounting

The compiler must not double-charge execution cost when a fast deviation is applied
to a slow anchor.

For a current weight `w0`, chosen slow anchor `a`, and final candidate `f`, the
implementation scores the actual final turnover cost from `w0 -> f`. If staged
objectives are used internally, the fast stage uses only the marginal difference
between final and anchor execution hurdle so total charged cost equals the actual
one-way turnover assumption exactly once.

This property requires direct tests.

## State-conditioned Uncertainty

V3 fit residual RMSE and horizon disagreement remain valid ingredients but are not
enough by themselves.

V4 adds a deterministic train-only residual calibration by causal market state.
The first design uses a small authored state taxonomy rather than a neural
uncertainty model:

- normal;
- high realized-volatility;
- low-liquidity;
- basis/positioning stress.

State thresholds are derived only from the eligible training prefix with fixed
quantile rules. If multiple stress states apply, deterministic severity precedence
is used.

For each horizon/state, estimate weighted final-forecast residual RMSE. A state
bucket below the authored effective-sample minimum falls back to the global
horizon RMSE; it is not filled from holdout data.

The target compiler consumes the state-conditioned final forecast uncertainty.
Persist both global and selected state RMSE plus support/effective sample size.

## Signal Liveness Evidence

V4 adds non-promotable but mandatory diagnostic evidence to distinguish a non-zero
intercept from a live market signal.

For every fit/symbol/horizon scope, persist at least:

- prediction mean/std/min/max and quantiles;
- prediction unique count under an authored numerical tolerance;
- maximum and median consecutive near-identical run length;
- intercept;
- `std(prediction - intercept)`;
- weighted residual RMSE;
- dynamic-signal-to-RMSE ratio;
- constant-feature count;
- available-feature count;
- contribution variance by feature family;
- contribution variance by timeframe for existing market features;
- common-market contribution variance;
- beta contribution variance;
- residual contribution variance;
- direction-score mean/std/sign balance;
- fast/slow target contribution counts.

A diagnostic artifact cannot by itself promote a model. Its purpose is to make
prediction collapse observable before a later economic failure.

## Artifact and Identity Contract

Introduce distinct V4 schemas; do not reinterpret V3 payloads.

Conceptually:

```text
causal_alpha_v4_global_context_v1
causal_alpha_v4_local_cross_market_v1
causal_alpha_v4_fit_config_v1
causal_alpha_v4_fit_v1
causal_alpha_v4_forecast_v1
causal_alpha_v4_target_path_v1
causal_alpha_v4_signal_diagnostic_v1
```

V4 fit identity binds at least:

- source/run/runtime/dataset identities;
- global-context schema and source digests;
- local cross-market schema and source digests;
- optional PIT-flow schema/source digests and explicit enabled/disabled state;
- ordered train-symbol tuple;
- feature and instrument-descriptor schema digests;
- knowledge cutoff;
- beta configuration/digest;
- all market/residual/direction model digests;
- overlap-weight digests;
- state-uncertainty configuration/support/RMSE;
- final forecast reconstruction evidence.

Any stale source, changed feature order, missing model, wrong beta, changed PIT mode,
or teacher/student information-set mismatch fails closed.

## Research Sequence

V4 is not allowed to consume the untouched Teacher-admission holdout for tuning.
The sequence is:

```text
1. preserve and execute the already-authored V3 market/residual counterfactual
2. retain its result as independent evidence
3. author/freeze V4 feature and model configuration before V4 outcomes are read
4. build V4 train-only counterfactual on earlier chronological Signal contracts
5. evaluate unchanged canonical Signal evidence plus new V4 component diagnostics
6. only after Signal admission, run economic selection
7. only after selection, open the untouched Teacher-admission holdout once
8. only after Teacher admission, run BC/DAgger/anchored PPO stages
```

V4 Signal failure stops the generation. It does not authorize threshold relaxation,
feature-family fishing, holdout inspection, or silent model-family substitution.

## Objective Quality Contract

### Objective

Produce a causal, auditable, student-reproducible hierarchical teacher whose
forecast and target behavior is materially state-dependent and whose admitted
candidate has positive after-cost economics under the existing reward/execution
contract.

### Acceptance Criteria

1. Reward, risk, execution, action semantics, and V3 historical artifacts are
   unchanged.
2. All V4 teacher current-time inputs are present in the declared student
   observation contract or are proven not to affect teacher actions.
3. Common + beta + residual predictions reconstruct final horizon prediction.
4. Beta uses only information available before each decision and reconstructs the
   authored residual label exactly.
5. 4h fast and 24h/72h slow components are separately persisted and attributable.
6. Direction heads are independently evaluated; return/direction disagreement can
   suppress exposure increase without blocking risk-reducing actions.
7. State uncertainty never reads a residual whose label end is not before its
   knowledge cutoff.
8. Execution cost is charged once for the actual final target change.
9. Added Spot/derivative/PIT features have explicit timing, availability,
   staleness, provenance, and immutable identity.
10. No non-PIT on-chain history enters backtest/training evidence.
11. Signal liveness diagnostics make a constant/intercept-dominated predictor
    directly observable.
12. The predeclared V4 generation passes every existing required Signal/economic/
    Teacher-admission gate before any downstream learner update.
13. BC/PPO is not claimed successful merely because Teacher V4 passes; learner
    contribution remains separately evaluated.

### Invariants

- no future label crosses the knowledge cutoff;
- validation/test/Teacher-admission outcomes never flow into fit/tuning;
- historical artifact semantics never change in place;
- final action remains one target weight for one concrete instrument;
- reward remains pure net-equity log growth;
- hard risk remains independent of teacher success;
- target generation is deterministic given identical immutable inputs;
- train-symbol row count cannot change symbol mass unintentionally;
- current-time teacher information is student-observable;
- unavailable source data is not silently converted into a meaningful zero.

### Failure Modes

- Spot/perpetual timestamp misalignment creates lookahead;
- historical OI/flow endpoint has insufficient retention and silently truncates
  early training history;
- on-chain provider retroactively relabels exchange wallets;
- common market label uses incomplete train universe;
- rolling beta uses future returns or unstable zero market variance;
- shared residual model leaks symbol identity through an unintended field;
- global context exists in Teacher but not BC/PPO observation;
- 4h fast head dominates and creates excessive turnover;
- slow and fast stages double-count execution costs;
- direction gate prevents necessary risk-reducing deleveraging;
- stress buckets have too little support and overfit uncertainty;
- fit has non-zero intercept but negligible dynamic prediction;
- duplicate/overlapping horizons inflate apparent sample support;
- missing Spot/OI/on-chain rows are interpreted as genuine zeros;
- source schema/order drift reuses stale cached evidence;
- closed-trade count is mistaken for executed-change count in economic admission.

### Risk

High. The change modifies teacher information, labels, model decomposition,
observation inputs, target generation, data provenance, and research artifacts.
A defect can create lookahead, false profitability evidence, an unreproducible BC
teacher, or unnecessary turnover while leaving ordinary unit tests green.

### Test Oracle

Correctness is judged from independently observable contracts, not from training
loss alone.

Required controlled oracles include:

- synthetic common-factor path where market/residual reconstruction is exact;
- two symbols with different known beta and identical residual alpha;
- residual-only path where market return is zero;
- market-only path where residual is zero;
- return head/direction head agreement and disagreement cases;
- risk-reducing action during direction disagreement;
- known 4h impulse on a stable 24h/72h anchor;
- exact one-way cost comparison showing no double charging;
- source row that becomes available just before/after a decision boundary;
- missing/stale Spot/OI/PIT-flow source;
- revised non-PIT on-chain history rejected by identity contract;
- state-RMSE fallback with insufficient support;
- intercept-only model flagged as dynamically inactive;
- identical model input in Teacher and student observation builder;
- old V3 artifacts still parse and retain their original digests/meaning.

Economic test oracles include gross return, net return, turnover, filled/executed
changes, closed trades as a separate metric, maximum drawdown, execution rejection,
hard-risk violations, and target-reason attribution.

### Required Test Layers

- static analysis / type check / lint / format;
- unit tests for source timing, feature arithmetic, beta, decomposition, direction,
  uncertainty, target composition, and identity;
- property/contract tests for reconstruction, causality, and digest drift;
- integration tests for Spot + USD-M + futures metrics + optional PIT source into
  immutable dataset artifacts;
- integration tests through the real V4 replay environment;
- regression tests proving reward/risk/execution/V3 schemas are unchanged;
- reporting/artifact corruption and resume tests;
- full repository test suite;
- frontend checks only if the implementation changes Studio/report UI;
- build/package checks and platform compatibility required by the repository CI;
- real train-only counterfactual before any production-teacher implementation is
  considered economically admitted.

### Quality Gate

Do not claim V4 complete unless all of the following are true on the same final
HEAD:

- Acceptance Criteria are mapped to concrete evidence;
- targeted tests pass;
- related module/integration tests pass;
- lint/format/type/static/import-boundary checks pass;
- full test suite passes or every unrelated pre-existing failure is documented and
  independently reproduced;
- build/package checks pass;
- changed lines and important failure paths are exercised by meaningful assertions;
- causality tests cover source latency and label cutoff;
- falsification review attempts to break reconstruction, identity, cost, and
  teacher/student information invariants;
- architecture self-review finds no unintended dependency reversal or hidden
  per-symbol teacher state;
- independent review is performed from the original requirements and final diff;
- CI/required checks are verified against the same final commit when CI is
  available;
- unverified items and residual risks are explicitly reported.

A green training run, a green unit suite, or a passing Signal Gate alone is not the
Quality Gate.

## Falsification Review Questions

Before promotion, explicitly try to prove V4 wrong:

- Can the Teacher see a current feature the student cannot reconstruct?
- Can a future Spot/OI/PIT value move an earlier prediction?
- Can changing symbol order alter a shared prediction without changing identity?
- Can beta accidentally use the same forward horizon it is decomposing?
- Can the return head be positive while its dynamic component is effectively zero?
- Can the fast head create turnover with no incremental after-cost edge?
- Can direction disagreement block flattening a dangerous position?
- Can unavailable OI or PIT flow be mistaken for zero state?
- Can a revised provider payload preserve the same artifact digest?
- Can slow and fast stages charge two costs for one executed delta?
- Can an implementation pass current tests while swapping common and residual
  contributions?
- Can overlapping 4h/24h/72h rows inflate confidence?
- Can a zero closed-trade count incorrectly reject a path with real filled
  turnover?

## External Evidence Informing the Design

The design uses external literature as supporting context, not as proof of Trade RL
profitability.

- `arXiv:2108.09750`, *Fragmentation, Price Formation, and Cross-Impact in Bitcoin
  Markets*, supports treating cross-market microstructure and leader/lagger effects
  as distinct information rather than assuming one venue always leads.
- `arXiv:2212.06888`, *Fundamentals of Perpetual Futures*, supports basis/funding as
  structural Spot-perpetual state rather than another arbitrary technical feature.
- `arXiv:2411.06327`, *Return and Volatility Forecasting Using On-Chain Flows in
  Cryptocurrency Markets*, motivates bounded exchange-flow features while also
  showing that effects differ by asset.
- Binance public-data documentation/issues expose historical USD-M `metrics`
  archives containing open-interest/ratio fields, while short-retention REST
  endpoints are not sufficient by themselves for a long immutable training
  history. V4 therefore binds historical coverage and source identity rather than
  assuming endpoint availability implies backtest availability.

## Implementation Boundary

This document is the design contract only. The implementation plan must decompose
work into independently reviewable stages, at minimum:

1. prerequisite activity-accounting correction, if still required;
2. immutable public Spot/derivative source extensions and feature pack;
3. Global Market Context + student observation contract;
4. causal beta + label decomposition;
5. common/residual/direction model bundle;
6. state-conditioned uncertainty;
7. slow-anchor/fast-impulse target compiler;
8. V4 artifact/reporting/liveness evidence;
9. train-only counterfactual and gate execution;
10. downstream BC/PPO only after Teacher admission.

No implementation stage may use later-stage empirical outcomes to rewrite the
predeclared acceptance threshold of an earlier stage.
