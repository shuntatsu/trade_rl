# Causal Scenario Action Evaluator Design

## Status

Approved design for Phase C. This specification defines an evaluation-only causal
scenario oracle and the evidence gates required before any Phase A teacher or
behavior-cloning integration is allowed.

## Goal

Add a research-only evaluator that answers the following question at a decision
state without reading the realized future of that state:

> If one residual action is applied now and the maintained Trend baseline is used
> afterwards, which candidate has the best distribution of baseline-relative
> finite-horizon outcomes under scenarios built only from earlier fold-training
> history?

The evaluator measures whether useful causal decisions exist between the
Perfect-Information Linear Bound and the maintained `residual-ppo-15m` policy.
It does not change the maintained training candidate, produce deployable orders,
or claim profitability.

## Architectural position

The research comparison is split into four distinct layers:

1. `PerfectInformationBound`: optimistic future-informed evaluation bound.
2. `CausalScenarioActionEvaluator`: train-only scenario-based causal planner.
3. maintained Trend baseline and residual PPO policies.
4. later Phase A students trained from state-action values, only after Phase C
   passes its declared evidence gate.

Future-informed paths from the perfect-information benchmark and the historical
DP teacher are not inputs to the causal evaluator. The evaluator shares the
maintained action composer, risk controls, stateful execution engine, and
finite-horizon terminal accounting contract, but remains outside maintained PPO
fitting, checkpoint/configuration selection, Serving, promotion, release
authorization, and direct execution.

## Scope decomposition

Phase C is implemented as three dependency-ordered pull requests.

### C1: contracts and artificial-market evaluator

C1 introduces immutable contracts for candidate actions, scenario rollouts,
value statistics, regret, deterministic selection, and the evaluation artifact.
It accepts already constructed causal scenarios through a narrow rollout-engine
protocol and proves the value calculation on artificial markets. It does not
build scenarios from historical data and does not modify walk-forward training.

### C2: train-only conditioned block scenario library

C2 builds a frozen scenario library exclusively from a fold's train range,
selects deterministic nearest historical regimes that ended before the query,
and creates replayable future blocks without accessing checkpoint, selection,
or outer-test futures.

### C3: walk-forward comparison and Phase A gate

C3 runs the frozen evaluator against Trend, residual PPO, and a compatible
Perfect-Information Bound on predeclared checkpoint and selection ranges. It
produces gap, calibration, ranking, cost, turnover, drawdown, and robustness
reports. It may open a sealed outer test only under the repository's existing
one-shot ledger and only after all evaluator configuration and gates are frozen.

## Decision semantics

### One-step residual deviation

At query time `t`, every candidate is a raw residual action under the maintained
`ActionSpec`. The selected candidate is applied for exactly one decision. For
all subsequent decisions in the scenario horizon, the residual action is zero,
which means the maintained Trend target is recomputed causally from each
scenario path and used without an additional residual.

For candidate `a`, scenario `s`, and horizon `H`, the value is:

```text
apply a at t
apply zero residual at t+1 ... t+H-1
use the same action, risk, emergency, pending-order, execution, and accounting
contracts at every step
liquidate under the maintained finite-horizon terminal contract at H
```

This estimates the marginal value of the current residual decision. C1 through
C3 do not implement multi-step scenario-tree MPC.

### Fixed initial horizon

The initial predeclared horizon is 96 decisions, equal to 24 hours at the
maintained 15-minute clock. The horizon is digest-bound and configurable for
research comparisons. C3 may choose among predeclared alternative horizons only
on the checkpoint range and freezes one horizon before selection evaluation.

### Current-state closure

Each query binds a complete causal snapshot sufficient to reproduce execution:

- causal policy observation and its schema digest;
- Trend state and current baseline target;
- hybrid book and shadow-book identities where applicable;
- persistent pending orders and execution state;
- reward, drawdown, risk, emergency, and tradability state;
- dataset, environment, action, execution-policy, and source-commit identities.

Missing state is an error. The evaluator does not reconstruct omitted state from
future bars or silently reset to cash.

## Candidate action generation

C1 defines a deterministic bounded candidate generator. Its mandatory set is:

1. zero residual;
2. for each asset, isolated raw residual magnitudes `-1.0`, `-0.5`, `+0.5`, and
   `+1.0` with all other residual coordinates zero;
3. two portfolio-wide baseline-reduction proposals that oppose the sign of the
   current Trend target with raw magnitudes `0.5` and `1.0`;
4. optional externally supplied proposals, initially the deterministic PPO mean
   action and half of that action in C3.

All raw actions must be finite and inside the action contract. They pass through
the canonical composer, emergency-risk and portfolio-risk controls before
execution. Candidates that produce identical submitted targets and identical
execution intent are deduplicated by canonical digest. Zero residual must remain
present after deduplication. The initial candidate count is capped at 32; excess
or nondeterministically ordered candidates fail closed.

The evaluator stores both raw residuals and projected submitted targets. It does
not treat projection-suppressed raw actions as economically distinct.

## Scenario library

### Train-only and past-only construction

For each fold, the library is built from the fold train interval only. An anchor
is eligible when:

- all condition features at the anchor are causally available;
- a complete 96-decision future block lies inside the train interval;
- all required execution and market arrays are finite and valid;
- the block does not cross a dataset discontinuity that the maintained simulator
  cannot represent.

Checkpoint, selection, purge, test, outer, or fresh-confirmation rows are never
library anchors and never affect library normalization, distances, scenario
weights, or hyperparameters.

Every scenario used for a query must also be historical relative to that query.
The source block's final bar must be strictly earlier than the query's causal
cutoff. For a development query inside the train interval, a full one-horizon
embargo is additionally required:

```text
source_block_stop <= query_index - horizon
```

Thus no train query can select its own future, an overlapping future, or a later
train block. For checkpoint and later queries, the fold train boundary already
provides a stronger chronological separation.

### Condition vector

The version-one condition vector is deterministic and uses only the last
causally closed market bar and state available at the query timestamp:

- maintained Trend fast, base, and slow signals per asset;
- 24-hour realized volatility per asset;
- 7-day pairwise return correlations;
- spread rates and log market-notional liquidity;
- funding rates and funding-due flags;
- tradable, buy-allowed, sell-allowed, borrow-available, and active masks.

Continuous components are median-centered and scaled by train-only median
absolute deviation with an epsilon floor. Binary masks remain unscaled. The
normalization statistics and feature ordering are immutable library evidence.
No processing-bar OHLCV or other post-decision value enters this vector.

### Conditioned block selection

Version one uses deterministic k-nearest-neighbor block selection:

- `scenario_count = 64`;
- squared Euclidean distance in normalized condition space;
- ascending anchor index as the final tie-break;
- uniform probability `1/64` for each selected scenario;
- no automatic fallback to an unconditional library.

Fewer than 64 eligible past anchors is a fail-closed error. Scenario identifiers
bind the dataset ID, train range, anchor index, horizon, condition configuration,
and library digest.

### Replay transformation

Historical blocks are transferred to the query state through relative market
paths rather than absolute historical prices. The scenario block stores and
replays:

- open, high, low, close, and mark-price relatives;
- volume and market-notional relatives with non-negative validation;
- spread, fee, funding, borrow, cash-rate, dividend, tradability, directional,
  active, delisting, split, minimum-notional, participation, tick, and lot paths;
- elapsed-time coordinates and funding-due events.

The first synthetic scenario bar is anchored to the query's current market
state. Corporate-action and delisting fields retain the source block's causal
ordering. Any transformed path that violates the `MarketDataset` or execution
contracts is rejected rather than repaired.

## Scenario rollout and objective

Every candidate starts from an independent clone of the same query snapshot and
is replayed through the maintained conservative stateful execution path.
Candidate scenarios share no mutable book, pending-order, random-number, or
telemetry state.

For scenario `s` and candidate `a`:

```text
g[s, a] = log(terminal_equity[s, a] / starting_equity)
d[s, a] = g[s, a] - g[s, zero]
```

`d` is the baseline-relative scenario advantage. Transaction fees, spread,
impact, funding, borrow, dividends, cash interest, partial fills, pending orders,
and terminal liquidation already enter terminal equity and are not subtracted a
second time.

Version one penalizes only downside relative to the baseline:

```text
downside_loss[s, a] = max(-d[s, a], 0)
mean_advantage[a] = mean(d[:, a])
downside_cvar_10[a] = mean(largest ceil(0.10 * 64) downside losses)
score[a] = mean_advantage[a] - 0.25 * downside_cvar_10[a]
regret[a] = max(score) - score[a]
```

The CVaR level `0.10`, penalty `0.25`, scenario count, and horizon are
configuration- and artifact-digest inputs. C3 may compare alternatives on the
checkpoint range, but must freeze them before selection.

### Confidence evidence

Each query records all scenario-level advantages. It also records a deterministic
90% bootstrap confidence interval for the mean advantage using 256 resamples
from a counter-based generator seeded by the query and configuration digest.
This interval is diagnostic and is not used to choose the candidate. Fold-level
and aggregate C3 inference uses the repository's maintained paired moving-block
bootstrap, not independent per-query intervals.

### Deterministic selection

Candidates within `1e-8` score of the maximum are tied. Ties are resolved by:

1. lower expected filled turnover;
2. lower raw residual L1 norm;
3. zero residual;
4. canonical candidate digest.

The selected index, tie set, and every tie-break input are stored. Repeated
execution on the same source head and inputs must produce the same result digest.

## Artifact contract

C1 introduces `causal_scenario_value_artifact_v1`, separate from the existing
single-action supervised teacher artifact. It stores:

- dataset, fold, train, query, source, environment, action, observation,
  execution, risk, Trend, scenario-library, candidate-generator, and evaluator
  digests;
- query timestamp/index and causal state-snapshot digest;
- scenario IDs, anchor indices, probabilities, distances, and condition vectors;
- raw candidate residuals, projected targets, and candidate digests;
- per-scenario gross log returns and baseline-relative advantages;
- feasibility, termination, fill, cost, turnover, and terminal-equity evidence;
- mean advantage, downside CVaR, score, regret, bootstrap interval, and effective
  scenario count per candidate;
- selected candidate, tie set, and deterministic selection evidence.

Arrays have explicit scenario and candidate axes, are numeric and finite where
their masks declare validity, become read-only after construction, and are
written in deterministic archives. The artifact has exact file closure,
canonical JSON, content digests, atomic writes, and fail-closed loading. It
contains no realized query-future outcome; realized C3 evaluation is a separate
paired-evaluation artifact.

## Data-access and leakage boundaries

The implementation makes the following boundaries structural:

- the scenario builder receives a train-range capability, not an unrestricted
  dataset view;
- scenario anchors and normalization statistics expose source indices for audit;
- the query-time selector filters the frozen train library to blocks that ended
  before the query's causal cutoff;
- the evaluator receives frozen scenarios and a causal query snapshot, not the
  query's subsequent realized bars;
- the C3 realized replay is invoked only after candidate selection has been
  persisted and digest-bound;
- checkpoint may tune only predeclared Phase C parameters;
- selection evaluates the frozen configuration;
- outer evaluation requires the existing sealed-test ledger and cannot change
  the evaluator, scenario library, candidates, horizon, or gates;
- none of the Phase C artifacts may enter Serving, policy observations, PPO
  rewards, maintained checkpoint/configuration selection, or release
  authorization.

Prefix-causality tests must prove that changing any row at or after the query's
decision cutoff does not change the condition vector, candidate set, eligible
past anchors, selected scenario anchors, or predicted value artifact.

## Error handling

The implementation fails closed for:

- missing zero-residual baseline;
- invalid candidate dimensions, bounds, order, count, or duplicate identity;
- incomplete causal state or incompatible identity digests;
- insufficient, future-relative, overlapping, or out-of-range scenario anchors;
- train-library overlap with forbidden ranges;
- non-finite normalization, distances, paths, values, costs, or probabilities;
- probabilities that are negative or do not sum to one within tolerance;
- transformed scenario paths that violate market-data contracts;
- mutable state shared across scenario or candidate rollouts;
- replay results that cannot be independently reconstructed;
- baseline-relative advantages inconsistent with stored gross returns;
- selected indices or regret inconsistent with recomputed statistics;
- artifact file-closure, digest, schema, or read-only violations.

There is no implicit substitution of cash, unconditional scenarios, a shorter
horizon, fewer scenarios, optimistic execution, or current realized future.

## C3 evaluation report

For each fold and aggregate range, C3 reports:

- a same-period compatible Perfect-Information relaxed bound and replay return;
- causal Scenario Oracle, Trend zero-residual, deterministic PPO mean, and
  predeclared random-candidate comparator returns;
- compatible `PerfectInfo - ScenarioOracle` information/relaxation gap;
- `ScenarioOracle - PPO` policy approximation gap;
- paired daily log-growth uplift, confidence interval, and p-value;
- predicted versus realized candidate ranking, top-one regret, Spearman
  correlation, and random-ranking comparison;
- predicted mean/CVaR calibration by score bucket;
- turnover, fees, spread, impact, funding, borrow, fills, pending-order events,
  termination reasons, and maximum drawdown;
- sensitivity under maintained nominal and adverse execution scenarios;
- scenario-neighbor distances, anchor concentration, and effective historical
  coverage.

Perfect-information ordering is asserted only when the bound and causal replay
use the same realized period, initial weights, return matrix, exposure limits,
and a documented relaxation whose feasible set contains the executable causal
path. When those dominance conditions are not proven, the report marks the gap
`not_comparable` instead of asserting an upper-bound violation.

## Phase A entry gate

Phase A teacher development is prohibited until a frozen C3 configuration passes
all of the following on selection evidence:

1. no leakage, identity, replay, artifact, or deterministic-reproduction failure;
2. at least six independently reset folds covering at least 180 selection days;
3. at least four of six folds have positive Scenario-Oracle uplift over the
   zero-residual Trend baseline;
4. the aggregate paired moving-block 95% lower confidence bound for daily
   Scenario-Oracle-minus-Trend log growth is strictly positive;
5. the worst fold's Scenario-Oracle maximum drawdown is no greater than 20% and
   no more than two percentage points worse than Trend;
6. selected-candidate realized regret is lower than the random-candidate
   comparator with a strictly positive paired 95% confidence margin;
7. predicted candidate ranking has positive aggregate Spearman correlation with
   realized finite-horizon ranking and a strictly positive lower confidence
   bound;
8. every asserted perfect-information comparison satisfies the documented
   dominance conditions and bound ordering within tolerance;
9. nominal and required adverse execution scenarios satisfy their predeclared
   cost, turnover, drawdown, and positive-uplift gates.

Failing this gate leaves Phase A unimplemented. Passing it authorizes a separate
Phase A design; it does not automatically enable BC, PPO fine-tuning, Serving,
or production.

## Testing strategy

All production behavior is introduced with RED/GREEN tests.

### C1 tests

- immutable configuration, candidate, result, and artifact contracts;
- zero candidate presence and canonical deduplication;
- monotonic-up, monotonic-down, flat, high-cost, asymmetric-tail, and
  infeasible artificial markets with known rankings;
- downside CVaR, score, regret, bootstrap, and tie-break recomputation;
- isolated state cloning and no cross-candidate mutation;
- malformed rollout evidence and artifact tampering;
- deterministic digests across process and platform.

### C2 tests

- train-only and past-only anchor closure plus embargo enforcement;
- prefix causality under arbitrary at-or-after-cutoff mutations;
- robust train-only normalization and exact feature ordering;
- nearest-neighbor distance and anchor-index tie-breaks;
- insufficient-anchor failure;
- scenario-path anchoring and relative-path reconstruction;
- preservation of multivariate return, volatility, correlation, funding,
  liquidity, and tradability structure on controlled fixtures;
- rejection of discontinuous or invalid transformed datasets.

### C3 tests

- candidate selection persisted before realized replay;
- exact same environment/action/execution identity for all compared policies;
- independent fold resets and no continuous-return mislabeling;
- compatible upper-bound ordering and gap recomputation;
- explicit `not_comparable` behavior for incompatible bounds;
- paired inference and ranking/calibration reports;
- sealed-ledger behavior for any outer evaluation;
- no Phase C dependency from training, Serving, promotion, or release packages.

Every PR must pass focused statement and branch coverage for new modules, Ruff,
format, repository-wide MyPy, import architecture, dead-code checks, complete
Pytest and critical coverage ratchets, Windows/Ubuntu compatibility, PostgreSQL
catalog integration where applicable, and the complete training-image build.

## Non-goals

Phase C does not:

- train a BC, AWR, IQL, ranking, or PPO student;
- change the maintained `residual-ppo-15m` configuration;
- use the Perfect-Information or historical DP action as a causal label;
- implement neural forecasting, generative price models, multi-step MPC, or
  online scenario sampling;
- route, reconcile, or authorize exchange orders;
- change Serving bundles, release attestations, production gates, or the current
  `NO-GO` classification.

## Decision log

1. Chose C before A to establish that train-only causal state contains useful
   decision information before attempting policy distillation.
2. Chose one-step residual deviation followed by zero residual to isolate the
   marginal value of the current decision and avoid scenario-tree overfitting.
3. Chose the maintained residual action contract rather than reintroducing a
   direct-target experimental path.
4. Chose a 24-hour initial horizon and 64 uniform nearest-neighbor scenarios as
   a bounded, auditable first configuration.
5. Chose train-only, strictly earlier conditioned historical blocks before neural
   scenario models so evaluator correctness and prediction-model error remain
   separable.
6. Chose a state-action value artifact rather than extending the single-label
   teacher artifact, preserving semantic and trust boundaries.
7. Chose downside-only CVaR so uniformly positive advantage is not rewarded a
   second time through a negative loss penalty.
8. Chose strict Phase A evidence gates; weak or uncalibrated scenario values must
   not become training targets merely because the software pipeline passes CI.
