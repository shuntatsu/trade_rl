# Causal Alpha V3 Market/Residual Decomposition Design

## Objective

Replace the rejected Causal Alpha V3 pooled ridge forecast with a deterministic,
train-only, two-stage linear forecast that separates common market direction from
symbol-specific relative return. The change exists to repair the observed Signal
Gate direction failure while preserving the canonical Signal V2 evaluator and all
downstream economic, Teacher, BC, RL, reward, risk, execution, and admission
contracts.

The required end state remains a complete real PostgreSQL-backed Binance
multi-timeframe training run. This design is only the next predictor hypothesis.
It is not a success claim and does not authorize downstream learning unless the
new immutable generation passes every maintained gate in order.

## Current Evidence and Root Cause

The fresh `sidecar-fresh-20260822-r1` generation used the immutable Universal
runtime and completed all 144 Signal scopes without OOM or artifact corruption.
It was correctly rejected before selection because neither pooled ridge fit passed
all three lower-confidence-bound requirements.

The strongest existing pooled fit, ridge `1.0`, had positive mean rank IC, spread,
and direction excess, but its lower confidence bounds crossed zero. The 72-hour
horizon was materially stronger than the 24-hour horizon, while late 24-hour rank
quality decayed.

Thirteen bounded train-only counterfactuals were then evaluated with the maintained
Signal Gate thresholds and independent-episode bootstrap. They covered:

- 72-hour-only forecasts;
- 120-day rolling fits;
- ridge strengths from `0.1` through `100.0`;
- intercept removal and prior-episode calibration;
- slow-feature-only fits;
- per-symbol return fits;
- pooled/per-symbol blends;
- per-symbol sign targets; and
- eight versus twelve Signal contracts.

The strongest per-symbol alternatives made rank IC and top/bottom spread lower
bounds positive, including the per-symbol sign target with twelve contracts, but
direction-accuracy-excess still had a negative lower bound. Increasing evidence
also exposed a sharply negative late episode rather than stabilizing direction.

The evidence therefore supports two distinct forecast components:

1. symbol-specific structure is strong enough to rank returns and separate the
   top and bottom buckets; and
2. absolute market direction remains unstable and must be modeled as a common
   component rather than inferred independently by each symbol fit.

Changing Gate thresholds, bootstrap semantics, evidence count, reward, or controller
parameters would hide this failure rather than repair it and is out of scope.

## Options Considered

### Option A — deterministic market plus symbol-residual ridge

Fit one common market model per horizon and one residual model per train symbol per
horizon. The final prediction is their exact sum, with no learned or hand-authored
blend coefficient.

Chosen. It directly matches the observed decomposition, reuses the audited ridge
primitive, remains deterministic and dependency-free, and permits each component
to be persisted and falsified independently.

### Option B — jointly regularized multi-task ridge

Fit shared coefficients and symbol deviations in one block linear system with a
second shrinkage parameter.

Deferred. It may provide better partial pooling, but it adds a new solver,
additional regularization semantics, and a harder artifact/test oracle before the
simpler decomposition has been falsified.

### Option C — nonlinear temporal direction head

Train a small neural or boosted direction model for the common market component and
retain a linear residual head.

Deferred. It increases dependency, determinism, overfitting, calibration, and
audit complexity. The current evidence does not justify introducing a nonlinear
learner before testing the explicit common-factor hypothesis.

### Option D — continue bounded tuning of the current pooled ridge

Rejected. Ridge, horizon, rolling-window, intercept, feature-subset, target, and
evidence-count variants have already failed to make all three Signal lower bounds
positive.

## Immutable Boundaries

The following remain unchanged:

- the DB-backed immutable Universal runtime, dataset digest, time interval,
  partitions, feature schema, and 15m/1h/4h/1d inputs;
- nine train, three validation, and three test symbols;
- chronological expanding knowledge cutoffs and label realization rule;
- the canonical Signal V2 scope cohort and one-interval/one-independent-episode
  semantics;
- rank IC, top/bottom realized spread, direction accuracy, moving-block bootstrap,
  coverage, threshold, and pass/fail numerics;
- the fixed 24h/72h 24-hour-equivalent forecast and realized-return fusion;
- target compilation, execution-cost accounting, portfolio risk, and selection;
- untouched Teacher-admission holdout, validation, and test boundaries;
- BC, critic warm start, PPO, Lagrangian PPO, and Discounted Lagrangian PPO;
- the pure-growth reward
  `reward_t = 100 * log(net_equity_after / net_equity_before)`; and
- the rule that failed Signal or downstream admission performs zero PPO updates.

Historical V3 artifacts retain their original meaning. They are not migrated,
refitted, overwritten, or interpreted as evidence for the new model.

## Data and Leakage Contract

Only the nine training-symbol sample blocks may enter either head. For a knowledge
cutoff `K`, every training label used by either head must have a label-end index
strictly less than `K`. Scaling statistics and constant masks are fitted from the
same eligible prefix only.

For each horizon `h` in `{24h, 72h}` and decision index `t`, define the common
market label as the equal-weight mean over the fixed nine-symbol train universe:

```text
market_label_h(t) = mean_s(symbol_label_s,h(t))
```

The row is eligible only when all nine symbols have the matching decision index,
finite features required by the aggregate input contract, a finite label, and a
label end strictly before `K`. Requiring the complete authored universe avoids a
changing-universe label and makes the equal-weight meaning exact. Missing or
misaligned labels fail the row closed rather than being filled.

For each train symbol `s`, define:

```text
residual_label_s,h(t) = symbol_label_s,h(t) - market_label_h(t)
```

No holdout, validation, or test label may enter either definition. Holdout market
features may use contemporaneous train-symbol features because they are available
at the decision and contain no future outcome, but holdout labels are used only by
the existing canonical evaluator after prediction.

### Verified alignment feasibility

A read-only reconstruction from the immutable runtime content digest
`6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
confirmed that every train symbol has exactly 51,840 sample decisions spanning
indices 5,664 through 57,503. The complete decision-index, 24-hour label-end, and
72-hour label-end arrays are identical across all nine symbols. Each symbol has
51,743 realized 24-hour rows and 51,551 realized 72-hour rows.

The retained fresh Signal sidecars independently confirm that every one of the
eight Signal intervals contains the same 2,880 prediction decisions for all nine
symbols with zero non-actionable rows. Within each interval, all symbols also have
identical realized-row index sets: 2,784 rows for 24 hours and 2,592 rows for 72
hours. The strict fixed-universe construction is therefore supported by the actual
runtime rather than assumed from schema shape.

## Market Input Construction

At each decision, align all nine training-symbol feature rows by decision index.
For every ordered source feature, construct two deterministic market channels:

```text
market_mean::<feature>
market_available_fraction::<feature>
```

`market_mean` is the equal-weight mean of finite available values. It is zero when
no symbol provides the feature, and its availability mask is false in that case.
`market_available_fraction` is the number of available train-symbol values divided
by nine and is always finite and available.

All source features remain in their original schema order. This includes static
instrument/execution descriptors: their cross-sectional means are causal and are
normally constant within a fit, so the existing constant-column handling removes
unsupported variation without a new feature-role heuristic.

The aggregate feature schema binds:

- the ordered source feature names;
- the ordered derived feature names;
- the fixed train-symbol tuple;
- aggregation formula and missingness rule;
- source sample digests and feature-schema digest; and
- the knowledge cutoff.

The same constructor is used for fitting, Signal prediction, target generation,
selection, and Teacher admission. Reconstructing aggregate inputs independently in
callers is prohibited.

## Estimator Semantics

### Market heads

Fit one overlap-aware ridge model for the 24-hour market label and one for the
72-hour market label. A market row has one temporal overlap weight; it is not
duplicated once per symbol. The existing uniqueness-weight formula and strict
knowledge cutoff are reused.

### Residual heads

Fit one overlap-aware ridge model for each `(symbol, horizon)` pair from that
symbol's target-local feature vector and residual label. Each residual head uses
the existing deterministic scaler, unavailable-feature handling, uniqueness
weights, and finite/rank/shape checks.

Per-symbol residual heads are research-teacher components only. Their predictions
produce immutable teacher action paths for the nine training symbols. A symbol ID
is not added to policy observations, and the deployed Universal policy remains the
generic instrument-conditioned policy evaluated on validation and test symbols.

### Authored first hypothesis

The first counterfactual contains exactly one new fit hypothesis:

```text
model_kind = market_residual_ridge_v1
market_ridge_strength = 1.0
residual_ridge_strength = 0.1
```

These values correspond to the strongest observed pooled-direction and
per-symbol-ranking regimes. They are fixed before the new experiment. A failed
counterfactual does not authorize a silent grid expansion; another model family or
parameter family requires a new documented hypothesis and immutable generation.

## Forecast Composition and Uncertainty

For symbol `s`, horizon `h`, and decision `t`:

```text
prediction_s,h(t) = market_prediction_h(t) + residual_prediction_s,h(t)
```

There is no mixing weight, calibration offset, sign override, or post-hoc
direction threshold.

For each horizon, compute the composite weighted residual RMSE directly from the
final summed in-sample forecast and original symbol return labels over the exact
eligible prefix. Do not estimate composite uncertainty by assuming independent
market and residual errors. The existing `causal_alpha_v3_forecast` primitive then
receives the two final prediction arrays and the two directly measured composite
RMSE values. Its 24h/72h fusion, horizon disagreement term, uncertainty, and
signal-to-uncertainty formulas remain unchanged.

Market-head RMSE, residual-head RMSE, and composite RMSE are all persisted, but
only the unchanged composite forecast bundle is consumed by target compilation.

## Fit and Configuration Identity

Keep `CausalAlphaV3FitConfig` and the current pooled fit schema unchanged for
historical and compatibility paths. Introduce a distinct immutable configuration,
conceptually:

```text
CausalAlphaV3MarketResidualFitConfig
    model_kind = market_residual_ridge_v1
    market_ridge_strength
    residual_ridge_strength
    aggregate_feature_schema
```

The authored research-config schema is bumped for the new generation and uses a
discriminated fit-config union. Old V2 research configs continue to parse under
their existing strict schema; they are not accepted as the identity of a new
market/residual run.

The composite fit digest binds at least:

- configuration and aggregate-feature-schema digests;
- train-symbol tuple, sample-scope digest, and knowledge cutoff;
- 24h and 72h market-model digests;
- ordered `(symbol, model digest)` tuples for both residual horizons;
- market and per-symbol overlap-weight digests;
- market, residual, and composite RMSE values; and
- all eligible-index/cutoff identity already bound by the ridge models.

The fit cache remains keyed by `(fit_config_digest, knowledge_cutoff)` and returns
the complete immutable two-head bundle. Target-only candidates share that bundle.
Any missing symbol head, order drift, schema drift, or cache identity mismatch
fails closed.

## Prediction API and Dependency Direction

Introduce one workflow-owned prediction-input constructor that returns aligned
target-local inputs and aggregate market inputs for the requested symbol and
decision indices. Both the legacy pooled fit and new composite fit are exposed
through a typed forecast port. Runtime type branching must not leak into the
Signal Gate or target compiler.

Required direction:

```text
immutable samples + decisions
          |
          v
typed prediction-input constructor
          |
          v
pooled fit or market/residual fit
          |
          v
unchanged CausalAlphaV3Forecast
       /                 \
      v                   v
Signal metric      target compilation
```

The canonical metric builder continues to depend only on the final forecast,
aligned realized labels, and canonical cohort. It must not understand market or
residual model internals.

## Diagnostic Sidecar V2

The existing pooled diagnostic schema V1 remains valid and unchanged. A new
`causal_alpha_v3_signal_diagnostic_scope_v2` is written only for the decomposed
fit. The canonical Signal metric remains V2 and remains the only Gate input.

For each scope, the new sidecar persists:

- final 24h/72h predictions and unchanged fused forecast fields;
- market and target-symbol residual prediction components per decision;
- exact equality evidence that each final prediction is the component sum;
- market-model and target-symbol residual-model summaries for both horizons;
- aggregate-market and target-local feature availability summaries;
- market, residual, and composite weighted RMSE;
- aggregate feature schema, component model, fit, forecast, metric, run, contract,
  candidate, dataset, and partition identities;
- realized 24h, 72h, fused, and canonical cohort rows under the existing timing
  rules; and
- `research_only=true` and `promotion_eligible=false`.

The scope sidecar need not duplicate the other eight symbols' residual coefficient
arrays. The shared composite fit digest binds the full ordered model bundle, while
the scope sidecar persists the market and target-symbol state needed to reproduce
that scope's final forecast. This matches the existing boundary where a sidecar
does not independently reconstruct every canonical fit-digest input.

Strict validation reconstructs final horizon predictions from the persisted
components, reconstructs the unchanged forecast payload using composite RMSE, and
requires exact `forecast_digest` equality. A self-consistent outer artifact with a
forged component, forged sum, wrong symbol head, or stale forecast digest fails
closed.

Metric/sidecar paired write, partial-resume, corruption, wrong-path, wrong-run,
wrong-contract, and numerical-backend semantics remain the same as the existing
paired producer.

## Counterfactual Gate Before Production Implementation

Before changing tracked production code, run a disposable, train-only
counterfactual against the same immutable runtime and chronological Signal scopes.
The experiment must:

1. use exactly the authored two-stage fit hypothesis above;
2. use the existing eight Signal contracts and nine train symbols;
3. reuse the exact canonical cohort, metric, moving-block bootstrap, coverage,
   thresholds, and random seed;
4. read no Teacher holdout, validation, or test result for tuning;
5. persist the experiment script, source HEAD, runtime/manifest/config digests,
   per-scope metrics, component diagnostics, Gate evidence, logs, and completion
   status under a new retained generation directory; and
6. fail closed on missing scopes, non-finite values, identity drift, or incomplete
   component evidence.

Production implementation is justified only if this single predeclared
counterfactual passes all maintained Signal requirements, including non-negative
lower confidence bounds for rank IC, top/bottom spread, and direction-accuracy
excess. Positive means, individual passing metrics, or an almost-zero negative
bound are still rejection.

If it fails, preserve the evidence and return to architecture diagnosis. Do not
adjust Gate thresholds, add contracts, sweep decomposed ridge strengths, inspect
holdouts, or continue to selection.

## New Immutable Production Generation

After a passing counterfactual and verified TDD implementation:

1. build a Docker image from the exact clean source HEAD;
2. bind the image digest, source identity, authored config digest, runtime content
   digest, and code/manifests into a new run manifest;
3. start from an empty generation directory rather than resuming a pooled-fit run;
4. persist a metric plus diagnostic V2 sidecar for every expected Signal scope;
5. require the canonical Signal Gate before candidate freeze;
6. run economic selection only for Signal-admitted fit configs;
7. evaluate the selected teacher exactly once on the untouched causal holdout;
8. require BC reconstruction/economic admission before critic warm start;
9. run the maintained PPO, Lagrangian PPO, and Discounted Lagrangian PPO sequence;
10. complete validation/test, risk, robustness, reproducibility, and artifact audits;
    and
11. report learned-policy contribution separately from any baseline fallback.

Any failure preserves the immutable generation and performs zero unauthorized
downstream updates.

## Error Handling

Fail closed on:

- train-symbol order or universe drift;
- missing, duplicate, or misaligned cross-symbol decision rows;
- a market label built from fewer than all nine authored train symbols;
- a label end at or after the knowledge cutoff;
- scaling, aggregate-feature, or weight computation crossing the cutoff;
- missing symbol residual heads or a residual head used for the wrong symbol;
- non-finite component, summed prediction, RMSE, coefficient, or scaler state;
- component sums that do not exactly reproduce persisted final predictions;
- old/new config, fit, sidecar, checkpoint, or generation schema confusion;
- rebuilding or overwriting a corrupt paired artifact during resume;
- diagnostic objects reaching the Signal Gate;
- any holdout/validation/test evidence entering model or parameter selection; and
- any attempt to continue selection, Teacher admission, BC, or RL after a failed
  prerequisite gate.

## Test-First Strategy

### Unit tests

Prove:

- cross-symbol rows align by decision index and fixed train-symbol order;
- market labels are exact equal-weight means and residuals sum back to original
  symbol labels;
- incomplete universe labels and future-crossing labels are excluded/fail closed;
- aggregate means and availability fractions follow the authored missingness rule;
- market and residual ridge fits are deterministic in the same numerical runtime;
- final prediction equals market plus residual component exactly;
- composite RMSE is measured from summed forecasts rather than an independence
  approximation;
- fit/config/cache digests change for every relevant semantic identity;
- a residual head cannot be applied to another symbol; and
- legacy pooled fit/config behavior remains unchanged.

### Signal and sidecar tests

Prove:

- the canonical Signal metric/evaluator receives only the final unchanged forecast
  interface;
- cohort, realized fusion, bootstrap, thresholds, and sample-count semantics do not
  change;
- diagnostic V2 component rows reconstruct final predictions and forecast digest;
- forged component sums, RMSE, model identity, or outer digest fail closed;
- paired normal/resume/one-member/corrupt states retain existing semantics;
- diagnostic evidence is research-only and cannot affect Gate pass/fail; and
- no raw feature matrix, holdout, validation, or test evidence is serialized.

### Workflow tests

Prove:

- only train samples reach fitting and aggregate construction;
- all candidates sharing the fit semantic digest share one fit per cutoff;
- Signal rejection freezes zero candidates and performs zero economic/RL work;
- selection starts only after a passing complete Signal set;
- Teacher holdout remains one latest complete episode per train symbol and is
  evaluated exactly once;
- failed Teacher or BC admission performs zero PPO updates; and
- a new generation cannot resume or reuse pooled-run checkpoints under decomposed
  identities.

### Verification layers

Run targeted tests first, followed by format, Ruff, Mypy, import architecture,
dead-code checks, full pytest with coverage, compatibility/package/build checks,
the applicable PostgreSQL workflow, exact-image counterfactual reproduction, and
the canonical real-data generation. Completion claims require evidence from the
exact final HEAD and image digest.

## Acceptance Criteria

1. The predeclared train-only counterfactual passes every unchanged Signal Gate
   requirement on the immutable real-data runtime.
2. Market and residual labels, weights, fits, predictions, and identities are
   causal, deterministic in the controlled runtime, and cutoff-bound.
3. Final predictions are exact component sums with directly measured composite
   uncertainty inputs.
4. Canonical Signal V2 cohort, metrics, bootstrap, thresholds, and Gate authority
   are unchanged.
5. Diagnostic V2 evidence explains and reconstructs each component without
   entering promotion decisions.
6. Historical pooled artifacts/configs remain valid and are not migrated or
   reinterpreted.
7. The new production generation is bound to exact source, image, runtime, config,
   data, partition, fit, forecast, and artifact identities.
8. Selection, untouched Teacher admission, BC, and all learner families run only
   after their prerequisite gates pass.
9. Reward, costs, risk, execution, and holdout/validation/test boundaries remain
   unchanged and are verified on final HEAD.
10. The final report distinguishes predictor, teacher, BC, learned-policy, and
    baseline outcomes and does not infer success from an intermediate positive
    metric.

## Non-goals

- Relaxing or redefining any Signal, selection, Teacher, BC, risk, or promotion
  gate.
- Adding return, drawdown, turnover, cost, or baseline penalties to scalar reward.
- Selecting parameters from Teacher holdout, validation symbols, or test symbols.
- Adding a minimum holding period or direction override.
- Expanding the first market/residual ridge grid after seeing its result.
- Adding symbol identity to deployed policy observations.
- Replacing the DB-backed multi-timeframe runtime with a simplified dataset.
- Treating a passing Signal model, profitable teacher, or completed learner run as
  final research success without downstream audits.

## Review Decision

Approve this specification only if the intended next action is the single
train-only market/residual counterfactual described above, followed by production
implementation and the complete canonical workflow only when every prerequisite
gate passes unchanged.
