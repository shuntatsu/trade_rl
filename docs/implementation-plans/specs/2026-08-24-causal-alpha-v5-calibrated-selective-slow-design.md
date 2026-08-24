# Causal Alpha V5 Calibrated Selective Slow Anchor Design

## Status

Approved for implementation as a research-only successor to the observed Causal Alpha V4 r15 checkpoint.

This document does not reinterpret, mutate, or promote retained V4 evidence. V4 artifacts, schemas, thresholds, reward semantics, execution semantics, and pass/fail outcomes remain immutable research history. V5 uses new versioned contracts and remains `research_only=true` and `promotion_eligible=false` until its own gates pass.

## Objective

Build a causal slow-lane teacher that preserves V4's positive relative-ranking information while avoiding forced long/short decisions when slow absolute direction is insufficiently calibrated.

V5 must:

1. reuse V4 context, base 4h/24h/72h forecasts, uncertainty, execution, risk, and net-log reward contracts;
2. fit one shared descriptor-conditioned slow zero-point calibrator only on a causal train-prefix suffix;
3. combine calibrated slow return, independent slow direction evidence, uncertainty, and execution hurdle into a predeclared selective activation rule;
4. distinguish HOLD, position retention, risk reduction, entry, add, exit, and flip in immutable evidence;
5. preserve the V4 fast lane and its maximum absolute deviation of `0.05`;
6. evaluate fixed Signal, Selection, and untouched Admission contracts before BC or RL.

## Evidence boundary

The operator-provided r15 checkpoint reports full completion of 72 fast, 72 slow, and 72 liveness scopes without OOM. Fast 4h direction, Rank IC, and spread passed. Slow fused Rank IC and spread passed, while slow direction-accuracy excess had a positive mean but a negative lower confidence bound. Selection, Admission, BC, and RL did not run.

This supports a calibration/abstention hypothesis. It does not establish profitability, learning success, or production readiness. Because r15 influenced this design, reusing those same Signal scopes is exploratory. A confirmatory V5 claim requires a new sealed interval or another predeclared untouched split.

## Non-goals

V5 does not:

- lower or remove any V4 gate;
- reinterpret r15 as a pass;
- exclude APT, LTC, failed episodes, or any scope after observing outcomes;
- tune a threshold, ridge, fusion-weight, or feature grid on Signal, Selection, or Admission;
- introduce symbol-ID lookup, symbol-specific intercepts, or symbol-specific calibrators;
- change `reward_t = scale * log(net_equity_after / net_equity_before)`;
- change hard-risk, margin, liquidation, execution-cost, latency, partial-fill, participation, or target-weight accounting;
- change the V4 4h fast forecast, gate, or maximum `±0.05` fast deviation;
- add 1-minute features to the slow forecast;
- start BC or RL before V5 Signal, Selection, and Admission pass;
- authorize live routing, Production GO, or profitability claims.

## Architecture

```text
V4 context and base fitting
        |
        v
V4 forecast: 4h / 24h / 72h returns, direction scores, uncertainty
        |
        +--------------------------+
        |                          |
        v                          v
unchanged fast 4h lane      V5 slow calibrator
                                   |
                                   v
                         calibrated slow return
                                   |
                         independent slow direction
                                   |
                        confidence + cost hurdle
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
       active slow decision                      abstain / HOLD
              |                                         |
              +--------------------+--------------------+
                                   v
                         V5 selective target path
                                   |
                                   v
                    unchanged execution and risk replay
                                   |
                                   v
                 Signal -> Selection -> untouched Admission
```

V4 remains directly executable and testable. V5 imports V4 contracts; V4 must not import V5.

## Information-set invariants

1. Every V5 decision-time input is reconstructible from public information available to the student and serving path.
2. A label is fit-eligible only when `label_end < knowledge_cutoff`.
3. Base fitting, calibration, Signal, Selection, and Admission are strictly ordered.
4. Signal, Selection, and Admission labels or metrics cannot affect calibrator coefficients, scale, threshold, support, features, ridge strength, or fusion formula.
5. Feature rows at or after a fit cutoff are excluded from the fitted design matrix even when labels are masked.
6. Calibrator features contain no string symbol identity, one-hot identity, symbol-specific intercept, or symbol dispatch.
7. Missing calibration evidence fails closed; it is not replaced with zero, a global default, or an in-sample estimate.

## Slow forecast and direction

```text
slow_return_raw_t
  = 0.5 * (prediction_24h_t + prediction_72h_t / 3)

slow_direction_raw_t
  = 0.5 * (direction_score_24h_t + direction_score_72h_t)

slow_realized_t
  = 0.5 * (label_24h_t + label_72h_t / 3)
```

The existing V4 slow Signal contract remains unchanged and continues to evaluate the sign of `slow_return_raw_t`. V5 uses `slow_direction_raw_t` as independent confirmation evidence.

## Calibration split

The authored train interval is split chronologically:

```text
base-fit prefix       = first 80% of eligible causal train decisions
calibration suffix    = final 20% of eligible causal train decisions
```

The split is determined before outcome metrics are evaluated. The 72h label end controls purging:

```text
base label:        label_end_72h < calibration_start
calibration row:   decision >= calibration_start
calibration label: label_end_72h < train_stop
```

The calibration suffix is split into four non-empty chronological blocks `B1..B4`:

- fit `B1`, predict `B2`;
- fit `B1+B2`, predict `B3`;
- fit `B1+B2+B3`, predict `B4`.

Only out-of-fold residuals from `B2..B4` estimate calibration residual scale and direction-score scale. The final Signal calibrator is fit on `B1..B4`.

Minimum support before publication:

- pooled non-overlapping calibration rows: `>= 256`;
- every train symbol: `>= 16` rows;
- forward evaluation blocks: exactly `3`;
- every evaluation block contains at least two symbols;
- descriptor names and order equal the maintained public descriptor contract.

## Calibrator

The shared feature vector is exactly:

```text
[
  slow_return_raw,
  slow_direction_raw,
  log(slow_uncertainty_raw + 1e-12),
  public instrument descriptors in maintained order,
]
```

The maintained ridge primitive supplies the intercept. No target-local technical features or global context columns are added in the first V5 hypothesis.

Target:

```text
calibration_residual = slow_realized - slow_return_raw
```

Estimator:

```text
ridge_strength = 1.0
normalize_objective = true
working_memory_rows = 4096
```

The calibration artifact binds feature names, boundaries, final and forward model digests, overlap-weight digests, forward residual digests, per-symbol/per-block support, residual RMSE, direction-score RMSE, and source V4 identities.

Calibrated values:

```text
slow_return_calibrated
  = slow_return_raw + residual_prediction

slow_uncertainty_calibrated
  = sqrt(slow_uncertainty_raw ** 2 + calibration_residual_rmse ** 2)
```

Calibrated uncertainty may not be smaller than V4 slow uncertainty.

## Selective activation

```text
direction_scale
  = max(direction_score_rmse, 1e-12)

return_confidence
  = abs(slow_return_calibrated)
    / max(slow_uncertainty_calibrated, 1e-12)

direction_confidence
  = abs(slow_direction_raw) / direction_scale

selective_confidence
  = min(return_confidence, direction_confidence)

execution_hurdle
  = 1.5 * one_way_cost_rate + 0.001
```

The first V5 threshold is fixed:

```text
minimum_selective_confidence = 1.0
```

Exposure increase or flip is active only when all hold:

```text
sign(slow_return_calibrated) == sign(slow_direction_raw) != 0
selective_confidence >= 1.0
abs(slow_return_calibrated)
  - slow_uncertainty_calibrated
  - execution_hurdle > 0
```

Confidence equality is active. Hurdle equality abstains. Risk reduction, exit, and liquidity deleveraging remain allowed while inactive.

## Target compilation

When inactive, candidates are limited to current weight, zero, same-sign smaller exposure, and mandatory risk/liquidity projections. When active, the unchanged V4 slow magnitudes are also available:

```text
(0.0, 0.025, 0.05, 0.10, 0.25)
```

The existing V4 objective, uncertainty charge, cost multiplier, edge margin, cadence, maximum final delta, and deterministic tie-breaking are reused. The unchanged V4 fast impulse is applied after slow-anchor choice, with maximum absolute deviation `0.05`.

Every decision receives one reason:

```text
hold_flat
hold_position
entry
add
reduce
exit
flip
unactionable_hold
confidence_abstain
direction_disagreement_hold
edge_below_hurdle_hold
cadence_hold
liquidity_deleverage
risk_projection
```

Operational overrides take precedence over transition-derived reasons.

## Signal contract

The V4 fast 4h Signal evidence and gate are reused unchanged.

The V5 selective slow lane requires:

- raw slow scopes: exactly `72`;
- independent episode clusters: exactly `8`;
- all nine authored symbols represented;
- all eight episodes represented;
- unconditional Rank IC 95% lower CI `>= 0`;
- unconditional top-bottom spread 95% lower CI `>= 0`;
- unconditional direction-accuracy excess mean `>= 0`;
- active direction-accuracy excess 95% lower CI `>= 0`;
- overall active coverage `>= 0.25`;
- every symbol × episode scope active support `>= max(3, ceil(0.20 * raw_direction_support))`;
- every inactive direction-support row accounted for by one abstention reason.

Bootstrap settings remain `10000` resamples, seed `20260823`, block size `2`.

## Selection contract

Selection uses the unchanged simulator, risk, execution, and cost model.

Per symbol/episode evidence includes gross/net wealth and return, drawdown, turnover/day, cost, submitted/executed changes, sign flips, action-reason counts, active coverage, flat-time fraction, time-weighted absolute exposure, and completed holding duration.

Aggregate metrics include:

```text
symbol_balanced_net_wealth
  = exp(mean(log(symbol_net_wealth)))

median_symbol_net_wealth
positive_net_scope_fraction
worst_symbol_episode_net_return
scope_net_return_cvar_10
turnover_p50
turnover_p95
total_execution_cost
net_to_gross_retention
```

Pass conditions:

```text
symbol_balanced_net_wealth > 1.0
median_symbol_net_wealth >= 1.0
positive_net_scope_fraction >= 0.5
all authored symbols represented
at least one meaningful executed scope
no hard-risk violation
no unexplained execution rejection
```

## Admission contract

Admission opens only after V5 Signal and Selection pass. It keeps the existing untouched holdout economic semantics and requires `fit_knowledge_cutoff == holdout_start`. No Admission result feeds back into calibration or thresholds.

## Versioned schemas

```text
causal_alpha_v5_calibration_config_v1
causal_alpha_v5_calibration_fit_v1
causal_alpha_v5_selective_forecast_v1
causal_alpha_v5_target_v1
causal_alpha_v5_signal_scope_v1
causal_alpha_v5_signal_lane_evidence_v1
causal_alpha_v5_signal_evidence_v1
causal_alpha_v5_replay_metric_v1
causal_alpha_v5_selection_evidence_v1
causal_alpha_v5_admission_evidence_v1
causal_alpha_v5_research_config_v1
causal_alpha_v5_research_package_v1
```

Every artifact is content-addressed and binds upstream V4 identities. No V4 schema or payload changes.

## Failure modes

V5 fails closed for boundary leakage, insufficient support, missing descriptors, symbol identity in features, mismatched V4 identities, non-finite arrays, reduced calibrated uncertainty, malformed reasons, unaccounted abstentions, low coverage, missing symbol/episode support, stage bypass, or BC/RL invocation before Admission. A failed contract publishes no final package.

## Test Oracle

Tests observe exact artifact bytes/digests, publication and non-publication, label-end boundaries, feature names, forward-block identities, support, calibrated prediction and uncertainty, threshold equality, inactive risk reduction, action reasons, active/inactive accounting, Signal coverage, Selection wealth/cost/risk summaries, stage order, and V4 compatibility.

## Quality gate

Completion requires targeted and full tests, Ruff, format, Mypy, Import Linter, Vulture, branch and critical coverage, Ubuntu/Windows compatibility, training capability audit, training image and non-root probe, exact-head CI, falsification review, and a final diff without threshold relaxation, symbol exclusion, test skip, assertion weakening, debug code, or temporary workflow.

## One-minute data decision

V5 does not add 1-minute inputs. A separate execution-only hypothesis may consider them only after V5 Signal passes and Selection shows positive gross wealth but non-positive net wealth with execution costs consuming at least half of positive gross edge, or bar-path sensitivity changes at least 10% of scope decisions.