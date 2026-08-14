# Causal Alpha V3 research path design

## Status and objective

This design adds a **research-only** path for diagnosing and improving the Universal causal-alpha teacher without weakening the maintained U6 admission gates or changing the canonical reward. The immediate objective is to separate three failure boundaries that are currently conflated by long selection runs:

1. whether the fitted signal contains out-of-sample gross alpha;
2. whether signal-to-target compilation preserves that alpha after uncertainty, turnover, liquidity and execution costs;
3. whether a learned policy adds value beyond a deterministic causal teacher.

The current canonical U6 path remains fail-closed. No canonical example config is changed by this work.

## Non-goals

- Do not relax the `-5%` symbol-episode lower-tail floor or any BC/teacher admission gate.
- Do not add drawdown, turnover, baseline or execution-cost shaping to the scalar reward. Net log growth remains authoritative.
- Do not use teacher-admission holdouts, validation symbols or test symbols for model selection.
- Do not turn historical checkpoint diagnostics into resumable/promotion evidence.
- Do not replace the current production execution simulator or the hard `max_position_to_market_notional=0.02` risk rule.
- Do not make cross-sectional portfolio allocation part of the maintained single-instrument Universal contract in this change.

## Invariants

- `target_weight` remains the maintained economic action semantic.
- A decision may only use information available through its decision close; execution remains delayed according to the environment contract.
- Historical diagnostics must preserve checkpoint generator/grid provenance and may never be consumed by the selection-resume loader.
- Cost is counted exactly once in net equity/reward. A research target compiler may use cost as a decision hurdle, but never adds that hurdle to reward.
- Existing action modes and their digests/semantics are unchanged when the new research action mode is not selected.
- Existing `fit_causal_alpha_ridge` behavior is unchanged when new weighting/normalization options are omitted.

## Architecture

### 1. Historical V2 checkpoint diagnostics

Add a diagnostic-only reader for `causal_alpha_selection_checkpoint_metric_v2` rows. Unlike the resume loader, it does not require the *current* generator digest. It validates that all rows have one consistent historical `grid_digest` and `generator_code_digest`, reconstructs every `CausalAlphaCandidateEpisodeMetricsV2`, and verifies each metric digest through the existing dataclass contract.

The diagnostic report exposes two views:

- **prediction-identity view**: de-duplicates repeated signal evidence shared by controller candidates using `(symbol, episode_index, signal_24h.digest, signal_72h.digest)`;
- **paired candidate view**: compares candidates only on exact matched `(symbol, episode_index)` scopes, so partially completed candidates cannot gain an unfair aggregate advantage.

This report is explicitly non-promotable and never calls `load_causal_alpha_selection_checkpoint_v2`.

### 2. V3 overlap-aware fit primitives

Extend the low-level ridge fitter with optional `sample_weights` and optional mean-objective normalization. Defaults preserve current behavior.

A new research module builds weights using label-interval uniqueness:

- only labels fully realized before the requested knowledge cutoff participate;
- concurrency is calculated independently per symbol;
- each label receives the average inverse concurrency over its information interval;
- weights are normalized so every train symbol contributes equal total weight;
- each train symbol contributes equal total weight mass; absolute global scale is immaterial because the V3 weighted objective is normalized by total eligible weight.

V3 ridge uses weighted feature statistics and solves the weighted mean squared-error objective plus ridge regularization. This prevents the regularization strength from implicitly scaling with raw sample count and reduces pseudo-replication caused by heavily overlapping 24h/72h labels.

### 3. Multi-horizon forecast bundle

The V3 fit keeps independent 24h and 72h models. Predictions are converted to a common 24h economic unit before combination:

`mu_24eq = 0.5 * (prediction_24h + prediction_72h / 3)`

Weighted in-sample residual RMSE for each horizon is retained in fit evidence. Per-decision forecast uncertainty combines horizon residual uncertainty and cross-horizon disagreement. The result is a deterministic bundle containing raw horizon predictions, 24h-equivalent expected return, uncertainty, and a signal-to-uncertainty ratio.

This is a research heuristic, not a statistical confidence interval claim. Its exact formula is artifact-bound.

### 4. Uncertainty-aware target compiler

Add a discrete target optimizer rather than extending the existing threshold/tanh controller. For each decision it evaluates a predeclared target grid plus the current target, zero, and the current liquidity cap.

For candidate target `w`, previous target `w_prev`, and incremental change `d = w - w_prev`:

`objective = d * mu - z * abs(d) * sigma - abs(d) * (cost_multiplier * one_way_cost + edge_margin)`

HOLD has `d = 0` and objective `0`, so already-paid execution cost is not charged again and an explicit HOLD decision requires no ad-hoc reward shaping. Target changes occur only at a configured alpha rebalance interval, except that a falling causal liquidity cap may force immediate deleveraging and a sufficiently strong sign reversal may be evaluated early. `max_target_delta` remains available as an independent action-smoothing bound.

The compiler emits audit arrays for chosen objective, stay objective, expected return, uncertainty, liquidity cap, target, forced deleveraging and rebalance reason.

### 5. Teacher-anchored residual RL action mode

Add a new research action mode, `anchored_target_residual`, whose policy output is a bounded residual around a causal target-weight alpha provider:

`proposal = normalize_gross(alpha_anchor + residual_scale * policy_action)`

The alpha provider must use `AlphaSignalKind.TARGET_WEIGHT`. Zero policy action exactly reproduces the anchor. Existing `residual` and `target_weight` modes are unchanged.

For this mode the environment shadow target is the alpha anchor rather than the trend baseline, so paired research telemetry measures incremental RL value against the deterministic teacher. This mode is not enabled by any canonical config in this change.

### 6. DAgger learner-state collection primitive

Add a research collector that resets one declared episode, then for each learner-visited state:

1. records the current observation and decision index;
2. asks a teacher callback for the label **without stepping the environment with that label**;
3. obtains the learner action and steps the environment with the learner action;
4. records the teacher label against the learner-state observation.

The resulting immutable rollout evidence includes teacher identity, action/observation digests, decision indices, and episode bounds. A merge helper can append these learner-state samples to an existing episode-supervised dataset while assigning new contiguous episode IDs. This prevents teacher-forced state distribution from being silently reintroduced.

## Failure modes and required handling

- **Historical checkpoint identity mixing**: reject mixed grid/generator identities.
- **Duplicate signal evidence across candidates**: de-duplicate by signal digest and exact episode scope.
- **Partial candidate bias**: paired comparisons use intersection scopes only and report coverage.
- **Future-label leakage**: V3 weights and fits filter by `label_end_index < knowledge_cutoff` before any statistics are computed.
- **Cross-symbol imbalance**: equal total symbol weight after overlap correction.
- **Degenerate weights**: reject non-finite, negative, zero-total or shape-mismatched weights.
- **Cost double counting**: target compiler uses cost only in action selection; reward contract remains untouched.
- **Overconfident target**: uncertainty term can make HOLD/flat optimal.
- **Liquidity contraction**: immediate deterministic deleveraging is recorded separately from alpha rebalancing.
- **Residual action hides teacher baseline**: zero residual must reproduce anchor exactly before downstream hard-risk projection.
- **DAgger accidentally teacher-forced**: tests prove the environment receives learner actions while labels come from teacher actions.

## Test oracle

Correctness is judged through observable contracts, not only return values:

- diagnostic snapshot identities, unique prediction count and paired coverage;
- exact ridge coefficients for unweighted legacy behavior and expected weighted solutions for synthetic data;
- weight sums per symbol and overlap uniqueness ordering;
- forecast unit conversion and uncertainty monotonicity;
- target state transitions, HOLD choice, immediate liquidity deleveraging and rebalance cadence;
- environment submitted/shadow targets for anchored residual mode;
- DAgger environment action history versus stored teacher labels;
- artifact/config digests changing when V3 semantics change while legacy digests remain stable.

## Required test layers

- Unit: diagnostic reader/report, overlap weighting, weighted ridge, V3 forecast, target compiler, DAgger merge.
- Integration: environment decision planner/composer for anchored residual mode.
- Contract/architecture: canonical U6 example configs remain `target_weight`; new mode is not promoted/served implicitly.
- Static: Ruff, format, Mypy/import boundaries as required by repository CI.
- Full regression: repository pytest and exact-head GitHub Actions before the PR is considered ready.

## Acceptance criteria

1. Historical V2 checkpoint files from older generators can be read for diagnostics without being accepted for resume or promotion.
2. Shared signal evidence is counted once per unique prediction identity and candidate deltas are paired on exact common scopes.
3. Legacy unweighted ridge behavior remains unchanged; V3 can fit overlap-corrected, symbol-balanced, objective-normalized weighted ridge models.
4. V3 returns a deterministic 24h-equivalent forecast and uncertainty bundle with immutable evidence.
5. The target compiler selects HOLD/flat when conservative expected edge does not clear uncertainty and cost, respects causal liquidity caps, and records every forced/stateful transition.
6. `anchored_target_residual` has zero-action teacher identity and a bounded residual; existing action modes are unchanged.
7. DAgger samples learner-visited states while stepping only learner actions.
8. No canonical U6 config, reward weight, admission threshold or hard-risk limit is weakened or silently changed.
9. Targeted tests, static checks, full tests and exact-head CI are green on the final commit; any unavailable check is reported as unverified rather than inferred.

## Rollout

The new path is research-only. First use the historical diagnostic reader on preserved r3 evidence. Only if the V3 signal gate shows positive out-of-fold information should the deterministic V3 compiler be replayed through production execution. Teacher admission remains untouched and exactly-once. Anchored residual RL and DAgger are eligible only after deterministic teacher admission; they do not bypass the canonical gate.
