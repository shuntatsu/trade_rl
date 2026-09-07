# Causal Alpha V11 Policy Research Design

## Understanding summary

- Build a new V11 policy-research lane around the profitable-but-fragile V9 nonlinear signal.
- Freeze V10/r21 execution lifecycle, reward, costs, gates, symbols, 15-minute simulator, and 4-hour V11 prediction horizon.
- First reproduce the exact V9 action stream with live step metadata and decompose entry, neutral hold, and exit economics without changing behavior.
- Test loss containment, after-cost entry, pooled long/short calibration, and only then executable position sizing as separate hypotheses.
- Treat r21 and the existing eight economic folds as development evidence because their results have already influenced V11.
- Continue to V12 horizon research only if no V11 policy candidate clears the unchanged development Selection gates.
- Continue to V13 one-minute execution fidelity only after a 15-minute strategy is accepted and genuine one-minute data is available.

The user accepted this staged design by requesting that every item in the supplied research tree be executed and consolidated into one report.

## Assumptions and non-functional requirements

- PostgreSQL-backed Binance data and the frozen V4 context remain authoritative.
- V11 development uses the existing nine train symbols and eight economic scopes. Validation/test symbols and Admission remain untouched until one policy is locked.
- Every replay is restart-safe, atomically persisted, digest-bound, and independently parseable.
- Determinism is required for fit, target, trace, diagnostics, gates, and resume identity.
- Memory usage must remain within the existing Docker Desktop limit; controls and shared fits are computed once and reused.
- No symbol identity, symbol-specific coefficient, symbol exclusion, relaxed gate, reward change, or unreported runtime-config change is allowed.
- A stopped or rejected stage preserves durable evidence and does not silently fall through to Admission, BC, or RL.
- Maintenance ownership follows the existing `trade_rl.learning` domain / `trade_rl.workflows` orchestration split.

## Evidence and constraints

r21 completed Signal 72/72 and Selection 216/216. V9 had balanced net wealth `1.013175`, median symbol wealth `1.004164`, minimum symbol wealth `0.960973`, and positive scope fraction `0.375`. Hierarchical economics were unchanged from r20 after lifecycle hardening, so hierarchical policy tuning is frozen.

r21 V9 replay leaves contain authoritative realized exposure and lifecycle evidence, but their model trace metadata is unavailable because V9 was evaluated through a declared action matrix rather than a model exposing `last_step_trace_metadata`. D1 therefore cannot infer neutral observations from the stored r21 leaf alone.

The runtime contains only 15m, 1h, 4h, and 1d market data through 2026-07-05. No one-minute data exists. The current execution contract has `entry_threshold=0.1`, `no_trade_band=0.05`, and V9 `target_magnitude=0.1`.

## Chosen architecture

### 1. V11 exact-control trace policy

Build an immutable V11 target artifact from the existing V9 fit and forecast. Wrap its precomputed target path in a sequential policy whose `predict()` returns exactly the same action at every step and whose `last_step_trace_metadata` exposes:

- fast mean, standard deviation, qualified direction, and raw edge;
- after-cost entry objective;
- active risk/liquidity caps;
- policy reason and position origin;
- reduce-only intent for exits.

The V11 exact control is valid only if every replay action, gross/net return, cost, turnover, and final wealth matches the corresponding r21 V9 replay. Any mismatch is a hard failure, not a candidate result.

### 2. D1 diagnostics

From exact-control traces, identify realized entries and exits from lifecycle transitions. At each owned position, inspect only actionable four-hour cadence observations and split economics at the first neutral/unqualified observation after entry.

Persist pooled, long, short, per-symbol, and per-scope summaries for:

- entry-to-first-neutral gross/net log return and cost;
- first-neutral-to-exit gross/net log return and cost;
- trades with no neutral observation before exit;
- fixed four-hour entry edge `direction * labels_4h - 2 * one_way_cost`;
- mean/median/positive fraction/CVaR10 entry edge, MAE, MFE, net per exposure-hour, and net per turnover.

Entry count is coverage only. The unchanged Selection gates remain the success boundary.

### 3. Independent V11 candidates

All candidates reuse the exact V9 final wave fit, features, 4h cadence, target magnitude, execution contract, and reward unless explicitly stated.

`v9_control`
: Exact V9 action path with complete metadata. It is the behavior-neutral control.

`neutral_expiry_2`
: While a native position is owned, two consecutive actionable cadence observations with qualified direction zero request flat. Same-direction observations reset the neutral count. Opposite-signal exit remains the original two-confirmation rule. No direct flip is allowed.

`after_cost_entry`
: Entry qualification requires `abs(mean) - std - edge_margin - 2 * one_way_cost > 0`. Exit and fixed target magnitude remain exact V9 behavior. Cost filtering applies only when flat and does not create a new exit rule.

`sign_calibrated_entry`
: Fit two pooled, symbol-free ridge calibrations for long and short. For each outer cutoff, fit a source V9 model at the start of a fixed one-week calibration interval, predict that subsequent week out of sample, and require every calibration label end to precede the outer cutoff. Refit the normal V9 model at the outer cutoff for candidate forecasts. Regress signed 4h label on `[1, raw_edge]` independently by sign with fixed ridge strength `1.0`; entry requires calibrated edge minus round-trip cost to be positive. Exit and size remain exact V9 behavior.

`calibrated_edge_sizing`
: Uses the accepted sign-calibrated entry edge and the formula from the supplied design. Before replay, a feasibility gate checks the generated non-zero targets against the unchanged `entry_threshold=0.1` and `no_trade_band=0.05`. Because the formula is strictly below 0.1 for finite positive denominators, the current contract is expected to mark S1 structurally non-executable. V11 must persist that result and must not lower execution thresholds silently. A future sizing replay requires an explicitly versioned execution-contract experiment.

### 4. Study arms, replay, and gates

Each V11 output root has one immutable `study_arm` and exactly three candidates: V8 cash sanity, exact V9 control, and one treatment. L1, E1, C1, and S1 never share a Selection envelope or output root. The study-arm value and digest are part of every config, leaf, diagnostic, and terminal identity.

Pre-registered conditional order is D1, L1, E1, C1, S1. D1 determines whether the observed loss is primarily post-neutral hold or already present before neutral. L1 is run only when post-neutral deterioration is supported; E1 is run when entry quality is non-positive; C1 requires a persistent pooled long/short difference after E1; S1 requires positive out-of-inner-fold calibrated edge and an executable sizing contract.

All eight r21-influenced economic scopes are development evidence. A treatment that passes their unchanged numerical gates may be frozen, but it is not a confirmatory result. After the code, config, candidate, and artifact identities are locked, the untouched temporal holdout may be opened once through the existing Admission boundary. Failure after opening that holdout is terminal for the V11 generation; it cannot be tuned and retried against the same holdout.

## V12 and V13 boundaries

If no executable V11 candidate is eligible, write a separate V12 design before changing horizons. H1 compares fast horizons 4h, 8h, 16h, and 24h with slow 72h fixed; H2 compares slow 72h, 120h, and 168h after locking the H1 winner. Horizon choice occurs on inner walk-forward evidence before one untouched outer evaluation. Missing 8h/16h/120h/168h labels must be materialized causally from frozen 15-minute prices and artifact-bound; they cannot be interpolated from existing labels.

V13 is blocked until a 15-minute candidate is accepted. It also requires real one-minute klines, which are absent from the current PostgreSQL tables. V13 may compare only execution outcomes from a frozen 15-minute target stream; it cannot change signals, directions, entry/exit decisions, target magnitude, or confirmations.

## Stop conditions

- Stop before Admission if no V11/V12 candidate passes unchanged Selection.
- Stop S1 if the preflight proves all non-zero targets non-executable under the frozen contract.
- Stop V13 if the 15-minute strategy has not passed or if one-minute data is absent.
- A stop condition is a completed, evidence-backed result and must appear in the consolidated report.

## Decision log

1. **Freeze V10.** Extending V10 artifacts was rejected because r21 is the final lifecycle reference and schema v3 evidence must remain reproducible.
2. **Use a V11 exact-control trace policy.** Post-hoc inference from r21 was rejected because neutral signals are not recoverable from numeric targets or `unavailable` metadata.
3. **Keep candidates independent.** A combined loss-containment/cost/calibration candidate was rejected because it would not identify which hypothesis changed wealth.
4. **Use one-week out-of-sample pooled sign calibration.** In-sample calibration and symbol-specific calibration were rejected for leakage and hidden symbol tuning.
5. **Preflight S1.** Lowering `entry_threshold` was rejected because it changes the execution contract and would confound sizing with accessibility.
6. **Treat current folds as development evidence.** Reusing r21-influenced folds for a confirmatory claim was rejected.
7. **Make V12/V13 conditional.** Running horizon or one-minute experiments before policy quality is resolved was rejected as an attribution failure.
8. **Use one treatment per run.** A single multi-treatment Selection was rejected because sequential inspection would make the shared outer evidence an implicit tuning set.
9. **Produce one final report.** Plans, intermediate artifacts, failures, stop conditions, and final gate outcomes are consolidated rather than emitted as separate handoffs.
