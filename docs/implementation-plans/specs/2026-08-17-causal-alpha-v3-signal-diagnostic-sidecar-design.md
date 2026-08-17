# Causal Alpha V3 Signal Diagnostic Sidecar Design

## Objective

Persist the diagnostic evidence that is currently discarded while building Causal Alpha V3 Signal scopes so fresh Signal Contract V2 runs can explain *why* a fit passes or fails without changing the maintained Signal Gate, refitting a completed run, replaying environments, or using diagnostic evidence for promotion.

The sidecar must make the following currently-unobservable questions answerable from a fresh run:

- Is 24h predictive quality materially different from 72h predictive quality?
- Is the fixed 24h/72h blend hiding one good and one bad horizon?
- Are predictions collapsing toward zero or carrying a large intercept/base-rate component?
- Are fitted coefficients/scalers unstable across expanding chronological fits?
- Are low feature-availability rows associated with poor predictions?
- Is raw fitted row count materially overstating weighted effective sample size?
- Are conclusions sensitive to the canonical non-overlapping cohort phase?

The immediate consumer is a future extension of the read-only Signal Forensics work in PR #411. This producer change must not depend on PR #411 and must remain independently mergeable onto `main`.

## Non-goals

- Do not change Signal Gate thresholds, bootstrap semantics, independent-episode semantics, candidate ordering, or pass/fail results.
- Do not change ridge fitting, label formulas/timing, 24h/72h forecast blend, target compilation, economic selection, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, or execution semantics.
- Do not reopen, tune against, or otherwise use the untouched Teacher-admission holdout.
- Do not migrate or reinterpret old Signal V2 runs. Existing runs without sidecars remain truthful historical artifacts.
- Do not post-hoc refit old runs to manufacture missing diagnostics.
- Do not add new model candidates or use diagnostic values to select a candidate in this change.
- Do not duplicate the general run-report surface in PR #410 or the read-only historical analyzer in PR #411.

## Alternatives considered

### 1. Expand the canonical Signal leaf to a new schema

Store predictions, coefficients, and availability directly in `CausalAlphaV3SignalScopeMetric` and bump the leaf schema.

Rejected for this change. The maintained Signal leaf is a compact gate evidence contract. Enlarging it would mix diagnostic-only observations with promotion-adjacent evidence, force broader schema/parser/store changes, and make a numerical no-op instrumentation change look like a gate-contract change.

### 2. Refit completed runs inside the forensic analyzer

Load the historical run inputs, rebuild samples, refit each chronological ridge, and reconstruct the missing predictions and coefficients.

Rejected. It would make a supposedly read-only analyzer depend on a second execution of the training code, create provenance ambiguity between persisted and reconstructed evidence, and cannot guarantee byte-for-byte equivalence to the original ephemeral computation.

### 3. Persist a separate sidecar during the original Signal computation

Chosen. The Signal builder already has the fitted models, per-horizon predictions, labels, availability mask, and canonical cohort in memory before reducing them to the compact Signal metric. Persist those observations in a separate run-bound research-only sidecar while leaving the canonical V2 metric and Gate unchanged.

## Architecture

### Artifact boundary

Add one immutable diagnostic record for each canonical Signal leaf identity:

```text
signal/records/<fit_config_digest>/<symbol>/<episode>.json
signal/diagnostics/<fit_config_digest>/<symbol>/<episode>.json
```

The canonical `signal/records` leaf remains `causal_alpha_v3_signal_scope_v2` and remains the **only** input to Signal Gate aggregation.

The new diagnostic record uses a separate versioned schema, initially:

```text
causal_alpha_v3_signal_diagnostic_scope_v1
```

It is always:

```text
research_only = true
promotion_eligible = false
```

and is never accepted as a substitute for a canonical Signal metric.

### Identity binding

Each diagnostic record must bind to all relevant immutable identities:

- `run_manifest_digest`
- `fit_config_digest`
- `symbol`
- `episode_index`
- `contract_start`
- `contract_stop`
- `contract_digest`
- canonical `signal_metric_digest`
- pooled `fit_digest`
- `forecast_digest`
- 24h model digest
- 72h model digest
- feature schema / ordered feature names identity
- 24h and 72h overlap-weight digests already carried by the fit

The path is derived from `(fit_config_digest, symbol, episode_index)` exactly as the canonical Signal record path is. Unknown, duplicate, wrong-path, wrong-run, wrong-contract, wrong-fit, or digest-drifted sidecars fail closed.

### Computation boundary

Refactor the current Signal-scope build path so one internal computation produces both:

1. `CausalAlphaV3SignalScopeMetric` — unchanged numerical semantics;
2. `CausalAlphaV3SignalDiagnosticScope` — diagnostic-only evidence.

The existing public `build_causal_alpha_v3_signal_scope_metric(...)` remains available as a compatibility wrapper that returns only the metric. The maintained pipeline uses the paired computation path so no second fit or second forecast pass is required in the normal case.

The metric generated by the paired path must be byte/digest-equivalent to the metric generated by the existing path for the same inputs **when evaluated under the same numerical execution environment** (dependency set, numerical backend, and thread configuration).

## Diagnostic record contents

The record should preserve *sufficient observations* to derive future forensic summaries without persisting the full training dataset or raw market feature matrix.

### A. Contract prediction rows

For every decision in the Signal contract, persist:

- decision index;
- whether the target symbol row is actionable/present;
- available feature count and available feature fraction;
- `prediction_24h`;
- `prediction_72h`;
- `prediction_72h_24h_equivalent = prediction_72h / 3`;
- fused `expected_return_24h_equivalent`;
- fused `uncertainty_24h_equivalent`;
- `signal_to_uncertainty`.

This is prediction evidence only and does not require the forward label to be realized.

### B. Realized 24h rows

For every actionable/matched row whose 24h label is fully realized inside the contract, persist:

- decision index;
- 24h label end index;
- 24h prediction;
- realized 24h log return;
- available feature count/fraction.

### C. Realized 72h rows

For every actionable/matched row whose 72h label is fully realized inside the contract, persist:

- decision index;
- 72h label end index;
- 72h prediction and 24h-equivalent 72h prediction;
- realized 72h log return and 24h-equivalent realized return;
- available feature count/fraction.

### D. Fused realized rows

For every row where both horizons are realized inside the contract, persist:

- decision index;
- maximum label end index;
- fused prediction;
- fused realized return using the maintained formula;
- available feature count/fraction.

Persist the canonical non-overlapping cohort indices used by `CausalAlphaV3SignalScopeMetric` and require them to match the canonical metric exactly.

Keeping all realized rows, not only the canonical cohort, allows a later read-only analyzer to evaluate alternative non-overlap phases descriptively without changing the canonical Gate or treating those phases as additional independent episodes.

### E. Feature-availability summary

Persist, in the same ordered feature-name space as the fitted models:

- per-feature available fraction across contract prediction rows;
- count of rows with complete feature availability;
- count of rows with incomplete feature availability;
- minimum/mean/maximum available feature fraction.

Do **not** persist the raw feature values in this change.

### F. Model diagnostics

Persist one diagnostic summary for the 24h model and one for the 72h model:

- model digest;
- ordered feature names;
- intercept;
- coefficients;
- scaler location;
- scaler scale;
- constant-feature mask;
- fitted raw row count;
- weighted residual RMSE already computed by the fit;
- pooled weighted effective sample size;
- per-symbol weighted effective sample size;
- overlap-weight digest.

Weighted effective sample size is descriptive and defined as:

```text
ESS = (sum(w) ** 2) / sum(w ** 2)
```

using the exact positive overlap/symbol-balanced weights corresponding to the fit and horizon. Recomputing those deterministic weights for diagnostics is allowed, but their content digest must match the fit's stored horizon weight digest before the ESS values are accepted.

The sidecar must not mutate `CausalAlphaV3Fit` or its digest merely to attach diagnostics.

## Resume and partial-write semantics

Signal metric and diagnostic sidecar form a logically paired scope artifact but remain separate files.

The store/pipeline must distinguish four states for an expected scope:

1. **metric + sidecar present and valid** — reuse both with no rebuild;
2. **neither present** — build once and write both;
3. **only one present after an interrupted write** — recompute the scope once, require the recomputed artifact to match the persisted member's identity/digest, then write only the missing member;
4. **present member is corrupt, stale, wrong-path, or identity-drifted** — fail closed; do not silently replace it.

This preserves crash recovery without turning corruption into an implicit rebuild.

Writes remain under the existing single-writer run lock. The exact physical write order is not a correctness assumption because the paired resume logic covers either one-file partial state.

## Relationship to Signal Gate

The Signal Gate remains exactly:

```text
CausalAlphaV3SignalScopeMetric
    -> chronological cross-symbol episode aggregation
    -> moving-block bootstrap
    -> existing lower-CI thresholds
```

No diagnostic field is passed into `evaluate_causal_alpha_v3_signal_gate_clustered`.

The following must remain numerically identical before and after this change for identical inputs:

- Signal metric sample count;
- rank correlation;
- direction accuracy;
- top/bottom realized spread;
- canonical cohort indices;
- metric digest;
- fit-level Signal Gate evidence and pass/fail result.

## Relationship to PR #411 Signal Forensics

This producer change does not modify PR #411 and does not depend on it.

After the producer is verified, a separate follow-up may extend the forensic analyzer so:

- old Signal V2 runs without sidecars continue to report the existing analyses and explicit unavailable reasons;
- fresh runs with valid sidecars additionally expose 24h-vs-72h, coefficient/scaler stability, prediction distributions, availability effects, ESS, and non-overlap phase sensitivity.

The analyzer must never require sidecars for historical V2 validity.

## Implementation discovery: numerical-backend reproducibility boundary

During implementation, the unchanged pre-sidecar commit was executed more than once with the same controlled synthetic inputs on separate GitHub-hosted runners. The canonical Gate observations (`sample_count`, cohort indices, rank correlation, direction accuracy, and top/bottom spread) were identical, but low-bit ridge outputs caused the existing `fit_digest`, `forecast_digest`, and therefore metric artifact digest to differ across runners. This behavior predates the sidecar change.

The quality contract is therefore refined as follows rather than silently weakening the implementation criteria:

- **No-op refactor equivalence** is tested by running the pre-sidecar tree and candidate tree side-by-side on the **same runner**, using the same dependency environment and fixed numerical thread settings, and requiring the full canonical metric payload/digest to match exactly.
- The unit regression fixes stable Gate observations and stable scope/config identities; it does not treat a cross-host floating-point digest as a portable oracle.
- Cross-host bitwise reproducibility of the existing ridge solver is **not** introduced as a new guarantee by this diagnostic change.
- If partial-resume recomputation occurs on a numerical backend that produces a different persisted-member digest, the pipeline must fail closed rather than overwrite or reinterpret the existing evidence.
- Deterministic sidecar bytes are required for the same computed fit/forecast and numerical execution environment; raw cross-host BLAS reproducibility remains a separate residual risk.

## Acceptance criteria

1. Fresh Signal scope computation produces the same canonical V2 metric/digest as the existing implementation when compared under the same numerical execution environment, and one deterministic diagnostic sidecar.
2. The sidecar contains separate 24h, 72h, and fused prediction/realized observations sufficient to recompute descriptive horizon diagnostics without refitting.
3. The sidecar contains model coefficients/scalers/intercepts and exact model/fit identities sufficient to compare expanding fits across episodes.
4. The sidecar contains per-row and per-feature availability observations without persisting raw market feature values.
5. The sidecar contains horizon-specific pooled and per-symbol weighted ESS whose source weights reproduce the fit's stored weight digest.
6. Canonical cohort indices in the sidecar exactly equal the canonical Signal metric cohort indices.
7. Signal Gate receives only canonical metrics; all existing metric values, Gate evidence, and pass/fail semantics remain unchanged.
8. Complete paired artifacts resume without rebuilding.
9. A crash leaving exactly one valid member of a pair recomputes once and writes only the missing member after digest/identity agreement.
10. Corrupt/stale/wrong-run/wrong-path/wrong-contract/wrong-fit/duplicate diagnostic artifacts fail closed before reuse.
11. Old V2 runs without diagnostics are not migrated, post-hoc refitted, or reinterpreted.
12. No selection, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, execution, or promotion numerical behavior changes.
13. Exact final HEAD passes the required targeted tests, full suite/coverage, Ruff/format, Mypy, import architecture, dead-code checks, compatibility/build/package identity checks, and any applicable database workflow; unavailable checks are reported as unverified rather than assumed successful.

## Invariants

- Run manifest remains the run identity authority.
- Canonical Signal V2 metric remains the Signal Gate evidence authority.
- One chronological interval remains one independent temporal episode regardless of diagnostic row count or phase count.
- Diagnostic rows and alternate phases never increase independent evidence or change Gate sample counts.
- Pooled fit identity remains shared across symbols in the same chronological interval.
- Diagnostic artifacts remain research-only and non-promotable.
- No future label outside the contract is persisted as realized evidence.
- No raw feature matrix or untouched Teacher holdout is exposed by the sidecar.
- A diagnostic instrumentation failure may fail the fresh research run rather than silently proceeding with incomplete sidecar evidence; it may not alter a Gate result to compensate.

## Failure modes

Important failure modes to test explicitly:

- metric numerical drift introduced by the paired-computation refactor;
- 24h/72h label timing or eligibility drift relative to maintained label endpoints;
- canonical cohort mismatch between metric and sidecar;
- fit/model/forecast digest mismatch;
- incorrect ESS from weights that do not reproduce the fit weight digest;
- prediction/realized array misalignment;
- NaN/inf or non-JSON-safe diagnostic values;
- incorrect availability counts or feature-order drift;
- duplicate or wrong-path sidecar records;
- cross-run or cross-contract sidecar copy accepted as valid;
- partial crash causing unnecessary rebuild of already-valid paired evidence;
- corrupt artifact being silently overwritten during resume;
- diagnostic data accidentally entering Signal Gate evaluation;
- sidecar creation reading future labels outside the contract;
- sidecar exposing raw feature values or holdout evidence;
- excessive artifact size caused by accidental persistence of the entire training sample matrix.

## Test oracle

Correctness is observed through more than return values:

- exact equality of old-vs-new canonical Signal metric payload/digest for controlled inputs executed side-by-side under the same numerical execution environment;
- exact alignment of persisted decision indices, predictions, label endpoints, and realized returns against controlled synthetic samples;
- independently recomputed ESS and weight-digest equality;
- persisted path and content-digest checks;
- paired resume builder-call counts and file-state transitions;
- source run identity/contract/fit/forecast/model identity checks;
- Signal Gate evidence equality before/after diagnostic instrumentation;
- absence of sidecar objects from Gate evaluator arguments;
- byte-stable deterministic sidecar output for identical computed fit/forecast evidence under the same numerical execution environment;
- explicit tests proving no raw feature matrix or Teacher holdout fields are serialized.

## Required test layers

- Unit/contract tests for diagnostic dataclasses, JSON payload validation, ESS math, and horizon row extraction.
- Regression test proving canonical metric payload/digest is unchanged by the refactor.
- Store tests for exact paths, identity validation, corrupt payload rejection, and duplicate/cross-run/cross-contract copies.
- Workflow integration tests for normal paired persistence and all four resume/partial-write states.
- Falsification tests for metric/sidecar mismatch, weight-digest mismatch, future-label inclusion, and diagnostic-to-Gate leakage.
- Static analysis, Ruff, format check, Mypy, import architecture, and dead-code analysis.
- Full pytest and branch coverage, including changed-line execution and important error paths.
- Compatibility/build/package identity checks on exact final HEAD.
- PostgreSQL/catalog workflow only if the final diff touches an applicable path; otherwise record it as not applicable.

## Quality gate

Do not call the implementation complete merely because the new tests are green. Completion requires:

- all acceptance criteria mapped to observable tests or verified diff properties;
- final diff restricted to the diagnostic producer/store/pipeline/tests/docs needed for this objective;
- canonical Signal metric and Gate numerics demonstrated unchanged;
- paired resume semantics falsified against partial/corrupt states;
- architecture/self-review of responsibility boundaries and artifact ownership;
- independent/falsification review rebuilt from this design rather than from implementation assumptions;
- exact-final-HEAD CI/required checks inspected;
- remaining unverified items and residual risks reported explicitly.
