# Causal Alpha V3 Signal Forensics V2 Design

## Status and objective

This specification extends the read-only Causal Alpha V3 Signal Forensics design so that fresh runs containing `causal_alpha_v3_signal_diagnostic_scope_v1` sidecars can be diagnosed at the 24h / 72h / fused forecast, fitted-model, effective-sample-size, prediction-distribution, feature-availability, and chronological-stability levels.

The objective is diagnostic observability only. V2 must not refit a ridge model, rebuild source data, replay an environment, change Signal Gate thresholds/bootstrap semantics, re-rank candidates, open Teacher-admission evidence early, modify BC/RL behavior, or make a rejected run promotable.

Integration context for this design:

- `main` is `d27b4ecf35ec1ccf1647425640ca9936df1054f3` and contains the merged Signal diagnostic producer (#413) and owner-only manual research launcher (#412).
- PR #411 contains the verified read-only Signal Forensics V1 implementation for canonical Signal V2 leaves but is not merged. Its current head is `32803a50678374f3eef2bf111dc2dff8e18d3d57`.
- PR #411 is currently non-mergeable against the newer `main`; this V2 design therefore starts from current `main` and treats V1 behavior as a compatibility contract rather than mutating #411 in place.
- V2 implementation may reuse the already-verified V1 implementation and tests, but must not require #411 to be merged first.

## Approaches considered

### A. Expand the V1 report in place and conditionally add sidecar fields

This minimizes the number of public types, but changes the meaning and shape of the existing `causal_alpha_v3_signal_forensics_v1` contract depending on whether a sidecar directory happens to exist. Existing historical-output consumers would lose a stable schema boundary.

Rejected.

### B. Create a completely separate sidecar-only analyzer

This keeps V1 untouched, but duplicates run-manifest/config/canonical-Signal validation and risks two analyzers disagreeing about fit, symbol, episode, rejection, and path identities.

Rejected.

### C. Preserve V1 as the canonical base analysis and add an opt-in V2 wrapper

V2 first obtains a valid V1 report, then optionally binds strict diagnostic sidecars to the exact canonical scope set and computes additional research summaries. The V1 API and default CLI behavior remain unchanged. V2 has its own versioned schema and embeds or binds the V1 report digest.

Selected. This preserves historical behavior, centralizes canonical Signal validation, and makes the sidecar extension fail closed without turning research diagnostics into Gate evidence.

## Quality contract

### Objective

Provide deterministic, read-only, fail-closed Signal diagnostics for fresh Causal Alpha V3 runs with diagnostic sidecars while retaining the verified V1 historical-analysis contract unchanged.

### Non-goals

- no Signal Gate threshold, bootstrap, independence-unit, or aggregation change;
- no candidate re-ranking or tuning based on a rejected run;
- no data rebuild, model refit, or environment replay;
- no historical sidecar backfill or post-hoc reconstruction;
- no Teacher-admission, Teacher-package, BC, critic warm-start, PPO/Lagrangian, reward, risk, execution, or promotion change;
- no bull/bear/regime classifier invented from the sidecar;
- no per-decision attribution to one missing feature because the sidecar does not persist a row-by-feature missingness bitset;
- no claim that canonical `CausalAlphaRidgeModel.digest` can be reconstructed from the sidecar alone;
- no profitability, alpha, RL-uplift, or Production GO claim;
- no merge of PR #410 or PR #411 as an implicit side effect of this work.

### Acceptance criteria

1. The existing V1 public analysis contract and default V1 CLI output remain unchanged for the same canonical Signal V2 artifacts.
2. V2 has a distinct schema, `causal_alpha_v3_signal_forensics_v2`, and explicitly binds the V1 base-report digest.
3. V2 supports two deterministic modes only:
   - `historical_unavailable`: no `signal/diagnostics` path exists; V1 analysis remains authoritative and sidecar-only analyses remain explicitly unavailable;
   - `sidecar_complete`: the diagnostics path exists and every canonical Signal metric has exactly one valid diagnostic sidecar.
4. If `signal/diagnostics` exists, even as an empty directory, V2 never falls back to historical mode. Missing, extra, duplicate, corrupt, stale, wrong-path, wrong-run, or identity-drifted sidecars fail closed.
5. Sidecar parsing reuses the strict #413 codec / contract. V2 does not introduce a permissive parallel JSON parser for diagnostic artifacts.
6. The canonical metric and sidecar identity must agree exactly on run manifest, fit config, symbol, episode, contract interval/digest, Signal metric digest, fit digest, forecast digest, and canonical cohort indices.
7. 24h, 72h, and fused realized-forecast diagnostics reuse `evaluate_causal_alpha_signal_diagnostics`; correlation, direction, quantile, and bin semantics are not redefined in V2.
8. Direct 24h-vs-72h comparison uses only identical decision indices for which both horizons are fully realized within the contract. 72h prediction and realized return are compared in 24h-equivalent units (`value / 3.0`).
9. Overlapping raw realized rows may be used for descriptive diagnostics only. They are never presented as independent Gate samples and are never fed back into Signal Gate confidence intervals.
10. Pooled model evidence duplicated across symbol sidecars is deduplicated to exactly one model snapshot per `(fit_config_digest, contract_start, contract_stop)` after exact cross-symbol consistency validation.
11. Model stability is computed separately for 24h and 72h horizons and includes coefficient cosine similarity, coefficient sign-flip rate, normalized scaler-location drift, log-scale drift, fitted weighted residual RMSE, pooled weighted ESS, per-symbol weighted ESS, and overlap-weight identity transitions.
12. Prediction distributions cover 24h prediction, 72h 24h-equivalent prediction, fused expected return, uncertainty, and signal-to-uncertainty using one fixed quantile grid.
13. Availability analysis reports complete-vs-incomplete row counts/performance, row availability-fraction summaries, and per-feature availability fractions. It never claims that a particular feature caused a prediction error for a row because that evidence is not persisted.
14. Chronological sensitivity uses only authored episode ordering and deterministic early/late plus slope summaries. It does not label market regimes that were not persisted.
15. Source artifacts remain byte-for-byte read-only and V2 output may only be written outside the source run root.
16. V2 output remains `research_only=true` and `promotion_eligible=false`; no V2 object can be consumed as Signal Gate, selection, Teacher, learner, or promotion evidence.
17. Report identity is deterministic and independent of the local absolute source path.
18. V2 must preserve explicit limitations for analyses not justified by the persisted evidence.

### Invariants

- `causal_alpha_v3_signal_scope_v2` remains the sole canonical Signal Gate leaf contract.
- Diagnostic sidecars remain research-only and are not promotion evidence.
- Signal Gate receives only `CausalAlphaV3SignalScopeMetric` evidence; V2 never modifies or regenerates those metrics.
- One `(contract_start, contract_stop)` interval remains one chronological independent episode regardless of train-symbol count.
- Cross-symbol pooled fit identity remains exact within one chronological episode.
- Historical V2 runs without sidecars are never refitted or migrated to manufacture V2 diagnostics.
- The maintained scalar reward, position-risk cap, decision delay, selection ordering, and Teacher-admission chronology remain unchanged.

## Public compatibility boundary

### V1 remains unchanged

The existing V1 API remains the compatibility authority:

```python
load_causal_alpha_v3_signal_forensics(root: Path) -> CausalAlphaV3SignalForensicsReport
```

Its schema remains:

```text
causal_alpha_v3_signal_forensics_v1
```

The existing analyzer CLI keeps V1 as its default behavior. A historical run analyzed through the V1 path must produce the same deterministic payload/digest as before this feature.

### V2 is explicit

Add a distinct V2 entry point:

```python
load_causal_alpha_v3_signal_forensics_v2(
    root: Path,
) -> CausalAlphaV3SignalForensicsReportV2
```

The existing CLI may expose V2 through an explicit schema selector such as:

```text
--schema v1|v2
```

with `v1` remaining the default for compatibility. The exact CLI spelling is part of the implementation plan, but V2 must never silently replace the V1 output contract.

The V2 report binds the V1 result rather than reimplementing its canonical logic. Its top-level contract contains at least:

```text
schema_version = causal_alpha_v3_signal_forensics_v2
base_forensics_digest
base_forensics
sidecar_mode
sidecar_analysis
unavailable_analyses
research_only = true
promotion_eligible = false
artifact_digest
```

`base_forensics` is the deterministic V1 payload and `base_forensics_digest` must equal its artifact digest. The V2 artifact digest covers the full V2 payload.

## Sidecar mode and strict pairing

### Historical mode

If `root / "signal" / "diagnostics"` does not exist, V2 returns:

```text
sidecar_mode = historical_unavailable
sidecar_analysis = null
```

The V1 unavailable analyses remain explicit, including 24h-vs-72h, coefficient stability, prediction distributions, and residual-RMSE analysis.

No model/data reconstruction is attempted.

### Complete sidecar mode

If the diagnostics path exists, V2 requires a complete pair graph.

For every canonical metric identity:

```text
(fit_config_digest, symbol, episode_index)
```

there must be exactly one diagnostic at:

```text
signal/diagnostics/<fit_config_digest>/<symbol>/<episode_index>.json
```

The set of sidecar identities must equal the set of canonical metric identities exactly. No subset, superset, duplicate, or alternate path is accepted.

Each pair must agree on:

- `run_manifest_digest`;
- `fit_config_digest`;
- `symbol`;
- `episode_index`;
- `contract_start` / `contract_stop`;
- `contract_digest`;
- `signal_metric_digest == metric.digest`;
- `fit_digest`;
- `forecast_digest`;
- `canonical_cohort_indices == metric.cohort_indices`.

The diagnostic object itself remains responsible for strict schema/content validation, including the #413 forecast reconstruction check from persisted predictions plus horizon residual RMSE.

## Analysis architecture

The V2 path is split into four responsibilities.

### 1. Canonical base loader

Use V1 unchanged to validate and summarize:

- run manifest / authored config;
- canonical Signal records;
- chronological clusters;
- rejection evidence;
- fit/symbol/episode summaries;
- paired fit comparisons.

V2 does not duplicate these validations.

### 2. Diagnostic sidecar loader / binder

A small V2-specific loader discovers sidecar paths and uses `signal_diagnostic_scope_from_payload` as the parser authority. It validates exact path identity and the complete canonical-to-diagnostic bijection.

This layer returns typed, paired scopes only. Downstream analysis never receives unbound raw JSON.

### 3. Episode model-snapshot deduplicator

For each `(fit_config_digest, contract_start, contract_stop)` cluster, all symbol diagnostics must contain exactly equal pooled model evidence for each horizon.

Exact equality covers the model fields that are expected to be pooled-fit identical:

- model digest;
- feature names/order;
- intercept;
- coefficients;
- scaler location/scale;
- constant mask;
- fitted row count;
- weighted residual RMSE;
- pooled weighted ESS;
- per-symbol weighted ESS;
- overlap-weight digest.

Any cross-symbol disagreement fails closed. Only after equality succeeds is one snapshot retained for longitudinal model analysis. This prevents symbol count from multiplying fit-level evidence.

### 4. Pure summary builders

Pure functions consume typed paired scopes / deduplicated model snapshots and produce deterministic summaries. They do not access the filesystem, mutate source artifacts, or call model fitting/runtime code.

## Horizon diagnostics

For every sidecar scope, use the persisted `realized_24h_rows`, `realized_72h_rows`, and `realized_fused_rows`.

### Per-horizon units

- 24h: stored prediction and realized return as-is.
- 72h: use the persisted 24h-equivalent prediction/realized values for direct comparison. Raw 72h values remain available as descriptive fields but are not mixed numerically with 24h units.
- fused: use the persisted fused 24h-equivalent prediction/realized values.

### Existing diagnostic semantics

Call `evaluate_causal_alpha_signal_diagnostics(prediction, realized)` for each horizon. This preserves:

- sample count;
- prediction / realized mean, standard deviation, extrema, and fixed quantiles;
- Pearson correlation;
- rank correlation;
- explicit undefined-correlation reason;
- direction accuracy;
- sign-rate summaries;
- fixed prediction bins and per-bin realized summaries.

No new correlation/rank/bin definitions are introduced.

### Paired 24h-vs-72h comparison

Build maps keyed by `decision_index` from 24h and 72h realized rows, intersect those keys, and preserve chronological decision order. The paired set must contain at least two rows to calculate diagnostics; otherwise the comparison is explicitly unavailable for that scope rather than silently widened to unmatched samples.

The paired comparison records the same-sample 24h and 72h diagnostic summaries and descriptive deltas for metrics already defined by the existing diagnostic contract. Nullable correlations remain nullable; no numeric sentinel substitutes for an undefined correlation.

### Independence labeling

Raw sidecar rows may overlap in label windows. V2 labels these as descriptive rows and never calls them independent episodes. Independent chronological aggregation remains at the authored contract interval level inherited from V1.

## Model stability diagnostics

Model stability is evaluated per authored fit config and separately for `24h` and `72h`.

### Consecutive snapshot pairing

Order deduplicated model snapshots by `(contract_start, contract_stop)`. Compare only consecutive snapshots within the same fit config and horizon.

Feature names/order must match exactly; a schema/order change fails closed instead of aligning by guesswork.

### Coefficient cosine similarity

For consecutive coefficient vectors `a` and `b`:

```text
cosine = dot(a, b) / (||a|| * ||b||)
```

If either coefficient norm is `<= 1e-15`, cosine similarity is unavailable with an explicit reason rather than returning an unstable number.

### Coefficient sign-flip rate

A feature participates in the sign-flip denominator only when both consecutive coefficients are non-zero. A flip occurs when the two non-zero coefficients have opposite signs.

V2 records both:

- active paired coefficient count;
- sign-flip fraction among those active pairs.

If there are no active paired coefficients, the rate is unavailable.

### Scaler drift

Use dimensionless drift measures so unlike feature units are not averaged directly.

For previous location `mu0`, previous positive scale `s0`, current location `mu1`, and current positive scale `s1`:

```text
location_shift_rms = sqrt(mean(((mu1 - mu0) / s0) ** 2))
log_scale_ratio_rms = sqrt(mean(log(s1 / s0) ** 2))
```

These are descriptive stability measures only. They do not alter prediction scaling.

### Fit residual RMSE and ESS

For each snapshot/horizon, report:

- `weighted_residual_rmse` explicitly named as fit/training residual evidence, not realized OOS forecast RMSE;
- pooled weighted ESS;
- per-symbol weighted ESS;
- fitted row count;
- overlap-weight digest.

Chronological summaries include min/mean/max and deterministic early/late/slope where meaningful.

The overlap-weight digest unique count and transition count are also reported per fit/horizon. Digest changes are identity changes, not automatically interpreted as degradation.

## Prediction-distribution diagnostics

Prediction distribution uses all persisted prediction rows for descriptive geometry, not only the non-overlapping Gate cohort.

Use the fixed quantile grid already established by `CAUSAL_ALPHA_SIGNAL_QUANTILES`:

```text
0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0
```

For each scope and aggregated fit/episode view, summarize:

- `prediction_24h`;
- `prediction_72h_24h_equivalent`;
- `expected_return_24h_equivalent` (fused);
- `uncertainty_24h_equivalent`;
- `signal_to_uncertainty`.

Each distribution records count, mean, standard deviation, minimum, maximum, and fixed quantiles. No histogram bucket choice is introduced beyond existing fixed diagnostic bins where realized outcomes are available.

## Feature-availability diagnostics

The sidecar preserves both row-level availability counts/fractions and per-feature aggregate availability fractions.

### Row-level analysis

For each horizon, partition fully realized rows into:

```text
complete: available_feature_fraction == 1.0
incomplete: available_feature_fraction < 1.0
```

When a partition has at least two realized rows, evaluate it through the existing signal-diagnostic function. Smaller partitions remain explicit unavailable subsets with their observed row count.

Report complete-vs-incomplete differences descriptively; do not infer causality.

### Availability geometry

Report:

- complete and incomplete prediction-row counts;
- min/mean/max and fixed-quantile row availability fraction;
- per-feature availability fraction keyed by exact feature name/order;
- chronological early/late/slope summaries of mean availability.

### Explicit limitation

The sidecar does not persist the exact unavailable-feature set for each decision row. Therefore V2 cannot state that a specific feature's absence caused one row's error or calculate exact per-feature conditional forecast error. That remains an explicit unavailable analysis unless the producer schema changes in a future contract.

## Chronological sensitivity

V2 uses authored chronological Signal contracts as the only phase axis. It does not create bull/bear/high-volatility labels from price data or external information.

For each fit config, produce chronological episode summaries and deterministic early/late/slope views for at least:

- 24h / 72h / fused direction accuracy;
- available horizon rank/Pearson correlation, retaining undefined counts;
- prediction standard deviation;
- fit weighted residual RMSE by horizon;
- pooled weighted ESS by horizon;
- mean feature availability.

The slope is descriptive over episode order. It is not a statistical trend test and does not imply stationarity or causality.

## V2 unavailable analyses

In `historical_unavailable` mode, retain the V1 unavailable analyses.

In `sidecar_complete` mode, remove analyses that are now directly observable and retain at least these explicit limitations:

- exact per-decision, per-feature missingness attribution;
- causal attribution of availability loss to forecast error;
- independent reconstruction of canonical `CausalAlphaRidgeModel.digest` from sidecar state alone;
- market-regime labels not persisted by the run;
- profitability / Production GO inference from Signal diagnostics.

## Error handling and fail-closed rules

V2 fails instead of downgrading evidence when any of the following occurs in sidecar mode:

- diagnostics directory exists but contains zero sidecars while canonical Signal metrics exist;
- one canonical metric lacks a sidecar;
- an extra sidecar has no canonical metric;
- duplicate sidecar identity;
- invalid sidecar JSON/schema/content digest;
- wrong sidecar filesystem path;
- run/fit/symbol/episode/contract identity drift;
- `signal_metric_digest`, fit digest, forecast digest, or canonical cohort mismatch;
- cross-symbol pooled model state mismatch inside one chronological episode;
- feature name/order mismatch across horizons or consecutive snapshots where a comparison requires equality;
- non-finite values reaching a new summary calculation;
- attempted output inside the source run root.

V2 does not catch these errors and silently return V1-only results when the diagnostics path exists. Presence of that path commits the run to the sidecar contract.

## Test oracle

Correctness is observed through independent, inspectable artifacts and calculations rather than merely successful execution.

Primary oracles:

- V1 payload/digest is unchanged for the same historical fixtures before and after V2 implementation;
- exact canonical metric identity set equals exact sidecar identity set in sidecar mode;
- strict #413 codec accepts every consumed sidecar;
- paired metric/sidecar identity fields reproduce the persisted canonical relationships;
- cross-symbol pooled model snapshots compare exactly before deduplication;
- 24h/72h/fused summary values are independently recomputable from persisted realized rows;
- 72h comparison values use `/3.0` 24h-equivalent units;
- model stability calculations are checked against small hand-calculable vectors;
- ESS/RMSE evidence is counted once per pooled fit episode, never once per symbol;
- source file byte digests are unchanged after analysis;
- V2 output digest is identical for identical evidence copied under different absolute filesystem roots.

## Required test layers

### Unit

- fixed numeric/quantile summaries;
- paired horizon alignment;
- cosine/sign-flip/scaler drift formulas;
- model-snapshot deduplication;
- availability partitioning;
- deterministic chronological trend summaries.

### Contract

- V1 compatibility/no-op fixtures;
- sidecar all-or-none pairing;
- exact path and identity binding;
- historical-vs-sidecar mode selection;
- V2 schema/digest/research-only flags.

### Falsification / regression

Explicitly attempt:

- empty diagnostics directory fallback;
- one missing sidecar;
- one extra sidecar;
- self-consistent outer sidecar digest with wrong canonical metric binding;
- wrong forecast/fit/cohort identity;
- duplicated identity under alternate path;
- one symbol carrying a different pooled coefficient/scaler/RMSE/ESS snapshot;
- symbol-count multiplication of ESS/model evidence;
- unmatched 24h/72h decision sets being compared as if paired;
- 72h raw-unit values being mixed with 24h-equivalent values;
- undefined correlation being coerced to zero;
- source artifact mutation;
- output written under source root;
- diagnostics being passed toward Gate/selection/Teacher/learner interfaces.

### Static / architecture

- Ruff;
- format;
- Mypy;
- import architecture;
- dead-code scan;
- source-shape assertion that V2 remains a read-only workflow/analysis surface and does not import model fitting, runtime replay, selection, Teacher generation, learner training, or promotion writers.

### Integration / full regression

- V1 CLI historical fixture remains byte/digest compatible;
- V2 CLI historical mode;
- V2 CLI complete-sidecar mode;
- full repository pytest/branch coverage;
- critical branch coverage;
- Ubuntu/Windows compatibility where applicable;
- package/build/training-image checks required by repository CI;
- PostgreSQL Catalog when path filters or integration surfaces require it.

## Quality gate

V2 is not complete unless all of the following are true on the exact final HEAD:

- all acceptance criteria have direct tests or independent structural evidence;
- a valid TDD RED demonstrates the new sidecar behavior is absent before implementation;
- V1 compatibility is demonstrated with exact payload/digest evidence, not inferred from code review;
- all new targeted unit/contract/falsification tests pass;
- Ruff, format, Mypy, import architecture, and dead-code checks pass;
- full repository tests and required coverage pass;
- required build/package/compatibility checks pass;
- final diff contains no Signal Gate/model-fitting/selection/Teacher/BC/RL/reward/risk/execution/promotion semantic changes;
- falsification review explicitly searches for sidecar partial-fallback, identity forgery, model-evidence multiplication, unit mismatch, unavailable-value coercion, and source mutation;
- independent reviewer/subagent verification is used if available; otherwise a specification-first independent-style review is documented;
- exact final HEAD and CI/required-check HEAD are identical;
- remaining empirical limitations are documented.

## What this design can and cannot establish

If implemented and verified, V2 can establish that the persisted diagnostic sidecars are internally consistent with canonical Signal records and can deterministically describe horizon quality, model-state stability, forecast geometry, ESS, RMSE, availability, and chronological variation for the observed research generation.

It still cannot establish causal economic alpha, future profitability, a valid Production GO decision, or that changing any observed unstable coefficient/feature/horizon will improve out-of-sample performance. Those require separate experiments and promotion-grade evidence.