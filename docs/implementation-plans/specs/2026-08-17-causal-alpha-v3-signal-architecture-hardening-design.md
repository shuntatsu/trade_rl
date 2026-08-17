# Causal Alpha V3 Signal Architecture Hardening Design

## Objective

Harden the Causal Alpha V3 Signal diagnostic sidecar architecture after review without changing Signal Gate numerics, candidate selection, Teacher admission, BC/RL, reward, risk, execution, or historical artifact meaning.

This follow-up closes three architecture weaknesses found after the sidecar implementation:

1. the pipeline injection seam is typed as `Callable[..., Any]`, so an incompatible metric-only builder can pass static analysis and fail only at runtime;
2. the public canonical metric-only API currently reaches the diagnostic builder, so diagnostic-only failure can make canonical metric computation unavailable even when the canonical metric itself is valid;
3. the sidecar stores enough forecast state to independently reproduce `forecast_digest`, but the strict reader does not currently perform that cross-check.

The result must preserve the existing paired fresh-run behavior while making the canonical/diagnostic dependency direction explicit and strengthening static and artifact-level validation.

## Non-goals

- Do not change Signal Gate thresholds, bootstrap semantics, independent episode semantics, canonical cohort construction, metric values, Gate pass/fail, candidate order, or freeze semantics.
- Do not change ridge fitting, symbol-balanced weighting, label timing/formulas, 24h/72h blend, uncertainty formula, target compilation, selection, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, or execution semantics.
- Do not create fit-level diagnostic artifacts or migrate the existing per-scope sidecar schema in this change.
- Do not add per-row feature-availability bitsets or raw feature values.
- Do not merge PR #411 forensics into this producer PR.
- Do not post-hoc refit or migrate historical Signal V2 runs.
- Do not claim cross-host bitwise BLAS reproducibility.

## Architecture options considered

### Option A — minimal boundary hardening

Keep the existing public files and artifact schema, but introduce a typed builder protocol, split the shared ephemeral Signal computation from metric/diagnostic materialization, and reconstruct the persisted forecast inside strict validation.

Chosen. It closes the observed type-boundary defect and removes canonical-to-diagnostic coupling without expanding the artifact graph.

### Option B — broad Signal module extraction

Move all Signal scope computation out of `universal_causal_alpha_v3_teacher.py` into a new dedicated module and reorganize all callers now.

Rejected for this change. The direction is reasonable, but it increases unrelated file movement and makes a small architecture correction harder to review. A future extraction remains possible once the boundary is stable.

### Option C — normalize fit-level diagnostic artifacts now

Split shared model/scaler/ESS evidence into one fit-level artifact and keep only symbol-specific observations in scope sidecars.

Rejected for this change. It would require a new artifact schema/path, additional resume states, and consumer changes. The current symbol count does not justify that migration before real sidecar volume is observed.

## Components and dependency direction

### 1. Typed pipeline port

Define an explicit callable protocol for the paired Signal producer. The pipeline injection seam must statically require the current contract rather than `Callable[..., Any]`.

Conceptually:

```python
class CausalAlphaV3SignalScopeBuilder(Protocol):
    def __call__(
        self,
        *,
        run_manifest_digest: str,
        symbol: str,
        train_symbols: tuple[str, ...],
        samples: Mapping[str, CausalAlphaSymbolSamples],
        contract: OracleEpisodeContract,
        candidate: CausalAlphaV3Candidate,
        fit_cache: CausalAlphaV3FitCache | None = None,
    ) -> CausalAlphaV3SignalScopeBuild: ...
```

The exact home may remain near the Signal scope contract so it does not create a circular import. The important contract is that `run_universal_causal_alpha_v3_research_pipeline(..., signal_scope_builder=...)` no longer accepts an untyped metric-only callable.

`episode_batch_builder` is outside the scope of this hardening unless touching it is required to avoid a circular typing dependency.

### 2. Shared ephemeral Signal computation

Introduce a private immutable computation object or equivalent private helper that contains only the data already produced by the canonical calculation:

- fitted V3 model bundle;
- forecast;
- symbol sample block;
- decision indices;
- actionable mask;
- feature-availability matrix;
- matched mask;
- 24h/72h labels and label-end indices;
- canonical eligible mask/cohort rows.

The helper may compute the canonical cohort once. It must not construct or import a diagnostic artifact.

Required dependency direction:

```text
prediction / label alignment / canonical cohort
                  |
                  v
       shared ephemeral computation
            /                 \
           v                   v
canonical metric builder   diagnostic builder
           |                   |
           +--------+----------+
                    v
             paired producer
```

The canonical metric branch must not depend on diagnostic construction.

### 3. Public metric-only compatibility API

`build_causal_alpha_v3_signal_scope_metric(...)` remains public and numerically backward compatible.

Its behavior becomes:

```text
shared computation -> canonical metric -> return
```

It must not call `build_causal_alpha_v3_signal_scope(...)` and must not instantiate `CausalAlphaV3SignalDiagnosticScope`.

This restores the semantic meaning of a metric-only API: diagnostic instrumentation failure cannot make an otherwise valid canonical metric unavailable to an explicit metric-only caller.

### 4. Paired fresh-run producer

`build_causal_alpha_v3_signal_scope(...)` remains the production research-run producer and continues returning `CausalAlphaV3SignalScopeBuild(metric, diagnostic)`.

Its behavior becomes:

```text
shared computation
    -> canonical metric
    -> diagnostic(metric digest + same ephemeral evidence)
    -> pair validation
```

There must still be exactly one fit/forecast pass in the normal paired path.

The pipeline continues to require paired evidence for fresh generation and paired resume repair. The canonical-only compatibility API is not substituted into the maintained runner.

### 5. Strict forecast re-binding

The diagnostic sidecar already persists every field required to reconstruct `CausalAlphaV3Forecast`:

- all contract `prediction_24h` values;
- all contract `prediction_72h` values;
- fused expected returns;
- uncertainty values;
- signal-to-uncertainty values;
- 24h and 72h weighted residual RMSE via the diagnostic model summaries.

Strict diagnostic validation must reconstruct a `CausalAlphaV3Forecast` (or compute its exact canonical payload/digest through the same contract primitive) and require the reconstructed digest to equal `diagnostic.forecast_digest`.

This is an internal consistency check over already-persisted evidence. It must not refit a model, read raw feature values, or alter the sidecar schema.

The reconstructed arrays must follow the persisted prediction-row order, which is already required to be strictly increasing and contract-local.

### 6. Model digest boundary

Do not claim that the sidecar can independently reconstruct canonical `CausalAlphaRidgeModel.digest`. The canonical model digest also binds fields not currently serialized by the sidecar, including eligible fitted indices, knowledge cutoff, and ridge config identity.

The sidecar's model state remains bound by:

- outer sidecar content digest;
- producer-time copy from the fitted model;
- `fit_digest` / `model_digest` identities;
- exact weight-digest reproduction for ESS.

A separate model-state digest or fit-level artifact can be designed later if coefficient evidence needs an independently reconstructable canonical model identity.

## Data flow

Fresh paired scope:

```text
samples + contract + candidate
    -> prefix fit/cache
    -> forecast
    -> label alignment
    -> canonical eligibility + cohort
    -> CausalAlphaV3SignalScopeMetric
    -> CausalAlphaV3SignalDiagnosticScope
    -> CausalAlphaV3SignalScopeBuild
    -> metric + diagnostic persistence
    -> Gate receives metric only
```

Metric-only compatibility call:

```text
samples + contract + candidate
    -> same shared computation
    -> CausalAlphaV3SignalScopeMetric
    -> return
```

Diagnostic load:

```text
JSON
    -> strict field/type decoding
    -> diagnostic contract validation
    -> outer artifact digest validation
    -> reconstruct persisted forecast bundle
    -> forecast digest equality
    -> store run/path/contract identity validation
```

## Error handling

- A wrong builder type must be detectable by Mypy when wired through the maintained typed seam. Runtime type validation remains defense in depth.
- Metric-only computation may fail only for canonical Signal reasons such as insufficient non-overlapping realized labels or invalid canonical inputs; diagnostic-only validation is not allowed to create a new metric-only failure path.
- Paired fresh-run computation may still fail if diagnostic construction fails. A fresh maintained research run requires the paired sidecar by design.
- A sidecar whose outer `artifact_digest` is self-consistent but whose persisted prediction/model RMSE state does not reproduce `forecast_digest` must fail closed.
- Existing corrupt/stale/wrong-run/wrong-path/wrong-contract/pair-drift behavior remains fail closed.
- Partial resume on a numerically divergent backend continues to fail closed instead of overwriting the valid persisted member.

## Quality contract

### Acceptance Criteria

1. `signal_scope_builder` is statically typed to return `CausalAlphaV3SignalScopeBuild`; the old metric-only builder is rejected by static type checking in a representative assignment/wiring test or equivalent Mypy-verifiable fixture.
2. `build_causal_alpha_v3_signal_scope_metric(...)` no longer invokes diagnostic construction.
3. For the same inputs and numerical environment, the metric-only API and paired API return exactly equal canonical metric payloads/digests.
4. The same-runner pre-sidecar vs hardened-tree canonical metric payload/digest oracle remains byte-identical.
5. The maintained runner/pipeline still uses the paired producer, so fresh maintained runs cannot silently omit sidecars.
6. Paired fresh computation still performs one shared fit/forecast per scope subject to the existing fit cache; the refactor does not introduce a second fit or forecast pass.
7. Strict sidecar validation reconstructs the canonical forecast from persisted sidecar values and requires exact `forecast_digest` equality.
8. A sidecar with internally modified predictions/RMSE and a recomputed outer artifact digest, but stale `forecast_digest`, is rejected.
9. Existing pair identity, resume, corrupt artifact, cross-run, cross-contract, wrong-path, future-label, Gate-leakage, and raw-feature-exclusion tests remain Green.
10. Signal Gate receives only canonical `CausalAlphaV3SignalScopeMetric` objects and all Gate numerics/pass-fail semantics remain unchanged.
11. No selection, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, execution, or promotion logic changes.
12. PR #413 returns to Ready only after exact-final-HEAD targeted tests, Mypy, import architecture, full pytest/coverage, compatibility/build/package checks, applicable PostgreSQL workflow, self-review, and falsification review succeed.

### Invariants

- Run manifest is the run identity authority.
- Canonical Signal V2 metric remains the only Signal Gate evidence authority.
- Diagnostic evidence remains `research_only=true` and `promotion_eligible=false`.
- Metric-only API does not depend on diagnostic construction.
- Paired maintained producer always returns and persists both canonical metric and diagnostic sidecar for a newly built scope.
- One chronological interval remains one independent Signal episode.
- No future label outside the contract is serialized as realized evidence.
- No raw feature matrix or Teacher-admission holdout evidence is serialized.
- The existing per-scope sidecar schema/path remains unchanged in this hardening.

### Failure Modes

Explicitly test or re-verify:

- metric-only builder accidentally calls diagnostic builder;
- paired and metric-only canonical metric drift;
- incompatible metric-only builder accepted at typed pipeline boundary;
- recursive/circular imports introduced by protocol placement;
- diagnostic refactor causes a second fit or forecast pass;
- self-consistent outer sidecar with forged forecast observations accepted under stale `forecast_digest`;
- persisted prediction row order misused during forecast reconstruction;
- residual RMSE from the wrong horizon used in forecast reconstruction;
- Gate evaluator receives paired/diagnostic objects;
- resume semantics regress when only one paired member exists;
- corrupt diagnostic gets silently rebuilt/overwritten;
- sidecar validation starts depending on raw feature data or historical refit;
- architecture contract break inside existing macro-layer rules.

### Risk

Primary risk is a regression in canonical Signal computation caused by refactoring shared ephemeral state. Impact is high because Signal sits upstream of candidate freeze, selection, Teacher admission, and all downstream learner work.

Secondary risk is over-coupling strict diagnostic validation to backend-sensitive values. The forecast reconstruction is safe because it validates persisted values against the persisted forecast digest; it does not assert cross-host reproduction from a new model fit.

### Test Oracle

Correctness is observed through:

- Mypy-visible incompatibility of the legacy metric-only builder with the paired builder protocol;
- call-spy proving metric-only API does not invoke diagnostic construction;
- exact metric payload/digest equality between metric-only and paired paths;
- same-runner old-tree vs final-tree canonical metric byte comparison;
- fit-cache/producer call counts proving no additional fit pass;
- strict parser/contract rejection of forged forecast state with recomputed outer artifact digest;
- Gate spy receiving exact metric objects only;
- persisted file bytes and builder-call counts across all resume states;
- unchanged final Gate evidence/pass-fail for controlled fixtures.

### Required Test Layers

- Unit tests for typed contract-adjacent helpers and forecast reconstruction.
- Regression tests for metric-only/paired exact equivalence and diagnostic independence.
- Integration tests for runner/pipeline typed seam and paired resume.
- Falsification tests for forged forecast state and diagnostic-to-Gate leakage.
- Mypy, Ruff, format, import architecture, dead-code analysis.
- Full pytest with branch coverage and critical coverage ratchet.
- Ubuntu/Windows compatibility, Training image, package/uv identity.
- PostgreSQL Catalog if triggered by the exact final diff/workflow path filters; otherwise record as not applicable.

### Quality Gate

Do not mark the hardening complete unless:

- all Acceptance Criteria are mapped to observable evidence;
- no canonical Signal numerical drift is observed;
- the typed seam catches the previously observed wrong-builder class of defect;
- metric-only execution is demonstrably independent of diagnostic construction;
- forged forecast evidence is rejected fail closed;
- targeted and full verification both pass on the exact final HEAD;
- architecture/self-review finds no new circular or responsibility coupling;
- falsification review is rebuilt from this specification rather than implementation assumptions;
- PR head, CI head, and reported final HEAD are identical;
- remaining limitations and unverified items are explicitly reported.

## Deferred follow-ups

The following remain deliberately deferred until fresh real sidecar evidence shows they are worth the complexity:

- fit-level diagnostic artifact normalization and symbol-sidecar deduplication;
- per-decision feature missingness bitsets;
- independently reconstructable diagnostic model-state digest;
- broader extraction of Signal scope computation out of `universal_causal_alpha_v3_teacher.py`;
- extension of PR #411 forensics to consume valid fresh sidecars.
