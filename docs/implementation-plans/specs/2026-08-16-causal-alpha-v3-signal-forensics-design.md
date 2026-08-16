# Causal Alpha V3 Signal Forensics Design

## Objective

Add a deterministic, read-only diagnostic path for completed or rejected Causal Alpha V3 Signal runs so existing `signal/records/**/*.json` artifacts can be decomposed by fit, chronological episode, and symbol without rebuilding datasets, refitting ridge models, replaying environments, or changing any gate.

The immediate target is a local run such as `var/causal-alpha-v3-post-v2-full`, where Signal Contract V2 completed full raw coverage but no fit cleared the Signal Gate.

## Non-goals

- Do not change Signal Gate thresholds, bootstrap semantics, candidate ordering, or pass/fail state.
- Do not change ridge fitting, labels, horizon blending, target compilation, economic selection, Teacher admission, BC, critic warm start, PPO, reward, risk, or execution.
- Do not make a rejected run resumable, promotable, or production eligible.
- Do not infer 24h-only/72h-only metrics or coefficient stability from hashes when the persisted artifacts do not contain the underlying values.
- Do not duplicate the general deterministic run-report work in PR #410; this feature is Signal-specific research diagnostics.

## Alternatives considered

1. **Extend PR #410 reporting.** Rejected because that PR is a stage-status/reporting abstraction, while this work needs model-diagnostic statistics and would create unnecessary coupling/conflict.
2. **Revise Signal leaf schema to persist richer predictions/fits.** Useful for a future run, but it cannot retroactively diagnose the existing post-V2 run and would invalidate/rebuild artifacts. Not part of this change.
3. **Read-only forensic analyzer over current artifacts.** Chosen. It extracts every diagnostic that is actually present, validates identity strictly, and explicitly reports unavailable analyses instead of fabricating them.

## Inputs and trust boundary

Required input root:

- `run-manifest.json`
- `authored-config.json`
- `signal/records/<fit_config_digest>/<symbol>/<episode>.json`

Optional but validated when present:

- `signal/rejection.json`

The analyzer parses Signal leaves through the maintained `signal_scope_metric_from_payload` contract and parses the V2 run manifest through `CausalAlphaV3RunManifestV2.from_payload`.

It rejects:

- missing required run metadata;
- malformed JSON;
- stale/unsupported Signal leaf schema;
- artifact digest mismatch;
- run-manifest digest drift;
- path identity drift;
- duplicate `(fit_config_digest, symbol, episode_index)` records;
- symbol scope outside the run manifest;
- incomplete cross-symbol chronological clusters;
- more than one `fit_digest` inside a chronological episode cluster;
- rejection evidence whose fit/result metric digests disagree with the loaded leaves.

The analyzer never writes inside the source run root. `--output` may write a separate report path chosen by the caller.

## Output contract

Schema: `causal_alpha_v3_signal_forensics_v1`.

The report is content-addressed and non-promotable. It contains:

- source run-manifest digest and config digest;
- train symbols and observed fit-config digests;
- total raw record count and chronological episode count;
- one `fit_summary` per fit config;
- one `episode_summary` per fit and chronological interval;
- one `symbol_summary` per fit and symbol;
- fit-digest transition/uniqueness diagnostics across chronological episodes;
- explicit `unavailable_analyses` entries for information not represented by V2 persisted leaves.

### Fit summary

For each fit config:

- raw scope count;
- independent episode count;
- symbol count;
- sample-count min/mean/max;
- raw-scope mean/std/min/max and negative fraction for rank IC, top-bottom spread, and direction-accuracy excess;
- chronological-episode mean series for the same metrics;
- early-half versus late-half means and least-squares slope across chronological episode order;
- number of distinct pooled `fit_digest` values and number of transitions between consecutive episodes.

The trend values are descriptive only. They do not re-run or reinterpret the Signal Gate.

### Episode summary

For each chronological episode cluster:

- interval start/stop;
- representative episode index;
- shared pooled `fit_digest`;
- symbol count;
- total and mean cohort sample count;
- cross-symbol mean rank IC;
- cross-symbol mean top-bottom spread;
- cross-symbol mean direction accuracy and direction-accuracy excess;
- counts of symbols with negative rank, spread, and direction excess.

### Symbol summary

For each fit/symbol:

- episode count;
- sample-count min/mean/max;
- mean/std/min/max and negative fraction for rank IC, top-bottom spread, and direction-accuracy excess.

## Explicitly unavailable analyses for existing Signal V2 leaves

The current persisted Signal V2 metric contains a blended forecast digest but not separate 24h and 72h predictions/realized diagnostics, and it contains `fit_digest` but not ridge coefficients. Therefore the report must state these as unavailable:

- `horizon_24h_vs_72h`: requires persisted per-horizon predictions/realized outcomes or a new diagnostic sidecar;
- `coefficient_cosine_similarity`: requires persisted model coefficients;
- `coefficient_sign_flip_rate`: requires persisted model coefficients;
- `prediction_distribution`: requires persisted prediction values/summary diagnostics;
- `residual_rmse_by_episode`: requires persisted fit payload/diagnostic sidecar.

A future instrumentation change may add sidecars, but this analyzer must remain truthful for existing runs.

## CLI

Add `scripts/analyze_universal_causal_alpha_v3_signal.py`:

```text
python scripts/analyze_universal_causal_alpha_v3_signal.py \
  var/causal-alpha-v3-post-v2-full \
  --output var/causal-alpha-v3-post-v2-full-signal-forensics.json
```

With no `--output`, canonical compact JSON is written to stdout. With `--output`, the source run remains untouched and the report is written to the requested path.

## Acceptance criteria

1. A complete synthetic V2 run produces deterministic fit/episode/symbol summaries from persisted leaves only.
2. Chronological episode aggregation is cross-symbol mean and does not count symbol copies as independent episodes.
3. Direction accuracy is reported both as raw accuracy and excess over 0.5.
4. Early/late and slope diagnostics are descriptive and deterministic.
5. The report explicitly marks unsupported 24h/72h and coefficient analyses unavailable.
6. Corrupt leaf digest, wrong run identity, wrong path identity, duplicate scope, incomplete cluster, mixed cluster fit identity, or inconsistent rejection evidence fails closed.
7. No source artifact is modified.
8. Existing Signal Gate, selection, admission, learning, reward, risk, and execution numerical behavior is unchanged.
9. Exact final HEAD passes targeted tests, Ruff, format, Mypy, import architecture, vulture, full pytest/coverage, compatibility/build jobs, and any required database workflow, or unavailable checks are reported as unverified rather than assumed successful.

## Invariants

- The run manifest remains the run identity authority.
- Signal V2 leaf parser remains the leaf schema/digest authority.
- One chronological interval is one independent episode regardless of symbol count.
- All symbols in a chronological episode cluster share one pooled fit digest.
- Diagnostics are read-only, research-only, and never promotion eligible.
- The analyzer never changes or weakens a gate to make a historical run look better.

## Failure modes and test oracle

Failure modes include silent use of stale/corrupt artifacts, accidental independent-sample inflation, averaging mismatched fits, missing-symbol clusters being treated as complete, report non-determinism, and invented diagnostics from unavailable data.

The test oracle observes strict exceptions for invalid evidence; exact summary values for controlled synthetic leaves; equality of repeated report payloads/digests; source-file bytes unchanged before/after analysis; and explicit unavailable-analysis reasons.

## Required test layers

- Unit/contract tests for summary math and validation.
- Integration-style filesystem tests for realistic `run-manifest.json`, `authored-config.json`, Signal leaf paths, and rejection evidence.
- CLI smoke test.
- Ruff and format check.
- Mypy.
- Import architecture.
- 100%-confidence vulture scan.
- Full pytest/coverage.
- Repository CI compatibility/build/database checks on exact final HEAD.
