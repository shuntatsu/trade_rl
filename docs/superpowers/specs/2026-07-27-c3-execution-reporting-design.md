# C3 Execution and Reporting Lane Design

## Context

PR #196 combines the C3 evaluation algorithm, execution entrypoints, report publication, gate evidence, GPU workflow assets, and unrelated training-performance changes. This lane extracts only the execution and reporting surface so it can be reviewed and merged independently from the fold-local evaluation core.

## Goals

- Provide a stable, evaluation-only CLI entrypoint for C3 requests.
- Keep the C3 evaluation core behind one explicit backend contract.
- Accept a strict aggregate summary document produced by the core lane.
- Publish deterministic JSON, Markdown, and Phase A gate evidence.
- Fail closed with `production_status = "NO-GO"` when the core backend is absent, malformed, or inconsistent.
- Provide an opt-in GPU workflow that executes only through the same CLI entrypoint.
- Preserve exact file closure, canonical JSON, content digests, and idempotent artifact writes.

## Non-goals

- Implement fold-local scenario selection, replay, realized PnL comparison, bootstrap statistics, or decision construction. Those belong to lane B.
- Change training, Serving, promotion, release, or direct execution paths.
- Add Studio UI. The deterministic report artifact is the future Studio read model boundary.
- Carry over market-data cache or training-performance changes from PR #196.

## Architecture

### 1. Core backend boundary

`trade_rl.workflows.causal_scenario.c3_execution` owns the execution lifecycle. It resolves one optional backend function:

```python
def execute_c3_core_request(
    request_path: Path,
    *,
    output_root: Path,
) -> Path:
    """Return the path to a strict C3 aggregate summary JSON document."""
```

The default implementation is imported lazily from `trade_rl.workflows.causal_scenario.c3_core`. Until lane B provides that module, `evaluate` fails closed with a machine-readable error. Tests inject a backend explicitly; production code never silently substitutes fake evidence.

### 2. Aggregate summary read model

`trade_rl.evaluation.causal_scenario_c3_reporting` defines an immutable `C3AggregateSummary` and focused nested records. The summary contains only reporting and gate inputs, not raw candidate arrays or replay internals.

The loader requires:

- schema `causal_scenario_c3_aggregate_summary_v1`;
- exact top-level and nested field closure;
- finite numeric values;
- SHA-256 identities for source run, core report, and config;
- unique, sorted fold and execution-summary identities;
- coherent support counts, confidence intervals, drawdowns, quantiles, and scenario coverage;
- `production_status = "NO-GO"`;
- a `summary_digest` equal to the digest of the canonical payload excluding that field.

### 3. Pure Phase A gate

The gate evaluates the approved nine conditions using only the aggregate summary and a frozen `C3PhaseAGateConfig`:

1. integrity and diagnostics complete;
2. required fold and day support;
3. positive uplift in at least four folds, bounded by required fold count;
4. positive lower confidence bound for aggregate uplift;
5. worst scenario-oracle drawdown no greater than 20% and no more than two percentage points above trend;
6. positive lower confidence bound for realized regret improvement over random;
7. positive mean and lower confidence bound for predicted-realized ranking;
8. perfect-information compatibility valid for every asserted comparison;
9. nominal plus at least one required adverse execution scenario, all passed.

The gate never changes production status. A passing Phase A gate authorizes only the next evaluation phase.

### 4. Deterministic artifacts

The report artifact directory contains exactly:

- `manifest.json` — schema, file digests, source identities, summary digest, artifact digest, and `NO-GO` status;
- `summary.json` — canonical aggregate summary;
- `report.md` — deterministic human-readable report generated from the same summary and gate.

The gate artifact directory contains exactly:

- `manifest.json` — file digest, report artifact identity, gate digest, artifact digest, and `NO-GO` status;
- `gate.json` — canonical condition evidence.

Writers are atomic and idempotent. Rewriting identical content succeeds; conflicting existing content fails.

### 5. CLI

The lightweight dispatcher recognizes:

```text
trade-rl causal-scenario evaluate --request REQUEST.json --output OUTPUT_DIR
trade-rl causal-scenario publish --summary SUMMARY.json --output OUTPUT_DIR
trade-rl causal-scenario gate --report REPORT_ARTIFACT --output GATE_ARTIFACT
```

All success and failure results are one-line JSON. Failures use exit code 1 and include `error`, `error_type`, `schema`, `status`, and `production_status`.

`evaluate` calls the backend, validates the returned summary, publishes the report and gate artifacts, and prints their identities. `publish` is the deterministic lane boundary used to validate lane B output independently. `gate` re-verifies an existing report artifact before publishing gate evidence.

### 6. GPU workflow

`.github/workflows/causal-scenario-c3-gpu.yml` is manual-only. It accepts request and output paths, runs the same CLI command inside the existing training image contract, uploads the report/gate artifacts, and does not promote or release models. Missing GPU capability or missing core backend fails the workflow rather than downgrading execution.

## Error handling

- Every parser rejects unknown and missing fields.
- Paths returned by the core backend must resolve inside the supplied core output root.
- Artifact directories reject symlinks, subdirectories, and extra files.
- Invalid digests, non-canonical JSON, conflicting idempotent writes, and backend import failures are hard failures.
- No failure path emits a production authorization.

## Testing

- Unit tests cover strict summary parsing, digest checks, invariants, gate conditions, Markdown determinism, artifact closure, and idempotence.
- Workflow tests inject a backend to prove request-to-artifact execution and verify fail-closed behavior without lane B.
- CLI tests assert machine-readable success and failure output.
- Asset tests parse the GPU workflow and prove it is manual-only, calls the CLI, uploads evidence, and contains no promotion or release step.
- Full repository CI remains the integration gate.

## Integration order

1. Merge lane A independently.
2. Merge this lane C into main with the backend unavailable but safely fail-closed.
3. Lane B adds `trade_rl.workflows.causal_scenario.c3_core.execute_c3_core_request` and tests the same summary contract.
4. Run the manual GPU workflow and record verification evidence before changing `NO-GO`.