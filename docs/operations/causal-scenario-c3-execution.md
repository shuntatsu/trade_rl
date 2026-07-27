# Causal Scenario C3 Execution and Reporting

This runbook covers the evaluation-only lane C boundary. Every command and artifact in this document remains `NO-GO` for production. A passing Phase A gate authorizes only the next research evaluation phase and does not authorize production.

## Commands

### Publish an aggregate summary

Use this while lane B is being developed or when its output has already been copied into a strict aggregate summary file.

```bash
uv run trade-rl causal-scenario publish \
  --summary var/c3-core/summary.json \
  --output var/c3-evidence
```

The command validates the complete summary, evaluates all nine Phase A conditions, and creates both report and gate artifacts.

### Re-evaluate and publish a gate

```bash
uv run trade-rl causal-scenario gate \
  --report var/c3-evidence/report \
  --output var/c3-gate
```

The report artifact is fully re-verified before the gate artifact is written.

### Run the full evaluation boundary

```bash
uv run trade-rl causal-scenario evaluate \
  --request var/c3-request.json \
  --output var/c3-run
```

The full command lazily imports `trade_rl.workflows.causal_scenario.c3_core.execute_c3_core_request`. Until lane B provides that function, the command fails with `C3CoreBackendUnavailable`. This is intentional fail-closed behavior, not a fallback to synthetic or incomplete evidence.

## Artifact closure

The report directory contains exactly:

- `manifest.json`: artifact identity, source identities, file digests, gate digest, and `NO-GO` status.
- `summary.json`: canonical lane B aggregate summary.
- `report.md`: deterministic human-readable report derived from the same summary and gate.

The gate directory contains exactly:

- `manifest.json`: artifact identity, report artifact binding, gate digest, and `NO-GO` status.
- `gate.json`: canonical condition-by-condition Phase A evidence.

Extra files, subdirectories, symbolic links, non-canonical JSON, digest substitutions, and conflicting rewrites are rejected. Rewriting byte-identical evidence is idempotent.

## GPU execution

The `Causal Scenario C3 GPU Evidence` workflow is manual-only and runs on a self-hosted runner labelled `linux`, `x64`, `gpu`, and `nvidia`.

Before dispatching it:

1. Merge lane B so the C3 core backend exists.
2. Place the request JSON at the repository-relative path supplied as `request_path`.
3. Confirm the runner can execute `nvidia-smi` and install the frozen `uv.lock` environment.
4. Dispatch the workflow from `main` as the repository owner.

The workflow runs the same `trade-rl causal-scenario evaluate` command used locally and uploads the requested output directory as `c3-reporting-evidence`. Missing CUDA capability, a missing request, a missing backend, malformed evidence, or absent output fails the workflow.

## Interpretation

`phase_a_authorized` means all nine evidence conditions passed. It does not authorize production, model promotion, release approval, Serving packaging, direct exchange execution, or automatic trading. Production remains `NO-GO` until the later evaluation and release processes independently approve it.
