# Causal Scenario C3 Execution and Reporting

This runbook covers the evaluation-only lane C surface. The authoritative machine-readable evidence remains the C3 core `report.json` and `gate.json` artifacts. The Markdown artifact is a derivative read model bound to both core artifact digests.

Every command in this document remains `NO-GO` for production. A passing Phase A gate authorizes only the next research phase and does not authorize production.

## Input requirements

The request is canonical JSON using schema `causal_scenario_c3_evaluation_request_v2`. It points only to safe relative paths under the request directory.

The source walk-forward run must contain and validate:

- `manifest.json` and its complete artifact closure;
- `walk-forward.json`;
- `walk-forward-config.json`, whose digest must equal the manifest workflow-config digest;
- `execution-sensitivity.json`, used to derive source-bound adverse evidence.

The request does not contain `selection_days` or `required_adverse_passed`. Those values are derived from the validated source run. Every requested fold must include `nominal` and the source-declared required adverse execution scenario.

## Full evaluation

```bash
uv run trade-rl causal-scenario evaluate \
  --request var/c3-request.json \
  --output var/c3-run
```

The command validates the source run, C2 libraries, C1 value artifacts, persisted decisions, prediction evidence, replay outcomes, and source-bound adverse evidence. It then executes the authoritative C3 batch workflow.

The output contains:

- `decisions/<decision-digest>/...` — immutable decision evidence;
- `report/report.json` — authoritative C3 aggregate report artifact;
- `gate/gate.json` — authoritative Phase A gate artifact;
- `markdown/manifest.json` and `markdown/report.md` — deterministic read model bound to the report and gate artifacts.

## Publish Markdown from existing evidence

```bash
uv run trade-rl causal-scenario publish \
  --report var/c3-run/report \
  --gate var/c3-run/gate \
  --output var/c3-run/markdown
```

Both core artifacts are fully loaded and verified before Markdown is written. Extra files, symbolic links, non-canonical manifests, digest substitutions, report/gate identity mismatches, and conflicting rewrites are rejected.

## Verify core evidence

```bash
uv run trade-rl causal-scenario verify \
  --report var/c3-run/report \
  --gate var/c3-run/gate
```

The command emits one-line JSON containing the report artifact digest, gate artifact digest, report digest, gate digest, pass state, failed conditions, and `production_status: NO-GO`.

## GPU execution

The `Causal Scenario C3 GPU Evidence` workflow is manual-only and runs on a self-hosted runner labelled `linux`, `x64`, `gpu`, and `nvidia`.

Before dispatching:

1. Place the canonical request and all referenced source/C1/C2 artifacts at repository-relative paths.
2. Confirm the runner can execute `nvidia-smi` and install the frozen `uv.lock` environment.
3. Dispatch from `main` as the repository owner.

The workflow runs the same `trade-rl causal-scenario evaluate` command used locally and uploads the complete output directory as `c3-execution-evidence`. Missing GPU capability, malformed or substituted evidence, incomplete replay outcomes, or absent output fails the workflow.

## Interpretation

`phase_a_authorized` means the authoritative nine-condition gate passed. It does not authorize model promotion, release approval, Serving packaging, direct exchange execution, or automatic trading. Production remains `NO-GO` until later evaluation and release processes independently approve it.
