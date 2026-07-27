# Integrated C3 Execution and Reporting Lane

## Context

The C3 evaluation core is now authoritative on `main`. It already owns immutable contracts, decision persistence, realized comparison, aggregate reports, Phase A gate evaluation, and canonical `report.json` / `gate.json` artifacts. Lane C must not duplicate those models or calculations.

## Scope

Lane C owns only:

- request-driven orchestration over published walk-forward, C1, and C2 artifacts;
- source-bound adverse evidence loading and binding;
- the lightweight CLI surface;
- deterministic Markdown read-model publication from verified core artifacts;
- a manual-only GPU evidence workflow;
- operations and verification documentation.

## Non-goals

- Reimplement report or gate calculations.
- Add training, Serving, promotion, release, direct exchange routing, or Studio write paths.
- Trust request-provided support counts or adverse pass booleans.
- Introduce market-data cache or training-performance changes.

## Authoritative core boundary

Lane C consumes only the public evaluation APIs:

- `write_c3_aggregate_report_artifact` / `load_c3_aggregate_report_artifact`;
- `write_phase_a_gate_artifact` / `load_phase_a_gate_artifact`;
- `evaluate_phase_a_entry_gate`;
- C3 decision, prediction, replay, report, gate, and adverse-evidence contracts.

The canonical core report and gate artifacts remain the source of truth. Markdown is derivative and must bind both artifact digests.

## Request lifecycle

`execute_c3_evaluation_request(request_path, output_root=...)`:

1. Requires canonical JSON with schema `causal_scenario_c3_evaluation_request_v2`.
2. Resolves only safe, non-symlink relative paths under the request directory.
3. Validates the published walk-forward run and loads `walk-forward.json` plus `walk-forward-config.json`.
4. Recomputes and verifies the walk-forward config digest against the run manifest.
5. Loads source-bound adverse evidence from `execution-sensitivity.json`.
6. Derives fold selection support and required adverse evidence from the source run. The request cannot supply either value.
7. Loads frozen C2 libraries and C1 value artifacts, persists decisions before realized replay, reconstructs immutable replay outcomes, and builds batch queries.
8. Requires nominal and the source-declared adverse scenario for every fold.
9. Calls `execute_c3_batch`, which publishes the authoritative report and gate artifacts.
10. Writes a deterministic Markdown read-model artifact bound to those artifacts.

Production status remains `NO-GO` regardless of gate result.

## Markdown read model

The Markdown artifact contains exactly:

- `manifest.json` — schema, report artifact digest, gate artifact digest, report digest, gate digest, Markdown SHA-256, artifact digest, and `NO-GO`;
- `report.md` — deterministic human-readable content rendered from the verified report and gate.

Loaders reject extra files, symlinks, non-canonical manifests, digest mismatches, report/gate binding mismatches, and conflicting rewrites.

## CLI

```text
trade-rl causal-scenario evaluate --request REQUEST --output OUTPUT
trade-rl causal-scenario publish --report REPORT --gate GATE --output MARKDOWN
trade-rl causal-scenario verify --report REPORT --gate GATE
```

All outcomes are one-line JSON. Expected validation failures return exit code 1 and one-line JSON on stderr. Commands remain lightweight and do not import the SB3 training runtime.

## GPU workflow

The GPU workflow is `workflow_dispatch` only, pinned to immutable third-party action revisions, runs on the existing self-hosted NVIDIA runner labels, executes the same CLI entrypoint, uploads the complete evidence directory, and contains no promotion, release, Serving, or scheduled execution.

## Integration safety

Lane C may merge independently after lane A and B. A passing Phase A gate authorizes only the next research phase. Production remains `NO-GO`.