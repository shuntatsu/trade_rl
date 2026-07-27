# C3 Execution and Reporting Verification

## Scope

This record covers lane C only:

- source-bound request orchestration over the merged C3 evaluation core;
- authoritative report and Phase A gate artifact publication;
- deterministic Markdown read-model evidence;
- lightweight `evaluate`, `publish`, and `verify` CLI commands;
- manual-only NVIDIA GPU workflow assets.

The change does not implement or modify C3 scoring, realized comparison, fold aggregation, Phase A gate mathematics, training, Serving, promotion, release, or direct execution.

## Required verification gates

The exact pull-request head must pass:

- Ruff and formatter checks;
- Mypy;
- import architecture and dead-code diagnostics;
- full pytest and coverage;
- critical branch coverage;
- CLI smoke tests;
- Ubuntu and Windows compatibility suites;
- Studio fixed-viewport regression checks;
- complete training-image build and packaged non-root runtime probe.

## Security and evidence invariants

- Request JSON and walk-forward configuration must be canonical.
- Request paths must remain inside the request root and contain no symbolic-link components.
- Request fold identities must be valid SHA-256 digests and match the frozen C1 value artifacts.
- Fold support and required adverse status are derived from the validated source run, never accepted from request self-reporting.
- `report.json` and `gate.json` from the merged C3 core are authoritative.
- `report.md` is derivative and binds the report and gate artifact digests.
- Workflow execution is manual-only and third-party actions are pinned by full commit SHA.
- A passing Phase A gate authorizes only the next research phase.
- Production status remains `NO-GO`.

## Exact-head evidence

The final commit SHA, workflow run IDs, test count, coverage, and all job conclusions are recorded here only after the immutable PR head completes every required check.
