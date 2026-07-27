# Integrated C3 Execution and Reporting Implementation Plan

> **Required process:** implement with TDD and verify the exact PR head before merge.

**Goal:** Add request-driven C3 orchestration, lightweight CLI, deterministic Markdown read model, and a manual GPU workflow without duplicating the merged C3 evaluation core.

## Task 1: Batch workflow over authoritative core

- Create `trade_rl/workflows/causal_scenario/c3.py`.
- Add focused workflow tests.
- Use `C3AdverseFoldEvidence`, core report/gate builders, and core artifact writers directly.
- Reject duplicate decision/scenario pairs, identity mismatches, and incomplete fold mappings.

## Task 2: Published walk-forward request lifecycle

- Create `trade_rl/workflows/causal_scenario/c3_evaluation.py`.
- Add request lifecycle tests.
- Require canonical schema v2 and safe non-symlink relative paths.
- Validate `walk-forward-config.json` against the manifest digest.
- Load adverse evidence from the source run and derive fold support/adverse bindings.
- Require nominal plus the source-required adverse scenario for every fold.
- Persist decisions before realized replay and publish core artifacts via the batch workflow.

## Task 3: Markdown read-model artifact

- Create `trade_rl/evaluation/causal_scenario_c3_markdown.py`.
- Add deterministic rendering and artifact-closure tests.
- Bind the authoritative report and gate artifact digests.
- Reject substitutions, extra files, symlinks, and conflicting rewrites.

## Task 4: CLI

- Create `trade_rl/cli/causal_scenario.py` and update the lightweight dispatcher.
- Add `evaluate`, `publish`, and `verify` tests.
- Keep outputs one-line JSON and production status `NO-GO`.
- Prove commands do not import the SB3 training runtime.

## Task 5: GPU workflow and runbook

- Add `.github/workflows/causal-scenario-c3-gpu.yml`.
- Add `docs/operations/causal-scenario-c3-execution.md`.
- Add static workflow tests for manual-only execution, pinned actions, GPU labels, evidence upload, and absence of promotion/release/Serving paths.

## Task 6: Verification and integration

- Run focused tests, Ruff, formatter, Mypy, import architecture, full pytest/coverage, critical coverage, CLI smoke, Ubuntu/Windows compatibility, and training-image probe.
- Record exact head, CI run, counts, coverage, and known limitation that real evidence still determines Phase A status.
- Close old PR #224 as superseded.
- Mark the replacement PR ready and squash merge only after all exact-head checks pass.
