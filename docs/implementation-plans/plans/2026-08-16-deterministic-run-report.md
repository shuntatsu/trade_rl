# Deterministic Run Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a paste-ready research/training report from persisted artifacts without LLM analysis, interpretation, or threshold re-evaluation.

**Architecture:** Read immutable/derived run artifacts into one versioned `RunReport` data contract, then render that contract deterministically as JSON or Markdown. V3 Signal/Selection/Teacher Admission are collected from their maintained artifact graph; downstream BC/critic/PPO/zero-shot/sealed stages are present in the same schema and can consume optional strict stage-evidence files without changing the reporter.

**Tech Stack:** Python 3.12, dataclasses, pathlib/json, existing V3 artifact contracts, pytest, Ruff, Mypy, GitHub Actions.

## Global Constraints

- No LLM/API call or natural-language interpretation in report generation.
- Do not recompute Signal/Selection/Admission thresholds or rank candidates again; persisted evidence is authoritative.
- Do not mutate training/research artifacts.
- Fail closed on malformed known artifacts; represent the affected stage as `INVALID` instead of silently skipping it.
- Distinguish `IN_PROGRESS`, `MISSING`, and `NOT_RUN`; absence alone must not be reported as a successful/failed stage.
- Preserve deterministic ordering and serialization.
- No changes to reward, risk, execution, teacher, BC, critic, PPO, or promotion semantics.

## Quality Contract

### Objective

Provide one machine-generated report that can be pasted into a chat or stored as audit evidence without an LLM inventing analysis around the raw run data.

### Non-goals

- No recommendation about what experiment to run next.
- No profitability, alpha-quality, or Production GO inference.
- No migration of legacy artifacts into current evidence.
- No new training-stage persistence contract beyond an optional generic read-only stage-evidence adapter.

### Acceptance Criteria

- `RunReport` has a fixed schema version and fixed ordered stages: `signal`, `selection`, `teacher_admission`, `teacher_package`, `behavior_cloning`, `critic_warm_start`, `ppo`, `zero_shot`, `sealed_evaluation`.
- Stage status is one of `PASS`, `REJECT`, `IN_PROGRESS`, `NOT_RUN`, `MISSING`, `INVALID`.
- V3 run/execution identities are validated with maintained `from_payload` contracts and cross-checked against each other.
- Signal fit-result/rejection artifacts are summarized from persisted evidence only.
- Selection terminal evidence/rejection or `selection/progress.json` is reflected deterministically, including candidate and symbol aggregates when progress exists.
- Admission evidence/rejection is reflected deterministically; downstream stages are `NOT_RUN` only when an upstream persisted rejection proves they were blocked.
- Teacher package presence is represented independently from Teacher Admission.
- Optional `reporting/stages/<stage>.json` files use a strict generic stage-evidence schema and can populate downstream stage metrics.
- Missing downstream evidence after a passed upstream stage is `MISSING`, not guessed as `NOT_RUN`.
- JSON output is canonical/deterministic for the same artifacts.
- Markdown output contains only report facts/tables/identities/statuses, not recommendation or diagnosis prose.
- CLI supports `--root`, `--profile chat|json`, and `--output PATH|-`.

### Invariants

- Reporter is read-only.
- Existing artifact digests and stage decisions remain the sole source of truth.
- Candidate order and symbol order are preserved from persisted progress/evidence where available.
- Re-running the reporter does not change the source artifact tree.

### Failure Modes

Malformed JSON, stale schema, digest mismatch, run/execution identity mismatch, terminal evidence contradicting rejection marker, progress outside expected schema, generic downstream stage evidence with invalid status/schema/metrics, partial run with no terminal evidence, upstream rejection followed by impossible downstream PASS evidence.

### Test Oracle

Observe the exact `RunReport.to_payload()` structure, stage statuses, copied metrics/reasons/digests, deterministic serialization, Markdown tables, CLI exit code/output, and source-tree immutability in tests.

### Required Test Layers

Unit/contract tests for collector and renderer, CLI integration tests, static checks, import architecture, full pytest/coverage, compatibility/build CI.

### Quality Gate

Do not mark complete until the exact final HEAD has targeted/full tests, Ruff/format, Mypy, import architecture, coverage gates, compatibility/build checks, and required GitHub Actions successful; then perform a fresh falsification review against the quality contract.

---

### Task 1: Report contract and deterministic rendering

**Files:**
- Create: `trade_rl/reporting/__init__.py`
- Create: `trade_rl/reporting/run_report.py`
- Create: `trade_rl/reporting/markdown.py`
- Test: `tests/reporting/test_run_report.py`

**Interfaces:**
- Produces `RunStageStatus`, `RunStageReport`, `RunReport`, `render_run_report_markdown(report)`.

- [ ] Write RED tests for fixed stage ordering/status validation/deterministic payload and Markdown.
- [ ] Run exact-head CI and confirm failure because the reporting package does not exist.
- [ ] Implement the minimal immutable contracts and pure renderer.
- [ ] Verify targeted tests green.

### Task 2: V3 artifact collector

**Files:**
- Modify: `trade_rl/reporting/run_report.py`
- Test: `tests/reporting/test_run_report.py`

**Interfaces:**
- Produces `build_run_report(root: Path) -> RunReport`.

- [ ] Add tests using synthetic V3 artifact trees for Signal reject, Selection in-progress/terminal, Admission reject/pass, and identity corruption.
- [ ] Confirm RED before collector implementation.
- [ ] Implement strict JSON loading, maintained V3 identity validation, status propagation, and progress/evidence extraction.
- [ ] Keep persisted evidence fields as-is; do not re-evaluate thresholds or re-rank candidates.
- [ ] Verify targeted tests green.

### Task 3: Generic downstream stage evidence

**Files:**
- Modify: `trade_rl/reporting/run_report.py`
- Test: `tests/reporting/test_run_report.py`

**Interfaces:**
- Consumes optional `reporting/stages/{behavior_cloning,critic_warm_start,ppo,zero_shot,sealed_evaluation}.json`.
- Schema: `run_report_stage_evidence_v1` with `stage`, `status`, `metrics`, `reasons`, `artifact_digests`.

- [ ] Add strict-schema and downstream-population RED tests.
- [ ] Implement exact-field validation and status consistency.
- [ ] Reject downstream PASS if a persisted upstream rejection proves the stage was blocked.
- [ ] Verify targeted tests green.

### Task 4: CLI

**Files:**
- Create: `scripts/build_run_report.py`
- Test: `tests/scripts/test_build_run_report.py`

**Interfaces:**
- `uv run python scripts/build_run_report.py --root ROOT --profile chat --output -`
- `--profile json` emits the deterministic JSON payload.

- [ ] Add RED tests for stdout/file output and invalid-root exit behavior.
- [ ] Implement CLI without network/LLM dependencies.
- [ ] Verify targeted tests green.

### Task 5: Documentation and full verification

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Test: `tests/test_causal_alpha_v3_documentation_contract.py`

- [ ] Document that the report is program-generated, read-only, and non-interpretive.
- [ ] Document CLI examples and generic downstream stage-evidence location.
- [ ] Run targeted reporting tests.
- [ ] Run Ruff, format, Mypy, import architecture, dead-code check, full pytest/coverage, compatibility, training-image/package identity, and PostgreSQL Catalog as applicable on the exact final HEAD.
- [ ] Re-read final diff for training/economic semantic drift.
- [ ] Falsification review malformed/cross-run/partial/contradictory artifact cases before completion claim.
