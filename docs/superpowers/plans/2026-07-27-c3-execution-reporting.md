# C3 Execution and Reporting Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a main-mergeable C3 execution and reporting lane with a strict summary boundary, deterministic report/gate artifacts, machine-readable CLI, and manual GPU workflow.

**Architecture:** Lane C accepts a canonical aggregate summary JSON from an optional lane B backend. It validates the summary independently, evaluates the pure Phase A gate, writes closed deterministic artifacts, and fails with `NO-GO` when the backend or evidence is unavailable. The backend is loaded lazily so main remains importable and testable before lane B lands.

**Tech Stack:** Python 3.12, dataclasses, pathlib, canonical JSON helpers already in `trade_rl.artifacts`, pytest, GitHub Actions YAML.

## Global Constraints

- Production status is always exactly `NO-GO`.
- The lane must be mergeable into `main` before lane B exists.
- No training, Serving, promotion, release, or direct-execution path may import lane C.
- All artifact JSON is canonical, content-addressed, atomic, and exact-closure verified.
- Unknown or missing fields fail closed.
- GPU execution is manual-only and uses the same CLI path as local execution.

---

### Task 1: Strict aggregate summary contract and Phase A gate

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_c3_reporting.py`
- Test: `tests/evaluation/test_causal_scenario_c3_reporting.py`

**Interfaces:**
- Produces: `load_c3_aggregate_summary(path: str | Path) -> C3AggregateSummary`
- Produces: `evaluate_phase_a_gate(summary: C3AggregateSummary, *, config: C3PhaseAGateConfig | None = None) -> PhaseAGateEvidence`
- Produces: `render_c3_markdown(summary: C3AggregateSummary, gate: PhaseAGateEvidence) -> str`

- [ ] **Step 1: Write failing tests for strict parsing**

Cover a valid canonical summary, unknown fields, digest mismatch, non-finite values, duplicate fold IDs, unsorted execution summaries, invalid confidence intervals, invalid quantiles, and production status other than `NO-GO`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/evaluation/test_causal_scenario_c3_reporting.py`

Expected: collection/import failure because `trade_rl.evaluation.causal_scenario_c3_reporting` does not exist.

- [ ] **Step 3: Implement immutable records and strict loader**

Implement `C3FoldSummary`, `C3CalibrationBucketSummary`, `C3ExecutionSummary`, `C3AggregateSummary`, `C3PhaseAGateConfig`, `GateConditionResult`, and `PhaseAGateEvidence`. Compute digests with `content_digest` and validate SHA-256 values with `require_sha256`.

- [ ] **Step 4: Implement the nine-condition gate and deterministic Markdown renderer**

The Markdown must include identities, support, uplift/ranking/regret intervals, drawdown, scenario execution summaries, calibration/neighbor diagnostics, all gate conditions, failure reasons, and an explicit `Production status: NO-GO` line.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest -q tests/evaluation/test_causal_scenario_c3_reporting.py`

Commit: `feat: add strict C3 reporting contract`

---

### Task 2: Deterministic report and gate artifacts

**Files:**
- Modify: `trade_rl/evaluation/causal_scenario_c3_reporting.py`
- Test: `tests/evaluation/test_causal_scenario_c3_artifacts.py`

**Interfaces:**
- Consumes: `C3AggregateSummary`, `PhaseAGateEvidence`, `render_c3_markdown`
- Produces: `write_c3_report_artifact(root, summary, gate) -> LoadedC3ReportArtifact`
- Produces: `load_c3_report_artifact(root) -> LoadedC3ReportArtifact`
- Produces: `write_phase_a_gate_artifact(root, gate, *, report_artifact_digest) -> LoadedPhaseAGateArtifact`
- Produces: `load_phase_a_gate_artifact(root) -> LoadedPhaseAGateArtifact`

- [ ] **Step 1: Write failing artifact tests**

Assert exact file sets, canonical JSON bytes, digest verification, Markdown determinism, identical rewrite idempotence, conflicting rewrite failure, extra-file rejection, symlink rejection, and report-to-gate identity binding.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/evaluation/test_causal_scenario_c3_artifacts.py`

Expected: missing artifact functions.

- [ ] **Step 3: Implement atomic writers and loaders**

Report closure is exactly `manifest.json`, `summary.json`, and `report.md`. Gate closure is exactly `manifest.json` and `gate.json`. Use temporary files, flush, `os.fsync`, and `os.replace`.

- [ ] **Step 4: Run focused tests and commit**

Run: `pytest -q tests/evaluation/test_causal_scenario_c3_artifacts.py`

Commit: `feat: publish deterministic C3 evidence`

---

### Task 3: Execution boundary and fail-closed backend loading

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_execution.py`
- Modify: `trade_rl/workflows/causal_scenario/__init__.py`
- Test: `tests/workflows/test_causal_scenario_c3_execution_entrypoint.py`

**Interfaces:**
- Consumes backend signature: `execute_c3_core_request(request_path: Path, *, output_root: Path) -> Path`
- Produces: `execute_c3_evaluation_request(request_path, *, output_root, backend=None) -> C3ExecutionResult`

- [ ] **Step 1: Write failing workflow tests**

Cover injected backend success, backend output escaping its root, missing backend module, malformed summary, request/output path normalization, artifact publication, and retained `NO-GO` status even when the gate passes.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/workflows/test_causal_scenario_c3_execution_entrypoint.py`

Expected: missing execution module.

- [ ] **Step 3: Implement lazy backend resolution and lifecycle**

Use `importlib.import_module("trade_rl.workflows.causal_scenario.c3_core")` only inside the resolver. Wrap missing module/function as `C3CoreBackendUnavailable`. Require the returned summary path to be a regular non-symlink file inside the backend output root.

- [ ] **Step 4: Run focused tests and commit**

Run: `pytest -q tests/workflows/test_causal_scenario_c3_execution_entrypoint.py`

Commit: `feat: add C3 evaluation execution boundary`

---

### Task 4: Machine-readable CLI

**Files:**
- Create: `trade_rl/cli/causal_scenario.py`
- Modify: `trade_rl/cli/__init__.py`
- Test: `tests/cli/test_causal_scenario_c3_reporting_cli.py`

**Interfaces:**
- Produces commands: `evaluate`, `publish`, and `gate`

- [ ] **Step 1: Write failing CLI tests**

Assert one-line JSON success for all three commands, JSON failure on stderr with exit code 1, no RL runtime import for these commands, artifact paths/digests in output, failed-condition names, and `production_status = "NO-GO"`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/cli/test_causal_scenario_c3_reporting_cli.py`

Expected: dispatcher does not recognize the commands.

- [ ] **Step 3: Implement parsers and dispatch**

Use `argparse`, explicit required paths, lazy workflow imports, and stable result/error schemas. Never print Python tracebacks for expected validation failures.

- [ ] **Step 4: Run focused tests and commit**

Run: `pytest -q tests/cli/test_causal_scenario_c3_reporting_cli.py`

Commit: `feat: expose C3 evidence CLI`

---

### Task 5: Manual GPU workflow and operations documentation

**Files:**
- Create: `.github/workflows/causal-scenario-c3-gpu.yml`
- Create: `docs/operations/causal-scenario-c3-execution.md`
- Test: `tests/workflows/test_causal_scenario_c3_gpu_assets.py`

**Interfaces:**
- Consumes: `trade-rl causal-scenario evaluate`
- Produces: uploaded report/gate evidence artifact

- [ ] **Step 1: Write failing asset tests**

Parse the YAML as text/YAML and require `workflow_dispatch`, request/output inputs, GPU self-hosted runner labels, exact CLI invocation, artifact upload, and absence of schedule, push, pull-request, promotion, release, and Serving commands.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/workflows/test_causal_scenario_c3_gpu_assets.py`

Expected: workflow and operations document are missing.

- [ ] **Step 3: Add workflow and runbook**

The runbook documents local `publish` and `gate`, full `evaluate`, artifact closure, backend-unavailable behavior, GPU prerequisites, and the fact that a passing gate does not change production status.

- [ ] **Step 4: Run focused tests and commit**

Run: `pytest -q tests/workflows/test_causal_scenario_c3_gpu_assets.py`

Commit: `docs: add C3 GPU evidence workflow`

---

### Task 6: Full verification and PR handoff

**Files:**
- Create: `docs/verification/2026-07-27-c3-execution-reporting.md`

- [ ] **Step 1: Run focused lane tests**

Run:

```bash
pytest -q \
  tests/evaluation/test_causal_scenario_c3_reporting.py \
  tests/evaluation/test_causal_scenario_c3_artifacts.py \
  tests/workflows/test_causal_scenario_c3_execution_entrypoint.py \
  tests/cli/test_causal_scenario_c3_reporting_cli.py \
  tests/workflows/test_causal_scenario_c3_gpu_assets.py
```

- [ ] **Step 2: Run repository quality gates**

Run the repository CI equivalents: Ruff, formatting, Mypy, import architecture, full pytest/coverage, critical branch coverage, CLI smoke, Ubuntu/Windows compatibility, and training-image probe.

- [ ] **Step 3: Record exact evidence**

Write command, commit SHA, test counts, coverage, CI run ID, artifact names, and known limitation that lane B is not yet installed and therefore default `evaluate` intentionally fails closed.

- [ ] **Step 4: Review scope**

Confirm the diff contains no lane B algorithm, market-data cache, training-performance optimization, Studio UI, promotion, release, Serving, or direct execution changes.

- [ ] **Step 5: Mark PR ready only after all checks pass**

Commit: `docs: record C3 execution reporting verification`