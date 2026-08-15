# Causal Alpha V3 Signal Contract V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous V3 signal-scope semantics with an auditable V2 contract that separates raw records from independent chronological episodes, validates shared chronology, binds runtime provenance, and resumes from strict signal leaf artifacts.

**Architecture:** Keep the existing V3 research stage ordering and clustered bootstrap numerics. Bump authored config, signal metric/evidence, and execution identity schemas where semantics change; make `(contract_start, contract_stop)` the independence key; validate shared chronology before fit; and make persisted signal leaves the resume source of truth while recomputing derived aggregate evidence.

**Tech Stack:** Python 3.12, NumPy, dataclasses, canonical content digests, strict JSON artifacts, pytest, Ruff, Mypy, import-linter, GitHub Actions.

## Global Constraints

- Do not change ridge fitting, forecast formulas, target controller semantics, label formulas, economic selection, admission, reward, risk, execution, DAgger, BC, critic warm start, PPO, or Lagrangian learning.
- Do not weaken signal-quality lower-CI thresholds or raw coverage.
- Same-episode symbol duplication must not increase independent evidence.
- V3 remains `research_only=true` and `promotion_eligible=false`.
- Old V1 authored signal semantics must not be silently reinterpreted as V2.
- All new behavior follows RED -> GREEN -> refactor and exact-final-HEAD verification.

---

### Task 1: Authored Signal Gate V2 semantics

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_config.py`
- Modify: `examples/binance/universal-causal-alpha-v3-research.json`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_config.py`
- Modify: `tests/test_causal_alpha_v3_runner_example_contract.py`

**Interfaces:**
- Produces `CausalAlphaV3SignalGate.minimum_independent_episode_count: int`.
- Produces `CausalAlphaV3SignalGate.minimum_raw_scope_coverage: float`.
- Authored schema is `universal_causal_alpha_v3_research_config_v2`.

- [ ] Add a failing test that loads a V1 payload and expects `ValueError("unsupported causal alpha V3 research config schema")`.
- [ ] Add a failing test that V2 `minimum_independent_episode_count=3` with `signal_contract_count=2` is rejected before runner execution.
- [ ] Add a failing example-contract assertion requiring `minimum_independent_episode_count == signal_contract_count == 8` and `minimum_raw_scope_coverage == 1.0`.
- [ ] Run the targeted config/example tests and record RED caused by the missing V2 field/schema behavior.
- [ ] Bump the schema constant to V2, rename the two fields in the dataclass/parser/authored payload, and retain the explicit cross-field invariant.
- [ ] Update the maintained example to V2 field names with `8` independent episodes and `1.0` raw coverage.
- [ ] Re-run targeted tests and confirm GREEN.

### Task 2: Signal leaf and aggregate Evidence V2

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_v2.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_signal.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_architecture_hardening.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_falsification.py`

**Interfaces:**
- `CausalAlphaV3SignalScopeMetric` persists `contract_start` and `contract_stop` under schema `causal_alpha_v3_signal_scope_v2`.
- `CausalAlphaV3SignalGateEvidence` persists raw and independent counts under schema `causal_alpha_v3_signal_gate_evidence_v2`.
- `evaluate_causal_alpha_v3_signal_gate_clustered(metrics, *, expected_raw_scope_count, expected_independent_episode_count, gate)` is the only current aggregate evaluator.

- [ ] Add a failing strict round-trip test requiring `contract_start`/`contract_stop` and rejecting a V1 signal-scope payload.
- [ ] Add a failing test with two metrics sharing `episode_index=0` but using different `(contract_start, contract_stop)` intervals; assert they count as two independent episodes rather than one.
- [ ] Add a failing test that duplicate symbols inside one interval increase `raw_scope_count` but not `independent_episode_count`.
- [ ] Add a failing evidence-payload test requiring `independence_unit="chronological_episode"`, `aggregation_mode="cross_symbol_episode_mean"`, explicit raw counts, and explicit independent counts.
- [ ] Run targeted signal tests and record RED.
- [ ] Add interval fields to the leaf metric and populate them from `OracleEpisodeContract` in `build_causal_alpha_v3_signal_scope_metric`.
- [ ] Change clustered grouping from `episode_index` to `(contract_start, contract_stop)` sorted by start/stop.
- [ ] Rename gate checks/rejection reasons to `independent_episode_count` and `raw_scope_coverage`; keep bootstrap inputs/numerics unchanged for aligned metrics.
- [ ] Remove the flat aggregate evaluator from the current Signal API so it cannot produce V2 aggregate evidence; retain only leaf/partition/non-overlap contracts in `signal.py`.
- [ ] Re-run targeted tests and confirm GREEN, including a regression that aligned existing metric fixtures preserve bootstrap numeric outputs.

### Task 3: Shared chronology and runtime provenance closure

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runtime.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_identity.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_architecture_hardening.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_falsification.py`
- Add/modify focused runtime tests under `tests/workflows/` as needed.

**Interfaces:**
- `CausalAlphaV3ExecutionIdentity` schema becomes `causal_alpha_v3_execution_identity_v2`.
- Adds `shared_clock_digest`, `dependency_lock_digest`, and `python_runtime_digest`.
- Runtime preparation fails when train-symbol timestamp arrays, partition schedules, or decision cadence differ.

- [ ] Add a failing identity round-trip test requiring the three new provenance digests and rejecting old V1 identity payloads.
- [ ] Add a failing runtime-preparation test with two train-symbol environments whose timestamp arrays differ by one bar; expect failure before fit construction.
- [ ] Add a failing runtime-preparation test whose episode contract schedules differ despite valid per-symbol partitions; expect failure before signal evaluation.
- [ ] Add a failing test that changed dependency-lock or Python-runtime digest changes execution identity and prevents same-output-root manifest reuse.
- [ ] Run targeted runtime/identity tests and record RED.
- [ ] Compute `shared_clock_digest` with `content_and_arrays_digest` over canonical `datetime64[ns]` timestamps and require equality across train symbols.
- [ ] Require identical chronological `(start, stop)` schedules and one common `decision_bars` value across train symbols.
- [ ] Compute `dependency_lock_digest` from source-checkout `pyproject.toml` and `uv.lock` file digests.
- [ ] Compute `python_runtime_digest` from Python implementation plus exact major/minor/micro version.
- [ ] Add the new fields to execution identity serialization/strict loader and ensure the run-manifest digest changes through `execution_identity_digest`.
- [ ] Re-run targeted tests and confirm GREEN.

### Task 4: Signal leaf resume wiring

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_pipeline.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py` only if a focused loader interface is required.
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py`
- Modify/add integration tests under `tests/workflows/`.

**Interfaces:**
- Pipeline constructs one complete expected signal identity map for all representative fit digests/symbols/signal contracts.
- Calls `load_signal_scope_metrics(expected=...)` once after immutable run identity is closed.
- Reuses valid leaves and invokes `signal_scope_builder` only for missing identities.

- [ ] Add a failing integration test that pre-populates one valid signal leaf and asserts the signal builder is not called for that identity.
- [ ] Add a failing partial-resume test with one persisted leaf and one missing leaf; assert exactly the missing identity is built and both contribute to aggregate evidence.
- [ ] Add a failing corruption/unknown-leaf test proving the pipeline fails closed rather than silently overwriting or ignoring the bad leaf.
- [ ] Run targeted orchestration/resume tests and record RED.
- [ ] Build the complete expected identity map before iterating fits and load validated existing leaves once.
- [ ] Select persisted metrics by identity; build/write only missing metrics; keep the existing identity checks on newly built metrics.
- [ ] Pass `expected_raw_scope_count` and `expected_independent_episode_count` explicitly to the V2 evaluator.
- [ ] Recompute aggregate fit evidence from the complete validated leaf set every run.
- [ ] Re-run targeted tests and confirm GREEN.

### Task 5: Architecture closure and migration documentation

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: architecture tests under `tests/workflows/`.
- Modify: `docs/implementation-plans/specs/2026-08-15-causal-alpha-v3-architecture-hardening-design.md` only to clarify the new V2 unit names if needed.
- Modify: PR #407 description.

**Interfaces:**
- Runner exposes only `evaluate_causal_alpha_v3_signal_gate_clustered` as the current gate.
- No current code path imports/calls a flat raw-record aggregate evaluator.

- [ ] Add/adjust an architecture test proving the runner current gate is the clustered V2 evaluator and the legacy flat evaluator is absent from current execution wiring.
- [ ] Search the repository for `minimum_scope_count`, `minimum_scope_coverage`, and legacy config schema references; every remaining occurrence must be an explicit migration/history assertion, not active V2 behavior.
- [ ] Search the final diff for learner/economic/reward/risk/execution changes and remove any unrelated change.
- [ ] Run targeted V3 config/signal/runtime/pipeline tests.
- [ ] Run Ruff, Ruff format, Mypy, import-linter, dead-code, compatibility, training-image probe, full pytest with branch coverage, critical branch coverage, package/uv identity, and PostgreSQL Catalog on the exact final HEAD.
- [ ] Perform falsification review against all 12 failure modes in the design spec.
- [ ] Update PR #407 with RED evidence, exact final HEAD verification, migration notes, and remaining statistical limitations; keep it Draft until every quality gate is observed.