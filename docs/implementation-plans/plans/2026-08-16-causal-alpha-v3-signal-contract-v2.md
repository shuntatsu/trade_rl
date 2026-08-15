# Causal Alpha V3 Signal Contract V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous V3 signal-scope semantics with an auditable V2 contract that separates raw records from independent chronological episodes, validates shared chronology, binds runtime provenance, and resumes from strict run-bound signal leaf artifacts.

**Architecture:** Keep the existing V3 research stage ordering and clustered bootstrap numerics. Bump authored config, signal metric/evidence, and execution identity schemas where semantics change; make `(contract_start, contract_stop)` the independence key; validate shared chronology before fit; and make persisted run-bound signal leaves the resume source of truth while recomputing derived aggregate evidence.

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

- [x] Add a failing test that loads a V1 payload and expects `ValueError("unsupported causal alpha V3 research config schema")`.
- [x] Add a failing test that V2 `minimum_independent_episode_count=3` with `signal_contract_count=2` is rejected before runner execution.
- [x] Add a failing example-contract assertion requiring `minimum_independent_episode_count == signal_contract_count == 8` and `minimum_raw_scope_coverage == 1.0`.
- [x] Run the targeted config/example tests and record RED caused by the missing V2 field/schema behavior.
- [x] Bump the schema constant to V2, rename the two fields in the dataclass/parser/authored payload, and retain the explicit cross-field invariant.
- [x] Update the maintained example to V2 field names with `8` independent episodes and `1.0` raw coverage.
- [x] Re-run through the exact-head CI sequence after downstream migrations.

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

- [x] Add a failing strict round-trip test requiring `contract_start`/`contract_stop` and rejecting a V1 signal-scope payload.
- [x] Add a failing test with two metrics sharing `episode_index=0` but using different `(contract_start, contract_stop)` intervals; assert they count as two independent episodes rather than one.
- [x] Add a failing test that duplicate symbols inside one interval increase `raw_scope_count` but not `independent_episode_count`.
- [x] Add a failing evidence test requiring `independence_unit="chronological_episode"`, `aggregation_mode="cross_symbol_episode_mean"`, explicit raw counts, and explicit independent counts.
- [x] Run targeted/CI validation and record RED from the old field/signature contract.
- [x] Add interval fields to the leaf metric and populate them from `OracleEpisodeContract` in `build_causal_alpha_v3_signal_scope_metric`.
- [x] Change clustered grouping from `episode_index` to `(contract_start, contract_stop)` sorted by start/stop.
- [x] Rename gate checks/rejection reasons to `independent_episode_count` and `raw_scope_coverage`; keep bootstrap inputs/numerics unchanged for aligned metrics.
- [x] Remove the flat aggregate evaluator from the current Signal API so it cannot produce V2 aggregate evidence; retain only leaf/partition/non-overlap contracts in `signal.py`.
- [x] Re-run through downstream exact-head verification.

### Task 3: Shared chronology and runtime provenance closure

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runtime.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_identity.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_architecture_hardening.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_falsification.py`
- Add: `tests/workflows/test_universal_causal_alpha_v3_runtime_contract.py`.

**Interfaces:**
- `CausalAlphaV3ExecutionIdentity` schema becomes `causal_alpha_v3_execution_identity_v2`.
- Adds `shared_clock_digest`, `dependency_lock_digest`, and `python_runtime_digest`.
- Runtime preparation fails when train-symbol timestamp arrays, partition schedules, or decision cadence differ.

- [x] Add failing identity/runtime tests requiring the three new provenance digests and rejecting old V1 identity payloads.
- [x] Add failing shared-chronology tests for timestamp, episode schedule, and decision-cadence drift.
- [x] Add a failing test that changed dependency lock changes its digest.
- [x] Run exact-head validation and record RED from the missing chronology/provenance APIs.
- [x] Compute `shared_clock_digest` with `content_and_arrays_digest` over canonical `datetime64[ns]` timestamps and require equality across train symbols.
- [x] Require identical chronological `(episode_index, start, stop)` schedules and one common `decision_bars` value across train symbols.
- [x] Compute `dependency_lock_digest` from source-checkout `pyproject.toml` and `uv.lock` file digests.
- [x] Compute `python_runtime_digest` from Python implementation plus exact major/minor/micro version.
- [x] Add the new fields to execution identity serialization/strict loader and ensure the run-manifest digest changes through `execution_identity_digest`.
- [x] Exact-head CI `a426c50dbf5d132684d6001d24f5c6dc4330d180` passed Ruff, format, Mypy, import architecture, dead-code, recovery smoke, full pytest/coverage, critical coverage, package identity, Windows/Ubuntu compatibility, and training-image/non-root probe.

### Task 4: Run-bound Signal leaf resume wiring

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_v2.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_pipeline.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_architecture_hardening.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_falsification.py`

**Interfaces:**
- Every `CausalAlphaV3SignalScopeMetric` persists a required `run_manifest_digest`.
- Every `CausalAlphaV3SignalGateEvidence` persists the common `run_manifest_digest` of all included leaves.
- The clustered evaluator rejects a metric set containing more than one run-manifest identity.
- `CausalAlphaV3ArtifactStore` rejects signal-leaf writes or loads whose run identity differs from the store's immutable `run_manifest_digest`.
- Pipeline constructs one complete expected signal identity map for all representative fit digests/symbols/signal contracts and calls `load_signal_scope_metrics(expected=...)` once after immutable run identity is closed.
- Valid run-bound leaves are reused; `signal_scope_builder` is invoked only for missing identities; aggregate gate evidence is always recomputed.

- [ ] Add a failing leaf/evidence contract test requiring `run_manifest_digest` and rejecting cross-run metric mixing.
- [ ] Add a failing artifact-store test that rejects a copied leaf whose `run_manifest_digest` differs from the active store.
- [ ] Add a failing integration test that runs once to persist a valid leaf, reruns the same root, and asserts the signal builder is not called for that identity.
- [ ] Add a failing partial-resume test that crashes after one of two leaves is persisted, then reruns and asserts exactly the missing identity is rebuilt and both leaves reach aggregate evaluation.
- [ ] Add a failing corruption test proving an invalid persisted leaf schema/digest fails closed before builder invocation.
- [ ] Run the Task-4 tests and record RED while current pipeline still recomputes leaves and leaves lack run binding.
- [ ] Add `run_manifest_digest` to leaf serialization/strict loader and builder input; add it to aggregate evidence and require one common run identity.
- [ ] Enforce signal-leaf run identity on artifact-store write and load.
- [ ] Build the complete expected identity map before iterating fits and load validated existing leaves once.
- [ ] Select persisted metrics by identity; build/write only missing metrics; keep all run/fit/symbol/episode/interval/contract identity checks on newly built and reused metrics.
- [ ] Recompute aggregate fit evidence from the complete validated leaf set every run.
- [ ] Re-run targeted tests and confirm GREEN.

### Task 5: Architecture closure and migration documentation

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: architecture tests under `tests/workflows/`.
- Modify: `docs/implementation-plans/specs/2026-08-15-causal-alpha-v3-architecture-hardening-design.md` to clarify the V2 unit/chronology/run-binding contract.
- Remove/supersede the old minimal scope-count plan if it contradicts the V2 architecture.
- Modify: PR #407 description.

**Interfaces:**
- Runner exposes only `evaluate_causal_alpha_v3_signal_gate_clustered` as the current gate.
- No current code path imports/calls a flat raw-record aggregate evaluator.

- [ ] Add/adjust an architecture test proving the runner current gate is the clustered V2 evaluator and the legacy flat evaluator is absent from current execution wiring.
- [ ] Search the final PR diff for active `minimum_scope_count`, `minimum_scope_coverage`, old Signal Evidence semantics, and legacy config-schema use; every remaining occurrence must be migration/history documentation only.
- [ ] Search the final diff for learner/economic/reward/risk/execution changes and remove any unrelated change.
- [ ] Run targeted V3 config/signal/runtime/pipeline tests.
- [ ] Run Ruff, Ruff format, Mypy, import-linter, dead-code, compatibility, training-image probe, full pytest with branch coverage, critical branch coverage, package/uv identity, and PostgreSQL Catalog on the exact final HEAD.
- [ ] Perform falsification review against all 13 failure modes in the design spec.
- [ ] Update PR #407 with RED evidence, exact final HEAD verification, migration notes, and remaining statistical limitations; keep it Draft until every quality gate is observed.
