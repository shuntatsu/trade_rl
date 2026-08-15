# Causal Alpha V3 Full Research Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an artifact-bound Causal Alpha V3 real-data workflow from train-only selection through teacher admission and conditional anchored residual RL.

**Architecture:** Add an opt-in research runner beside canonical U6. Reuse maintained Universal runtime assembly, chronological contracts, production replay, economic gates, BC, and telemetry; introduce V3-specific immutable candidates, evidence, batch generation, and stage orchestration without modifying canonical configs.

**Tech Stack:** Python 3.12, NumPy, Gymnasium/SB3, Docker, pytest, Ruff, Mypy.

## Global Constraints

- Reward remains pure net log growth.
- Hard market-notional liquidity risk remains `0.02`.
- Selection uses train-symbol selection contracts only.
- Teacher-admission holdouts are untouched until one candidate is frozen.
- Validation/test/sealed data never tune V3.
- Historical diagnostics are never promotable or resumable as V3 evidence.
- Admission failure blocks every learning stage.
- Canonical Universal configs remain byte-for-byte unchanged.

---

### Task 1: V3 candidate and evidence contracts

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_contracts.py`

**Interfaces:**
- Produces `CausalAlphaV3CandidateConfig`, `CausalAlphaV3EpisodeMetric`,
  `CausalAlphaV3SelectionEvidence`, `CausalAlphaV3TeacherAdmissionEvidence`, and
  `UniversalCausalAlphaV3TeacherPackage`.

- [ ] Write failing tests for digest closure, immutable arrays, duplicate scopes,
  mixed identities, promotion disabled, and admission consistency.
- [ ] Run the focused tests and verify the missing module failure.
- [ ] Implement the immutable contracts and canonical payloads.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit the task.

### Task 2: Expanding V3 fit, forecast, and target batch

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_batch.py`

**Interfaces:**
- Produces `CausalAlphaV3FitCache.resolve(knowledge_cutoff, config)` and
  `build_causal_alpha_v3_contract_targets(...)`.

- [ ] Write failing tests proving cutoff-safe fits, shared fit/prediction cache
  reuse, 24h-equivalent forecasts, causal liquidity caps, and compiler evidence.
- [ ] Run focused tests and verify RED.
- [ ] Implement expanding fit/cache and per-contract targets using existing V3
  primitives and maintained cost/liquidity functions.
- [ ] Run V3, legacy fitting, Ruff, format, and Mypy checks.
- [ ] Commit the task.

### Task 3: Resumable production selection

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_selection.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_selection.py`

**Interfaces:**
- Produces `evaluate_causal_alpha_v3_selection(...)`, checkpoint reader/writer,
  deterministic ranking, and irreversible-rejection evidence.

- [ ] Write failing tests for exact replay scopes, checkpoint resume, mixed
  generator/config rejection, paired candidate coverage, and economic gates.
- [ ] Run focused tests and verify RED.
- [ ] Implement candidate replay with production environments and atomic progress.
- [ ] Persist selection or complete rejection evidence before returning.
- [ ] Run selection/monitor/Ruff/format/Mypy checks.
- [ ] Commit the task.

### Task 4: Exact-once admission and package

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_teacher.py`

**Interfaces:**
- Produces `build_universal_causal_alpha_v3_teacher_package(...)`.

- [ ] Write failing tests proving selection precedes holdout access, each holdout
  is replayed once, admission persists before package return, and failure blocks
  consumers.
- [ ] Run focused tests and verify RED.
- [ ] Assemble samples/partitions, run selection, freeze targets, replay holdouts,
  and persist the V3 package.
- [ ] Run teacher/admission/legacy package/Ruff/format/Mypy checks.
- [ ] Commit the task.

### Task 5: Research CLI and runtime preflight

**Files:**
- Create: `scripts/run_universal_causal_alpha_v3_research.py`
- Create: `examples/binance-multitimeframe/universal-causal-alpha-v3-research.json`
- Test: `tests/scripts/test_run_universal_causal_alpha_v3_research.py`

**Interfaces:**
- CLI consumes runtime manifest, frozen metadata, V3 config, output root, and
  stage limit; it emits canonical progress and terminal artifacts.

- [ ] Write failing CLI tests for manifest/config identity, stage ordering,
  rejection exit, resume, and no canonical config mutation.
- [ ] Run tests and verify RED.
- [ ] Implement runtime assembly and selection/admission stage execution.
- [ ] Add generation heartbeat fields for fit/cache/replay/resource progress.
- [ ] Run CLI, contract, Ruff, format, and Mypy checks.
- [ ] Commit the task.

### Task 6: Admission-gated DAgger and anchored RL comparison

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_learning.py`
- Extend: `scripts/run_universal_causal_alpha_v3_research.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_learning.py`

**Interfaces:**
- Produces same-segment random/teacher/BC/DAgger-BC/critic/RL stage evidence.

- [ ] Write failing tests proving failed admission never constructs learning,
  DAgger steps learner actions, anchored zero residual reproduces teacher, and
  stage metrics share one segment identity.
- [ ] Run tests and verify RED.
- [ ] Connect existing DAgger, BC, critic warm-start, anchored residual PPO,
  Lagrangian, and discounted PPO primitives behind admission.
- [ ] Persist reward/gross/net/baseline/cost/turnover/drawdown/action/dual trends.
- [ ] Run learning/integration/Ruff/format/Mypy checks.
- [ ] Commit the task.

### Task 7: Exact-head verification and real-data execution

**Files:**
- Modify: `report/universal-real-data-training-2026-08-12.md`

- [ ] Verify canonical config hashes are unchanged.
- [ ] Run targeted tests, full pytest, Ruff/format, Mypy, and architecture checks.
- [ ] Build a provenance-bound non-root training image.
- [ ] Run real-data V3 selection to terminal evidence while monitoring OOM,
  fit/cache counts, gross/net, turnover, cost, and lower-tail.
- [ ] Run teacher admission only if selection passes.
- [ ] Run DAgger/anchored RL only if admission passes.
- [ ] Update the report with branch/commit/image/digests and economic results.
- [ ] Commit and push final evidence.
