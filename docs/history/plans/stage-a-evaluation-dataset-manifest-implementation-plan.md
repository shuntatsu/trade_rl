# Stage A Evaluation Dataset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global Stage A dataset identity with an immutable PostgreSQL-backed evaluation dataset manifest that binds every split/triplet/fold cell to the exact dataset and scored range used by execution evidence.

**Architecture:** Introduce one workflow-level manifest that owns source closure, triplet-to-symbol-to-dataset bindings, and fold ranges. Stage A plan v3 binds only the manifest digest plus feature/execution/evaluation identities; cell requests, observations, replay identities, execution artifacts, schedules, and access records carry the resolved triplet dataset ID and exact `IndexRange`. PostgreSQL construction resolves symbols and ranges only from the manifest inputs and fails closed on drift.

**Tech Stack:** Python 3.12+, frozen dataclasses, canonical content digests, NumPy market datasets, PostgreSQL integration adapters, pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Do not preserve the ambiguous pre-v3 Stage A plan/request/observation/evidence schemas.
- Do not let callers provide arbitrary symbols, dataset IDs, or evaluation ranges after a manifest is sealed.
- Preserve the full common dataset timeline for warm-up; bind scoring to an explicit half-open `IndexRange`.
- Validation uses each fold's `configuration_selection` range; sealed test uses each fold's `test` range.
- Policy and baseline executions for one split/triplet/fold/seed cell must share manifest, dataset, and range identities.
- SB3 environment assembly, durable PostgreSQL one-shot persistence, and final CLI wiring remain outside this lane.

---

### Task 1: Immutable Evaluation Dataset Manifest

**Files:**
- Create: `trade_rl/workflows/stage_a_evaluation_dataset_manifest.py`
- Create: `tests/workflows/test_stage_a_evaluation_dataset_manifest.py`

**Interfaces:**
- Consumes: `IndexRange`, symbol-disjoint manifest digests, triplet IDs, symbols, real dataset IDs.
- Produces: `StageAEvaluationDatasetTriplet`, `StageAEvaluationDatasetFold`, `StageAEvaluationDatasetManifest`, strict JSON load/write functions, `triplet_for()`, `range_for()`, and `dataset_id_for()`.

- [ ] Write failing tests for strict schema, digest closure, split/triplet uniqueness, symbol closure, dataset IDs, fold closure, non-overlapping ordered ranges, lookup behavior, and legacy schema rejection.
- [ ] Run the focused test file and verify RED failures are caused by the missing module.
- [ ] Implement the minimal frozen dataclasses and strict JSON codec.
- [ ] Run the focused test file and verify GREEN.
- [ ] Refactor validation helpers without changing behavior.

### Task 2: PostgreSQL-backed Manifest Construction

**Files:**
- Create: `trade_rl/workflows/stage_a_postgres_evaluation_dataset.py`
- Modify: `trade_rl/integrations/postgres_indicator_artifacts.py`
- Modify: `trade_rl/integrations/postgres_market_dataset.py`
- Create: `tests/workflows/test_stage_a_postgres_evaluation_dataset.py`
- Modify: `tests/integrations/test_postgres_market_dataset.py`

**Interfaces:**
- Consumes: `SymbolDisjointTripletManifest`, declared walk-forward folds, source metadata/evidence, execution-rule histories, PostgreSQL connection.
- Produces: `StageAPostgresEvaluationDatasets` containing the immutable manifest and one verified `MarketDataset` per validation/test triplet.

- [ ] Write failing tests proving callers cannot substitute symbols/ranges, all triplets share one timeline, dataset identities bind triplet provenance, and fold ranges are copied exactly from maintained folds.
- [ ] Verify focused RED failures.
- [ ] Add immutable indicator-bundle subset support that preserves requested symbol order and artifact evidence.
- [ ] Implement one full-timeline dataset build per declared validation/test triplet and verify each result against the manifest entry.
- [ ] Verify focused GREEN tests and existing PostgreSQL dataset tests.

### Task 3: Stage A Plan and Evidence v3

**Files:**
- Modify: `trade_rl/evaluation/_stage_a_zero_shot_contract_helpers.py`
- Modify: `trade_rl/evaluation/_stage_a_zero_shot_contract_values.py`
- Modify: `trade_rl/evaluation/_stage_a_zero_shot_plan.py`
- Modify: `trade_rl/evaluation/_stage_a_zero_shot_evidence.py`
- Modify: `trade_rl/evaluation/_stage_a_zero_shot_contract_io.py`
- Modify: `trade_rl/evaluation/stage_a_zero_shot_contracts.py`
- Modify: `tests/evaluation/test_stage_a_zero_shot_contracts.py`
- Modify: `tests/evaluation/test_stage_a_zero_shot_hardening.py`
- Modify: `tests/evaluation/test_stage_a_zero_shot_gate.py`

**Interfaces:**
- Consumes: `evaluation_dataset_manifest_digest`, per-observation `dataset_id`, and `evaluation_range`.
- Produces: plan/observation/evidence v3 contracts that reject pre-v3 JSON and validate every observation through a supplied manifest.

- [ ] Update tests first to require plan v3 and observation/evidence v3 fields and to reject legacy payloads.
- [ ] Verify RED failures.
- [ ] Replace `dataset_identity` on the plan with `evaluation_dataset_manifest_digest`.
- [ ] Replace observation `dataset_identity` with `evaluation_dataset_manifest_digest`, `dataset_id`, and `evaluation_range`.
- [ ] Make evidence construction validate split/triplet/fold closure against the manifest.
- [ ] Update strict JSON codecs and exports.
- [ ] Verify focused GREEN tests.

### Task 4: Cell Request, Test Schedule, and Orchestrator v2

**Files:**
- Modify: `trade_rl/workflows/stage_a_zero_shot_runner_contracts.py`
- Modify: `trade_rl/workflows/stage_a_zero_shot_runner.py`
- Modify: `trade_rl/workflows/stage_a_zero_shot_artifacts.py`
- Modify: `tests/workflows/test_stage_a_zero_shot_runner_contracts.py`
- Modify: `tests/workflows/test_stage_a_zero_shot_runner.py`
- Modify: `tests/workflows/test_stage_a_zero_shot_artifacts.py`

**Interfaces:**
- Consumes: one verified `StageAEvaluationDatasetManifest`.
- Produces: request v2 and schedule v2 resolved only from the manifest; Stage A-specific access records binding manifest, triplet dataset, and fold range.

- [ ] Update tests first for request/schedule/access-record v2 and exact manifest resolution.
- [ ] Verify RED failures.
- [ ] Bind the orchestrator constructor to both plan and manifest and require digest/closure equality.
- [ ] Resolve each request's dataset and range from `manifest.triplet_for()` and `manifest.range_for()`.
- [ ] Build observations from policy and baseline requests only after their manifest/dataset/range identities match.
- [ ] Derive the test schedule from the manifest and emit Stage A-specific access records.
- [ ] Verify focused GREEN tests.

### Task 5: Execution and Policy Source Identity Migration

**Files:**
- Modify: `trade_rl/workflows/stage_a_execution_observation.py`
- Modify: `trade_rl/workflows/stage_a_execution_producer.py`
- Modify: `trade_rl/workflows/stage_a_execution_replay.py`
- Modify: `trade_rl/workflows/stage_a_execution_store.py`
- Modify: `trade_rl/workflows/stage_a_policy_source.py`
- Modify: `trade_rl/workflows/stage_a_production_evaluator.py`
- Modify: relevant `tests/workflows/test_stage_a_*` files.

**Interfaces:**
- Consumes: request v2 identities.
- Produces: execution-cell/replay/source/evaluator paths that validate manifest digest, real dataset ID, and exact range and cannot be replayed under another cell.

- [ ] Update execution tests first to require manifest/dataset/range closure.
- [ ] Verify RED failures.
- [ ] Migrate candidate/runtime validation from plan-global dataset identity to request+manifest validation.
- [ ] Upgrade execution-cell and replay schemas and strict loaders.
- [ ] Bind event/evidence dataset IDs to `request.dataset_id` and include the range/manifest in replay identity.
- [ ] Verify focused GREEN tests across producer, replay, store, policy source, and production evaluator.

### Task 6: Regression Closure and Documentation

**Files:**
- Modify: remaining Stage A tests and fixtures.
- Modify: `docs/operations/stage-a-zero-shot-orchestrator-design.md`
- Modify: `docs/operations/stage-a-production-evaluator-design.md`
- Modify: `docs/operations/stage-a-production-evaluator-plan.md`

**Interfaces:**
- Consumes: completed v3/v2 contracts.
- Produces: repository-wide closure with no legacy Stage A dataset identity use.

- [ ] Search production code for Stage A `dataset_identity` references and remove all obsolete uses.
- [ ] Run all Stage A evaluation/workflow tests.
- [ ] Run PostgreSQL integration tests.
- [ ] Run Ruff, Ruff format check, MyPy, Import Linter, and full pytest with coverage.
- [ ] Inspect the final diff against the approved design and remove temporary or unrelated files.
