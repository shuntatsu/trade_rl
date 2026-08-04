# Architecture 15-Loop Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove fifteen identified architectural boundary defects while preserving all maintained numerical and serialized behavior.

**Architecture:** Introduce small public contracts below their consumers, retain compatibility facades, and enforce semantic boundaries with AST-based regression tests. Separate recipe identity from run transport/provenance and move optional accelerator registration to explicit composition roots.

**Tech Stack:** Python 3.12, pytest, AST source inspection, Import Linter, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Keep `training_run_config_v4`, `sb3_policy_identity_v4`, `structured_policy_export_v2`, and `serving_bundle_v6` unchanged.
- Do not change Bellman numerical behavior, action/reward/execution semantics, or Production `NO-GO` status.
- Preserve compatibility imports for evaluation metrics/series and RL training helpers.
- Write regression tests before production changes.
- Run the focused architecture gate after every loop and the complete repository CI on the final head.

---

### Task 1: Public training-environment contract

**Files:**
- Create: `trade_rl/rl/training_environment_contract.py`
- Modify: `trade_rl/rl/training.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `training_environment_identity(environment: object) -> dict[str, Any]`
- Produces: `validate_training_environment(identity: dict[str, Any], config: ResidualTrainingConfig) -> None`

- [ ] Write the failing contract test for the public framework-neutral module.
- [ ] Run the focused architecture test and confirm failure because the module is absent.
- [ ] Move the identity/validation implementation into the public module and keep private aliases in `rl.training` for compatibility.
- [ ] Run the focused test and confirm the task contract passes.
- [ ] Commit the task.

### Task 2: Integration consumes only public RL contracts

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Consumes: `training_environment_identity`, `validate_training_environment`

- [ ] Add the failing import-boundary assertion.
- [ ] Confirm it reports `_environment_identity` and `_validate_training_environment`.
- [ ] Replace both private imports and call sites with the public functions.
- [ ] Run the focused gate.
- [ ] Commit the task.

### Task 3: Generic cross-package private-import guard

**Files:**
- Modify: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: a repository-wide AST rule rejecting `from trade_rl.<other-package>... import _private`.

- [ ] Add the failing repository scan before production fixes.
- [ ] Confirm each reported import is a real cross-package private dependency.
- [ ] Remove or publish every reported production dependency.
- [ ] Run the focused gate.
- [ ] Commit the task.

### Task 4: Generic catalog excludes sealed-evaluation reservation

**Files:**
- Modify: `trade_rl/catalog/postgres.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Keeps: `PostgresArtifactCatalog` implementing only `ArtifactCatalog` operations.

- [ ] Assert that `postgres.py` imports no evaluation module and defines no sealed reservation method.
- [ ] Confirm the current method/import fails the test.
- [ ] Remove the evaluation import and delegation method.
- [ ] Run catalog and focused architecture tests.
- [ ] Commit the task.

### Task 5: Public PostgreSQL connection utility

**Files:**
- Create: `trade_rl/catalog/postgres_connection.py`
- Modify: `trade_rl/catalog/postgres.py`
- Modify: `trade_rl/catalog/postgres_sealed_test.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `import_psycopg() -> Any | None`
- Produces: `default_connection_factory(database_url: str) -> Any`

- [ ] Assert both adapters consume the public utility and no adapter imports a private symbol from another module.
- [ ] Confirm the current `_default_connection_factory` import fails.
- [ ] Extract the utility and update both adapters.
- [ ] Run focused and PostgreSQL unit tests.
- [ ] Commit the task.

### Task 6: RL terminal info stops importing evaluation

**Files:**
- Modify: `trade_rl/rl/environment_info.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Consumes: `trade_rl.simulation.performance` compatibility-neutral metrics contract.

- [ ] Assert the RL module imports no `trade_rl.evaluation` package.
- [ ] Confirm the current metrics/series imports fail.
- [ ] Redirect the imports after Task 7 creates the lower contract.
- [ ] Run environment-info tests.
- [ ] Commit the task.

### Task 7: Performance contract below RL and evaluation

**Files:**
- Create: `trade_rl/simulation/performance.py`
- Replace: `trade_rl/evaluation/metrics.py` with a compatibility facade
- Replace: `trade_rl/evaluation/series.py` with a compatibility facade
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `ReturnKind`, `ReturnSeries`, `PerformanceMetrics`, `compound_return`, `evaluate_performance`

- [ ] Assert the lower module owns all performance types/functions and evaluation reexports them.
- [ ] Confirm the module is absent.
- [ ] Move the unchanged implementation and add explicit `__all__` facades.
- [ ] Run evaluation and RL environment tests.
- [ ] Commit the task.

### Task 8: Neutral maintained policy identifiers

**Files:**
- Create: `trade_rl/domain/policy_contracts.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `SB3_POLICY_IDENTITY_SCHEMA`, `HIERARCHICAL_SEQUENCE_ENCODER`, `STRUCTURED_TIMEFRAMES`

- [ ] Assert exact maintained values in a standard-library-only domain module.
- [ ] Confirm the module is absent.
- [ ] Create the constants and export them.
- [ ] Run domain and focused tests.
- [ ] Commit the task.

### Task 9: Structured export consumes neutral identifiers

**Files:**
- Modify: `trade_rl/artifacts/structured_policy_contract.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Consumes: the Task 8 constants.

- [ ] Assert no local schema/timeframe duplication remains.
- [ ] Confirm local constants fail the test.
- [ ] Import and use the neutral constants.
- [ ] Run artifact/serving tests.
- [ ] Commit the task.

### Task 10: RL policy modules consume neutral identifiers

**Files:**
- Modify: `trade_rl/rl/policy_identity.py`
- Modify: `trade_rl/rl/sequence_observations.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Consumes: the Task 8 constants.

- [ ] Assert policy identity and sequence window ordering are tied to the neutral contract.
- [ ] Confirm duplicated identifiers fail the test.
- [ ] Replace local values with imported constants without changing serialized values.
- [ ] Run policy-identity and sequence-observation tests.
- [ ] Commit the task.

### Task 11: Recipe identity excludes export transport

**Files:**
- Modify: `trade_rl/rl/training_run_config.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `_recipe_identity_payload()` used by `candidate_digest_payload()`.

- [ ] Assert the candidate recipe payload contains no export fields.
- [ ] Confirm the shared `_identity_payload` currently fails.
- [ ] Extract the recipe payload and exclude transport-only fields.
- [ ] Run training-run-config tests.
- [ ] Commit the task.

### Task 12: Recipe identity excludes source provenance

**Files:**
- Modify: `trade_rl/rl/training_run_config.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Keeps: Git commit/dirty state outside recipe identity.

- [ ] Assert `_recipe_identity_payload()` does not include `git_commit` or `git_dirty`.
- [ ] Confirm the pre-refactor shared payload fails.
- [ ] Keep provenance only in full run identity.
- [ ] Run digest identity tests.
- [ ] Commit the task.

### Task 13: Full run identity retains transport and provenance

**Files:**
- Modify: `trade_rl/rl/training_run_config.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `_run_identity_payload(resume_checkpoint_digests, transfer_checkpoint_digests)`.

- [ ] Assert full identity contains exports, Git state, resume, and transfer digests.
- [ ] Confirm the method is absent before refactoring.
- [ ] Extend the recipe payload in the full run method and route `digest_payload()` through it.
- [ ] Run all training config/checkpoint identity tests.
- [ ] Commit the task.

### Task 14: Explicit Oracle accelerator registration

**Files:**
- Modify: `trade_rl/integrations/__init__.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Produces: `register_default_oracle_accelerators() -> None`

- [ ] Assert importing `trade_rl.integrations` has no top-level registration call.
- [ ] Confirm the current module-level call fails.
- [ ] Add an explicit idempotent registration function and call it only when a non-NumPy Oracle solver is requested.
- [ ] Run Oracle solver/integration tests.
- [ ] Commit the task.

### Task 15: Documentation and CI contract alignment

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/architecture/test_architecture_15_loop.py`

**Interfaces:**
- Documents: `sb3_policy_identity_v4`
- CI: PR target `main`; Windows/Linux compatibility includes architecture contracts.

- [ ] Assert docs do not claim v1, stale PR base branches are absent, and compatibility runs architecture tests.
- [ ] Confirm all three mismatches fail.
- [ ] Update documentation and CI without changing privileged GPU policy.
- [ ] Run the focused gate.
- [ ] Remove the temporary architecture-loop workflow from the final branch, open the final PR to `main`, and run complete CI.
