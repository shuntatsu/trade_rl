# Stage A CLI Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fail-closed Stage A validation, sealed-test, and complete-run CLI commands that consume immutable execution artifacts and atomically publish the maintained Stage A result packages.

**Architecture:** Add a dedicated lazy-loaded `trade_rl.cli.stage_a` module. The CLI loads the strict plan and evaluation-dataset manifest, builds the existing artifact-backed evaluator, delegates all candidate selection and sealed-test decisions to `StageAZeroShotEvaluationOrchestrator`, uses `PostgresStageASealedTestLedger` only for test opening, and publishes through `StageAZeroShotArtifactPublisher`. The complete-run command publishes validation first and opens the sealed test only when the recomputed validation selection passes.

**Tech Stack:** Python 3.12, argparse, immutable JSON artifacts, PostgreSQL ledger, pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Do not accept caller-supplied growth, candidate selection, test ranges, or authorization digests.
- Never access the sealed test before a passed validation selection has been recomputed.
- Use `TRADE_RL_DATABASE_URL` when `--database-url` is omitted.
- Do not print the PostgreSQL DSN in command output.
- Publish only through the existing atomic Stage A artifact publisher.
- Keep the CLI module free of SB3/Torch imports and preserve all Import Linter contracts.
- Validation and sealed-test packages remain immutable and cannot be overwritten.

---

### Task 1: Add the Stage A command surface

**Files:**
- Create: `trade_rl/cli/stage_a.py`
- Modify: `trade_rl/cli/__init__.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/cli/test_stage_a_cli.py`

**Interfaces:**
- Consumes: `load_stage_a_zero_shot_evaluation_plan(path)`, `load_stage_a_evaluation_dataset_manifest(path)`.
- Produces: `add_stage_a_parser(subparsers) -> None` and `main(argv, stdout, stderr) -> int`.

- [ ] **Step 1: Write failing parser and routing tests**

Add tests proving `trade_rl.cli.main(["stage-a", "validation", ...])` routes to the dedicated module and that `build_parser()` exposes `stage-a validation`, `stage-a sealed-test`, and `stage-a run`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py`

Expected: collection or parser failure because `trade_rl.cli.stage_a` and the commands do not exist.

- [ ] **Step 3: Implement the minimal parser and lazy routing**

The parser must define these exact common options:

```text
--plan PATH
--manifest PATH
--execution-store PATH
--baseline-config-digest SHA256
--output-root PATH
```

`sealed-test` additionally requires `--validation-package PATH`; `sealed-test` and `run` accept optional `--database-url`, falling back to `TRADE_RL_DATABASE_URL`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py`

- [ ] **Step 5: Commit**

```bash
git add trade_rl/cli/stage_a.py trade_rl/cli/__init__.py trade_rl/cli/app.py tests/cli/test_stage_a_cli.py
git commit -m "feat: add Stage A CLI command surface"
```

### Task 2: Wire validation and immutable publication

**Files:**
- Modify: `trade_rl/cli/stage_a.py`
- Test: `tests/cli/test_stage_a_cli.py`

**Interfaces:**
- Consumes: `StageAExecutionPromotionStore`, `ArtifactBackedStageAEvaluationCellEvaluator`, `StageAZeroShotEvaluationOrchestrator`, `StageAZeroShotArtifactPublisher`.
- Produces: a canonical JSON result with schema `stage_a_validation_cli_result_v1`.

- [ ] **Step 1: Write a failing validation command test**

Use real plan/manifest writers and the real orchestrator/publisher with an injected deterministic evaluator. Assert that the command creates `output-root/validation/evidence.json` and `selection.json`, reports the exact plan, manifest, run, evidence, and selection digests, and reports the selected candidate and pass decision.

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py::test_validation_command_publishes_complete_package`

Expected: failure because the validation handler is not implemented.

- [ ] **Step 3: Implement validation wiring**

Load and cross-validate plan/manifest, build the existing artifact-backed evaluator from `--execution-store` and `--baseline-config-digest`, run `evaluate_validation()`, publish through `publish_validation()`, and emit only content identities and the published path.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py`

- [ ] **Step 5: Commit**

```bash
git add trade_rl/cli/stage_a.py tests/cli/test_stage_a_cli.py
git commit -m "feat: publish Stage A validation from CLI"
```

### Task 3: Wire PostgreSQL sealed-test authorization and complete run

**Files:**
- Modify: `trade_rl/cli/stage_a.py`
- Test: `tests/cli/test_stage_a_cli.py`

**Interfaces:**
- Consumes: `load_stage_a_evaluation_evidence`, `load_stage_a_validation_selection`, `StageAValidationRun`, `PostgresStageASealedTestLedger`.
- Produces: schemas `stage_a_sealed_test_cli_result_v1` and `stage_a_complete_run_cli_result_v1`.

- [ ] **Step 1: Write failing sealed-test and complete-run tests**

Prove that:

```text
sealed-test loads validation/evidence.json and validation/selection.json strictly;
sealed-test passes the explicit or environment DSN into the PostgreSQL ledger factory;
sealed-test publishes evidence.json, decision.json, and access-records.json;
run publishes validation first;
run does not construct a ledger or evaluate test when validation fails;
run evaluates and publishes test only after validation passes.
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py`

Expected: failures because sealed-test and complete-run handlers are not implemented.

- [ ] **Step 3: Implement strict validation-package loading**

Load the validation evidence with `load_stage_a_evaluation_evidence`, load the selection with `load_stage_a_validation_selection(plan=plan, evidence=evidence)`, construct `StageAValidationRun`, and reject any mismatch before ledger construction.

- [ ] **Step 4: Implement sealed-test and complete-run handlers**

Use one `PostgresStageASealedTestLedger` per command. `run` must return a valid result with `sealed_test: null` when validation does not pass; this is a completed scientific decision, not an operational exception. Never print the DSN.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py`

- [ ] **Step 6: Commit**

```bash
git add trade_rl/cli/stage_a.py tests/cli/test_stage_a_cli.py
git commit -m "feat: wire Stage A sealed test and complete run"
```

### Task 4: Documentation and full verification

**Files:**
- Modify: `docs/operations/stage-a-zero-shot-evaluation-plan.md`
- Modify: `README.md` only when an existing CLI section has an appropriate Stage A entry.

**Interfaces:**
- Consumes: the finalized CLI syntax.
- Produces: operator instructions that distinguish preproduced execution artifacts from validation/test orchestration.

- [ ] **Step 1: Document exact commands and trust boundary**

Document that the execution promotion store must already contain every requested baseline/policy cell, validation does not need PostgreSQL, sealed-test and run require migrated catalog schema v3, and output packages are immutable.

- [ ] **Step 2: Run focused static and architecture checks**

```bash
PYTHONPATH=. pytest -q tests/cli/test_stage_a_cli.py
ruff check trade_rl/cli/stage_a.py trade_rl/cli/__init__.py trade_rl/cli/app.py tests/cli/test_stage_a_cli.py
ruff format --check trade_rl/cli/stage_a.py trade_rl/cli/__init__.py trade_rl/cli/app.py tests/cli/test_stage_a_cli.py
mypy trade_rl/cli/stage_a.py trade_rl/cli/__init__.py trade_rl/cli/app.py
lint-imports
```

- [ ] **Step 3: Run repository-wide verification**

Run full pytest with branch coverage, critical coverage ratchet, Studio tests/typecheck/build/layout, Ubuntu/Windows compatibility, training-image probe, and PostgreSQL Catalog workflow on the exact final head.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/operations/stage-a-zero-shot-evaluation-plan.md README.md
git commit -m "docs: document Stage A evaluation CLI"
```
