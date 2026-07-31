# Stage A Production Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed artifact-backed implementation of `StageAEvaluationCellEvaluator` that derives Stage A log growth only from verified conservative execution artifacts bound to the exact A6a request.

**Architecture:** Introduce an immutable execution-cell identity and a content-addressed replay artifact that references the maintained order-event artifact and execution-promotion evidence. A request-indexed store publishes and reloads those artifacts, and a small evaluator compares the complete replay identity to the A6a request before returning recomputed log growth.

**Tech Stack:** Python 3.12, dataclasses, pathlib, canonical JSON, SHA-256 content identities, existing execution promotion validation, pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Do not load checkpoints or run market episodes in A6b-1.
- Do not duplicate A6a selection, evidence aggregation, sealed-test authorization, or gate logic.
- Do not trust caller-supplied log growth.
- Do not accept symlinks, unsafe relative paths, non-canonical JSON, or mutable request rebinding.
- Baseline candidate configuration identity must be independent of policy candidates.
- Keep public workflow modules dependent only on public A6a contracts and maintained simulation APIs.

---

## Implementation status

A6b-1 implementation is complete locally through explicit RED/GREEN cycles. The focused A6a/A6b workflow suite passes, including request rebinding, byte tampering, symlink, terminal-equity, candidate/checkpoint substitution, and shared-baseline integration tests. Repository-wide static and CI verification remains the final gate before merge.

### Task 1: Exact execution-cell identity and replay artifact

**Files:**
- Create: `trade_rl/workflows/stage_a_execution_replay.py`
- Test: `tests/workflows/test_stage_a_execution_replay.py`

**Interfaces:**
- Consumes: `StageAEvaluationCellRequest`, `ExecutionEvidence`, existing order-event artifact path.
- Produces: `StageAExecutionCellIdentity`, `StageAExecutionReplayArtifact`, `build_stage_a_execution_replay_artifact`, `load_stage_a_execution_replay_artifact`.

- [ ] **Step 1: Write failing identity tests**

Add tests that construct policy and baseline identities and assert candidate/checkpoint nullability, SHA-256 validation, finite positive equity, canonical field closure, and deterministic digest behavior.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/workflows/test_stage_a_execution_replay.py -q
```

Expected: collection failure because `trade_rl.workflows.stage_a_execution_replay` does not exist.

- [ ] **Step 3: Implement the immutable contracts**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StageAExecutionCellIdentity:
    request_digest: str
    plan_digest: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    candidate_id: str | None
    checkpoint_digest: str | None
    candidate_config_digest: str
    dataset_identity: str
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    schema_version: str = "stage_a_execution_cell_identity_v1"
    digest: str = ""
```

Implement `StageAExecutionReplayArtifact` with canonical actions, observation digests, equity curve, event/evidence digests and sizes, strict JSON encoding, and `log_growth = log(equity_curve[-1] / equity_curve[0])`.

- [ ] **Step 4: Verify GREEN and static checks**

Run:

```bash
pytest tests/workflows/test_stage_a_execution_replay.py -q
ruff check trade_rl/workflows/stage_a_execution_replay.py tests/workflows/test_stage_a_execution_replay.py
mypy trade_rl/workflows/stage_a_execution_replay.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/stage_a_execution_replay.py tests/workflows/test_stage_a_execution_replay.py
git commit -m "feat: add Stage A execution replay contract"
```

### Task 2: Content-addressed request store

**Files:**
- Create: `trade_rl/workflows/stage_a_execution_store.py`
- Test: `tests/workflows/test_stage_a_execution_store.py`

**Interfaces:**
- Consumes: `StageAExecutionReplayArtifact`, event artifact path, execution evidence path.
- Produces: `StageAExecutionPromotionStore.publish(...)` and `StageAExecutionPromotionStore.load(request_digest)`.

- [ ] **Step 1: Write failing store tests**

Cover canonical layout, idempotent identical publication, request rebinding rejection, index/path traversal rejection, symlink rejection, event/evidence byte tampering, and non-canonical manifest rejection.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
pytest tests/workflows/test_stage_a_execution_store.py -q
```

Expected: collection failure because the store module is absent.

- [ ] **Step 3: Implement exclusive content-addressed publication**

Use canonical paths:

```text
root/events/<digest>.order-events.json
root/evidence/<digest>.execution-evidence.json
root/cells/<request-digest>/<artifact-digest>.stage-a-cell.json
root/by-request/<request-digest>.json
```

Open existing files through the maintained verified regular-file helper or an equivalent no-follow implementation. Create request indexes with exclusive creation; accept an existing index only when its canonical bytes are identical.

- [ ] **Step 4: Verify GREEN and static checks**

```bash
pytest tests/workflows/test_stage_a_execution_store.py -q
ruff check trade_rl/workflows/stage_a_execution_store.py tests/workflows/test_stage_a_execution_store.py
mypy trade_rl/workflows/stage_a_execution_store.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/stage_a_execution_store.py tests/workflows/test_stage_a_execution_store.py
git commit -m "feat: add content-addressed Stage A execution store"
```

### Task 3: Artifact-backed A6a evaluator

**Files:**
- Create: `trade_rl/workflows/stage_a_production_evaluator.py`
- Test: `tests/workflows/test_stage_a_production_evaluator.py`

**Interfaces:**
- Consumes: `StageAEvaluationCellRequest`, `StageAExecutionPromotionStore`, `StageAZeroShotEvaluationPlan`.
- Produces: `ArtifactBackedStageAEvaluationCellEvaluator.evaluate(request) -> StageAEvaluationCellResult`.

- [ ] **Step 1: Write failing evaluator tests**

Cover valid policy and baseline results, request digest mismatch, candidate/config/checkpoint substitution, split/triplet/fold/seed substitution, dataset/feature/execution/evaluation substitution, baseline-config substitution, and recomputed growth.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/workflows/test_stage_a_production_evaluator.py -q
```

Expected: collection failure because the evaluator module is absent.

- [ ] **Step 3: Implement exact request comparison**

Implement:

```python
class ArtifactBackedStageAEvaluationCellEvaluator:
    def __init__(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        store: StageAExecutionPromotionStore,
        baseline_candidate_config_digest: str,
    ) -> None: ...

    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult: ...
```

Compare every identity field. Policy candidate configuration comes from `plan.candidate(candidate_id).candidate_config_digest`; baseline configuration comes only from `baseline_candidate_config_digest`.

- [ ] **Step 4: Verify GREEN and static checks**

```bash
pytest tests/workflows/test_stage_a_production_evaluator.py -q
ruff check trade_rl/workflows/stage_a_production_evaluator.py tests/workflows/test_stage_a_production_evaluator.py
mypy trade_rl/workflows/stage_a_production_evaluator.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/stage_a_production_evaluator.py tests/workflows/test_stage_a_production_evaluator.py
git commit -m "feat: connect verified execution artifacts to Stage A"
```

### Task 4: A6a integration and repository verification

**Files:**
- Modify: `docs/operations/stage-a-zero-shot-evaluation-plan.md`
- Create: `tests/workflows/test_stage_a_production_orchestration.py`
- Modify: `docs/operations/stage-a-production-evaluator-design.md`
- Modify: `docs/operations/stage-a-production-evaluator-plan.md`

**Interfaces:**
- Consumes: A6a orchestrator, artifact-backed evaluator, in-memory sealed-test ledger.
- Produces: end-to-end validation evidence from verified policy and shared baseline artifacts.

- [ ] **Step 1: Write integration tests**

Create complete two-fold, two-seed policy and baseline artifacts for one validation triplet. Run `StageAZeroShotEvaluationOrchestrator.evaluate_validation()` and assert the evidence uses the same baseline digest for every candidate in each triplet/fold/seed cell.

- [ ] **Step 2: Run focused workflow tests**

```bash
pytest tests/workflows/test_stage_a_execution_replay.py tests/workflows/test_stage_a_execution_store.py tests/workflows/test_stage_a_production_evaluator.py tests/workflows/test_stage_a_production_orchestration.py -q
```

Expected: all pass.

- [ ] **Step 3: Update operations documentation**

Record A6b-1 completion and leave checkpoint execution, canonical loader, CLI, schedule source, and PostgreSQL construction explicitly assigned to A6b-2.

- [ ] **Step 4: Run full verification**

```bash
ruff check .
ruff format --check .
mypy trade_rl
lint-imports
pytest -q
```

Then run the repository PostgreSQL and compatibility CI workflows on one unchanged head.

- [ ] **Step 5: Commit and open a draft PR**

```bash
git add docs/operations tests/workflows trade_rl/workflows
git commit -m "test: verify Stage A production evaluator integration"
git push -u origin agent/stage-a6b-production-evaluator
gh pr create --draft --base main --head agent/stage-a6b-production-evaluator --title "Connect Stage A to verified execution artifacts"
```
