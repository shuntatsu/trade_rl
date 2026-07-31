# Stage A Policy-Bound Producer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that each policy execution published to the A6b-1 store came from the exact retained checkpoint declared by the Stage A plan, with optional canonical serving-bundle inference.

**Architecture:** Add an immutable request-indexed policy-source binding, a source-identified runtime handle, a strict episode-result contract, and a producer that validates all identities before publishing the existing A6b-1 replay artifact. Keep concrete PostgreSQL dataset slicing and Stable-Baselines3 environment assembly outside this PR.

**Tech Stack:** Python 3.12, dataclasses, pathlib, canonical JSON, SHA-256 content identities, existing checkpoint and serving-bundle loaders, existing A6a/A6b-1 contracts, pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Do not move model loading or execution into `ArtifactBackedStageAEvaluationCellEvaluator`.
- Do not accept caller-supplied log growth.
- Do not select a latest, highest-scoring, nearby, or replacement checkpoint.
- Policy source identity must include the exact request, candidate, seed, checkpoint manifest, candidate configuration, and checkpoint policy digest.
- Baseline execution must never resolve a policy source.
- All request indexes are immutable and may be retried only with identical canonical bytes.
- Reject absolute paths, `..`, symlinks, non-canonical JSON, extra fields, missing fields, digest mismatch, and size mismatch.
- Preserve the current Import Linter rule that workflows do not directly import model frameworks.

---

### Task 1: Policy-source binding contract

**Files:**
- Create: `trade_rl/workflows/stage_a_policy_source.py`
- Test: `tests/workflows/test_stage_a_policy_source.py`

**Interfaces:**
- Consumes: `StageAZeroShotEvaluationPlan`, `StageAEvaluationCellRequest`, `CheckpointManifest`, optional `ServingBundle`.
- Produces: `StageAPolicySourceBinding`, `StageAPolicySourceStore.publish(...)`, and `StageAPolicySourceStore.load(request_digest)`.

- [ ] **Step 1: Write the failing checkpoint-only tests**

Add tests that create one plan candidate, one policy request, and one real checkpoint manifest fixture. Assert that construction and reload succeed only when:

```python
binding.checkpoint_digest == request.checkpoint_digest
binding.seed == request.seed
binding.candidate_config_digest == plan.candidate(request.candidate_id).candidate_config_digest
binding.checkpoint_policy_digest == checkpoint_manifest.policy_digest
```

Also assert that baseline requests are rejected.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/workflows/test_stage_a_policy_source.py -q
```

Expected: collection failure because `trade_rl.workflows.stage_a_policy_source` does not exist.

- [ ] **Step 3: Implement the immutable binding**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StageAPolicySourceBinding:
    plan_digest: str
    request_digest: str
    candidate_id: str
    seed: int
    checkpoint_digest: str
    candidate_config_digest: str
    checkpoint_policy_digest: str
    checkpoint_manifest_path: str
    serving_bundle_path: str | None = None
    serving_bundle_digest: str | None = None
    schema_version: str = "stage_a_policy_source_binding_v1"
    digest: str = ""
```

The constructor validates field closure, SHA-256 values, normalized relative paths, serving-bundle nullability, and deterministic content digest.

- [ ] **Step 4: Implement plan and checkpoint validation**

Add:

```python
def validate(
    self,
    *,
    root: Path,
    plan: StageAZeroShotEvaluationPlan,
    request: StageAEvaluationCellRequest,
) -> CheckpointManifest:
    ...
```

Reload `checkpoint.json` through `load_checkpoint_manifest`, require the exact manifest digest, seed, training-config digest, and policy digest, and compare the request to the immutable plan before returning the manifest.

- [ ] **Step 5: Implement the request-indexed store**

Use canonical paths:

```text
root/bindings/<request-digest>/<binding-digest>.json
root/by-request/<request-digest>.json
```

`publish()` writes canonical bytes exclusively and accepts an existing file only when bytes are identical. `load()` opens regular non-symlink files, validates the index, reloads the binding, and revalidates checkpoint and optional bundle bytes.

- [ ] **Step 6: Add tamper and substitution tests**

Cover:

- wrong plan, request, candidate, seed, checkpoint, config, and policy digest;
- checkpoint manifest and policy file tampering;
- binding and index tampering;
- absolute path, `..`, and symlink rejection;
- request rebinding rejection;
- identical retry success.

- [ ] **Step 7: Verify Task 1 GREEN**

Run:

```bash
pytest tests/workflows/test_stage_a_policy_source.py -q
ruff check trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
ruff format --check trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
mypy trade_rl/workflows/stage_a_policy_source.py
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
git commit -m "feat: bind Stage A requests to retained checkpoints"
```

### Task 2: Optional canonical serving-bundle source

**Files:**
- Modify: `trade_rl/workflows/stage_a_policy_source.py`
- Test: `tests/workflows/test_stage_a_policy_source.py`

**Interfaces:**
- Consumes: `load_serving_bundle`, `canonical_policy_loader`.
- Produces: validated checkpoint-plus-bundle bindings and `load_serving_policy(binding) -> StageAPolicyRuntimeHandle`.

- [ ] **Step 1: Write failing bundle-binding tests**

Create a minimal valid serving-bundle fixture and require:

```python
bundle.manifest.bundle_digest == binding.serving_bundle_digest
bundle.manifest.policy_digest == checkpoint_manifest.policy_digest
bundle.manifest.environment_digest == checkpoint_manifest.environment_digest
```

Test bundle file tampering, undeclared files, policy substitution, environment substitution, and bundle digest substitution.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest tests/workflows/test_stage_a_policy_source.py -q -k serving_bundle
```

Expected: failures because bundle validation and runtime loading are absent.

- [ ] **Step 3: Implement runtime handle and loader protocol**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StageAPolicyRuntimeHandle:
    policy: object
    source_binding_digest: str
    checkpoint_digest: str
    checkpoint_policy_digest: str
    serving_bundle_digest: str | None

class StageAPolicyRuntimeLoader(Protocol):
    def load(
        self,
        binding: StageAPolicySourceBinding,
        *,
        checkpoint_manifest: CheckpointManifest,
    ) -> StageAPolicyRuntimeHandle: ...
```

- [ ] **Step 4: Implement canonical serving-bundle loading**

Add `CanonicalServingBundleStageAPolicyLoader`. It reloads the bundle, resolves `canonical_policy_loader` using the bundle's architecture identity, loads the policy, and returns a handle bound to the checkpoint and bundle identities. Flat bundles require an explicitly supplied flat fallback loader; no implicit fallback is allowed.

- [ ] **Step 5: Verify Task 2 GREEN**

Run:

```bash
pytest tests/workflows/test_stage_a_policy_source.py -q
ruff check trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
ruff format --check trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
mypy trade_rl/workflows/stage_a_policy_source.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/stage_a_policy_source.py tests/workflows/test_stage_a_policy_source.py
git commit -m "feat: load Stage A policies from canonical bundles"
```

### Task 3: Strict episode-result contract

**Files:**
- Create: `trade_rl/workflows/stage_a_execution_producer.py`
- Test: `tests/workflows/test_stage_a_execution_producer.py`

**Interfaces:**
- Consumes: `StageAEvaluationCellRequest`, `StageAPolicyRuntimeHandle`.
- Produces: `StageAEvaluationEpisodeResult` and `StageAEvaluationEpisodeExecutor`.

- [ ] **Step 1: Write failing result-contract tests**

Cover policy and baseline results, requiring:

```python
policy_result.policy_source_digest == binding.digest
baseline_result.policy_source_digest is None
result.request_digest == request.digest
```

Reject non-finite actions, empty observations, action/observation length mismatch, non-positive equity, missing order events, and source nullability violations.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/workflows/test_stage_a_execution_producer.py -q
```

Expected: collection failure because the producer module is absent.

- [ ] **Step 3: Implement result and executor interfaces**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StageAEvaluationEpisodeResult:
    request_digest: str
    policy_source_digest: str | None
    candidate_config_digest: str
    actions: tuple[tuple[float, ...], ...]
    observation_digests: tuple[str, ...]
    equity_curve: tuple[float, ...]
    order_events: tuple[OrderEvent, ...]
    terminal_book: BookState
    terminal_order_book: OrderBookState

class StageAEvaluationEpisodeExecutor(Protocol):
    def execute(
        self,
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
        candidate_config_digest: str,
    ) -> StageAEvaluationEpisodeResult: ...
```

- [ ] **Step 4: Verify Task 3 GREEN**

```bash
pytest tests/workflows/test_stage_a_execution_producer.py -q
ruff check trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_execution_producer.py
ruff format --check trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_execution_producer.py
mypy trade_rl/workflows/stage_a_execution_producer.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_execution_producer.py
git commit -m "feat: add Stage A episode result contract"
```

### Task 4: Policy-bound artifact producer

**Files:**
- Modify: `trade_rl/workflows/stage_a_execution_producer.py`
- Test: `tests/workflows/test_stage_a_execution_producer.py`
- Test: `tests/workflows/test_stage_a_policy_producer_orchestration.py`

**Interfaces:**
- Consumes: `StageAPolicySourceStore`, `StageAPolicyRuntimeLoader`, `StageAEvaluationEpisodeExecutor`, `StageAExecutionPromotionStore`.
- Produces: `StageAExecutionArtifactProducer.produce(request) -> StoredStageAExecutionReplay`.

- [ ] **Step 1: Write failing policy producer tests**

Use fake runtime loader and executor implementations. Test:

- valid source-bound policy publication;
- loader handle source substitution;
- loader checkpoint and policy substitution;
- executor request and source substitution;
- candidate-config substitution;
- policy source missing for a policy request;
- request rebinding with different execution bytes.

- [ ] **Step 2: Write failing baseline producer tests**

Assert that baseline production never calls the policy-source store or runtime loader, requires the independent baseline config digest, and publishes an artifact accepted by A6b-1.

- [ ] **Step 3: Implement producer validation order**

Implement:

```python
class StageAExecutionArtifactProducer:
    def produce(
        self,
        request: StageAEvaluationCellRequest,
    ) -> StoredStageAExecutionReplay:
        ...
```

For policy requests: validate plan, load binding, validate checkpoint, load handle, verify handle, execute, verify result, build A6b-1 replay, publish, reload, and return.

For baseline requests: validate plan, execute with `policy=None`, verify the baseline config digest and null source, build, publish, reload, and return.

Use `ExecutionCostConfig` reconstructed from an injected maintained cost resolver and require its digest to equal `plan.execution_identity` before constructing promotion evidence.

- [ ] **Step 4: Add A6a integration test**

Produce all baseline and policy validation cells for a small plan into one store, then run `ArtifactBackedStageAEvaluationCellEvaluator` through `StageAZeroShotEvaluationOrchestrator.evaluate_validation()`. Assert that validation evidence uses complete replay digests and one shared baseline per triplet/fold/seed.

- [ ] **Step 5: Verify Task 4 GREEN**

```bash
pytest tests/workflows/test_stage_a_execution_producer.py tests/workflows/test_stage_a_policy_producer_orchestration.py tests/workflows/test_stage_a_execution_replay.py tests/workflows/test_stage_a_execution_store.py tests/workflows/test_stage_a_production_evaluator.py tests/workflows/test_stage_a_production_orchestration.py -q
ruff check trade_rl/workflows/stage_a_policy_source.py trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_policy_source.py tests/workflows/test_stage_a_execution_producer.py tests/workflows/test_stage_a_policy_producer_orchestration.py
ruff format --check trade_rl/workflows/stage_a_policy_source.py trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_policy_source.py tests/workflows/test_stage_a_execution_producer.py tests/workflows/test_stage_a_policy_producer_orchestration.py
mypy trade_rl/workflows/stage_a_policy_source.py trade_rl/workflows/stage_a_execution_producer.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/stage_a_execution_producer.py tests/workflows/test_stage_a_execution_producer.py tests/workflows/test_stage_a_policy_producer_orchestration.py
git commit -m "feat: publish checkpoint-bound Stage A executions"
```

### Task 5: Documentation and full verification

**Files:**
- Modify: `docs/operations/stage-a-production-evaluator-design.md`
- Modify: `docs/operations/stage-a-zero-shot-evaluation-plan.md`
- Modify: `docs/superpowers/plans/2026-07-31-stage-a-policy-producer.md`

**Interfaces:**
- Consumes: completed A6b-2a implementation.
- Produces: documented trust boundary and exact-head verification record.

- [ ] **Step 1: Update operations documentation**

Document that A6b-2a proves retained-checkpoint provenance before A6b-1 publication, while exact PostgreSQL fold materialization and concrete SB3 execution remain A6b-2b.

- [ ] **Step 2: Run focused verification**

```bash
pytest tests/workflows/test_stage_a_policy_source.py tests/workflows/test_stage_a_execution_producer.py tests/workflows/test_stage_a_policy_producer_orchestration.py tests/workflows/test_stage_a_execution_replay.py tests/workflows/test_stage_a_execution_store.py tests/workflows/test_stage_a_production_evaluator.py tests/workflows/test_stage_a_production_orchestration.py -q
```

Expected: all pass.

- [ ] **Step 3: Run repository verification**

```bash
ruff check .
ruff format --check .
mypy .
lint-imports
pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
python .github/check_critical_coverage.py coverage.json pyproject.toml
trade-rl --version
```

Expected: all pass with total coverage at least 80%.

- [ ] **Step 4: Verify platform and database workflows**

Require the unchanged final head to pass:

- Ubuntu compatibility;
- Windows compatibility;
- complete training image and non-root runtime probe;
- PostgreSQL Compose validation, migration, unit, and integration tests.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/operations/stage-a-production-evaluator-design.md docs/operations/stage-a-zero-shot-evaluation-plan.md docs/superpowers/plans/2026-07-31-stage-a-policy-producer.md
git commit -m "docs: record Stage A policy producer boundary"
```
