# Walk-Forward Manifest and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish every new market walk-forward run with the dedicated walk-forward manifest and real runtime provenance.

**Architecture:** Keep the existing market walk-forward orchestration and artifact layout. Replace only the terminal evidence-publication boundary: capture `RuntimeProvenance`, persist it, build `WalkForwardRunManifest`, validate with the walk-forward validator, and publish through the existing `ArtifactStore`.

**Tech Stack:** Python 3.12, pytest, dataclasses, canonical JSON artifacts, existing Git/source-tree provenance utilities.

## Global Constraints

- Production status remains `NO-GO`.
- No change to candidate training, selection, sealed-test access, execution sensitivity, or fold evaluation.
- Existing legacy artifacts remain readable.
- New behavior must be introduced test-first.
- No new dependency.

---

### Task 1: Add the failing publication contract test

**Files:**
- Create: `tests/workflows/test_walk_forward_manifest_provenance.py`
- Reuse: `tests/workflows/test_market_walk_forward.py`

**Interfaces:**
- Consumes: `execute_market_walk_forward(...) -> WalkForwardRunResult`
- Produces: a regression contract for `walk_forward_run_v1` and `provenance.json`

- [ ] **Step 1: Write the failing test**

Create a focused test that runs the existing minimal walk-forward fixture through public workflow APIs, then asserts:

```python
manifest = validate_walk_forward_run_directory(published)
assert manifest.schema_version == WALK_FORWARD_RUN_MANIFEST_SCHEMA
assert manifest.evaluation_digest == result.evaluation_digest
assert manifest.fold_count == 1
provenance = json.loads((published / "provenance.json").read_text())
assert provenance["digest"] == manifest.provenance_digest
with pytest.raises(ValueError, match="unsupported training run schema"):
    validate_training_run_directory(published)
```

Monkeypatch `capture_runtime_provenance` only to provide deterministic valid provenance input; do not mock manifest construction or directory validation.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/workflows/test_walk_forward_manifest_provenance.py -q
```

Expected: FAIL because current `run.json` uses `training_run_v3` and no `provenance.json` is written.

- [ ] **Step 3: Commit the RED test**

```bash
git add tests/workflows/test_walk_forward_manifest_provenance.py
git commit -m "test: require walk-forward provenance manifest"
```

### Task 2: Replace the publication boundary

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Test: `tests/workflows/test_walk_forward_manifest_provenance.py`

**Interfaces:**
- Consumes: `capture_runtime_provenance(root, deterministic_seed_config=...) -> RuntimeProvenance`
- Produces: `WalkForwardRunManifest` through `write_walk_forward_run_manifest()`

- [ ] **Step 1: Update imports**

Replace the training-manifest imports with:

```python
from trade_rl.artifacts.provenance import capture_runtime_provenance
from trade_rl.artifacts.run_manifest import (
    WalkForwardRunManifest,
    validate_walk_forward_run_directory,
    write_walk_forward_run_manifest,
)
```

- [ ] **Step 2: Capture and persist provenance**

After configuration normalization and before candidate training, compute:

```python
config_digest = content_digest(config.digest_payload())
provenance = capture_runtime_provenance(
    Path(__file__).resolve().parents[2],
    deterministic_seed_config={
        "candidate_seeds": tuple(
            seed
            for candidate in config.candidates
            for seed in candidate.run.training.seeds
        ),
        "workflow_config_digest": config_digest,
    },
)
_write_json(stage / "provenance.json", asdict(provenance))
```

Use deterministic candidate ordering from the normalized configuration.

- [ ] **Step 3: Build the dedicated manifest**

Replace `TrainingRunManifest.build(...)` with:

```python
run_manifest = WalkForwardRunManifest.build(
    root=stage,
    run_id=resolved_run_id,
    dataset_id=dataset.dataset_id,
    environment_digest=environment_digest,
    evaluation_digest=result.evaluation_digest,
    workflow_config_digest=config_digest,
    policy_set_digest=policy_digest,
    provenance_digest=provenance.digest,
    fold_count=len(result.folds),
    artifact_paths=_artifact_paths(stage),
    created_at=resolved_created_at,
)
write_walk_forward_run_manifest(stage, run_manifest)
validate_walk_forward_run_directory(stage)
```

- [ ] **Step 4: Update the store validator**

Change `_validate_for_store()` to call `validate_walk_forward_run_directory()`.

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/workflows/test_walk_forward_manifest_provenance.py tests/workflows/test_market_walk_forward.py tests/artifacts/test_run_manifests_v2.py tests/studio/test_run_catalog.py -q
```

Expected: PASS.

- [ ] **Step 6: Run static and architecture gates**

```bash
ruff check trade_rl/workflows/market_walk_forward.py tests/workflows/test_walk_forward_manifest_provenance.py
ruff format --check trade_rl/workflows/market_walk_forward.py tests/workflows/test_walk_forward_manifest_provenance.py
mypy trade_rl/workflows/market_walk_forward.py
pytest tests/architecture -q
```

Expected: PASS.

- [ ] **Step 7: Commit implementation**

```bash
git add trade_rl/workflows/market_walk_forward.py tests/workflows/test_walk_forward_manifest_provenance.py
git commit -m "fix: publish walk-forward provenance manifest"
```

### Task 3: Verify repository-wide compatibility

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: completed Task 2 branch
- Produces: exact-head CI evidence

- [ ] **Step 1: Run the complete Python suite**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run repository static gates**

```bash
ruff check .
ruff format --check .
mypy
```

Expected: PASS.

- [ ] **Step 3: Open a draft PR and verify exact-head workflows**

The PR description must state that no training, selection, reward, execution, or Serving semantics changed. Required workflows: standard CI, PostgreSQL Catalog when path filters apply, Ubuntu, Windows, training-image probe, critical coverage, and CLI smoke.
