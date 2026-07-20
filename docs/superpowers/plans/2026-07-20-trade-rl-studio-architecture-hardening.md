# Trade RL Studio Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Studio identities, job control, comparison, validation, and frontend state fail closed across restarts, duplicate roots, malformed artifacts, and asynchronous requests.

**Architecture:** Introduce focused resource-ID, catalog, error, and job-store boundaries behind the existing Studio facade. Replace path-based write requests with validated identities, attach comparison eligibility, and make the frontend validate every payload while persisting navigation state in the URL.

**Tech Stack:** Python 3.12, standard-library filesystem primitives, FastAPI, Pydantic v2, React 19, strict TypeScript, Vitest, Testing Library.

## Global Constraints

- Bind only to loopback; remove remote override.
- Add no exchange order, API-key, live activation, or fund-mutation capability.
- Preserve `NO-GO` on every response.
- Use canonical repository validators; do not duplicate training or artifact validation logic.
- Write a failing test before each behavior change.
- Keep browser document scrolling disabled at 1536×1024 and 1440×900.

---

### Task 1: Typed errors and collision-free resource identities

**Files:**
- Create: `trade_rl/studio/errors.py`
- Create: `trade_rl/studio/resource_ids.py`
- Modify: `trade_rl/studio/contracts.py`
- Test: `tests/studio/test_resource_ids.py`

**Interfaces:**
- Produces `resource_id(kind: str, relative_path: str, identity: str) -> str`.
- Produces `StudioError`, `ResourceNotFound`, `InvalidStudioRequest`, `IdentityConflict`, `ArtifactInvalid`, and `JobOwnershipLost`.
- Dataset, run, and config summaries expose a unique `id` and the canonical identity separately.

- [ ] Write tests proving identical human run IDs under different roots receive different resource IDs and malformed IDs never resolve.
- [ ] Run `python -m pytest -q tests/studio/test_resource_ids.py` and confirm failure.
- [ ] Implement resource IDs, exception classes, and contract fields.
- [ ] Run the focused test and confirm success.

### Task 2: Focused validated catalogs with cache invalidation

**Files:**
- Create: `trade_rl/studio/catalog_cache.py`
- Create: `trade_rl/studio/dataset_catalog.py`
- Create: `trade_rl/studio/config_catalog.py`
- Create: `trade_rl/studio/run_catalog.py`
- Create: `trade_rl/studio/system_probe.py`
- Create: `trade_rl/studio/overview.py`
- Modify: `trade_rl/studio/catalog.py`
- Test: `tests/studio/test_catalog.py`

**Interfaces:**
- `DatasetCatalog.resolve(resource_id)` returns a fully validated path and summary.
- `ConfigCatalog.resolve(resource_id)` returns `TrainingRunConfig`, path, digest, and summary.
- `RunCatalog.resolve(resource_id)` returns an exact validated run.
- Cache entries are reused only while manifest and declared-file stat fingerprints are unchanged.

- [ ] Add tests for canonical config rejection, duplicate-root resolution, cache reuse, and invalidation after artifact mutation.
- [ ] Run catalog tests and confirm the new assertions fail.
- [ ] Implement focused catalogs, cache, probe, overview service, and compatibility facade.
- [ ] Run catalog tests and confirm success.

### Task 3: Restart-safe job store and identity-only submission

**Files:**
- Create: `trade_rl/studio/job_store.py`
- Modify: `trade_rl/studio/jobs.py`
- Modify: `trade_rl/studio/api.py`
- Test: `tests/studio/test_jobs.py`
- Test: `tests/studio/test_api.py`

**Interfaces:**
- `JobStore.reserve`, `create`, `transition`, `read`, `list`, and `release` are cross-process atomic.
- `JobSupervisor.submit_training` accepts only validated resource IDs through `TrainingJobRequest`.
- Detached workers cannot be cancelled and are returned with `cancellable=false`.

- [ ] Add failing tests for two supervisors racing on one run ID, restart cancellation, invalid canonical config/dataset, and legal transitions.
- [ ] Implement exclusive reservations, unique atomic writes, instance ownership, PID tokens, and identity resolution.
- [ ] Run job and API tests and confirm success.

### Task 4: Comparison eligibility and evidence integrity

**Files:**
- Modify: `trade_rl/studio/comparison.py`
- Modify: `trade_rl/studio/evidence.py`
- Modify: `trade_rl/studio/serving_monitor.py`
- Test: `tests/studio/test_comparison.py`
- Test: `tests/studio/test_evidence.py`
- Test: `tests/studio/test_serving_monitor.py`

**Interfaces:**
- `RunComparison.eligibility` reports comparability and reasons.
- Not-comparable runs expose no decision metrics, folds, or wealth.
- Evidence nodes verify internal proposal and authorization identities.
- Paper snapshot digest mismatches produce a failed check.

- [ ] Add failing tests for dataset/test-range mismatch, partial legacy alignment, proposal binding mismatch, and snapshot digest mismatch.
- [ ] Implement eligibility, aligned labels, evidence binding, and canonical telemetry digest checks.
- [ ] Run focused audit tests and confirm success.

### Task 5: Loopback-only CLI and explicit API errors

**Files:**
- Modify: `trade_rl/studio/cli.py`
- Modify: `trade_rl/studio/api.py`
- Test: `tests/studio/test_studio_cli.py`
- Test: `tests/studio/test_api.py`

**Interfaces:**
- Studio rejects every non-loopback host; no remote override exists.
- API maps each Studio exception to a stable HTTP status and error code.

- [ ] Add failing tests for removed remote override and structured errors.
- [ ] Implement CLI and error mapping.
- [ ] Run focused tests and confirm success.

### Task 6: Frontend runtime contracts, modes, URL state, and request ordering

**Files:**
- Create: `studio/src/api/guards.ts`
- Create: `studio/src/state/urlState.ts`
- Modify: `studio/src/api/studioApi.ts`
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/main.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/components/TopBar.tsx`
- Modify: `studio/src/pages/DataLabPage.tsx`
- Modify: `studio/src/pages/ExperimentsPage.tsx`
- Modify: `studio/src/pages/RunCenterPage.tsx`
- Modify: `studio/src/pages/ComparePage.tsx`
- Modify: `studio/src/pages/EvidencePage.tsx`
- Test: existing Studio frontend tests plus new guard and state tests.

**Interfaces:**
- Every API adapter validates its full response at runtime.
- Overview source is `live`, `offline`, or explicit `demo`.
- URL parameters restore workspace and selected resources.
- Sequence guards ignore stale responses.

- [ ] Add failing tests for malformed list payloads, offline startup, explicit demo mode, URL restoration, and stale comparison responses.
- [ ] Implement guards, modes, URL state, identity fields, and request ordering.
- [ ] Run frontend tests, typecheck, build, and layout checks.

### Task 7: Documentation, architecture contracts, and complete verification

**Files:**
- Modify: `.importlinter`
- Modify: `README.md`
- Modify: `studio/README.md`
- Delete: `.github/workflows/export-architecture-fix-source.yml`

**Interfaces:**
- Import contracts prevent lower layers importing Studio job or API modules.
- Documentation describes identity-only requests, loopback-only binding, detached jobs, and comparison eligibility.

- [ ] Run `python -m pytest -q tests/studio`.
- [ ] Run full Ruff, format, Mypy, import-linter, Python coverage, frontend tests/typecheck/build/layout, compatibility, and training-image CI.
- [ ] Confirm the PR diff contains no transfer or export workflow.
