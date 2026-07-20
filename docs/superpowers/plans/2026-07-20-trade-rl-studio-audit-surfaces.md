# Trade RL Studio Audit Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only Compare, Evidence Explorer, and Serving Monitor surfaces backed by canonical run and serving artifacts.

**Architecture:** Extend the local FastAPI runtime with three focused readers and typed contracts. Keep artifact discovery in `StudioCatalog`, delegate cryptographic and file-closure validation to existing loaders, and render bounded React pages through the existing typed API adapter.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, React 19, Vite 8, strict TypeScript, Vitest, Testing Library, pytest.

## Global Constraints

- Do not add exchange credentials, order routing, order submission, or live-capital mutation.
- Keep the HTTP server loopback-only by default.
- Keep `productionStatus` equal to `NO-GO`.
- Treat canonical artifacts as the source of truth; do not recreate research metrics in TypeScript.
- Keep `html`, `body`, and `#root` free of document scrolling.
- Reject project-root escapes and unknown run identifiers.

---

### Task 1: Safe run lookup and comparison contracts

**Files:**
- Modify: `trade_rl/studio/contracts.py`
- Modify: `trade_rl/studio/catalog.py`
- Create: `trade_rl/studio/comparison.py`
- Test: `tests/studio/test_comparison.py`

**Interfaces:**
- Consumes: `StudioCatalog.list_runs()`, canonical `walk-forward.json`, and validated run roots.
- Produces: `StudioCatalog.resolve_run(run_id: str) -> Path` and `compare_runs(left: Path, right: Path) -> RunComparison`.

- [ ] **Step 1: Write failing tests for exact run resolution, unknown IDs, unsafe IDs, metric deltas, configuration diffs, and fold series.**
- [ ] **Step 2: Run `uv run pytest -q tests/studio/test_comparison.py` and verify the new imports fail.**
- [ ] **Step 3: Add Pydantic comparison models and a resolver that only returns directories already discovered under configured run roots.**
- [ ] **Step 4: Implement comparison extraction from `walk-forward.json`, `training-config.json`, and fold payloads; represent missing values as `None`.**
- [ ] **Step 5: Run the focused test and commit `feat: add Studio run comparison reader`.**

### Task 2: Evidence graph reader

**Files:**
- Modify: `trade_rl/studio/contracts.py`
- Create: `trade_rl/studio/evidence.py`
- Test: `tests/studio/test_evidence.py`

**Interfaces:**
- Consumes: a safe run path and existing run-manifest validators.
- Produces: `inspect_run_evidence(root: Path) -> EvidenceReport`.

- [ ] **Step 1: Write failing tests for exploratory, selected-final, invalid-closure, and optional evidence states.**
- [ ] **Step 2: Run `uv run pytest -q tests/studio/test_evidence.py` and verify RED.**
- [ ] **Step 3: Add typed evidence node, file-integrity, and report contracts.**
- [ ] **Step 4: Implement required-versus-optional node rules from `run_kind`, preserving exact validator errors.**
- [ ] **Step 5: Run the focused test and commit `feat: add Studio evidence inspection`.**

### Task 3: Read-only serving registry monitor

**Files:**
- Modify: `trade_rl/studio/contracts.py`
- Modify: `trade_rl/studio/settings.py`
- Create: `trade_rl/studio/serving_monitor.py`
- Test: `tests/studio/test_serving_monitor.py`

**Interfaces:**
- Consumes: configured serving registry root, `active.json`, `load_serving_bundle`, and optional `var/studio/paper-inference.json`.
- Produces: `inspect_serving(settings: StudioSettings) -> ServingMonitorReport`.

- [ ] **Step 1: Write failing tests for idle registry, valid active bundle, invalid pointer, digest mismatch, and optional paper snapshot.**
- [ ] **Step 2: Run `uv run pytest -q tests/studio/test_serving_monitor.py` and verify RED.**
- [ ] **Step 3: Add serving and paper-snapshot contracts plus a configurable registry root defaulting under `var/serving`.**
- [ ] **Step 4: Implement read-only pointer and bundle validation without activation or policy execution.**
- [ ] **Step 5: Run the focused test and commit `feat: add Studio serving monitor`.**

### Task 4: FastAPI endpoints

**Files:**
- Modify: `trade_rl/studio/api.py`
- Test: `tests/studio/test_api.py`

**Interfaces:**
- Consumes: comparison, evidence, and serving reader functions.
- Produces: the three read-only endpoints defined in the design.

- [ ] **Step 1: Add failing TestClient cases for success, 404, invalid artifact, and idle serving responses.**
- [ ] **Step 2: Run `uv run pytest -q tests/studio/test_api.py` and verify RED.**
- [ ] **Step 3: Add endpoint handlers and route all path/value failures through `_raise_http`.**
- [ ] **Step 4: Run `uv run pytest -q tests/studio/test_api.py tests/studio/test_comparison.py tests/studio/test_evidence.py tests/studio/test_serving_monitor.py`.**
- [ ] **Step 5: Commit `feat: expose Studio audit APIs`.**

### Task 5: TypeScript API contracts and adapters

**Files:**
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/api/studioApi.ts`
- Modify: `studio/src/api/studioApi.test.ts`

**Interfaces:**
- Consumes: backend camelCase JSON contracts.
- Produces: `loadRunComparison`, `loadEvidenceReport`, and `loadServingMonitor`.

- [ ] **Step 1: Add failing adapter tests for valid, HTTP-error, and malformed responses.**
- [ ] **Step 2: Run `npm test -- --run src/api/studioApi.test.ts` and verify RED.**
- [ ] **Step 3: Add strict TypeScript interfaces, guards, URL encoding, and typed loaders.**
- [ ] **Step 4: Run the focused adapter test and `npm run typecheck`.**
- [ ] **Step 5: Commit `feat: add Studio audit API adapters`.**

### Task 6: Compare page

**Files:**
- Create: `studio/src/pages/ComparePage.tsx`
- Modify: `studio/src/App.tsx`
- Create: `studio/src/pages/ComparePage.test.tsx`
- Modify: `studio/src/styles.css`

**Interfaces:**
- Consumes: run list and `loadRunComparison`.
- Produces: a bounded comparison workspace with selectors, metric deltas, config differences, folds, and wealth series.

- [ ] **Step 1: Write failing UI tests for default selection, selector changes, missing metrics, and API failure.**
- [ ] **Step 2: Run the focused Vitest file and verify RED.**
- [ ] **Step 3: Implement the page without document scroll and wire it in `App.tsx`.**
- [ ] **Step 4: Run focused tests, typecheck, and build.**
- [ ] **Step 5: Commit `feat: add Studio comparison workspace`.**

### Task 7: Evidence and Serving pages

**Files:**
- Create: `studio/src/pages/EvidencePage.tsx`
- Create: `studio/src/pages/ServingPage.tsx`
- Create: `studio/src/pages/AuditPages.test.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/styles.css`

**Interfaces:**
- Consumes: run list, `loadEvidenceReport`, and `loadServingMonitor`.
- Produces: evidence rail/detail view and read-only serving identity/telemetry view.

- [ ] **Step 1: Write failing UI tests for evidence statuses, invalid reports, idle serving, valid bundle, and absent telemetry.**
- [ ] **Step 2: Run the focused Vitest file and verify RED.**
- [ ] **Step 3: Implement both pages with bounded internal panels and no action that mutates serving or orders.**
- [ ] **Step 4: Run all frontend tests, typecheck, build, and the no-page-scroll browser check.**
- [ ] **Step 5: Commit `feat: add Studio evidence and serving workspaces`.**

### Task 8: Full verification and documentation

**Files:**
- Modify: `studio/README.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml` only if the existing Studio checks do not cover the new files.

**Interfaces:**
- Produces: documented read-only scope and reproducible validation evidence.

- [ ] **Step 1: Document endpoints, registry paths, optional paper snapshot schema, and the explicit no-order boundary.**
- [ ] **Step 2: Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy trade_rl`, and focused Studio pytest.**
- [ ] **Step 3: Run the full Python suite with coverage and critical-coverage validation.**
- [ ] **Step 4: Run `npm test --prefix studio -- --run`, `npm run typecheck --prefix studio`, `npm run build --prefix studio`, and `npm run check:layout --prefix studio`.**
- [ ] **Step 5: Confirm the PR contains no credential, exchange, or order-routing additions and commit `docs: document Studio audit surfaces`.**
