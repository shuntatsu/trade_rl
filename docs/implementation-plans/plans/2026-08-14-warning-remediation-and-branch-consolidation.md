# Warning Remediation and Branch Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the Vite >500 KiB production-chunk warning and Python 3.12 fork deprecation warnings at their causes, then verify all prior work remains consolidated in `integration/cost-aware-causal-teacher-final2`.

**Architecture:** Use workspace-level dynamic imports so non-default Studio pages form semantic chunks, backed by an emitted-file size gate. Replace fork-inherited normalizer closure state with a picklable worker specification initialized under `forkserver`/`spawn`, preserving serial semantics and observation order.

**Tech Stack:** React 19, Vite 8.1, Vitest, Node.js, Python 3.12, concurrent.futures, multiprocessing, pytest, GitHub Actions.

## Global Constraints

- No merge to `main`.
- No force-push or history rewrite.
- No warning filters or Vite warning-limit inflation.
- No trading/reward/risk/evaluation semantic changes.
- Final verification must target one exact HEAD.

---

### Task 1: Add a fail-closed emitted bundle-size gate

**Files:**
- Create: `frontend/scripts/check-bundle-size.mjs`
- Modify: `frontend/package.json`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `frontend/dist/assets/*.js` produced by `vite build`.
- Produces: exit 0 only when every emitted JS chunk is `<= 512000` bytes; non-zero when dist is missing, no chunks exist, or any chunk exceeds the limit.

- [ ] Add the check script and `check:bundle` package command without changing application imports.
- [ ] Add `npm run check:bundle --prefix frontend` immediately after the production build in Core CI.
- [ ] Push the RED commit and verify Core CI fails specifically because the existing `521.26 kB` chunk exceeds `512000` bytes.

### Task 2: Split non-default Studio workspaces

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx` only if async rendering requires `findBy*` assertions; do not weaken navigation assertions.

**Interfaces:**
- `DashboardPage` remains eager.
- Non-default pages are `React.lazy` dynamic imports under one `Suspense` fallback inside `AppShell`.

- [ ] Convert Data/Experiments/Runs/Live/Compare/Evidence/Serving/Settings pages to lazy imports.
- [ ] Preserve Dashboard eager rendering and the existing history-backed navigation contract.
- [ ] Run frontend tests, typecheck, production build, `check:bundle`, and fixed-layout checks.
- [ ] Inspect emitted chunk sizes and confirm no Vite >500 KiB warning remains.

### Task 3: Add RED coverage for safe multiprocessing start methods

**Files:**
- Modify: `tests/workflows/test_market_walk_forward.py`

**Interfaces:**
- Exercise the real normalizer collection path with more than one worker.
- Observe multiprocessing context selection and warnings rather than filtering them.

- [ ] Add a test asserting the parallel normalizer path does not request `fork` when `forkserver` is available.
- [ ] Add an integration test with a live background thread that captures deprecation warnings during normalizer fitting and requires no fork warning.
- [ ] Verify the tests fail against the current explicit `get_context("fork")` implementation for the intended reason.

### Task 4: Replace fork inheritance with a serializable normalizer worker contract

**Files:**
- Modify: `trade_rl/workflows/_market_walk_forward_core.py`
- Modify: `tests/workflows/test_market_walk_forward.py`

**Interfaces:**
- Add a frozen worker-spec dataclass containing the training dataset/view, authored run config, bound alpha/factor providers, and environment episode length needed to rebuild the worker environment.
- Worker initializer receives the spec through normal multiprocessing serialization.
- Worker task receives only partition bounds and reconstructs/uses the worker environment from the initialized spec.
- Start method resolves to `forkserver` when available, otherwise `spawn`.

- [ ] Implement a pure start-method resolver that never returns `fork`.
- [ ] Replace `_NORMALIZER_ENVIRONMENT_FACTORY` with explicit serialized worker state initialized by the process pool.
- [ ] Reconstruct the same `build_market_environment(...)` call inside workers.
- [ ] Preserve the serial path for one worker and finite-horizon collection.
- [ ] Add/retain serial-vs-parallel matrix parity assertions including exact row ordering.
- [ ] Run the targeted warning-producing workflow tests and confirm the fork warning is absent.

### Task 5: Full verification and falsification review

**Files:**
- Modify PR description only after exact-head evidence is complete.

- [ ] Run/observe exact-head Core CI, including frontend 127+ tests, typecheck, build, bundle gate, layout, Ruff, format, Mypy, Import Linter, full pytest/coverage, critical coverage, Windows, Ubuntu, and Training image.
- [ ] Verify Nautilus Capability and PostgreSQL Catalog on the same head.
- [ ] Inspect full pytest warning summary; require zero multiprocessing/fork warnings and enumerate any unrelated residual warnings.
- [ ] Inspect final diff for warning suppression, `chunkSizeWarningLimit`, `filterwarnings`, `PYTHONWARNINGS`, `get_context("fork")`, accidental debug code, and unrelated refactors.
- [ ] Recheck review threads and PR state.

### Task 6: Consolidation verification

**Files:** none unless PR description update is allowed.

- [ ] Compare `codex/universal-real-data-training` -> final2 and require final2 `behind_by=0`.
- [ ] Compare `integration/cost-aware-causal-teacher-final` -> final2 and require final2 `behind_by=0`.
- [ ] Compare `integration/cost-aware-causal-teacher-review` -> final2 and require final2 `behind_by=0`.
- [ ] Compare `main` -> final2 and require final2 `behind_by=0` relative to current main.
- [ ] Do not fabricate physical branch deletion if no delete-ref action is available; report old refs as redundant aliases whose commits are fully contained.
- [ ] Mark PR Ready only after every Quality Gate item has evidence; do not merge to main without a separate explicit merge request.
