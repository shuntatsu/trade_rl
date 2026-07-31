# Studio Data Truthfulness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct four high-priority Studio defects that can misrepresent research data or prevent live updates.

**Architecture:** Keep changes local to existing Studio components and hooks. Preserve API schemas and request-generation guards; replace only interval scheduling, chart-domain calculation, marker direction, and initial log orchestration.

**Tech Stack:** React 19, TypeScript 7, Vitest 4, Testing Library, Vite 8, GitHub Actions.

## Global Constraints

- Frontend-only repair; no backend schema or API changes.
- One PR based on `main` commit `1e66225db425f24b02d107f819c9660a9477b3ac`.
- Preserve Read-only, NO-GO, seed/environment/episode identity, and stream-generation behavior.
- Do not mix Serving Monitor stale-state UI, Settings, Dashboard navigation, responsive layout, typography, or README port work into this PR.
- Follow RED → GREEN for every behavioral change.

---

### Task 1: Dashboard shared Y-domain

**Files:**
- Modify: `studio/src/components/LineChart.tsx`
- Modify: `studio/src/pages/DashboardPage.test.tsx`

**Interfaces:**
- Consumes: `EquityPoint[]` with `rl` and `baseline` values.
- Produces: two SVG paths projected through one `{ minimum, maximum }` domain.

- [ ] **Step 1: Write the failing test**

Render `DashboardPage` with an overview whose equity points include RL wealth `3.0` and Baseline wealth `1.2`. Read the final Y coordinates from `.chart-line--rl` and `.chart-line--baseline`; assert the Baseline endpoint is projected using the shared 0.8–3.0 domain rather than its own 0.8–2.0 domain.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
npm test --prefix studio -- --run studio/src/pages/DashboardPage.test.tsx
```

Expected: the shared-domain coordinate assertion fails against current per-series scaling.

- [ ] **Step 3: Implement the minimal fix**

Add one `sharedDomain(points)` calculation over both keys. Change `pathFor` to consume that domain and pass the same value to both paths.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run the same command and require zero failures.

---

### Task 2: Primary-symbol marker direction

**Files:**
- Create: `studio/src/live/MarketReplayChart.test.tsx`
- Modify: `studio/src/live/MarketReplayChart.tsx`

**Interfaces:**
- Consumes: `TrainingTelemetryRecord.weightsBefore[0]` and `weightsAfter[0]`.
- Produces: BUY for positive primary delta and SELL for negative primary delta.

- [ ] **Step 1: Write the failing test**

Render one `position` record with weights changing from `[0.2, 0.1]` to `[0.1, 0.2]`. Assert that the chart contains `.live-marker--sell` and does not contain `.live-marker--buy`.

- [ ] **Step 2: Run the test to verify RED**

```bash
npm test --prefix studio -- --run studio/src/live/MarketReplayChart.test.tsx
```

Expected: current signed-sum logic returns zero and renders BUY, so the test fails.

- [ ] **Step 3: Implement the minimal fix**

Replace the multi-asset summation in `weightDelta` with index-0 after-minus-before.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run the same command and require zero failures.

---

### Task 3: Serial telemetry and metric polling

**Files:**
- Modify: `studio/src/live/useTrainingTelemetry.test.tsx`
- Modify: `studio/src/live/useTrainingTelemetry.ts`
- Modify: `studio/src/live/useTrainingMetrics.test.tsx`
- Modify: `studio/src/live/useTrainingMetrics.ts`

**Interfaces:**
- Consumes: existing async `refresh()` functions.
- Produces: immediate refresh followed by one `setTimeout` scheduled only after completion.

- [ ] **Step 1: Write telemetry overlap test**

Use fake timers and unresolved status/events promises. After the first request starts, advance 1,000 ms and assert each loader is still called once. Resolve the promises, flush microtasks, advance another 1,000 ms, and assert the second poll starts.

- [ ] **Step 2: Write metric overlap test**

Use an unresolved metrics-status promise. Advance 2,000 ms and assert only one status request exists. Resolve the first cycle, advance another 2,000 ms, and assert the second cycle starts.

- [ ] **Step 3: Run both tests to verify RED**

```bash
npm test --prefix studio -- --run \
  studio/src/live/useTrainingTelemetry.test.tsx \
  studio/src/live/useTrainingMetrics.test.tsx
```

Expected: fixed intervals start overlapping requests and both new assertions fail.

- [ ] **Step 4: Implement serial scheduling**

In each effect, create `active` and `timer` locals. Define an async `poll` that awaits `refresh()` and schedules the next timeout only when still active. Cleanup clears the timeout and increments the existing request generation.

- [ ] **Step 5: Run both tests to verify GREEN**

Run the same command and require zero failures.

---

### Task 4: Run Center initial log

**Files:**
- Modify: `studio/src/pages/RuntimePages.test.tsx`
- Modify: `studio/src/pages/RunCenterPage.tsx`

**Interfaces:**
- Consumes: `loadJobs()` result and existing selected job ID.
- Produces: immediate `loadJobLog(next)` for the resolved selection.

- [ ] **Step 1: Write the failing test**

Remove the initial job-row click from the Run Center test. Assert that `loadJobLog('job-1')` is called and the log contains `step 2` immediately after the jobs load.

- [ ] **Step 2: Run the test to verify RED**

```bash
npm test --prefix studio -- --run studio/src/pages/RuntimePages.test.tsx
```

Expected: the log remains at `ログを読み込んでいます…` because the initial selection does not load it.

- [ ] **Step 3: Implement the minimal fix**

Resolve `next` directly in `refreshJobs`, update selection and URL, then await `loadLog(next)`. When no jobs exist, clear lines and invalidate pending log requests.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run the same command and require zero failures.

---

### Task 5: Full verification and PR finalization

**Files:**
- No additional production files.

- [ ] **Step 1: Run the complete Studio suite**

```bash
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

- [ ] **Step 2: Review the final diff**

Confirm only the approved four defects, their tests, and Superpowers docs changed.

- [ ] **Step 3: Run repository CI on one unchanged head**

Require the Studio frontend tests/typecheck/build/layout portion and the complete repository matrix to pass.

- [ ] **Step 4: Keep the PR draft until verification is complete**

The final PR description must list the four root causes, regression tests, exact verification head, and any remaining unrelated review items.