# Trade RL Studio Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable, no-page-scroll Trade RL Studio shell with an approved dashboard design, typed API fallback, navigation, and verified React/Vite/TypeScript tooling.

**Architecture:** Create an independent `studio/` Vite application. Keep all network access behind one typed adapter, keep deterministic demo data separate from components, and compose the dashboard from small reusable panels. The existing Python core remains unchanged in this slice.

**Tech Stack:** React 19, Vite 8, TypeScript 7, Vitest 4, Testing Library, Lucide React, CSS Grid.

## Global Constraints

- The browser document must not vertically scroll.
- Header, sidebar, and status bar remain fixed inside the viewport grid.
- All source code uses strict TypeScript.
- The UI must show `NO-GO` and must not imply direct exchange execution is available.
- The frontend must work without a backend by using deterministic demo data.
- No training or artifact-authority logic is reimplemented in TypeScript.

---

### Task 1: Tooling and typed dashboard contract

**Files:**
- Create: `studio/package.json`
- Create: `studio/tsconfig.json`
- Create: `studio/tsconfig.app.json`
- Create: `studio/tsconfig.node.json`
- Create: `studio/vite.config.ts`
- Create: `studio/index.html`
- Create: `studio/src/data/types.ts`
- Create: `studio/src/data/demoOverview.ts`
- Test: `studio/src/api/studioApi.test.ts`
- Create: `studio/src/api/studioApi.ts`

**Interfaces:**
- Produces: `StudioOverview`, `StudioOverviewResult`, `loadStudioOverview(fetcher?)`.

- [ ] **Step 1: Write the failing API tests**

```ts
it('returns API data for a successful response', async () => {
  const result = await loadStudioOverview(async () =>
    new Response(JSON.stringify(demoOverview), { status: 200 }),
  )
  expect(result.source).toBe('api')
})

it('falls back to demo data on failure', async () => {
  const result = await loadStudioOverview(async () => {
    throw new Error('offline')
  })
  expect(result).toEqual({ source: 'demo', overview: demoOverview })
})
```

- [ ] **Step 2: Run the test and verify RED**

Run: `npm test -- --run src/api/studioApi.test.ts`
Expected: FAIL because `studioApi.ts` does not exist.

- [ ] **Step 3: Implement the typed adapter**

```ts
export async function loadStudioOverview(
  fetcher: typeof fetch = fetch,
): Promise<StudioOverviewResult> {
  try {
    const response = await fetcher('/api/studio/overview')
    if (!response.ok) return { source: 'demo', overview: demoOverview }
    const payload: unknown = await response.json()
    if (!isStudioOverview(payload)) return { source: 'demo', overview: demoOverview }
    return { source: 'api', overview: payload }
  } catch {
    return { source: 'demo', overview: demoOverview }
  }
}
```

- [ ] **Step 4: Run the API tests and verify GREEN**

Run: `npm test -- --run src/api/studioApi.test.ts`
Expected: 2 tests pass.

### Task 2: Fixed application shell

**Files:**
- Test: `studio/src/App.test.tsx`
- Create: `studio/src/App.tsx`
- Create: `studio/src/main.tsx`
- Create: `studio/src/components/AppShell.tsx`
- Create: `studio/src/components/TopBar.tsx`
- Create: `studio/src/components/Sidebar.tsx`
- Create: `studio/src/components/StatusBar.tsx`
- Create: `studio/src/styles.css`
- Create: `studio/src/test/setup.ts`

**Interfaces:**
- Consumes: `StudioOverviewResult`.
- Produces: fixed shell landmarks and page selection callback.

- [ ] **Step 1: Write failing shell tests**

```tsx
it('renders fixed shell and NO-GO status', async () => {
  render(<App initialOverview={{ source: 'demo', overview: demoOverview }} />)
  expect(screen.getByRole('banner')).toBeInTheDocument()
  expect(screen.getByRole('navigation')).toBeInTheDocument()
  expect(screen.getAllByText('NO-GO').length).toBeGreaterThan(0)
})
```

- [ ] **Step 2: Verify RED**

Run: `npm test -- --run src/App.test.tsx`
Expected: FAIL because `App.tsx` does not exist.

- [ ] **Step 3: Implement shell and viewport CSS**

Use `height: 100%; overflow: hidden` on `html`, `body`, and `#root`, plus a three-row/two-column application grid.

- [ ] **Step 4: Verify GREEN**

Run: `npm test -- --run src/App.test.tsx`
Expected: shell tests pass.

### Task 3: Dashboard composition

**Files:**
- Create: `studio/src/components/Panel.tsx`
- Create: `studio/src/components/MetricRing.tsx`
- Create: `studio/src/components/LineChart.tsx`
- Create: `studio/src/components/StabilityChart.tsx`
- Create: `studio/src/pages/DashboardPage.tsx`
- Test: `studio/src/pages/DashboardPage.test.tsx`

**Interfaces:**
- Consumes: `StudioOverview`.
- Produces: the complete approved dashboard surface.

- [ ] **Step 1: Write failing dashboard tests**

Assert the latest dataset, active jobs, latest runs, warnings, baseline comparison, stability chart, and production status are visible.

- [ ] **Step 2: Verify RED**

Run: `npm test -- --run src/pages/DashboardPage.test.tsx`
Expected: FAIL because the page does not exist.

- [ ] **Step 3: Implement the dashboard from reusable panels**

Keep chart SVGs code-native and labels accessible. Use bounded panel bodies with hidden overflow.

- [ ] **Step 4: Verify GREEN**

Run: `npm test -- --run src/pages/DashboardPage.test.tsx`
Expected: dashboard tests pass.

### Task 4: Navigation workspaces

**Files:**
- Create: `studio/src/pages/WorkspacePage.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/App.test.tsx`

**Interfaces:**
- Produces: page switching for all sidebar items without URL reload or document scroll.

- [ ] **Step 1: Add a failing navigation test**

```tsx
await user.click(screen.getByRole('button', { name: /Data Lab/i }))
expect(screen.getByRole('heading', { name: 'Data Lab' })).toBeInTheDocument()
expect(screen.queryByText('最新の実験結果サマリー')).not.toBeInTheDocument()
```

- [ ] **Step 2: Verify RED**

Run: `npm test -- --run src/App.test.tsx`
Expected: FAIL because navigation does not change pages.

- [ ] **Step 3: Implement page switching and bounded workspace placeholders**

Each workspace displays a compact header, stage rail, and next implementation actions inside the viewport.

- [ ] **Step 4: Verify GREEN**

Run: `npm test -- --run src/App.test.tsx`
Expected: all app tests pass.

### Task 5: Build and browser fidelity verification

**Files:**
- Create: `studio/README.md`
- Create: `studio/scripts/check-no-page-scroll.mjs`
- Modify: `studio/package.json`

**Interfaces:**
- Produces: reproducible verification commands and screenshot evidence.

- [ ] **Step 1: Add the browser layout check**

The script starts the built preview, opens Chromium at 1536×1024 and 1440×900, and asserts `document.documentElement.scrollHeight === window.innerHeight` and the same for `body`.

- [ ] **Step 2: Run full verification**

Run:

```bash
npm test -- --run
npm run typecheck
npm run build
npm run check:layout
```

Expected: all commands pass with no warnings or page overflow.

- [ ] **Step 3: Capture final screenshot**

Save the 1536×1024 screenshot as `/mnt/data/trade-rl-studio-dashboard.png` and compare it with the accepted concept for layout, density, color, typography, fixed chrome, and visible copy.
