# Research Trading Chart Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Live Training static SVG replay with a directly manipulable, desktop research chart workspace built on Lightweight Charts.

**Architecture:** React owns source selection, replay state, layer visibility, and inspector state. A pure model converts telemetry into timeframe buckets and chart series. One Lightweight Charts instance renders four synchronized panes and reports hover/click selections back to React.

**Tech Stack:** React 19, TypeScript 7, Vite 8, Vitest 4, Testing Library, Playwright, Lightweight Charts 5.2.0.

## Global Constraints

- Work on `agent/research-trading-chart-workspace`, based on PR #316 head `77e10276b980d6ac4940d774c9a4a05b3bda0b0a`.
- Keep the first release desktop-only and preserve the existing fixed-viewport/no-page-scroll contract.
- Do not show a decorative LIVE connection badge or a “接続 / LIVE / Seed / Env” metric card.
- Keep exactly three summary metrics: RL equity, baseline delta, and drawdown.
- Use one Lightweight Charts instance with four panes and one shared time scale.
- Keep `layout.attributionLogo: true`.
- A primary-asset zero weight delta must never produce BUY or SELL.
- Hover previews; click commits and pauses replay.
- Use TDD: create each failing behavior test and verify RED before production implementation.
- Do not merge PR #316 or the feature PR without an explicit user request.

---

### Task 1: Dependency and pure chart model

**Files:**
- Modify: `studio/package.json`
- Modify: `studio/package-lock.json`
- Create: `studio/src/live/researchChartModel.ts`
- Create: `studio/src/live/researchChartModel.test.ts`

**Interfaces:**
- Produces: `ResearchTimeframe`, `ResearchChartLayers`, `ResearchChartData`, `buildResearchChartData(records, symbol, timeframe)`, `previousEventIndex(records, cursor)`, and `nextEventIndex(records, cursor)`.
- `ResearchChartData` contains primitive numeric epoch-second times so the model remains independent of the renderer.

- [ ] **Step 1: Add failing model tests**

Cover:

```ts
it('aggregates 15 minute telemetry into hourly candles and keeps the latest metrics')
it('drops invalid timestamps instead of inventing chart times')
it('does not create a directional marker when the primary asset delta is zero')
it('filters records to the selected symbol')
it('navigates only non-rollout events')
```

Use telemetry fixtures containing nanosecond timestamps, multiple symbols, and a multi-asset position event.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test --prefix studio -- --run src/live/researchChartModel.test.ts
```

Expected: FAIL because `researchChartModel` does not exist.

- [ ] **Step 3: Pin Lightweight Charts 5.2.0**

Add exactly:

```json
"lightweight-charts": "5.2.0"
```

Regenerate `studio/package-lock.json` with npm lockfile version 3 and preserve all existing exact dependency versions.

- [ ] **Step 4: Implement the pure model**

Implement timeframe seconds:

```ts
const TIMEFRAME_SECONDS = { '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400 } as const
```

Normalize fractional seconds to milliseconds, append `Z` when no offset exists, bucket by floor division, sort by epoch then sequence, aggregate OHLC, keep the final finite metric values, and derive truthful event markers.

- [ ] **Step 5: Verify GREEN**

Run the focused test and then:

```bash
npm test --prefix studio -- --run
npm run typecheck --prefix studio
```

Expected: all existing tests plus model tests pass.

- [ ] **Step 6: Commit**

```bash
git add studio/package.json studio/package-lock.json studio/src/live/researchChartModel.ts studio/src/live/researchChartModel.test.ts
git commit -m "feat(studio): model research chart telemetry"
```

### Task 2: Replay toolbar and atomic source selection

**Files:**
- Create: `studio/src/live/ReplayToolbar.tsx`
- Create: `studio/src/live/ReplayToolbar.test.tsx`

**Interfaces:**
- Consumes: `JobSummary`, `ResearchChartLayers`, replay speed, source options, and callbacks.
- Produces: `ReplayToolbar` with Run selection, source popover, transport controls, speed, latest-follow, layer popover, and reset.

- [ ] **Step 1: Add failing component tests**

Cover:

```ts
it('keeps Seed and Environment inside the source popover')
it('applies Seed and Environment atomically')
it('emits first previous play next last and speed callbacks')
it('toggles latest-follow and chart layers')
it('does not render LIVE connection status')
```

- [ ] **Step 2: Verify RED**

```bash
npm test --prefix studio -- --run src/live/ReplayToolbar.test.tsx
```

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement toolbar**

Use native controls. Keep local draft Seed/Environment values while the popover is open. Apply both in one callback:

```ts
onSourceChange({ seed: draftSeed, environmentId: draftEnvironmentId })
```

Close on Apply, Escape, and outside click. Keep the current selection summarized in the “対象を変更” button accessible name without adding a status card.

- [ ] **Step 4: Verify GREEN and commit**

```bash
npm test --prefix studio -- --run src/live/ReplayToolbar.test.tsx
npm run typecheck --prefix studio
git add studio/src/live/ReplayToolbar.tsx studio/src/live/ReplayToolbar.test.tsx
git commit -m "feat(studio): add research replay toolbar"
```

### Task 3: Four-pane chart runtime and inspector

**Files:**
- Create: `studio/src/live/ResearchChartWorkspace.tsx`
- Create: `studio/src/live/ResearchChartWorkspace.test.tsx`
- Create: `studio/src/live/ResearchChartInspector.tsx`
- Create: `studio/src/live/ResearchChartInspector.test.tsx`
- Delete after replacement: `studio/src/live/MarketReplayChart.tsx`
- Delete after replacement: `studio/src/live/MarketReplayChart.test.tsx`

**Interfaces:**
- Consumes: `ResearchChartData`, committed sequence, layer visibility, follow-latest, reset token, range preset, and callbacks.
- Produces: direct zoom/pan/crosshair interaction, click-to-commit, synchronized inspector, timeframe/symbol/range controls, and scrubber.

- [ ] **Step 1: Mock the renderer and add failing tests**

Mock `lightweight-charts` at its public boundary. Assert that the component:

```ts
it('creates one chart with four pane indices')
it('loads candlesticks weights reward cost equity baseline and drawdown')
it('creates truthful event markers')
it('previews on crosshair and commits on chart click')
it('applies range presets and follows the newest record')
it('removes the chart and observers on unmount')
it('does not recreate the chart for cursor-only updates')
```

Add inspector tests proving hover preview wins over committed cursor and essential values remain visible without hover.

- [ ] **Step 2: Verify RED**

```bash
npm test --prefix studio -- --run src/live/ResearchChartWorkspace.test.tsx src/live/ResearchChartInspector.test.tsx
```

Expected: FAIL because the components do not exist.

- [ ] **Step 3: Implement chart lifecycle**

Create the chart once in an effect scoped to the container. Configure:

```ts
layout: { background: { color: '#07111a' }, textColor: '#8193a2', attributionLogo: true }
crosshair: { mode: CrosshairMode.Normal }
handleScroll: true
handleScale: true
```

Add candlestick and line series with pane indices 0, 1, 2, and 3. Give drawdown a separate left price scale in pane 3. Add markers with `createSeriesMarkers`.

Use one ResizeObserver. Subscribe to crosshair move, click, and visible logical range changes. Manual visible-range change disables latest-follow through `onManualNavigation` unless the change was initiated by the component.

- [ ] **Step 4: Implement chart header, inspector, and scrubber**

Symbol and timeframe controls sit in the chart header. Range presets are 1H, 24H, 7D, and all. The scrubber uses raw replay indices and commits on input. The right inspector shows action, executed target, reward, cost, drawdown, OHLC, portfolio, baseline, checkpoint evidence, and sequence/time.

- [ ] **Step 5: Verify GREEN and delete old SVG chart**

```bash
npm test --prefix studio -- --run src/live/ResearchChartWorkspace.test.tsx src/live/ResearchChartInspector.test.tsx
npm run typecheck --prefix studio
```

Delete the old `MarketReplayChart` only after the replacement tests pass.

- [ ] **Step 6: Commit**

```bash
git add studio/src/live
git commit -m "feat(studio): add interactive multi-pane research chart"
```

### Task 4: Integrate the workspace into Live Training

**Files:**
- Modify: `studio/src/pages/LiveTrainingPage.tsx`
- Modify: `studio/src/pages/LiveTrainingPage.test.tsx`
- Modify: `studio/src/liveTraining.css`
- Modify: `studio/scripts/check-live-training-layout.mjs`

**Interfaces:**
- Consumes: toolbar and workspace components from Tasks 2 and 3.
- Produces: the approved balanced desktop Live Training experience.

- [ ] **Step 1: Update page tests first**

Add or revise tests proving:

```ts
it('renders Run as the only always-visible source selector')
it('does not render LIVE connection chrome')
it('changes Seed and Environment through the source popover')
it('pauses and commits replay when the chart selects a record')
it('navigates to previous and next non-rollout events')
it('keeps exactly three summary metrics')
it('preserves checkpoint evidence selection')
it('isolates replay values by the selected environment and current episode')
```

- [ ] **Step 2: Verify RED**

```bash
npm test --prefix studio -- --run src/pages/LiveTrainingPage.test.tsx
```

Expected: FAIL against the old header and SVG workspace.

- [ ] **Step 3: Refactor page state**

Remove `replayMode`, `timelineMode`, connection-label UI, old transport, old event list, old four-card metric grid, and old direct chart geometry. Add:

```ts
symbol
researchTimeframe
rangePreset
followLatest
layers
chartResetToken
previewRecord
```

Keep telemetry polling independent from playback. Source changes reset environment/cursor safely. Previous/next controls use the pure event-index helpers.

- [ ] **Step 4: Compose the approved layout**

Header: title, NO-GO, view selector, and toolbar. Main area: three summary metrics, chart/inspector workspace, and no page scroll. Preserve diagnostics as the alternate view.

- [ ] **Step 5: Replace CSS and layout assertions**

Keep the existing dark research theme variables. Add fixed desktop grid rules, chart pane sizing, popovers, toolbar, inspector, and scrubber styling. Remove selectors used only by the deleted SVG transport/events. Update the static layout script to require the new workspace and forbid the removed LIVE badge/card.

- [ ] **Step 6: Verify GREEN and commit**

```bash
npm test --prefix studio -- --run src/pages/LiveTrainingPage.test.tsx
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
git add studio/src/pages/LiveTrainingPage.tsx studio/src/pages/LiveTrainingPage.test.tsx studio/src/liveTraining.css studio/scripts/check-live-training-layout.mjs
git commit -m "feat(studio): integrate research chart workspace"
```

### Task 5: Full verification, documentation, and PR

**Files:**
- Modify if required: `README.md`
- Modify if required: Studio documentation that describes Live Training.

**Interfaces:**
- Produces: review-ready Draft PR with RED/GREEN evidence and dependency disclosure.

- [ ] **Step 1: Run complete Studio verification**

```bash
npm ci --prefix studio --no-audit --no-fund
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

- [ ] **Step 2: Run repository verification**

Use the repository CI workflow to run Python tests, coverage, Ruff, format, MyPy, import architecture, dead-code, compatibility, and training-image checks.

- [ ] **Step 3: Review the diff**

Check for stale SVG code, duplicate replay state, unbounded effects, missing cleanup, renderer recreation, incorrect primary-asset markers, hidden essential values, attribution disabled, page scrolling, and unrelated changes.

- [ ] **Step 4: Update Draft PR**

Open against `main` so CI triggers. State that it is stacked on PR #316. Include focused RED evidence, final GREEN run IDs, changed-file scope, renderer version, attribution handling, and the exact excluded scope.

- [ ] **Step 5: Final verification commit if needed**

Commit only necessary documentation or verification fixes. Do not merge.
