# Synchronized Chart Workspace V6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Live Training around one existing Lightweight Charts instance with three synchronized analytical panes, direct chart interaction, trade lifecycle context, and low-frequency details hidden by default.

**Architecture:** React owns preset, range-selection, disclosure, and replay state. Lightweight Charts 5.2.0 continues to own chart lifecycle, panes, series, time scale, crosshair, markers, scrolling, and scaling. Existing `researchChartModel` remains the telemetry transformation source; the new workspace derives only contract-safe values such as gross exposure from existing weights and never invents cash balance or risk limits.

**Tech Stack:** React 19, TypeScript 7, Lightweight Charts 5.2.0, Vitest, Testing Library, existing Studio CSS tokens.

## Global Constraints

- Reuse the installed Lightweight Charts dependency; do not add or replace chart libraries.
- Keep `ResearchChartWorkspace` intact for compatibility and add the new workspace beside it.
- Use exactly three panes: Price & Execution, Portfolio, State & Risk.
- Presets change emphasis and pane proportions; they must not replace the shared time axis or destroy context.
- Range selection is opt-in and must not break ordinary pan/zoom.
- Do not display a cash balance or configured risk threshold because current telemetry does not provide either contract.
- Preserve keyboard-accessible controls, focus restoration, no page scroll, and existing NO-GO/evidence semantics.
- Tests must cover pane assignment, preset emphasis, range selection, latest-value labels, trade bands, page disclosure, and existing replay commits.

---

### Task 1: Extend the chart model with contract-safe state data

**Files:**
- Modify: `studio/src/live/researchChartModel.ts`
- Modify: `studio/src/live/researchChartModel.test.ts`

**Interfaces:**
- Produces `grossExposure: ResearchLinePoint[]` from `sum(abs(weightsAfter)) * 100`.
- Produces `tradeBands: ResearchTradeBand[]` from non-flat position intervals.

- [ ] Write failing model tests for gross exposure and closed/open trade bands.
- [ ] Run the focused model tests and verify failure.
- [ ] Implement the minimal model transformation.
- [ ] Run the focused model tests and verify success.

### Task 2: Add the synchronized Lightweight Charts workspace

**Files:**
- Create: `studio/src/live/SynchronizedResearchChartWorkspace.tsx`
- Create: `studio/src/live/SynchronizedResearchChartWorkspace.test.tsx`

**Interfaces:**
- Consumes the existing replay props used by `ResearchChartWorkspace`.
- Owns `overview | trade | risk` emphasis preset and opt-in range selection.
- Uses pane 0 for candles/target/executed, pane 1 for equity/baseline/drawdown, pane 2 for gross exposure/reward/cost.

- [ ] Write failing tests for three pane indices, stretch factors, preset option changes, and range selection.
- [ ] Implement chart creation, series lifecycle, markers, price labels, trade-band overlay, and range overlay.
- [ ] Verify chart teardown and replay callbacks remain correct.

### Task 3: Replace the always-visible summary and inspector layout

**Files:**
- Modify: `studio/src/pages/LiveTrainingPage.tsx`
- Modify: `studio/src/pages/LiveTrainingPage.test.tsx`

**Interfaces:**
- Replaces the old chart component with `SynchronizedResearchChartWorkspace`.
- Removes the three summary cards because latest values are labeled on the chart.
- Moves `ResearchChartInspector` into a collapsed native details disclosure beneath the chart.

- [ ] Write failing page tests for absence of summary cards and collapsed details.
- [ ] Implement the full-width chart layout and details disclosure.
- [ ] Verify chart selection still commits replay and the inspector updates after disclosure opens.

### Task 4: Add workspace-specific responsive styles

**Files:**
- Create: `studio/src/synchronizedResearchChart.css`
- Modify: `studio/src/pages/LiveTrainingPage.tsx`

**Interfaces:**
- Styles chart overlays without overriding Lightweight Charts internals.
- Keeps pointer events disabled on trade bands and enabled only for explicit range mode.

- [ ] Add layout, overlay, direct-summary, details, focus, reduced-motion, and forced-colors styles.
- [ ] Run layout checks at supported desktop breakpoints.

### Task 5: Verify and publish

- [ ] Run Studio unit tests.
- [ ] Run Studio typecheck and production build.
- [ ] Run layout checks.
- [ ] Inspect the final diff for invented metrics, duplicated controls, and chart-library replacement.
- [ ] Open a draft PR against `main` with verification evidence and explicitly list the deferred telemetry-contract work.
