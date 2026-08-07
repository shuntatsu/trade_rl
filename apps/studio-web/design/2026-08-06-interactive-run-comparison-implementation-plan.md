# Interactive Run Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the equal-weight Compare panel grid with an interactive, ordinal-axis comparison workspace that shows where the right run differs from the left run without inventing timestamps.

**Architecture:** Extend each canonical wealth point with its authoritative fold index, then derive chart geometry and selection summaries in a pure TypeScript model. Render one SVG workspace with synchronized cumulative-wealth and Right−Left panes, plus a fold strip and an overlay inspector sheet. Keep run selection, eligibility, NO-GO semantics, URL state, and read-only behavior.

**Tech Stack:** Python 3.12 dataclasses, React 19, TypeScript 7, SVG, Vitest, Testing Library, Playwright Chromium.

## Global Constraints

- Keep `productionStatus` at `NO-GO`.
- Do not synthesize wall-clock time; the horizontal axis is sealed evaluation index/order.
- Do not replace or modify Live Training, Lightweight Charts, or Trade lifecycle background bands.
- Hover/focus previews; click/Enter commits; normal drag pans; Range mode drag selects; wheel zooms; double-click resets; Escape clears.
- Essential Left/Right/current-delta values remain visible without hover.
- Desktop validation targets are 1536×1024, 1440×900, and 1180×800.

---

### Task 1: Authoritative fold identity in comparison points

**Files:**
- Modify: `trade_rl/studio/contracts.py`
- Modify: `trade_rl/studio/comparison.py`
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/api/guards.ts`
- Test: `tests/studio/test_comparison.py`
- Test: `studio/src/api/guards.test.ts`

**Interfaces:**
- Produces `ComparisonSeriesPoint.fold_index: int | None` / `foldIndex: number | null`.
- Start point has `foldIndex=null`; every return-derived point has its source `foldIndex`.

- [ ] Write failing Python and TypeScript contract tests for start/fold point identities.
- [ ] Run focused tests and confirm the new field is missing.
- [ ] Extend serialization, runtime guards, and series generation minimally.
- [ ] Run focused tests and confirm they pass.

### Task 2: Pure interactive comparison model

**Files:**
- Create: `studio/src/compare/comparisonWorkspaceModel.ts`
- Create: `studio/src/compare/comparisonWorkspaceModel.test.ts`

**Interfaces:**
- Produces `buildComparisonWorkspaceModel(comparison)` with points, fold spans, final labels, domains, metric verdicts, and `summarizeComparisonRange(model, start, end)`.
- Right−Left is computed only where both selected wealth values are present.

- [ ] Write failing tests for shared wealth domain, zero-inclusive delta domain, fold spans, direct-label collision offsets, and range summaries.
- [ ] Implement deterministic pure functions with no DOM dependency.
- [ ] Run focused model tests.

### Task 3: Interactive SVG comparison workspace

**Files:**
- Create: `studio/src/compare/InteractiveComparisonWorkspace.tsx`
- Create: `studio/src/compare/InteractiveComparisonWorkspace.test.tsx`
- Create: `studio/src/comparisonWorkspace.css`
- Modify: `studio/src/main.tsx`

**Interfaces:**
- Props include the model, committed point/range, and callbacks for point/range state.
- Exposes accessible `role="application"` with direct labels and an equivalent hidden table.

- [ ] Write failing tests for two synchronized panes, point commit, keyboard movement, Range selection, zoom reset, and fold selection.
- [ ] Implement SVG paths, crosshair, selection band, direct labels, fold strip, pan/zoom/range state, and reset controls.
- [ ] Run focused component tests and TypeScript typecheck.

### Task 4: Compare page composition and inspector

**Files:**
- Create: `studio/src/compare/ComparisonInspectorSheet.tsx`
- Rewrite: `studio/src/pages/ComparePage.tsx`
- Modify: `studio/src/pages/ComparePage.test.tsx`
- Modify: `studio/src/state/urlState.ts`
- Modify: `studio/src/state/urlState.test.ts`

**Interfaces:**
- URL preserves `left`, `right`, `comparePoint`, `compareStart`, and `compareEnd`.
- Inspector tabs: Selection, Metrics, Config.
- Inspector overlay must not change chart dimensions.

- [ ] Write failing page tests for the new workspace, URL restoration, selection summary, inspector tabs, and stale-request exclusion.
- [ ] Implement selector toolbar, eligibility ribbon, winner summary, workspace, and overlay sheet.
- [ ] Remove the legacy equal-weight metrics/config/fold panel grid.
- [ ] Run Compare page tests and full Studio tests.

### Task 5: Browser QA and repository verification

**Files:**
- Modify: `studio/scripts/check-no-page-scroll.mjs`

**Interfaces:**
- Saves `trade-rl-studio-compare-{viewport}.png` in `studio/qa-screenshots`.
- Verifies chart geometry is unchanged when the inspector opens.

- [ ] Add Chromium checks for two panes, direct labels, point selection, Range selection, wheel zoom, reset, inspector overlay, and no page overflow.
- [ ] Run `npm test --prefix studio -- --run`, `npm run typecheck --prefix studio`, `npm run build --prefix studio`, and `npm run check:layout --prefix studio`.
- [ ] Run Ruff, format, MyPy, Import Linter, full pytest/coverage, compatibility, and training-image CI on the exact PR head.
