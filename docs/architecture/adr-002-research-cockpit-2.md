# ADR-002: Research Cockpit 2.0 frontend architecture

Status: Proposed
Date: 2026-08-10
Scope: `frontend/` only unless an explicitly required read-only Studio API contract is missing

## 1. Decision

Trade RL Studio will be redesigned as **Research Cockpit 2.0**.

The redesign will preserve the existing research and evidence contracts while replacing page-by-page visual conventions with one operational workspace model:

- a compact global navigation rail,
- a context bar for current run/source identity,
- one dominant primary evidence viewport,
- a synchronized inspector for committed selection,
- a compact status/transport region,
- progressive disclosure for secondary diagnostics.

The redesign is not a new trading terminal. Production remains `NO-GO`. The Studio remains local-first and read-only with respect to checkpoint selection, sealed-test opening, release approval, bundle activation, credentials, and exchange orders.

## 2. Motivation

The current frontend already contains substantial high-value behavior:

- Dashboard is decision-oriented rather than a generic KPI page.
- Live Training uses `lightweight-charts` with synchronized panes, zoom/pan, crosshair preview, replay controls, chart-click commit, current-episode isolation, and bounded telemetry buffering.
- Live Training distinguishes target/executed exposure and preserves trade lifecycle bands such as `ENTRY -> REDUCE -> EXIT`.
- Compare has an interactive ordinal workspace rather than a static card grid.
- Evidence and Serving surfaces enforce read-only identity and validation semantics.
- URL state restores important analytical selections.
- fixed-viewport layout checks protect the desktop research workflow.

The main remaining weakness is not lack of features. It is inconsistency of information hierarchy and interaction grammar across workspaces. Different pages currently decide independently where identity, status, controls, evidence, details, and errors live. As functionality grows, this increases cognitive load and encourages repeated one-off CSS and component patterns.

Research Cockpit 2.0 therefore standardizes the shell and interaction model while reusing the strongest existing feature implementations.

## 3. Goals

### 3.1 Primary goals

1. Make the primary evidence visually dominant on every analytical workspace.
2. Make run/source identity, freshness, and `NO-GO` state continuously understandable without decorative status cards.
3. Standardize preview versus committed selection semantics.
4. Keep controls next to the evidence they affect.
5. Make Dashboard answer one question first: **what is the next research blocker or action?**
6. Make Live Training answer: **what did the policy do, why, and what happened next?**
7. Make Compare answer: **where do two runs/folds materially differ and why?**
8. Make Evidence answer: **can this artifact and its lineage be trusted?**
9. Preserve fail-closed evidence behavior and existing research isolation contracts.
10. Reduce CSS and component duplication without introducing a large design-system dependency.

### 3.2 Secondary goals

- Improve keyboard navigation and focus visibility.
- Make loading, stale, partial, offline, and invalid states visually distinct.
- Keep screenshots analytically meaningful without requiring hover.
- Retain bounded browser memory and predictable rendering cost.
- Keep the implementation compatible with the existing Vite/React/TypeScript stack.

## 4. Non-goals

The redesign will not:

- add order placement or exchange credential entry,
- add release or promotion approval actions,
- weaken `NO-GO` or evidence validation rules,
- replace `lightweight-charts` merely for visual novelty,
- rewrite the Studio backend unless a missing read-only field blocks a defined UI requirement,
- introduce a general-purpose state-management framework unless existing local/URL state becomes demonstrably insufficient,
- support mobile or tablet layouts in this phase,
- create a user-customizable dashboard builder,
- add decorative animation, 3D, particles, or ambient effects,
- perform unrelated repository restructuring.

## 5. Design principles

### 5.1 Evidence first

Every workspace gets one dominant evidence surface. Supporting metadata and diagnostics are subordinate.

Avoid equal-weight card grids when the user is trying to make a single analytical decision.

### 5.2 Preview is temporary; selection is committed

The interaction grammar is shared across analytical views:

- hover/focus: preview only,
- click/Enter: commit selection,
- committed selection drives the inspector and URL state where appropriate,
- Escape/reset clears or restores the defined default state,
- telemetry refresh must not silently overwrite a committed analytical selection.

Live Training's existing preview/commit model is the reference behavior.

### 5.3 Controls belong to their evidence

Global identity controls belong in the context bar. Chart-only controls belong immediately above the chart. Replay controls belong with replay state. Evidence filters belong with the evidence tree/list.

A distant generic control panel must not become the default dumping ground for unrelated toggles.

### 5.4 Color is semantic

The visual system uses restrained semantic roles:

- neutral surfaces for structure,
- cyan/blue for selection and informational focus,
- green for validated/success state,
- amber for warning/stale/attention,
- red for invalid/risk/`NO-GO`/failure.

Gradients, glows, and accent borders are not used as decoration where a neutral surface communicates hierarchy more clearly.

### 5.5 Last-known-good evidence remains visible when safe

For polling or streaming surfaces, a transient fetch failure should not erase previously validated evidence. The UI will distinguish:

- `LIVE`: current data is arriving within expected cadence,
- `STALE`: previously valid data exists but freshness is outside the expected window,
- `PARTIAL`: some required subresources are unavailable while the visible subset remains valid,
- `OFFLINE`: the Studio API cannot currently be reached and no current refresh is possible,
- `INVALID`: received evidence failed a contract/identity/digest/schema check.

`INVALID` is not softened into stale or partial. It fails closed.

## 6. Global application shell

### 6.1 Layout

Desktop target: 1440 x 900 baseline, with a minimum supported width consistent with the current desktop-only contract.

The shell becomes:

```text
+------------------------------------------------------------------------+
| Trade RL | Workspace / context identity | freshness | GPU | NO-GO      |
+--------+-----------------------------------------------+----------------+
|        |                                               |                |
| Nav    |              PRIMARY WORKSPACE                |   INSPECTOR    |
| rail   |                                               |                |
|        |                                               |                |
|        |                                               |                |
+--------+-----------------------------------------------+----------------+
| contextual status / replay / cursor / timestamp                        |
+------------------------------------------------------------------------+
```

The inspector is persistent only on workspaces where committed selection is central. On simpler pages it may collapse or be omitted so the main evidence surface retains priority.

### 6.2 Navigation

The current nine-item navigation remains functionally available, but the visual grouping becomes task-oriented:

- **Decide**: Dashboard
- **Research**: Data Lab, Experiments, Run Center
- **Analyze**: Live Training, Compare
- **Trust**: Evidence Explorer, Serving Monitor
- **System**: Settings

This is a visual grouping, not a route migration. Existing workspace IDs and deep links remain compatible unless a later implementation step proves a migration is necessary.

### 6.3 Context bar

The global context bar may show only globally meaningful identity/state:

- active workspace,
- current run/job identity when the workspace has one,
- source freshness,
- `NO-GO`,
- compact runtime environment summary when useful.

Seed, environment, symbol, timeframe, fold, and checkpoint do not automatically become global state. They stay feature-scoped unless two or more workspaces have a real shared contract for the same identity.

## 7. Workspace designs

## 7.1 Dashboard: decision cockpit

Primary question: **What is the single most important blocker or next action?**

The existing decision model is retained. The visual structure becomes:

1. decision headline/ribbon,
2. readiness pipeline,
3. ranked blocker/action queue,
4. compact latest-result context,
5. inspector/detail only for the committed blocker or stage.

Environment details move out of a prominent page-specific control unless they are required to interpret the selected decision.

The Dashboard must not regress into generic metric tiles. Metrics are shown only when they explain the decision.

## 7.2 Live Training: primary research workspace

Live Training remains the flagship visualization.

### Main structure

```text
context + replay transport
symbol / timeframe / range / layers
-------------------------------------------------
Market pane      candles + trade lifecycle + risk
Policy pane      target / executed exposure
Learning pane    reward / interval cost
Performance pane RL equity / baseline / drawdown
-------------------------------------------------
replay scrubber
-------------------------------------------------
right inspector: committed record / checkpoint evidence
```

### Preserved contracts

The redesign must preserve:

- synchronized time scale and crosshair,
- mouse-wheel zoom and drag pan,
- manual navigation disabling follow-latest,
- programmatic range changes not being misclassified as manual navigation,
- hover as preview only,
- chart click as commit and replay pause,
- bounded telemetry cache,
- current seed/environment/current-episode isolation,
- nanosecond timestamp normalization and invalid timestamp rejection,
- timeframe aggregation semantics,
- target/executed weight distinction,
- LONG/SHORT/CLOSE plus increase/reduce/rebuy/flip semantics,
- independent RISK and END events,
- `ENTRY -> REDUCE -> EXIT` and open-trade lifecycle bands,
- checkpoint evidence being explicitly selected rather than silently score-maximized.

### Changes

- The chart gains more visual area by removing redundant page chrome.
- The inspector becomes a real synchronized side panel instead of details being treated as a secondary document block.
- Replay identity and chart controls are visually separated by responsibility.
- Layer toggles use compact grouped disclosure rather than always consuming top-level space.
- Empty training state prioritizes Behavior Cloning progress or a precise no-record reason.
- Diagnostics remain a sibling analysis mode, not an unrelated page.

## 7.3 Compare: explain differences, not just show two runs

The existing interactive ordinal comparison remains the foundation.

The workspace standardizes:

- explicit left/right or baseline/candidate identity,
- a dominant comparison plot,
- linked fold/metric selection,
- committed selection reflected in an inspector,
- direct visibility of return, drawdown, cost, constraint, and eligibility differences where supported by authoritative data,
- invalid/ineligible comparisons failing closed instead of being normalized into ordinary rows.

The main view should answer: **where is the material difference?** The inspector answers: **what data/config/evidence explains it?**

## 7.4 Evidence Explorer: lineage-oriented explorer

Evidence becomes an explorer model rather than a set of equal-weight cards.

```text
Evidence outline/tree -> primary evidence view -> inspector
```

The outline groups authoritative artifacts by identity and lineage. The selected artifact view shows validation state and meaningful fields. The inspector emphasizes:

- identity,
- digest,
- source/parent relationships,
- authorization/closure state,
- reason for invalidity when rejected.

No UI inference may repair missing identity, digest, or lineage fields.

## 7.5 Data Lab

Data Lab is reorganized around dataset identity and quality evidence:

- dataset catalog/list,
- selected dataset summary,
- coverage/quality visualization when authoritative data exists,
- provenance/contract inspector,
- deep-link compatibility retained.

The redesign does not invent analytical charts for fields the API does not actually provide.

## 7.6 Experiments

Experiments remains an exploratory configuration surface. It should visually separate:

- immutable selected dataset/config identity,
- editable exploratory parameters,
- validation/error state,
- launch/result navigation.

This phase does not expand the experiment API or introduce a visual strategy builder.

## 7.7 Run Center

Run Center becomes an operational list/detail workspace:

- run/job list or compact outline,
- selected run status and progress,
- logs/artifact links in the main or inspector region,
- clear running/completed/failed/stale distinctions.

It must not imply that a completed training job is approved for production.

## 7.8 Serving Monitor

Serving remains read-only and trust-oriented. It distinguishes:

- active bundle identity,
- bundle validity,
- paper inference state,
- stale/offline state,
- production `NO-GO`.

No activation control is added.

## 8. Component and module boundaries

The implementation will avoid a full directory rewrite. Existing feature folders such as `live/`, `compare/`, `dashboard/`, `pages/`, `api/`, and `state/` remain valid.

Introduce only reusable boundaries that have at least two real consumers.

Suggested structure:

```text
frontend/src/
  components/
    workspace/
      WorkspaceHeader.tsx
      WorkspaceContextBar.tsx
      WorkspaceInspector.tsx
      WorkspaceStatus.tsx
    ui/
      SegmentedControl.tsx
      SelectField.tsx
      EmptyState.tsx
      ErrorState.tsx
      FreshnessBadge.tsx
  styles/
    tokens.css
    shell.css
    workspace.css
```

Feature-specific controls remain in their feature folders. For example, ReplayToolbar stays under `live/` rather than being generalized into an abstract command framework.

A component is promoted to shared UI only when its semantics are actually shared, not merely because two pieces look similar.

## 9. State model

State is divided intentionally:

### 9.1 URL-restorable analytical state

Use URL state for selections that a researcher reasonably expects to deep-link or restore, such as workspace, committed dashboard decision, selected comparison identity, or selected evidence artifact where the current API identity is stable.

### 9.2 Local interaction state

Keep ephemeral state local:

- hover preview,
- open/closed menus,
- temporary layer popovers,
- crosshair preview,
- playback timer internals.

### 9.3 Streaming state

Polling hooks own network freshness, generation/cursor continuity, and stale request protection. Rendering components receive normalized state and must not independently refetch the same resource.

### 9.4 No new global store by default

React state, existing hooks, and URL state remain the default. A new store library is allowed only if implementation evidence shows that cross-workspace synchronization creates brittle prop/state ownership that cannot be solved cleanly with the existing architecture.

## 10. Data flow

The preferred flow is:

```text
Studio API
  -> runtime guards / authoritative validation
  -> feature polling/load hook
  -> feature view model / visualization model
  -> committed + preview interaction state
  -> primary visualization + inspector
```

JSX should not repeatedly reinterpret raw API shapes. Complex transformation stays in tested model functions such as the existing research chart and comparison models.

## 11. Visualization architecture

### 11.1 Renderer choice

Keep `lightweight-charts` for synchronized market/time-series work. It already fits the current density, direct manipulation, and maintenance requirements.

Use DOM/SVG/simple CSS graphics for small categorical/status views where semantics and accessibility matter more than mark count.

Do not introduce WebGL or another chart library without a measured requirement the existing stack cannot satisfy.

### 11.2 Rendering budget

- One dominant heavyweight interactive chart workspace per analytical page.
- Secondary charts should be limited and lazy/hidden when not visible.
- Streaming transforms must remain bounded by the existing cache/window contracts.
- Avoid full React rerenders of the chart tree for crosshair-only preview where imperative chart APIs already own rendering.

### 11.3 Screenshot legibility

Critical state must remain understandable without hover:

- current identity,
- selected/committed state,
- lifecycle bands and event markers,
- direct series keys/labels,
- risk/invalid state,
- axes and units.

## 12. Accessibility and keyboard contract

The desktop application must support:

- visible focus indicators,
- keyboard activation of nav and committed selections,
- Escape/reset paths for overlays and committed selections where defined,
- semantic buttons instead of clickable generic containers,
- `aria-current` for navigation,
- `aria-pressed` for persistent toggles,
- text/status cues in addition to color,
- `prefers-reduced-motion` support,
- no critical evidence available only through hover.

Dense chart inspection may remain pointer-optimized, but the surrounding controls and selection lists must remain keyboard operable.

## 13. Failure behavior

### Fetch failure

Keep the last known valid view when available and surface freshness/error state without destroying analytical context.

### Contract failure

If a runtime guard, identity, digest, or schema check fails, show `INVALID` and do not render guessed replacement values.

### Empty state

State why no evidence exists when known: no jobs, no telemetry for selected seed/environment, behavior cloning still running, missing artifact, or unsupported authoritative field.

### Stale selection

If a refreshed dataset no longer contains a committed selection, clear or migrate it only according to a tested deterministic rule. Never silently select the highest-scoring checkpoint or a semantically different artifact merely to keep the panel populated.

## 14. Styling system

Create a small token layer for:

- background/surface elevations,
- semantic text,
- borders,
- selection,
- success/warning/danger,
- spacing,
- radius,
- focus ring,
- typography sizes,
- control heights.

The goal is consistency, not a general design system. Existing feature CSS is migrated incrementally as each workspace is redesigned.

Avoid CSS churn unrelated to a workspace currently being migrated.

## 15. Implementation order

Implementation will proceed in vertical slices so the application remains usable:

1. shell tokens and workspace primitives,
2. AppShell/navigation/context/status integration,
3. Live Training migration while preserving all chart contracts,
4. Dashboard migration,
5. Compare migration,
6. Evidence Explorer migration,
7. Data Lab / Run Center / Experiments / Serving alignment,
8. settings and remaining shared polish,
9. documentation and full verification.

Each slice includes its relevant tests before moving to the next.

## 16. Test strategy

### 16.1 TDD sequence

For each slice:

1. add/adjust a focused failing behavioral or layout regression test,
2. implement the minimum behavior,
3. run the nearest Vitest tests,
4. refactor only after green,
5. expand to related frontend tests.

### 16.2 Required regression contracts

Do not weaken tests covering:

- telemetry environment/current-episode isolation,
- bounded replay semantics,
- synchronized chart panes,
- crosshair preview,
- chart-click commit,
- manual viewport preservation,
- programmatic range notifications,
- event navigation,
- lifecycle bands including `ENTRY -> REDUCE -> EXIT`,
- explicit checkpoint selection,
- URL restoration/deep links,
- comparison eligibility/identity,
- runtime guards,
- fixed viewport/no browser page scroll.

### 16.3 New tests

Add focused coverage for:

- shell navigation grouping without route breakage,
- context bar source/freshness/NO-GO states,
- inspector preview-versus-commit behavior,
- stale/partial/offline/invalid visual distinctions,
- keyboard focus/activation for new shared primitives,
- workspace layout at 1440 x 900,
- Live Training chart area not shrinking when inspector/detail state changes.

### 16.4 Final verification

At the final implementation head run at minimum:

```bash
npm test --prefix frontend -- --run
npm run typecheck --prefix frontend
npm run build --prefix frontend
npm run check:layout --prefix frontend
```

Then run repository-required Python/static/architecture checks affected by any Studio API or documentation changes. If backend code is untouched, still run the repository's required final CI-equivalent checks before declaring the branch complete.

## 17. Migration and compatibility

This is an incremental migration on one feature branch.

Compatibility requirements:

- existing backend read-only contracts remain valid,
- existing workspace IDs remain stable unless an explicit migration is documented and tested,
- existing deep links are preserved where their identity remains meaningful,
- existing chart interaction contracts remain preserved,
- no production or order capability is introduced,
- no evidence validation is relaxed.

Temporary compatibility wrappers should be avoided. Prefer direct migration of one workspace at a time with tests that lock the externally observable behavior.

## 18. Success criteria

Research Cockpit 2.0 is complete when:

1. all maintained workspaces use the shared shell hierarchy,
2. Live Training remains fully interactive with preserved lifecycle and replay semantics,
3. Dashboard presents one dominant decision path instead of equal-weight dashboard chrome,
4. Compare uses linked selection + inspector semantics,
5. Evidence Explorer exposes lineage and validation without guessed values,
6. source states are consistently represented across relevant workspaces,
7. keyboard and focus behavior works for the redesigned controls,
8. 1440 x 900 fixed-viewport checks pass without browser page scroll,
9. frontend tests/typecheck/build/layout checks pass,
10. repository-wide required final verification passes on the same final head,
11. no new live-order, release-approval, or credential capability exists.

## 19. Rejected alternatives

### Cosmetic-only redesign

Rejected because it would leave inconsistent interaction and state semantics in place while spending effort on styling.

### Full frontend rewrite

Rejected because the current Live Training, Compare, runtime guards, telemetry isolation, and evidence behavior already represent significant validated functionality. Rewriting them would create risk without corresponding analytical value.

### New chart framework or WebGL-first architecture

Rejected because the current renderer already supports the required direct-manipulation time-series workload. No measured scale requirement currently justifies the maintenance and regression cost.

### Generic global state store

Rejected as a default because current URL state and feature hooks already define useful ownership boundaries. It can be reconsidered only from concrete implementation pressure.

## 20. Consequences

Positive consequences:

- a clearer research workflow,
- less visual and interaction inconsistency,
- stronger reuse of proven chart behavior,
- easier extension of future analytical views,
- more explicit failure/freshness semantics,
- improved regression-test ownership.

Costs and risks:

- shell changes touch every workspace and therefore require staged migration,
- CSS migration can create fixed-viewport regressions if done too broadly,
- moving details into synchronized inspectors can accidentally change selection ownership,
- shared primitives can become over-generalized if promoted before two real consumers exist.

These risks are controlled by vertical-slice TDD, preserving feature-level models, and refusing unrelated refactoring.