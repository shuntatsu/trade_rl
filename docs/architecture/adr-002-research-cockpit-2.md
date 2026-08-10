# ADR-002: Research Cockpit 2.0 frontend architecture

Status: Proposed
Date: 2026-08-10
Scope: `frontend/` only unless an explicitly required read-only Studio API contract is missing

## 1. Decision

Trade RL Studio will be redesigned as **Research Cockpit 2.0**, organized around the research lifecycle rather than the current page/file boundaries.

The primary navigation becomes:

- Home
- Data
- Runs
- Compare
- Evidence
- Serving
- Settings as a separated utility destination

The current Experiments, Run Center, and Live Training workflows become views under **Runs**:

- New
- Overview
- Replay
- Diagnostics
- Logs

The user's main object is a Research Run, not an implementation page. The redesign therefore keeps run identity and research context continuous while the user creates a job, observes execution, inspects market behavior, diagnoses optimization, compares results, audits evidence, and inspects paper serving.

The redesign is not a trading terminal. Production remains `NO-GO`. The Studio does not gain exchange credentials, order placement, release approval, sealed-test opening, bundle activation, or other production authorization actions.

## 2. Why this replaces the previous draft

The previous ADR correctly emphasized evidence-first visualization and a shared operational shell, but it over-standardized the product around page composition. Further review of the actual workflow and current code exposed five stronger product constraints:

1. Maintained research is now single-instrument: one maintained run equals one instrument, one target-weight action, and one checkpoint/evidence chain.
2. Experiments, Run Center, and Live Training are not independent user jobs. They are consecutive states of one Run workflow.
3. The existing Dashboard and Compare implementations are already strong and should be preserved rather than rebuilt.
4. The existing synchronized Replay uses a strong three-pane model. Adding a fourth learning pane would mix optimizer-step and market-time semantics and reduce the dominant market viewport.
5. Evidence API nodes do not expose authoritative lineage edges. The UI must not invent a graph or tree relationship that the backend does not state.

Research Cockpit 2.0 therefore standardizes interaction grammar and research identity, but not by forcing every view into the same visual skeleton.

## 3. Product mental model

The product lifecycle is:

```text
Data quality / identity
        |
        v
Create Research Run
        |
        v
Execution / observation / diagnosis
        |
        +--------> Compare runs
        |
        +--------> Audit evidence
        |
        v
Inspect paper serving
```

Each primary workspace answers one question:

- **Home:** What requires my attention next?
- **Data:** Can this Dataset be trusted as research input?
- **Runs:** What is this Run doing, and what did the policy do?
- **Compare:** Where do two Runs materially differ?
- **Evidence:** Is the result's required evidence complete and valid?
- **Serving:** What identity is paper inference using, and is it valid?

A workspace must not become a warehouse for unrelated metrics merely because the data is available.

## 4. Goals

### 4.1 Primary goals

1. Make the researcher's next decision obvious within the first few seconds of each workspace.
2. Keep the dominant evidence surface larger than supporting chrome.
3. Preserve authoritative identity, validation, and fail-closed behavior.
4. Unify preview/commit/reset interaction semantics across analytical views.
5. Reduce navigation caused by implementation boundaries.
6. Make maintained single-instrument identity explicit in the UI.
7. Preserve existing high-value Dashboard, Replay, and Compare behavior.
8. Keep URL-restorable analytical state deterministic and workspace-scoped.
9. Distinguish execution success, connection freshness, validation, and production authorization.
10. Keep the desktop research workflow usable at 1440x900 and 1180x800.

### 4.2 Secondary goals

- improve keyboard and focus behavior;
- reduce decorative gradients, repeated red warnings, and equal-weight card grids;
- avoid unnecessary polling for hidden views;
- keep last-known-good evidence visible on recoverable transport failures;
- reduce CSS duplication only where semantics are actually shared.

## 5. Non-goals

The redesign will not:

- add exchange orders, credentials, or broker actions;
- add production approval or bundle activation;
- make training completion imply profitability or release readiness;
- add a visual strategy builder;
- make training configuration fully editable from the UI unless a separate product requirement is approved;
- invent Dataset quality metrics absent from authoritative API/artifacts;
- invent Evidence lineage edges absent from the API;
- replace `lightweight-charts` without a measured rendering requirement;
- introduce a global state library by default;
- add mobile/tablet support in this phase;
- introduce decorative 3D, particles, ambient animation, or novelty rendering;
- perform unrelated backend or repository restructuring.

## 6. State semantics

No single badge represents overall system goodness. The UI separates four state axes.

| Axis | Examples | Question |
| --- | --- | --- |
| Execution | QUEUED, RUNNING, SUCCEEDED, FAILED | Did the process execute? |
| Connection | CONNECTED, LIVE, DELAYED, STALE, OFFLINE | Is current data arriving? |
| Validation | VALID, INVALID, VERIFIED, INCOMPLETE | Can this artifact/evidence be trusted? |
| Authorization | NO-GO | Is production use authorized? |

`SUCCEEDED` must not be styled as equivalent to `VALID`, and neither implies production approval.

`DEMO` is data provenance/mode, not a connection state.

### 6.1 Color semantics

- neutral: structure and inactive context;
- cyan/blue: selected, informational focus, running/current state;
- green: verified or validated evidence only;
- amber: attention, delayed, stale, incomplete;
- red: invalid evidence, explicit risk/failure, destructive action, `NO-GO`.

`RUNNING` is informational, not green. `SUCCEEDED` is execution completion, not evidence validation.

## 7. Global application shell

### 7.1 Desktop layout

Baseline: 1440x900. Supported compact desktop target: 1180x800.

```text
+---------+--------------------------------------------------------------+
|         | Runs / selected context   BTCUSDT   RUNNING   CONNECTED NO-GO|
| Nav     +--------------------------------------------------------------+
| rail    |                                                              |
|         |                    ACTIVE WORKSPACE                           |
|         |                                                              |
|         |                                                              |
+---------+--------------------------------------------------------------+
```

There is no persistent global bottom status bar. Replay owns a local transport region when Replay is active.

### 7.2 Navigation rail

Primary items:

```text
Home
Data
Runs
Compare
Evidence
Serving
-------
Settings
```

At compact desktop width the rail may collapse to icons/short labels, but navigation remains keyboard operable and uses `aria-current`.

### 7.3 Top context bar

The default context bar shows only research-relevant global context:

- workspace;
- selected Job/Run identity where applicable;
- maintained instrument identity where authoritative;
- execution state where applicable;
- source/freshness state;
- one persistent `NO-GO` production-boundary indicator.

CUDA, GPU name, Python version, and other environment details move to an Environment utility surface. They are not continuously given the same visual priority as research identity.

### 7.4 Global `NO-GO`

`NO-GO` appears persistently once in the global shell. Workspaces repeat it only when authorization itself is the subject of the view, such as Evidence or Serving.

This avoids turning red production-boundary styling into decorative background noise.

### 7.5 Current fixed StatusBar

The current fixed statement that all services are healthy must be removed or replaced with authoritative runtime state. No static message may claim health independently of actual source state.

## 8. URL and navigation contract

### 8.1 Canonical workspaces

Canonical workspace IDs become:

- `dashboard` (display label Home; ID may remain for backward compatibility)
- `data`
- `runs`
- `compare`
- `evidence`
- `serving`
- `settings`

### 8.2 Legacy route migration

Old links remain readable:

- `workspace=experiments` -> `workspace=runs&view=new`
- `workspace=runs` -> `workspace=runs&view=overview` when no view exists
- `workspace=live` -> `workspace=runs&view=replay`

After interpretation, the browser URL is replaced with canonical state.

### 8.3 Workspace-scoped parameter allowlists

Navigation must remove parameters belonging to another workspace.

Examples:

```text
Home:
workspace=dashboard&stage=...&decision=...

Data:
workspace=data&dataset=...

Runs replay:
workspace=runs&job=...&view=replay&seed=7&env=2&timeframe=15m&range=24h

Compare:
workspace=compare&left=...&right=...&comparePoint=...

Evidence:
workspace=evidence&evidenceRun=...&node=...
```

Playback state, hover state, crosshair state, open popovers, and temporary previews do not belong in the URL.

### 8.4 Job identity and Run artifact identity

The frontend must not equate `JobSummary.id`, `runId`, and `RunSummary.id` merely because strings look related.

- execution views use authoritative Job resource identity;
- completed research artifact views use authoritative Run resource identity;
- the frontend may join them only through an explicit backend contract or a deterministic identity rule already defined and tested by the API.

## 9. Shared interaction grammar

Analytical views use:

```text
DEFAULT
  -> hover/focus = PREVIEW
  -> click/Enter = COMMITTED
  -> committed selection drives inspector and optional URL state
  -> Escape/reset = DEFAULT
```

A refresh preserves a committed selection if the same authoritative identity still exists.

If the selected identity disappears, clear it according to a deterministic tested rule. Do not silently substitute a highest-score checkpoint or a semantically different artifact.

Overlay inspectors restore focus to their trigger when closed.

## 10. Home

Primary question: **What requires attention next?**

The current decision-cockpit model remains the foundation.

### 10.1 Reading order

1. current research context when available;
2. primary attention/decision ribbon;
3. Data -> Training -> Evaluation -> Evidence -> Release readiness pipeline;
4. ranked secondary decisions;
5. compact latest validated result context.

The page does not become a generic KPI dashboard.

### 10.2 Preserve

- deterministic primary decision;
- preview versus committed selection;
- URL-backed stage/decision state;
- browser history restoration;
- drill-through actions;
- keyboard-safe Environment shortcut behavior.

### 10.3 Remove/de-emphasize

- duplicate `NO-GO` if global shell already shows it;
- prominent environment details unrelated to the selected decision;
- decorative metric cards not required to explain the next action.

### 10.4 Acceptance

A screenshot without hover must answer:

- what is currently happening;
- what the highest-priority attention item is;
- which research stage blocks progress;
- where the user can drill through next.

## 11. Data

Primary question: **Can this Dataset be trusted as research input?**

### 11.1 Layout

Two-column desktop workspace:

- left: searchable/paged Dataset catalog;
- right: selected Dataset identity and validation evidence.

### 11.2 Default-visible fields

Only authoritative fields currently available through the Dataset contract are used initially:

- status;
- name;
- instrument/symbol list;
- market;
- timeframes;
- range;
- bar count;
- feature count;
- Dataset identity;
- artifact path;
- updated time;
- validation error.

Feature count is not a hero-quality metric. A larger feature count is not presented as better.

### 11.3 Quality visualization rule

Missingness, timestamp gaps, coverage percentages, quality scores, or similar charts appear only after an authoritative contract provides them. The frontend must not infer or fabricate such values from incomplete metadata.

### 11.4 INVALID state

If selected Dataset status is INVALID, validation failure and reason outrank normal metadata. The UI must not visually present an invalid Dataset as ordinary input with a small warning badge.

### 11.5 URL

`workspace=data&dataset=<resource-id>` restores the selected Dataset and opens the page containing it.

## 12. Runs information architecture

Runs is the center of Research Cockpit 2.0.

Subviews:

- New
- Overview
- Replay
- Diagnostics
- Logs

A selected execution Job remains selected while switching between Overview, Replay, Diagnostics, and Logs.

### 12.1 Run selector rail

When needed, a secondary Run/Job rail shows compact rows:

- run ID truncated visually but available in full accessibly;
- maintained instrument if authoritative;
- execution state;
- current phase if authoritative;
- invalid/error indication when applicable.

The secondary rail may collapse in chart-heavy Replay so the market viewport remains dominant.

## 13. Runs / New

Primary question: **What exactly will this research job execute?**

### 13.1 Inputs

The existing training request contract remains authoritative:

- Config resource;
- Dataset resource;
- Run ID.

Research Cockpit 2.0 does not add ad-hoc editable optimizer parameters to this page.

### 13.2 Preview

Show authoritative derived identity only where available:

- selected Dataset identity/status;
- instrument from Dataset;
- selected Config identity/status;
- algorithm;
- Run ID validation;
- maintained single-instrument preflight when supported by the existing contract.

Do not expose steps, seeds, folds, learning rate, or other values unless the Config API explicitly provides them.

### 13.3 Preflight

The launch button is enabled only when required existing validations pass. UI preflight is explanatory; server-side validation remains authoritative.

### 13.4 Submission transition

On successful Job creation, navigate to the created Job's Overview using its authoritative resource ID.

## 14. Runs / Overview

Primary question: **What is this Job/Run doing now, and what evidence is available to inspect?**

### 14.1 Default-visible sections

- execution state;
- submitted/started/completed timestamps where available;
- process ownership/cancellability where relevant;
- Dataset identity;
- Config identity;
- artifact root;
- available analytical views/evidence state.

### 14.2 Progress truthfulness

Do not invent a generalized percentage from unrelated overview data. `JobSummary` does not itself define a generic progress contract. Progress is shown only if an authoritative Job-specific API provides it.

### 14.3 Completion semantics

`SUCCEEDED` means the process exited successfully. It does not imply:

- artifact validity;
- checkpoint eligibility;
- profitability;
- Evidence completeness;
- production approval.

## 15. Runs / Replay

Primary question: **What did the policy do in market context, and what happened around that decision?**

Replay remains the flagship interactive visualization.

### 15.1 Preserve current three-pane architecture

Do not replace it with a four-pane design.

Recommended visual ratios:

- Price & Execution: 55-60%;
- Exposure / Portfolio: 22-27%;
- State & Risk: 16-20%.

Exact values are measured during browser QA and encoded as stretch factors rather than brittle fixed pixel heights.

### 15.2 Why learning metrics stay out of Replay

Market replay is indexed by market/event time. Optimization metrics are indexed by optimizer/training step. Combining them as equal synchronized panes would imply a stronger shared axis than the evidence supports.

Learning metrics remain in Diagnostics.

### 15.3 Preserve Replay behavior

The redesign must retain:

- synchronized time scale and crosshair;
- wheel zoom and drag pan;
- manual navigation disabling follow-latest;
- programmatic range changes not being classified as manual navigation;
- hover preview;
- chart click commit and replay pause;
- bounded telemetry buffer;
- seed/environment/current-episode isolation;
- nanosecond timestamp normalization and rejection of invalid timestamps;
- timeframe aggregation semantics;
- target versus executed exposure distinction;
- LONG/SHORT/CLOSE and increase/reduce/rebuy/flip event semantics;
- independent RISK and END events;
- trade lifecycle background bands;
- explicit checkpoint evidence selection, with no score-max auto-selection.

### 15.4 Trade lifecycle is a locked contract

The following behavior must remain:

```text
ENTRY #1 ---- REDUCE #1 ---- EXIT #1
ENTRY #2 -------------------- OPEN #2
```

Entry, reduction, exit/open markers remain visible, while events belonging to one position lifecycle remain connected by the same background Trade band.

A reduction must not split the lifecycle into a new Trade.

### 15.5 Maintained single-instrument behavior

For maintained one-instrument Runs, instrument is identity, not a normal interactive selector.

Example:

```text
Instrument  BTCUSDT
```

A symbol selector appears only for compatible historical/legacy multi-symbol artifacts.

### 15.6 Control placement

Run selection is outside Replay and is removed from ReplayToolbar.

Top-local controls contain analytical source/view choices such as:

- Seed / Environment;
- timeframe;
- range;
- Layers.

Playback transport belongs below or immediately adjacent to the replay scrubber:

- first;
- previous event;
- play/pause;
- next event;
- latest;
- speed;
- follow latest.

### 15.7 Inspector

The Replay Inspector is closed by default.

A committed Trade/point opens a 320-360px overlay inspector. Opening or closing it must not resize the chart canvas or alter analytical geometry.

Inspector content may include only available authoritative fields:

- selected timestamp/step;
- target/executed exposure;
- portfolio/risk values;
- event interpretation;
- explicitly selected checkpoint evidence identity.

### 15.8 Empty states

Empty Replay states state the reason when known, for example:

- Behavior Cloning still running;
- no telemetry for selected Seed/Environment;
- telemetry not yet initialized;
- source unavailable.

Generic `No data` is insufficient where a more precise state is known.

## 16. Runs / Diagnostics

Primary question: **Does optimization/training behavior require attention?**

Diagnostics is not profitability evidence.

### 16.1 Preserve metric groups

Retain the existing groups:

- Optimization;
- Policy;
- Value;
- Trading / Risk.

### 16.2 Diagnostic language

Diagnostic thresholds such as Approx KL guidance are presented as **ATTENTION**, not production failure or model invalidity, unless a separate authoritative gate says otherwise.

Use states such as:

- NO ATTENTION;
- ATTENTION;
- UNAVAILABLE.

Avoid a green `HEALTHY` label that could be mistaken for profitability or approval.

### 16.3 Polling ownership

Hidden Diagnostics must not continue polling merely because CSS `hidden` is applied.

- Replay active: Diagnostics component/hook is not polling.
- Diagnostics active: metric polling is enabled.

The same principle applies to other expensive hidden analytical views.

### 16.4 Charts

Charts remain step-indexed and directly labeled. Threshold guides are visually subordinate and explicitly described as diagnostic guidance, not release gates.

## 17. Runs / Logs

Primary question: **What is the owned process doing, and can I safely stop it?**

### 17.1 Preserve

- selected Job log;
- Run ID;
- PID;
- process ownership;
- execution state;
- stale-request protection;
- initial log load for the selected Job.

### 17.2 Log following

Default follows latest log output.

If the user manually scrolls away from the bottom, automatic following pauses. A `Jump to latest` action restores it. New log lines must not steal the user's scroll position.

### 17.3 Stop action

Stop is visible only for Jobs where the existing `cancellable` and state contract permits cancellation.

Stopping requires explicit confirmation containing the selected Run/Job identity and clarifying that existing artifacts are not deleted.

Destructive-action red is reserved for this type of operation, not used decoratively.

## 18. Compare

Primary question: **Where does the right Run differ materially from the left Run?**

The current interactive ordinal comparison is retained as the foundation.

### 18.1 Preserve

- Left/Right Run identity;
- comparison eligibility;
- ordinal evaluation axis without invented timestamps;
- cumulative comparison pane;
- Right-minus-Left difference pane;
- pan/zoom/range selection;
- click/keyboard selection;
- URL restoration;
- Metrics/Config inspector;
- stale-request exclusion;
- explicit `no automatic winner` semantics.

### 18.2 Summary language

Show preference-aware counts and differences, not a single winner badge.

Example:

```text
3 improved · 2 worse · 1 tie
No automatic winner
```

### 18.3 Cross-workspace actions

Inspector/context may expose `Open Left Run` and `Open Right Run`, linking to Runs with the correct authoritative resource identity.

## 19. Evidence

Primary question: **Are the required evidence artifacts present and valid?**

### 19.1 Closure path, not invented lineage graph

Current Evidence nodes expose status, required, digest, path, label, and detail, but not authoritative parent/edge relationships.

Therefore the default visualization is an **Evidence Closure Path** or ordered audit list, not a graph/tree that implies backend lineage.

Connector lines, if used, represent reading/audit order only.

### 19.2 Default view

Show:

- Run identity;
- required/verified summary;
- ordered Evidence nodes;
- required versus optional;
- status.

### 19.3 Committed node inspector

A committed node opens an overlay with available fields:

- status;
- required/optional;
- path;
- digest;
- interpretation/detail;
- file integrity context where supported.

INVALID status and validation reason take visual priority.

### 19.4 Future lineage

Only if the backend later returns explicit lineage edges may the UI promote the closure path into a real dependency graph.

## 20. Serving

Primary question: **What is paper inference using now, and is the identity valid?**

Serving remains read-only.

### 20.1 Reading order

1. runtime/paper state;
2. active Bundle/Run/Dataset/Policy identity;
3. latest paper decision;
4. validation checks;
5. production `NO-GO`.

Do not use four equal-weight panels when the information has a natural dependency order.

### 20.2 Single-instrument decision view

For maintained one-instrument bundles, visualize target exposure on one signed, zero-centered scale.

For historical multi-symbol data, retain an accessible multi-weight fallback.

### 20.3 Validation semantics

A release attestation warning must not overwrite otherwise-valid identity checks, and valid identity checks must not imply production approval.

If identity/schema validation fails, Serving is INVALID and fails closed.

## 21. Settings and Environment

Settings is visually separated from the primary research workflow.

Until real settings justify a larger surface, it may contain only actual utility functions such as:

- environment/runtime information;
- local appearance/preferences if implemented;
- Studio information.

Remove placeholder `FOUNDATION READY` content from primary navigation prominence.

GPU, CUDA, Python, and similar environment metadata belong here or in an Environment drawer rather than the persistent top bar.

## 22. Failure, freshness, and last-known-good behavior

### 22.1 Recoverable fetch failure

When a previously validated view exists and a refresh fails, keep the last-known-good evidence visible and mark source freshness/error explicitly.

Example:

```text
STALE
Last validated telemetry remains visible.
Refresh failed at <timestamp>.
```

### 22.2 Contract failure

Schema, identity, digest, generation, or other contract failures are `INVALID`, not `STALE`.

Do not keep rendering data as trustworthy when its contract has failed.

### 22.3 Partial state

Use `PARTIAL` only where the visible subset remains valid but an independent subresource is unavailable. Missing required evidence is not silently treated as ordinary success.

### 22.4 Empty states

Empty states explain the known reason and next action where possible.

Examples:

- no validated Dataset;
- Behavior Cloning in progress;
- no telemetry for selected source;
- no Evidence report produced;
- no active paper bundle.

## 23. State ownership

### 23.1 URL-restorable state

Persist analytical identity/navigation that a researcher may reasonably share or restore:

- workspace;
- selected Dataset;
- selected Job/Run;
- Runs view;
- Replay Seed/Environment/timeframe/range;
- committed Dashboard selection;
- Compare pair/selection;
- Evidence Run/node selection.

### 23.2 Local ephemeral state

Keep local:

- hover preview;
- crosshair preview;
- playback running/paused timer state;
- playback speed unless a later preference contract is added;
- open menus/popovers;
- temporary layer menus;
- drag gesture internals.

### 23.3 No new global store by default

Use React state, feature hooks, and URL state until a concrete cross-workspace ownership problem proves a store is necessary.

## 24. Component boundaries

Do not perform a directory rewrite.

Keep established feature folders such as:

- `dashboard/`;
- `live/`;
- `compare/`;
- `pages/`;
- `api/`;
- `state/`.

Create shared UI only after at least two real consumers share semantics, not merely appearance.

Likely shared candidates:

```text
components/workspace/
  WorkspaceContextBar
  OverlayInspector
  FreshnessStatus
  EmptyState
  ErrorState

runs/
  RunsWorkspace
  RunsNavigation
  RunOverview
  NewRun
```

ReplayToolbar remains feature-specific, but loses Run selection after Runs owns the selected Job.

## 25. Rendering and performance

### 25.1 Renderer ownership

- keep `lightweight-charts` for Replay market/time-series visualization;
- retain existing SVG/DOM comparison implementation unless measured scale demands otherwise;
- use DOM/SVG/CSS for small categorical/status graphics;
- no WebGL/new chart dependency without evidence.

### 25.2 Streaming bounds

Preserve bounded telemetry buffers and generation/cursor protection.

Hidden views must not continue unnecessary polling.

Crosshair-only interaction should avoid full React chart-tree rerenders where the chart library can own imperative rendering.

### 25.3 Inspector geometry

Analytical inspectors overlay the evidence viewport. Opening them must not resize the underlying chart/comparison geometry.

## 26. Accessibility and keyboard contract

Required:

- visible focus indicators;
- semantic buttons and form controls;
- `aria-current` for navigation;
- `aria-pressed` for persistent toggles;
- keyboard activation for list/selection controls;
- Escape closes overlays and restores trigger focus;
- text plus color for state;
- no critical evidence available only by hover;
- `prefers-reduced-motion` support;
- current chart controls remain keyboard reachable even if dense chart inspection is pointer-optimized.

Do not add global shortcuts that steal browser/OS-modified keys.

## 27. Desktop layout acceptance

Required browser QA:

- 1440x900;
- 1180x800;
- no whole-page overflow;
- primary evidence remains visible without scrolling chrome off-screen;
- compact navigation remains usable;
- overlay inspector does not alter chart dimensions;
- focused control is not clipped;
- important state is understandable in a screenshot without hover.

## 28. Test contract

### 28.1 Navigation and URL

- legacy `experiments` URL resolves to Runs/New;
- legacy `live` URL resolves to Runs/Replay;
- Runs without view resolves to Overview;
- cross-workspace navigation removes unrelated query parameters;
- browser history restores canonical analytical state.

### 28.2 Shell

- no static `all services healthy` assertion;
- one global `NO-GO` is visible;
- GPU/CUDA/Python are available through Environment/Settings rather than consuming normal top-bar priority.

### 28.3 Runs

- successful New submission opens the created Job Overview;
- Job/Run are not joined solely by `runId`;
- non-cancellable Job has no Stop action;
- Stop requires confirmation;
- hidden Diagnostics does not poll.

### 28.4 Replay

Preserve all existing regressions plus:

- maintained single-symbol Run has no ordinary Symbol dropdown;
- compatible legacy multi-symbol data can expose a selector;
- `ENTRY -> REDUCE -> EXIT` remains one closed Trade band;
- later open position remains a separate open Trade band;
- event markers remain visible with background bands;
- manual pan disables Follow latest;
- committed click pauses Replay;
- hover remains preview only;
- opening inspector preserves chart geometry.

### 28.5 Diagnostics

- metrics poll only while Diagnostics is active;
- diagnostic threshold attention is not rendered as profitability/release failure;
- unavailable metrics explain why.

### 28.6 Data

- Dataset deep link restores selection;
- INVALID Dataset is visually fail-closed;
- no unsupported quality metrics are fabricated.

### 28.7 Compare

Preserve current interaction and fail-closed tests:

- eligibility;
- URL restore;
- range/point selection;
- pan/zoom/reset;
- keyboard navigation;
- no automatic winner;
- stale request exclusion;
- inspector geometry stability.

### 28.8 Evidence

- required missing evidence remains visible;
- INVALID evidence fails closed;
- UI does not assert lineage edges absent from backend contract;
- committed node inspector restores focus.

### 28.9 Serving

- maintained single-instrument exposure uses signed zero-centered semantics;
- validation failure produces INVALID;
- release-attestation absence does not imply unrelated identity checks failed;
- valid identity does not remove `NO-GO`.

## 29. Implementation order

Implementation proceeds incrementally and keeps the application usable between commits.

1. Shell truthfulness and canonical navigation foundation.
2. Runs information architecture and legacy-route compatibility.
3. Runs/New and Runs/Overview.
4. Move existing Replay under Runs without changing chart semantics.
5. Replay chrome, single-instrument behavior, overlay inspector, and polling cleanup.
6. Runs/Diagnostics and Runs/Logs integration.
7. Data hierarchy improvements using only existing authoritative fields.
8. Evidence closure-path redesign.
9. Serving reading-order redesign.
10. Home visual integration with minimal behavioral change.
11. Compare visual integration with minimal behavioral change.
12. CSS/token cleanup limited to proven shared semantics.

For every step:

1. characterize current behavior with tests;
2. add the new failing acceptance test;
3. implement the smallest change;
4. run focused tests;
5. run TypeScript typecheck/build as appropriate;
6. run fixed-viewport browser checks for affected workspaces;
7. refactor only after green.

## 30. Verification before completion

Final verification must include at least:

```text
npm test --prefix frontend -- --run
npm run typecheck --prefix frontend
npm run build --prefix frontend
npm run check:layout --prefix frontend
```

Then run the broader repository checks required by the repository CI for the exact final PR head, including architecture/import checks and relevant Python tests.

Existing Dashboard, Replay lifecycle, Compare, URL restoration, and fail-closed regressions must remain green.

## 31. Definition of done

Research Cockpit 2.0 is complete only when:

- primary navigation reflects the six research decisions rather than implementation pages;
- Experiments, Run Center, and Live Training are coherently represented under Runs;
- current research identity is understandable without repeated selectors;
- maintained single-instrument Runs do not present misleading normal Symbol selection;
- Replay preserves its three-pane synchronized market-first model and lifecycle bands;
- optimizer metrics remain in Diagnostics;
- hidden analytical views do not perform unnecessary polling;
- Data does not invent quality metrics;
- Evidence does not invent lineage edges;
- Compare remains no-automatic-winner and fail-closed;
- Serving clearly separates identity validity from production authorization;
- the static false health StatusBar is gone;
- one global `NO-GO` remains continuously visible;
- URL migration and workspace parameter scoping are tested;
- 1440x900 and 1180x800 fixed-viewport checks pass;
- frontend tests, typecheck, build, and layout checks pass;
- broader repository CI is green on the exact PR head;
- no new dependency is introduced without a measured requirement.

## 32. Risks and mitigations

### Risk: Runs integration becomes a big-bang rewrite

Mitigation: preserve existing feature components and migrate composition/selection ownership before changing visualization internals.

### Risk: Job and Run identity are accidentally conflated

Mitigation: keep resource IDs distinct and require an explicit tested join contract.

### Risk: Replay regresses while moving under Runs

Mitigation: treat existing synchronized panes, viewport behavior, source isolation, timestamps, and lifecycle tests as locked characterization tests.

### Risk: Shared components over-abstract feature behavior

Mitigation: require at least two semantic consumers before promotion to shared UI.

### Risk: New status system becomes cosmetic complexity

Mitigation: status labels must map to existing authoritative contracts; no synthetic overall-health score is added.

### Risk: Evidence closure path visually implies lineage

Mitigation: documentation and accessible labels define connectors as audit order only until explicit backend edges exist.

### Risk: compact desktop loses chart space

Mitigation: collapse navigation/secondary run rail before shrinking the dominant evidence viewport; use overlay inspectors.

## 33. Deferred questions

The following are implementation-measurement questions, not design blockers:

- exact Run rail width at 1440x900 and 1180x800;
- exact Replay pane stretch factors within the approved ranges;
- exact overlay inspector width within 320-360px;
- whether an explicit backend Job-to-Run relationship already exists but is not exposed by the current frontend contract;
- whether future Dataset QA artifacts justify a dedicated coverage/missingness visualization;
- whether future Evidence contracts provide authoritative lineage edges.

If a required authoritative field is absent, implementation records the limitation rather than inferring the value in the frontend.
