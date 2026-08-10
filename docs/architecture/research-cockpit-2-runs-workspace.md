# Research Cockpit 2.0: Runs Workspace Detailed Specification

Status: Proposed
Date: 2026-08-10
Parent: `docs/architecture/adr-002-research-cockpit-2.md`
Scope: Runs workspace UX, routing, state ownership, data ownership, Replay/Diagnostics/Logs composition, and Job/Run-artifact identity boundaries.

## 1. Decision

Research Cockpit 2.0 uses one primary **Runs** workspace for research execution and result inspection, but it does **not** collapse all backend resources into one synthetic `Run` object.

The workspace exposes two authoritative resource families:

1. **Execution Jobs** — `JobSummary` resources used by Job execution, Replay telemetry, Diagnostics, checkpoint-evaluation inspection, Behavior Cloning progress, logs, and cancellation.
2. **Research Artifacts** — `RunSummary` resources used by immutable result inspection, Compare, and Evidence.

`runId` is a human-facing research identifier. It is not sufficient by itself to join these resource families.

The Runs workspace may visually place Jobs and Artifacts together, but it must preserve their type and resource identity at all times.

## 2. Why the split is required

The current Studio API has separate contracts:

- `loadJobs()` returns `JobSummary[]`.
- `loadRuns()` returns `RunSummary[]`.
- telemetry, training metrics, checkpoint evaluations, Behavior Cloning progress, logs, and cancellation are addressed by **Job ID**.
- Compare and Evidence are addressed by **Run resource ID**.

No current frontend API contract exposes a formal Job-to-Run-artifact relationship.

Therefore the frontend must not infer:

```text
JobSummary.runId === RunSummary.runId
=> same authoritative resource
```

Even when labels match, navigation and data loading remain bound to the resource type required by the endpoint.

## 3. User mental model

The user should understand Runs as one research area with two stages:

```text
NEW RESEARCH RUN
      |
      v
EXECUTION JOB
  Overview
  Replay
  Diagnostics
  Logs
      |
      | explicit backend relationship only
      v
RESEARCH ARTIFACT
  Overview
  -> Compare
  -> Evidence
```

The UI does not imply the downward relationship unless the backend explicitly provides it.

A successful Job can remain a Job-only context even if a similarly named Run artifact exists.

## 4. Resource terminology

### 4.1 Job

A Job is a process/execution resource.

Authoritative identity:

```text
JobSummary.id
```

Useful fields include:

- `runId`;
- `status`;
- `configResourceId` / `configDigest`;
- `datasetResourceId` / `datasetId`;
- `artifactRoot`;
- submitted/started/completed timestamps;
- PID/process ownership fields;
- `cancellable`;
- process error.

A Job status answers only whether execution occurred or is occurring.

### 4.2 Research Artifact

A Research Artifact is an immutable completed/registered result resource.

Authoritative identity:

```text
RunSummary.id
```

Useful fields include:

- `runId`;
- `manifestDigest`;
- `runKind`;
- `algorithm`;
- `datasetId`;
- period;
- timestamps;
- file count;
- available result metrics;
- validation status/error;
- `productionStatus`.

### 4.3 Human Run ID

`runId` is displayed prominently because it is meaningful to the researcher, but it is not used as an implicit foreign key.

Where both resource ID and `runId` are available:

- primary visible label: `runId`;
- secondary technical identity: resource ID;
- API calls: resource ID required by the endpoint.

## 5. Canonical Runs route model

Runs uses an explicit discriminated target model.

### 5.1 New

```text
?workspace=runs&view=new
```

No Job or Artifact target is selected.

### 5.2 Job target

```text
?workspace=runs&target=job&job=<job-resource-id>&view=overview
?workspace=runs&target=job&job=<job-resource-id>&view=replay
?workspace=runs&target=job&job=<job-resource-id>&view=diagnostics
?workspace=runs&target=job&job=<job-resource-id>&view=logs
```

Replay may additionally persist:

```text
seed=<n>
env=<n>
timeframe=15m|1h|4h|1d
range=<supported-range>
```

### 5.3 Artifact target

```text
?workspace=runs&target=artifact&run=<run-resource-id>&view=overview
```

Artifact targets do not expose Replay, Diagnostics, or Logs unless a future explicit API contract makes those views addressable by Run resource ID.

### 5.4 Why `target` is explicit

The route must not infer resource type from arbitrary ID shape.

`target=job|artifact` makes selection deterministic and prevents a query containing both `job` and `run` from creating ambiguous ownership.

## 6. Route canonicalization

### 6.1 Legacy links

Interpret and replace legacy routes as follows:

```text
workspace=experiments
-> workspace=runs&view=new

workspace=live&job=<id>
-> workspace=runs&target=job&job=<id>&view=replay

workspace=live
-> workspace=runs&view=replay

workspace=runs&job=<id>
-> workspace=runs&target=job&job=<id>&view=overview

workspace=runs&run=<id>
-> workspace=runs&target=artifact&run=<id>&view=overview
```

If legacy Live has no Job ID, target resolution follows the default-selection rule in section 8.

### 6.2 Invalid combinations

Canonicalization removes incompatible state.

Examples:

```text
target=job + run=<id>
-> remove run

target=artifact + job=<id>
-> remove job

target=artifact + view=replay
-> view=overview

view=new + job/run target
-> remove target and resource IDs
```

### 6.3 Workspace allowlist

Runs owns only its documented parameters. Navigation to another workspace removes Runs-specific parameters.

Runs never leaves `seed`, `env`, `timeframe`, `range`, `job`, `run`, or `target` behind when navigating to Data/Compare/Evidence/etc. unless the destination explicitly receives a translated destination parameter.

## 7. Runs workspace layout

Desktop default:

```text
+----------------------+------------------------------------------------+
| Runs rail            | Selected context                               |
|                      | Overview | Replay | Diagnostics | Logs         |
| + New Run            +------------------------------------------------+
|                      |                                                |
| EXECUTION JOBS       |               ACTIVE VIEW                      |
|  job A  RUNNING      |                                                |
|  job B  SUCCEEDED    |                                                |
|  job C  FAILED       |                                                |
|                      |                                                |
| RESEARCH ARTIFACTS   |                                                |
|  run X  VALID        |                                                |
|  run Y  INVALID      |                                                |
+----------------------+------------------------------------------------+
```

The rail is secondary to the active evidence viewport.

Recommended desktop width range: 220-260px at 1440px viewport.

At compact desktop width, Replay may collapse the rail to a narrow selected-context trigger or drawer before reducing the market viewport.

The exact width is a browser-QA measurement, not a hard-coded product requirement.

## 8. Default selection

When the user enters Runs without an explicit valid target, selection is deterministic.

Priority:

1. most recently submitted active Job (`queued`, `running`, or `cancelling`);
2. newest valid or invalid Research Artifact by authoritative creation timestamp;
3. newest terminal Job;
4. New Run when neither family contains resources.

This is a navigation default only. It does not assert that the selected resource is best, valid, or recommended.

When an explicit URL target is present but missing from the refreshed catalog, do **not** silently substitute another resource. Show a target-not-found state with actions to return to the Runs default or create a new Run.

## 9. Runs rail

### 9.1 Sections

The rail has three semantic elements:

1. `+ New Run` action;
2. **Execution Jobs** section;
3. **Research Artifacts** section.

Do not merge Job and Artifact rows into one undifferentiated chronological list.

### 9.2 Job row

Default-visible content:

```text
<runId, truncated visually>
<execution status> · <submitted/started recency if available>
<instrument only when authoritative>
```

Technical Job resource ID is available in accessible/title/detail context, not used as the main human label unless `runId` is absent.

Job color maps only to execution semantics:

- running/current: cyan/informational;
- queued/cancelling: neutral/amber as appropriate;
- failed: red;
- succeeded/cancelled: neutral completion semantics, not validation-green.

### 9.3 Artifact row

Default-visible content:

```text
<runId, truncated visually>
<VALID|INVALID> · <algorithm or runKind>
<period/created time when space permits>
```

Artifact green is permitted for `VALID`; invalid is red.

`NO-GO` is not repeated in every row.

### 9.4 Sorting

Jobs:

- active before terminal;
- within the same activity class, newest submitted first.

Artifacts:

- newest created first.

Sorting never ranks by Sharpe, total return, checkpoint score, or other performance metric.

### 9.5 Filtering/search

No new search subsystem is required in the first implementation unless current resource counts make the rail unusable during browser QA.

YAGNI: do not add fuzzy search, tags, saved filters, or pagination without a real current need/API contract.

## 10. Target-specific navigation

### 10.1 New target

Visible subview:

- New only.

### 10.2 Job target

Visible navigation:

- Overview;
- Replay;
- Diagnostics;
- Logs.

These tabs remain Job-scoped.

Replay/Diagnostics can explain unavailable data in-view; they do not disappear merely because telemetry or TensorBoard output has not started yet.

### 10.3 Artifact target

Visible navigation:

- Overview.

Cross-workspace actions may include:

- Compare this Artifact;
- Open Evidence.

Do not render disabled Replay/Diagnostics/Logs tabs for an Artifact when no API contract can satisfy them. Their presence would falsely imply an addressable relation.

## 11. New Run

Primary question: **What exactly will this Job execute?**

### 11.1 Data loading

New Run needs:

- Config catalog;
- Dataset catalog.

Only authoritative VALID options allowed by the current launch contract are selectable.

Invalid resources remain inspectable in their own workspaces; they are not launch options.

### 11.2 Inputs

The launch request remains exactly:

```text
configResourceId
datasetResourceId
runId
```

Do not create browser-only optimizer overrides.

### 11.3 Identity preview

Show:

- selected Config name/algorithm/digest/path;
- selected Dataset name/ID/market/symbols/timeframes/status;
- Run ID validity;
- exact instrument identity when the Dataset contract is unambiguous.

Do not display steps, folds, seeds, learning rate, batch size, or action shape unless an authoritative API explicitly exposes them for the selected Config.

### 11.4 Single-instrument preflight

The frontend may state that the selected Dataset is one-symbol when the exact Dataset resource has one authoritative symbol.

It must not infer the full maintained policy/action contract from Dataset cardinality alone.

If the maintained Config contract later becomes explicitly available through API metadata, a stronger preflight can be added.

### 11.5 Submission

On success:

```text
submitTrainingJob(...)
-> authoritative JobSummary
-> workspace=runs&target=job&job=<returned id>&view=overview
```

Do not leave the user on New Run with only a success card.

On submit failure, retain inputs and show the server error.

## 12. Job Overview

Primary question: **What is this execution doing, and what can I inspect now?**

### 12.1 Identity block

Show:

- `runId`;
- Job resource ID in secondary detail;
- exact Dataset resource/dataset ID;
- Config resource/digest;
- artifact root;
- authoritative instrument from the exact Dataset resource when available.

The Runs controller may join Job -> Dataset/Config only by exact `datasetResourceId` / `configResourceId`, which are explicit Job fields.

This exact-resource join is allowed and is distinct from prohibited Job -> Run-artifact matching by `runId`.

### 12.2 Execution block

Show available fields:

- status;
- submitted;
- started;
- completed;
- PID;
- process ownership/cancellability;
- exit code;
- process error.

Do not invent generalized progress.

### 12.3 Analysis availability

Provide direct actions/tabs to Replay, Diagnostics, and Logs.

Availability messages are derived from each subresource's own API status, not guessed from execution status alone.

Examples:

- running Job can have no telemetry yet;
- succeeded Job can still have readable telemetry/logs;
- failed Job can still have useful logs and partial diagnostics.

### 12.4 Completion

A Job becoming `succeeded` does not automatically navigate the user to a similarly named Artifact.

If a future API exposes an explicit resulting Run resource ID, Overview may show:

```text
Result artifact available -> Open Artifact
```

Until then, no implicit bridge is shown.

## 13. Artifact Overview

Primary question: **What immutable result is this, and is the registered artifact valid?**

### 13.1 Identity and validation

Show:

- `runId`;
- Run resource ID;
- validation status/error;
- manifest digest;
- run kind;
- algorithm;
- dataset ID;
- period;
- created/completed timestamps;
- relative artifact path/file count;
- production status.

### 13.2 Result summary

Available result metrics such as total return, max drawdown, and Sharpe may be shown as a compact result summary.

Rules:

- metrics do not override validation status;
- invalid Artifact remains visibly invalid even if performance values are positive;
- no automatic "good", "winner", or release-ready label is derived from metrics.

### 13.3 Actions

Primary cross-workspace actions:

- Compare;
- Evidence.

Compare receives the selected Run resource ID as one side of the pair.

Evidence receives the selected Run resource ID as `evidenceRun`/canonical Evidence selection.

Do not offer Logs/Replay/Diagnostics based solely on a matching `runId`.

## 14. Replay ownership

Replay is a Job view.

### 14.1 Job selection

ReplayToolbar no longer owns Run/Job selection.

The selected Job comes from the Runs target and URL.

Removing duplicate Job selection is a primary cognitive-load improvement.

### 14.2 Exact Dataset identity

Runs should resolve the selected Job's exact Dataset resource using `datasetResourceId` when the catalog contains it.

If the Dataset has exactly one authoritative symbol:

- render that symbol as read-only instrument identity;
- do not show a normal Symbol dropdown.

If the Dataset is historical/legacy multi-symbol:

- show a Symbol selector populated only from authoritative Dataset symbols;
- choose a deterministic default from the Dataset contract, not a hard-coded `BTCUSDT` fallback.

If the referenced Dataset resource cannot be resolved:

- do not guess maintained/legacy mode;
- preserve telemetry inspection where safe;
- mark instrument-contract context as unavailable.

### 14.3 Source selection

Seed and Environment remain Replay-local analytical source controls.

Their committed values may be URL-backed.

Changing source:

- pauses playback;
- enables follow-latest for the new source;
- clears preview/committed chart selection when identity is no longer valid;
- resets chart navigation deterministically.

### 14.4 Timeframe and range

Timeframe and range are URL-restorable because they materially affect the analytical view.

Layer toggles remain local in the initial version.

### 14.5 Transport

Playback controls belong by the scrubber/transport area rather than competing with source identity at the top.

Transport state such as `playing` and speed is local, not URL-backed.

### 14.6 Inspector

Committed point/Trade opens an overlay inspector.

Inspector open/close must not resize the chart canvas.

Escape closes and restores appropriate focus.

### 14.7 Locked Replay regressions

Preserve:

- three synchronized panes;
- market-first pane sizing;
- chart zoom/pan/crosshair;
- manual navigation -> follow-latest off;
- hover preview vs click commit;
- commit pauses Replay;
- current episode isolation;
- bounded telemetry cache;
- timestamp normalization/validation;
- target/executed exposure distinction;
- risk/end event semantics;
- lifecycle bands;
- `ENTRY -> REDUCE -> EXIT` as one Trade lifecycle;
- independent later open Trade lifecycle;
- explicit checkpoint evidence selection.

## 15. Diagnostics ownership

Diagnostics is a Job view.

### 15.1 Mount/poll contract

Training metric hooks are active only when `view=diagnostics` for the selected Job.

CSS hiding is not sufficient.

Switching away from Diagnostics unmounts or disables expensive metric polling.

### 15.2 Job changes

Changing Job clears metric generation/cursors/selection and establishes a fresh Job-specific metric stream.

### 15.3 Semantics

Optimization warnings remain diagnostic attention, not profitability or production validation.

Metric groups remain:

- Optimization;
- Policy;
- Value;
- Trading / Risk.

## 16. Logs ownership

Logs is a Job view.

### 16.1 Data

Logs load only for the selected Job.

A stale response from a previously selected Job must never replace current content.

### 16.2 Follow behavior

When Logs is active:

- latest output is followed by default;
- manual scroll away from bottom pauses follow;
- new lines do not steal scroll position;
- `Jump to latest` restores follow.

Because the current API exposes a bounded log response rather than a streaming cursor, polling may replace/merge the currently visible bounded log snapshot. Exact cadence is an implementation measurement; polling stops when Logs is not active.

### 16.3 Cancellation

Show Stop only when:

```text
job.cancellable
&& status in {queued, running}
```

Cancellation requires confirmation containing the selected human `runId` and technical Job identity where useful.

After cancellation request succeeds, update the selected Job from the authoritative response and refresh logs.

Do not delete artifacts.

## 17. Catalog/data ownership

Runs should avoid each subview loading the same catalogs independently.

### 17.1 Runs controller owns

At minimum:

- Job catalog;
- Run artifact catalog.

Config/Dataset catalogs may be loaded lazily when New Run or exact Job identity enrichment requires them, then reused within the Runs workspace session.

### 17.2 Subviews own

Replay:

- telemetry status/events;
- Behavior Cloning progress where relevant;
- checkpoint evaluations.

Diagnostics:

- training metric status/scalars.

Logs:

- Job log snapshot/polling.

Artifact Overview:

- no extra data required initially beyond RunSummary;
- Compare/Evidence load their own destination resources after navigation.

### 17.3 No duplicate hidden polling

Only the active analytical subview owns its expensive polling.

Changing:

```text
Replay -> Diagnostics
```

must stop Replay's high-frequency active-view polling unless a tiny shared status requirement is explicitly justified.

Global overview polling is separate application-shell behavior and must not cause duplicate feature streaming.

## 18. Catalog freshness

The selected Job's execution state should not require a full page reload.

The Runs controller may refresh the Job catalog while Runs is active, with a bounded cadence slower than Replay telemetry.

Rules:

- only one Job-catalog poller per Runs workspace;
- preserve last-known-good catalog on recoverable fetch failure;
- active selected resource stays selected if still present;
- do not reorder focus/scroll on each refresh;
- Run artifact catalog can refresh less frequently because artifacts are immutable once registered;
- exact polling intervals are implementation measurements, not product semantics.

## 19. Error and stale states

### 19.1 Catalog failure with prior data

Keep last-known-good rail entries visible and mark catalog freshness/error.

Do not clear selected context merely because refresh failed.

### 19.2 Explicit target not found

Show:

```text
Selected Job/Artifact is not present in the current Studio catalog.
```

Actions:

- Return to Runs;
- New Run when appropriate.

Do not select another resource silently.

### 19.3 Job subresource failure

Replay/Diagnostics/Logs failures stay local to the subview.

A telemetry failure does not mark the Job itself FAILED.

A training metrics failure does not invalidate the Artifact.

### 19.4 Artifact INVALID

Artifact remains selectable so the researcher can inspect the failure and open Evidence.

INVALID is not filtered out of the Artifact rail.

## 20. Cross-workspace navigation

### 20.1 From Home

Home actions targeting active execution should use:

```text
workspace=runs&target=job&job=<authoritative-job-id>&view=<appropriate-view>
```

Home actions targeting completed registered results use:

```text
workspace=runs&target=artifact&run=<authoritative-run-resource-id>&view=overview
```

Do not pass `runId` as a substitute resource ID.

### 20.2 From Compare

`Open Left Run` / `Open Right Run` opens the corresponding Artifact Overview because Compare identities are Run resource identities.

### 20.3 From Evidence

`Open Run` opens Artifact Overview using Evidence's `runResourceId`.

### 20.4 From Artifact Overview

`Compare` and `Evidence` translate the selected Artifact ID into the destination workspace's canonical parameter.

## 21. Back/forward history

Browser navigation is a supported analytical workflow.

Expected behavior:

```text
Job A Replay
-> Job A Diagnostics
-> Artifact X Overview
-> Back
= Job A Diagnostics
-> Back
= Job A Replay with restorable seed/env/timeframe/range
```

Ephemeral hover, open menus, playing state, and playback speed are not restored.

## 22. Focus and keyboard

### 22.1 Rail

Rail rows are semantic buttons/links with visible focus.

Arrow-key roving navigation is optional; Tab + Enter/Space is required.

### 22.2 Tabs

Job subview tabs use an accessible tab/navigation pattern.

Switching tabs does not unexpectedly move focus into chart internals.

### 22.3 Overlay Inspector

Escape closes the inspector and restores focus to the action/selection that opened it when practical.

### 22.4 Cancellation dialog

Focus is trapped while open, initial focus is non-destructive Cancel, and Escape cancels the dialog.

## 23. Desktop composition

### 23.1 1440x900

Default rail can remain expanded.

Replay chart receives the majority of horizontal and vertical space.

### 23.2 1180x800

Collapse in this order before shrinking evidence:

1. reduce global navigation rail;
2. collapse Runs rail to selected-context trigger/drawer in Replay;
3. preserve three-pane market workspace;
4. use overlay inspector rather than fixed right column.

The product is desktop-only in this phase; no mobile layout is required.

## 24. Visual hierarchy rules

### 24.1 Job context

Strongest visible semantics:

- human `runId`;
- execution state;
- instrument when authoritative;
- selected subview.

Dataset/config digests are secondary but readily inspectable.

### 24.2 Artifact context

Strongest visible semantics:

- human `runId`;
- VALID/INVALID;
- algorithm/run kind;
- key result context;
- actions to Compare/Evidence.

Positive return must not visually overpower INVALID evidence.

### 24.3 Avoid equal-weight cards

Overview pages should use reading order and grouped definitions rather than a dashboard of equal-size KPI cards.

## 25. Acceptance tests

### 25.1 Routing

- `workspace=experiments` canonicalizes to Runs/New.
- legacy `workspace=live&job=X` canonicalizes to Job X Replay.
- legacy `workspace=live` chooses the deterministic default Job when available.
- Job route strips Run-artifact parameters.
- Artifact route strips Job/Replay parameters.
- Artifact + Replay canonicalizes to Artifact Overview.
- workspace exit removes Runs-only parameters.

### 25.2 Default selection

- active Job outranks Artifact when no explicit target exists.
- newest Artifact outranks terminal Job when no active Job exists.
- explicit missing target is not silently replaced.

### 25.3 Identity

- Job APIs receive `JobSummary.id`.
- Compare/Evidence receive `RunSummary.id`.
- no test permits Job/Artifact join solely by equal `runId`.
- Job -> Dataset/Config enrichment uses exact resource IDs only.

### 25.4 Rail

- Job and Artifact sections are visually/semantically distinct.
- terminal Job does not receive validation-green merely because it succeeded.
- invalid Artifact remains listed and clearly invalid.

### 25.5 New Run

- only selectable valid launch resources are offered.
- submit request contains only current authoritative contract fields.
- success navigates to returned Job ID Overview.
- submit error preserves form selection.

### 25.6 Job tabs

- Job exposes Overview/Replay/Diagnostics/Logs.
- Artifact does not expose unsupported Job-only tabs.
- Job selection survives tab changes.

### 25.7 Replay

Preserve all existing Replay characterization tests plus:

- no duplicate Job selector inside Replay;
- exact one-symbol Dataset suppresses normal Symbol selector;
- legacy multi-symbol Dataset uses Dataset symbols rather than hard-coded symbol defaults;
- unresolved Dataset identity does not fabricate maintained-mode claims;
- opening inspector does not change chart dimensions.

### 25.8 Polling

- Replay streaming active only while Replay view is active.
- Diagnostics metrics active only while Diagnostics view is active.
- Logs polling active only while Logs is active.
- changing Job invalidates old request generations/cursors.
- no hidden view continues expensive feature polling.

### 25.9 Logs/cancel

- stale old-Job log response is ignored.
- manual scroll pauses log follow.
- Stop appears only for cancellable queued/running Job.
- Stop requires confirmation.
- cancel response updates selected Job without switching target.

### 25.10 Artifact

- INVALID artifact stays inspectable.
- positive metrics do not override INVALID state.
- Compare/Evidence drill-through uses Run resource ID.
- no Job-only action is exposed without explicit relationship contract.

### 25.11 History/layout

- browser Back/Forward restores target type, resource ID, view, and Replay analytical URL state.
- 1440x900 passes without whole-page overflow.
- 1180x800 preserves Replay primary evidence.
- collapsed Runs rail does not lose selected context identity.

## 26. Implementation boundaries

This specification intentionally does not require a big-bang rewrite.

Recommended internal composition:

```text
runs/
  RunsWorkspace
  RunsRail
  runsRouteState
  runsSelectionModel
  NewRunView
  JobOverviewView
  ArtifactOverviewView
```

Existing feature components remain reusable:

```text
live/
  SynchronizedResearchChartWorkspace
  ResearchChartInspector
  TrainingDiagnosticsPanel
  BehaviorCloningProgressPanel
  telemetry/metric hooks
```

Current `ExperimentsPage`, `RunCenterPage`, and `LiveTrainingPage` behavior should be characterized by tests, then their responsibilities migrated under Runs incrementally.

Shared abstractions are introduced only after semantic duplication is demonstrated.

## 27. Migration sequence

1. Add route/selection model with no visual behavior change.
2. Add Runs shell and two-section rail using current Job/Run catalogs.
3. Move current Experiments behavior into New Run; navigate to returned Job.
4. Add Job Overview using existing authoritative fields.
5. Mount current Live Training Replay under selected Job; remove duplicate Job selector only after route ownership is tested.
6. Split Diagnostics mount lifecycle from Replay and stop hidden polling.
7. Move Run Center logs/cancellation under Logs with confirmation and follow behavior.
8. Add Artifact Overview from existing RunSummary fields.
9. Wire Compare/Evidence drill-through by Run resource ID.
10. Add compact-width rail collapse and final visual harmonization.

At every step preserve a working application and run focused regression tests before refactoring.

## 28. Deferred capabilities

Not part of this implementation unless a separate authoritative contract appears:

- automatic Job -> Run-artifact linkage by `runId`;
- visual editing of training hyperparameters;
- Artifact Replay through guessed historical Job;
- result promotion/release actions;
- rail search/filtering beyond current demonstrated need;
- pagination without API support;
- user-created tags/groups;
- synthetic overall Run health score.

## 29. Definition of done

Runs is complete when:

- the user can create a Job, inspect execution, Replay, Diagnostics, and Logs without navigating between three primary pages;
- Job and Artifact resource identities remain distinct and truthful;
- completed Artifacts are accessible in the same research area without pretending they are Job resources;
- Compare/Evidence drill-through uses authoritative Artifact IDs;
- maintained one-symbol Replay removes misleading normal Symbol selection only when exact Dataset identity supports it;
- hidden analytical views stop unnecessary polling;
- cancellation remains process-owner/cancellable gated and requires confirmation;
- legacy URLs canonicalize safely;
- browser history and workspace parameter scoping are deterministic;
- existing Replay lifecycle and interaction contracts remain green;
- 1440x900 and 1180x800 desktop checks pass;
- no implicit Job/Artifact join exists without an explicit backend relationship.