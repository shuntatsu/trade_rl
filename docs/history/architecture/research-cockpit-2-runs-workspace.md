# Research Cockpit 2.0: Runs Workspace Detailed Specification

Status: Proposed
Date: 2026-08-10
Parent: `docs/architecture/adr-002-research-cockpit-2.md`
Scope: Runs workspace UX, routing, state ownership, data ownership, Replay/Diagnostics/Logs composition, and Job/Run-artifact identity boundaries.

## 1. Decision

Research Cockpit 2.0 uses one primary **Runs** workspace for research execution and result inspection, but it does **not** collapse all backend resources into one synthetic `Run` object.

The workspace exposes two authoritative resource families:

1. **Execution Jobs** — `JobSummary` resources used for execution state, Replay telemetry, Diagnostics, checkpoint-evaluation inspection, Behavior Cloning progress, logs, and cancellation.
2. **Research Artifacts** — `RunSummary` resources used for immutable result inspection, Compare, and Evidence.

`runId` is a human-facing research identifier. It is not sufficient by itself to join these resource families.

The user sees one research area, but the UI must preserve resource type and authoritative resource ID at all times.

## 2. Why this split is required

The current Studio API is explicitly split:

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

Even when labels match, data loading and navigation remain bound to the resource type required by the endpoint.

## 3. User mental model

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

The visual layout may suggest lifecycle progression, but the connector between Job and Artifact must not be shown as established lineage unless a backend relationship exists.

A successful Job can remain a Job-only context even if a similarly named Artifact exists.

## 4. Resource terminology

### 4.1 Job

Authoritative identity:

```text
JobSummary.id
```

A Job answers execution questions.

Important fields:

- `runId`;
- status;
- Config resource/digest;
- Dataset resource/ID;
- artifact root;
- submitted/started/completed timestamps;
- PID/process ownership fields;
- `cancellable`;
- exit code/process error.

`SUCCEEDED` means execution completed successfully. It does not mean Artifact VALID, Evidence complete, profitable, or production-approved.

### 4.2 Research Artifact

Authoritative identity:

```text
RunSummary.id
```

An Artifact answers immutable-result and validation questions.

Important fields:

- `runId`;
- manifest digest;
- run kind;
- algorithm;
- dataset ID;
- evaluation period;
- timestamps;
- file count;
- result metrics when present;
- VALID/INVALID;
- validation error;
- `productionStatus`.

### 4.3 Human Run ID

`runId` is the primary visible research label where available.

Resource ID remains available in technical detail and is always used for APIs that require it.

`runId` must not function as an implicit frontend foreign key.

## 5. Canonical route model

Runs uses an explicit discriminated target.

### 5.1 New Run

```text
?workspace=runs&view=new
```

No Job or Artifact is selected.

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

Artifact targets expose only views supported by Run resource APIs. Replay, Diagnostics, and Logs are not shown without an explicit API relationship.

### 5.4 Why `target` is explicit

The UI must not infer type from ID format. `target=job|artifact` is the route discriminator.

If both `job` and `run` appear, canonicalization keeps only the one compatible with `target`.

## 6. Legacy route migration

Interpret old routes as follows:

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

After interpretation, replace the URL with canonical state.

## 7. Route canonicalization rules

Examples:

```text
target=job + run=<id>
-> remove run

target=artifact + job=<id>
-> remove job

target=artifact + view=replay|diagnostics|logs
-> view=overview

view=new + target/job/run
-> remove target/job/run
```

Runs owns only its documented query parameters. Navigating away removes Runs-only state unless explicitly translated into the destination workspace's canonical parameters.

Examples of Runs-only state:

- `target`;
- `job`;
- `run`;
- Runs `view`;
- `seed`;
- `env`;
- `timeframe`;
- `range`.

## 8. Default target resolution

Default resolution depends on the requested view intent.

### 8.1 Job-only view requested

If the URL requests:

- Replay;
- Diagnostics;
- Logs;

but has no explicit Job target, choose only from Jobs.

Priority:

1. newest active Job (`queued`, `running`, `cancelling`);
2. newest terminal Job.

If there is no Job, remain in a **Job-required empty state** for that requested view.

Do not redirect to an Artifact Overview because that changes the user's requested task.

This rule is especially important for legacy `workspace=live` migration.

### 8.2 Generic Runs/Overview entry

If the user enters Runs without an explicit target and without a Job-only view intent:

1. newest active Job;
2. newest Research Artifact by authoritative creation timestamp;
3. newest terminal Job;
4. New Run if no resources exist.

This is only a navigation default. It does not imply recommendation, quality, or validity.

### 8.3 Explicit target missing

If an explicit URL target is absent from the refreshed catalog, do not substitute another resource.

Show:

```text
Selected Job/Artifact is not present in the current Studio catalog.
```

Actions:

- Return to Runs;
- New Run where appropriate.

## 9. Runs workspace layout

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

The rail is subordinate to the evidence viewport.

At 1440px width, target rail range is approximately 220-260px.

At compact desktop width, Replay collapses the rail before shrinking the dominant market viewport.

Exact dimensions are browser-QA measurements, not hard product constants.

## 10. Runs rail semantics

### 10.1 Sections

The rail contains:

1. `+ New Run`;
2. **Execution Jobs**;
3. **Research Artifacts**.

Do not merge these into one undifferentiated list.

### 10.2 Job row

Default-visible content:

```text
<runId>
<execution state> · <recency>
<instrument when authoritative>
```

Technical Job ID is secondary detail.

Execution styling:

- RUNNING/current: cyan/informational;
- QUEUED/CANCELLING: neutral or amber depending on attention semantics;
- FAILED: red;
- SUCCEEDED/CANCELLED: neutral completion semantics, not validation-green.

### 10.3 Artifact row

Default-visible content:

```text
<runId>
<VALID|INVALID> · <algorithm or runKind>
<period/created time when space permits>
```

VALID may use validation-green. INVALID uses red.

`NO-GO` is not repeated in every row.

### 10.4 Sorting

Jobs:

- active before terminal;
- newest submitted first within activity class.

Artifacts:

- newest created first.

Never rank by Sharpe, return, checkpoint score, or other performance metric.

### 10.5 Search/filter scope

No fuzzy search, saved filters, tags, or pagination are added initially unless real resource counts prove the rail unusable.

## 11. View availability by target type

### 11.1 New

Only New Run is shown.

### 11.2 Job

Show:

- Overview;
- Replay;
- Diagnostics;
- Logs.

Replay/Diagnostics remain addressable even before telemetry/metrics exist. Their content explains the unavailable state.

### 11.3 Artifact

Show:

- Overview.

Expose cross-workspace actions:

- Compare;
- Evidence.

Do not show disabled Job-only tabs; that would falsely imply an addressable relationship.

## 12. New Run

Primary question:

**What exactly will this Job execute?**

### 12.1 Data inputs

New Run needs Config and Dataset catalogs.

Only resources permitted by the current launch contract are selectable. Invalid Dataset/Config resources remain inspectable elsewhere but are not launch options.

### 12.2 Request contract

Submission remains exactly:

```text
configResourceId
datasetResourceId
runId
```

No browser-only optimizer override is introduced.

### 12.3 Preview

Show only authoritative values:

- Config name/algorithm/digest/path;
- Dataset name/ID/market/symbols/timeframes/status;
- Run ID validity;
- exact instrument identity when Dataset is unambiguous.

Do not show steps, folds, seeds, learning rate, batch size, or action shape unless an API explicitly exposes them for the selected Config.

### 12.4 Single-instrument wording

If the selected exact Dataset has one authoritative symbol, the UI may state that the Dataset is one-symbol.

Do not infer the complete maintained action/policy contract from Dataset cardinality alone.

### 12.5 Success transition

On successful submit:

```text
submitTrainingJob(...)
-> returned JobSummary.id
-> workspace=runs&target=job&job=<id>&view=overview
```

Do not leave the user on New Run with only a success card.

Submission failure retains form state and shows the server error.

## 13. Job Overview

Primary question:

**What is this execution doing, and what can I inspect now?**

### 13.1 Exact identity enrichment

The Runs controller may enrich a Job by matching:

```text
JobSummary.datasetResourceId -> DatasetSummary.id
JobSummary.configResourceId  -> ConfigSummary.id
```

These are explicit resource references and are safe joins.

This is intentionally different from prohibited Job -> Artifact matching by `runId`.

### 13.2 Identity block

Show:

- `runId`;
- Job resource ID as secondary technical detail;
- Dataset resource / Dataset ID;
- Config resource / digest;
- artifact root;
- instrument when exact Dataset identity is available.

### 13.3 Execution block

Show available authoritative values:

- status;
- submitted;
- started;
- completed;
- PID;
- process ownership/cancellability;
- exit code;
- process error.

Do not invent a generic progress percentage.

### 13.4 Analysis availability

Replay, Diagnostics, and Logs remain reachable.

Their state comes from their own APIs, not from assumptions such as:

```text
RUNNING => telemetry must exist
SUCCEEDED => metrics must exist
FAILED => Replay is useless
```

### 13.5 Completion behavior

When a Job becomes SUCCEEDED, stay on the Job context.

Do not auto-switch to a similarly named Artifact.

If a future backend contract provides an explicit result Artifact resource ID, show an `Open Result Artifact` action.

## 14. Artifact Overview

Primary question:

**What immutable registered result is this, and is it valid?**

### 14.1 Identity/validation

Show:

- `runId`;
- Run resource ID;
- VALID/INVALID and validation error;
- manifest digest;
- run kind;
- algorithm;
- dataset ID;
- period;
- created/completed timestamps;
- relative path/file count;
- production status.

### 14.2 Result summary

Available values such as total return, max drawdown, and Sharpe may be shown compactly.

Rules:

- validation status outranks performance;
- INVALID remains visually dominant even when metrics are positive;
- no `good`, `winner`, or release-ready label is derived from performance.

### 14.3 Compare action

`Compare` navigates with this Artifact as one explicit side, preferably canonical `left=<selected-run-resource-id>`.

The Compare resolver must choose the first other eligible VALID Run as the opposite side.

It must never silently compare a Run with itself when another eligible candidate exists.

If no second valid Run exists, Compare shows a pair-required state rather than fabricating a comparison.

### 14.4 Evidence action

Evidence navigation uses the selected Run resource ID directly.

Do not offer Replay/Diagnostics/Logs based on a matching `runId`.

## 15. Replay

Replay is a Job-only view.

Primary question:

**What did the policy do in market context, and what happened around that decision?**

### 15.1 Job selection ownership

ReplayToolbar loses its Job selector.

The selected Job comes from Runs route/rail state.

This removes duplicate context selection.

### 15.2 Instrument contract

Resolve the exact Dataset using `datasetResourceId` when possible.

If exact Dataset has one symbol:

- show instrument as read-only identity;
- do not show a normal Symbol dropdown.

If exact Dataset is historical/legacy multi-symbol:

- show Symbol selector populated only from Dataset symbols;
- choose deterministic Dataset-derived default;
- never hard-code `BTCUSDT` as the generic fallback.

If Dataset resource cannot be resolved:

- do not guess maintained/legacy mode;
- preserve telemetry inspection where safe;
- mark instrument contract unavailable.

### 15.3 Source state

Seed and Environment remain Replay-local analytical controls and may be URL-backed.

Changing source:

- pauses playback;
- follows latest for the new source;
- clears invalid preview/committed selection;
- resets chart navigation deterministically.

### 15.4 Timeframe/range/layers

Timeframe and range are URL-restorable.

Layer toggles remain local initially.

### 15.5 Transport

Playback controls live by the scrubber/transport region:

- first;
- previous event;
- play/pause;
- next event;
- latest;
- speed;
- follow latest.

`playing` and speed are local state, not URL state.

### 15.6 Three-pane architecture

Preserve market-first three-pane composition.

Recommended ranges:

- Price & Execution: 55-60%;
- Exposure / Portfolio: 22-27%;
- State & Risk: 16-20%.

Learning metrics remain in Diagnostics because optimizer-step and market-time semantics differ.

### 15.7 Locked lifecycle contract

Preserve:

```text
ENTRY #1 ---- REDUCE #1 ---- EXIT #1
ENTRY #2 -------------------- OPEN #2
```

A reduction does not create a new Trade lifecycle.

Markers remain visible together with background lifecycle bands.

### 15.8 Interaction contract

Preserve:

- synchronized time scale/crosshair;
- wheel zoom and drag pan;
- manual navigation disables follow-latest;
- programmatic range changes do not count as manual navigation;
- hover = preview;
- click = commit;
- commit pauses Replay;
- bounded telemetry cache;
- seed/environment/current-episode isolation;
- timestamp normalization/validation;
- target vs executed exposure;
- LONG/SHORT/CLOSE and increase/reduce/rebuy/flip semantics;
- independent RISK/END events;
- explicit checkpoint evidence selection.

### 15.9 Inspector

Committed Trade/point opens a 320-360px overlay inspector.

Opening/closing must not resize chart geometry.

Escape closes it and restores appropriate focus.

### 15.10 Empty states

Prefer precise reason:

- Behavior Cloning still running;
- telemetry not initialized;
- no telemetry for selected Seed/Environment;
- source unavailable;
- no Job exists for requested Replay.

Generic `No data` is insufficient when reason is known.

## 16. Diagnostics

Diagnostics is Job-only.

Primary question:

**Does optimization/training behavior require attention?**

### 16.1 Semantic boundary

Diagnostics is not profitability evidence.

Retain metric groups:

- Optimization;
- Policy;
- Value;
- Trading / Risk.

Thresholds such as Approx KL guidance are `ATTENTION`, not production failure unless a separate authoritative gate says otherwise.

Preferred group state labels:

- NO ATTENTION;
- ATTENTION;
- UNAVAILABLE.

Avoid green `HEALTHY` if it could imply model quality/profitability.

### 16.2 Poll ownership

Training metric polling is active only while Diagnostics is the active Runs view for the selected Job.

CSS `hidden` is not sufficient.

Changing Job resets generation/cursors/selection.

## 17. Logs

Logs is Job-only.

Primary question:

**What is this owned process doing, and can it be safely stopped?**

### 17.1 Request ownership

Logs load only for the selected Job.

A stale response from a previously selected Job must not replace current content.

### 17.2 Follow latest

Default: follow latest.

If the user manually scrolls away from bottom:

- pause follow;
- do not steal scroll position on new lines;
- show `Jump to latest`.

Because current API exposes a bounded log snapshot rather than cursor streaming, active-view polling may refresh the bounded snapshot. Exact cadence is implementation-measured.

Logs polling stops when Logs is not active.

### 17.3 Cancellation

Stop is visible only when:

```text
job.cancellable
&& job.status in {queued, running}
```

Confirmation includes human `runId` and enough Job identity to prevent mistakes.

Initial dialog focus is Cancel; Escape cancels the dialog.

After successful cancellation:

- update selected Job from authoritative response;
- remain on same Job target;
- refresh log;
- do not delete artifacts.

## 18. Catalog/data ownership

Runs avoids duplicate catalog fetching across subviews.

### 18.1 Runs controller owns

At minimum:

- Job catalog;
- Run Artifact catalog.

Dataset/Config catalogs may load lazily when New Run or exact Job identity enrichment requires them, then be reused within the Runs session.

### 18.2 Replay owns

Only while active:

- telemetry status/events;
- Behavior Cloning progress where needed;
- checkpoint evaluations.

### 18.3 Diagnostics owns

Only while active:

- training metric status/scalars.

### 18.4 Logs owns

Only while active:

- Job log snapshot/polling.

### 18.5 Artifact Overview owns

No additional fetch is required initially beyond `RunSummary`.

Compare/Evidence fetch their own authoritative resources after navigation.

## 19. Polling and freshness

### 19.1 Feature polling

Only active analytical views perform expensive feature polling.

Switching:

```text
Replay -> Diagnostics
```

stops Replay feature polling unless a separately justified tiny status request exists.

Switching away from Diagnostics stops metrics polling.

Switching away from Logs stops log polling.

### 19.2 Catalog polling

Runs may refresh Job catalog while Runs is active so execution state does not require page reload.

Rules:

- exactly one Job catalog poller per Runs workspace;
- cadence slower than Replay telemetry;
- keep last-known-good catalog on recoverable failure;
- preserve selected target if still present;
- do not steal focus/scroll because list order refreshed;
- Artifact catalog may refresh less frequently because registered artifacts are immutable.

Exact interval values are implementation measurements.

### 19.3 Global overview polling

Application-shell overview polling is separate. Feature components must not duplicate streaming merely because overview is also refreshing.

## 20. Error semantics

### 20.1 Catalog failure with prior data

Keep prior valid rail/context visible and mark catalog freshness/error.

Do not clear selection because refresh failed.

### 20.2 Missing explicit target

Show target-not-found state; do not substitute another Job/Artifact.

### 20.3 Job subresource failure

A Replay failure does not mark Job FAILED.

A training metrics failure does not invalidate an Artifact.

A log load error does not erase Job identity.

### 20.4 Artifact INVALID

INVALID Artifacts remain listed and selectable for inspection/Evidence.

They are not filtered out merely because Compare uses only valid candidates.

## 21. Cross-workspace navigation

### 21.1 Home -> Runs

Execution action:

```text
workspace=runs&target=job&job=<authoritative-job-id>&view=<view>
```

Artifact action:

```text
workspace=runs&target=artifact&run=<authoritative-run-resource-id>&view=overview
```

Do not substitute `runId` for resource ID.

### 21.2 Compare -> Runs

`Open Left Run` / `Open Right Run` opens Artifact Overview using the comparison's Run resource IDs.

### 21.3 Evidence -> Runs

`Open Run` opens Artifact Overview using `EvidenceReport.runResourceId`.

### 21.4 Artifact -> Compare

Set selected Artifact as one side.

Opposite side selection must prefer the first other valid Run rather than the same Run.

### 21.5 Artifact -> Evidence

Pass selected Artifact resource ID to the Evidence workspace canonical selection parameter.

## 22. Back/forward history

Browser navigation is a supported analysis workflow.

Example:

```text
Job A Replay
-> Job A Diagnostics
-> Artifact X Overview
-> Back
= Job A Diagnostics
-> Back
= Job A Replay with restored seed/env/timeframe/range
```

Do not restore:

- hover preview;
- crosshair preview;
- open popovers;
- `playing`;
- playback speed.

## 23. Focus and keyboard

### 23.1 Rail

Rows use semantic button/link behavior with visible focus.

Tab + Enter/Space is required. Roving arrow navigation is optional.

### 23.2 View navigation

Job subview navigation is keyboard-operable and exposes current state semantically.

Switching tabs must not unexpectedly move focus into chart internals.

### 23.3 Inspector

Escape closes overlay and returns focus when practical.

### 23.4 Cancellation dialog

Focus stays within modal; initial focus is Cancel; Escape closes without action.

## 24. Desktop layout contract

### 24.1 1440x900

- expanded global rail;
- expanded Runs rail;
- dominant evidence viewport;
- overlay inspector.

### 24.2 1180x800

Collapse in this order:

1. global navigation rail;
2. Runs rail in Replay;
3. preserve Replay chart area;
4. overlay inspector remains overlay.

Do not shrink market visualization first.

Mobile/tablet support is outside this phase.

## 25. Visual hierarchy

### 25.1 Job context

Strongest visible semantics:

1. `runId`;
2. execution state;
3. instrument when authoritative;
4. current Job subview.

Dataset/Config digests are secondary details.

### 25.2 Artifact context

Strongest visible semantics:

1. `runId`;
2. VALID/INVALID;
3. run kind/algorithm;
4. compact result context;
5. Compare/Evidence actions.

Positive performance must not visually overpower INVALID status.

### 25.3 Avoid KPI-card grids

Use ordered sections/definition groups. Do not turn Overview into equally weighted metric tiles.

## 26. Acceptance tests

### 26.1 Routing

- `workspace=experiments` -> Runs/New.
- `workspace=live&job=X` -> Job X Replay.
- legacy `workspace=live` selects only a Job; no Job produces Replay-specific empty state.
- Job route strips Artifact params.
- Artifact route strips Job/Replay params.
- Artifact + Job-only view -> Artifact Overview.
- leaving Runs removes Runs-only params.

### 26.2 Default resolution

- Job-only requested view chooses active/newest Job only.
- generic Runs entry prefers active Job, then newest Artifact, then terminal Job.
- explicit missing target is never silently replaced.

### 26.3 Identity

- Job APIs receive `JobSummary.id`.
- Compare/Evidence receive `RunSummary.id`.
- equal `runId` alone never joins Job and Artifact.
- Job -> Dataset/Config enrichment uses exact resource IDs.

### 26.4 Rail

- Job and Artifact sections are distinct.
- SUCCEEDED Job does not use validation-green.
- INVALID Artifact remains visible.
- sorting never depends on performance score.

### 26.5 New Run

- launch options follow current validation contract.
- request contains only authoritative current request fields.
- success opens returned Job Overview.
- failure preserves user selection/input.

### 26.6 View availability

- Job exposes Overview/Replay/Diagnostics/Logs.
- Artifact exposes only supported Artifact views/actions.
- Job selection survives Job subview changes.

### 26.7 Replay

Preserve current Replay regression suite plus:

- Replay contains no duplicate Job selector.
- one-symbol exact Dataset removes normal Symbol dropdown.
- multi-symbol exact Dataset uses Dataset symbol list and deterministic Dataset-derived default.
- unresolved Dataset identity does not fabricate maintained-mode claim.
- opening inspector preserves chart geometry.
- lifecycle band contracts remain unchanged.

### 26.8 Polling

- Replay feature polling active only in Replay.
- Diagnostics polling active only in Diagnostics.
- Logs polling active only in Logs.
- Job changes invalidate old request generations/cursors.
- hidden analytical view does not keep expensive polling alive.

### 26.9 Logs/cancel

- stale old-Job log response ignored.
- manual scroll pauses follow.
- `Jump to latest` restores follow.
- Stop visible only for cancellable queued/running Job.
- Stop requires confirmation.
- successful cancel remains on same Job target.

### 26.10 Artifact

- INVALID remains inspectable.
- positive metrics do not override INVALID.
- Compare/Evidence drill-through uses Run resource ID.
- no Job-only action from `runId` matching.
- Compare opposite side never resolves to same selected Run when another valid candidate exists.
- one valid Run only -> pair-required state, no self-comparison.

### 26.11 History/layout

- Back/Forward restores target type, resource ID, Job view, and Replay analytical URL state.
- 1440x900 no whole-page overflow.
- 1180x800 preserves dominant Replay evidence.
- collapsed Runs rail retains selected context identity.

## 27. Implementation boundaries

Avoid a big-bang rewrite.

Likely new composition:

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

Reuse existing feature internals:

```text
live/
  SynchronizedResearchChartWorkspace
  ResearchChartInspector
  TrainingDiagnosticsPanel
  BehaviorCloningProgressPanel
  telemetry/metric hooks
```

Current `ExperimentsPage`, `RunCenterPage`, and `LiveTrainingPage` behavior is first characterized with tests, then responsibilities migrate under Runs incrementally.

Shared abstractions require at least two semantic consumers.

## 28. Migration sequence

1. Add route/selection model while old pages still function.
2. Add Runs shell and two-section rail from current Job/Run catalogs.
3. Move Experiments behavior into New Run and navigate to returned Job.
4. Add Job Overview from current Job fields.
5. Mount existing Replay under selected Job with chart semantics unchanged.
6. Remove duplicate Job selector from Replay after route ownership is tested.
7. Split Diagnostics lifecycle so hidden metrics stop polling.
8. Move Run Center Logs/cancellation under Job Logs, adding confirmation/follow behavior.
9. Add Artifact Overview from current RunSummary fields.
10. Wire Artifact -> Compare/Evidence and Compare/Evidence -> Artifact drill-through.
11. Add compact Replay rail collapse.
12. Harmonize visuals only after behavior is green.

Each migration step:

1. characterize current behavior;
2. add failing acceptance test;
3. implement smallest change;
4. run focused tests;
5. typecheck/build where relevant;
6. run affected fixed-viewport checks;
7. refactor only after green.

## 29. Deferred capabilities

Not part of this implementation without a new authoritative contract/need:

- Job -> Artifact linkage by matching `runId`;
- Artifact Replay through guessed historical Job;
- editing training hyperparameters in browser;
- result promotion/release actions;
- synthetic overall Run health score;
- rail fuzzy search/tags/saved filters without demonstrated need;
- pagination without API support;
- mobile/tablet layout.

## 30. Definition of done

Runs is complete only when:

- New Run, execution inspection, Replay, Diagnostics, and Logs no longer require three separate primary workspaces;
- Job and Artifact resource identities remain distinct;
- registered Artifacts remain available in Runs without pretending they are Jobs;
- Compare/Evidence use authoritative Artifact IDs;
- legacy Live intent does not unexpectedly redirect to an Artifact when no Job exists;
- maintained one-symbol Replay removes misleading normal Symbol selection only when exact Dataset identity supports it;
- hidden analytical views stop unnecessary polling;
- cancellation remains safely gated and confirmed;
- browser history and query scoping are deterministic;
- Artifact Compare entry avoids self-comparison;
- existing Replay lifecycle/interaction regressions remain green;
- 1440x900 and 1180x800 desktop checks pass;
- no implicit Job/Artifact join exists without an explicit backend relationship.