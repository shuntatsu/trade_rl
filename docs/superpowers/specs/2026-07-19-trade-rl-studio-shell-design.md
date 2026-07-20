# Trade RL Studio Shell Design

## Purpose

Add a local-first research console to `trade_rl` as an independent `studio/` subproject built with Vite, React, and strict TypeScript. The shell must make the repository approachable without weakening the existing research, artifact, or evidence contracts.

## Primary UX constraint

The browser page itself must never vertically scroll. The application uses a fixed top bar, fixed left navigation, fixed bottom status bar, and a main workspace that always fits within the viewport. Dense content is handled with bounded panels, pagination, tabs, drawers, and split views rather than document scrolling.

## Initial vertical slice

The first deliverable includes:

- a fixed no-page-scroll application shell;
- desktop navigation for Dashboard, Data Lab, Experiments, Run Center, Compare, Evidence Explorer, Serving Monitor, and Settings;
- a dashboard matching the approved dark research-console concept;
- responsive sidebar compaction for narrower desktop windows;
- working client-side page switching;
- an API adapter that requests `/api/studio/overview` and falls back to deterministic demo data when the backend is unavailable;
- unit and interaction tests;
- build, lint, and browser screenshot verification.

## Architecture

`studio/` is an independent Node project. React owns presentation and local navigation state. `src/api/studioApi.ts` is the only network boundary for dashboard data. Components consume typed `StudioOverview` data and do not know whether it came from a server or the deterministic fallback fixture.

The Python core remains authoritative. No training logic, dataset identity logic, evidence verification, or artifact mutation is duplicated in TypeScript.

## Layout

The viewport is a CSS grid:

```text
rows:    48px 1fr 34px
columns: 184px 1fr
```

The top bar spans the workspace column. The sidebar spans all rows. The main dashboard uses a nested grid with four horizontal bands. Every grid child sets `min-height: 0` and `overflow: hidden` so content cannot force document growth.

## Visual system

- Background: near-black blue.
- Panels: low-contrast navy surfaces with one-pixel borders.
- Primary accent: cyan.
- Positive: green.
- Warning: amber.
- Critical and NO-GO: red.
- Typography: system UI stack with compact labels and tabular numbers.
- Corners: 7–10px, not oversized cards.
- Motion: subtle selected-state and hover transitions only, disabled under `prefers-reduced-motion`.

## Error behavior

The API adapter returns `{ source: "api" | "demo", overview }`. Network errors, non-2xx responses, invalid JSON, and shape mismatches all return demo data and expose `source: "demo"`. The UI displays a visible “DEMO DATA” state in the top bar when fallback data is active.

## Testing

- API adapter falls back on failed requests.
- App renders fixed shell landmarks and the NO-GO research status.
- Navigation changes the visible workspace without changing document layout.
- CSS contract includes viewport-height root and hidden document overflow.
- Production build succeeds with strict TypeScript.

## Out of scope for this slice

- starting or cancelling real training jobs;
- writing datasets;
- reading private keys;
- direct exchange routing;
- release approval;
- persistent user preferences;
- mobile phone layout.
