# Trade RL Studio

Local-first research console for `trade_rl`, implemented with Vite, React, and strict TypeScript.

## Current slice

- fixed top bar, sidebar, workspace, and status bar;
- no browser page scrolling at 1536×1024 and 1440×900;
- dashboard for system capacity, dataset identity, active jobs, run summary, alerts, baseline comparison, fold stability, and production assessment;
- working workspace navigation;
- typed `/api/studio/overview` adapter with deterministic demo fallback;
- explicit `NO-GO` research status and no direct exchange-order controls.

## Run locally

```bash
cd studio
npm install
npm run dev
```

Open `http://127.0.0.1:4173`.

## Verify

```bash
npm test -- --run
npm run typecheck
npm run build
npm run check:layout
```

`check:layout` uses Chromium to verify that neither the document nor body exceeds the viewport height at the supported desktop sizes.

## Backend contract

The frontend requests `GET /api/studio/overview`. Until the Python studio API is connected, failed or invalid responses automatically use deterministic demo data and display `DEMO DATA` in the top bar.
