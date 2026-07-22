# Live Training Stream Isolation Verification — 2026-07-23

## 1. Finding and scope

This note records remediation of architecture-audit finding `AUD-STUDIO-001` from `docs/verification/2026-07-22-post-merge-architecture-audit.md`.

The confirmed defect was limited to exploratory Studio visualization. Training telemetry for multiple vector environments is written into one seed-scoped stream with a global sequence. The previous Live Training page treated the complete seed buffer as one price, equity, PnL, baseline, drawdown, event, and replay path. That joined independent environments and could also join a completed episode to the environment's auto-reset episode.

This was diagnostic misrepresentation. No path was found from Live Training telemetry into model fitting, checkpoint selection, sealed evaluation, release authorization, Serving activation, simulated execution, or direct exchange routing.

## 2. Root cause

`LiveTrainingPage` previously derived all replay state directly from `telemetry.records`:

- active and latest record;
- first portfolio value and replay-period PnL;
- equity, baseline, and drawdown series;
- event list and market chart;
- cursor bounds, playback, jumps, and latest-position behavior.

`TrainingTelemetryRecord.environmentId` was available, but the browser did not select a single environment. The telemetry schema also has no persistent episode identifier, so the browser did not delimit an environment's auto-reset episodes.

## 3. Implemented boundary

PR #85 introduces one browser-side stream-selection boundary before presentation calculations:

1. Enumerate available `environmentId` values in stable order.
2. Select one environment explicitly; default to the environment of the latest received record.
3. Sort that environment's records by global telemetry sequence.
4. Select only its current episode.
5. Start a new episode after `episode_end`, `terminated`, or `truncated`.
6. Fail conservatively into a new episode when `environmentStep` or non-null `marketIndex` moves backwards.
7. Use the resulting `replayRecords` consistently for the chart, price, portfolio, PnL, baseline comparison, drawdown, event list, cursor, playback, jumps, and latest-record display.

The page now shows the selected seed, environment, and current-episode boundary. The stream/total counter keeps the selected replay length distinct from the complete received seed buffer.

## 4. Regression contracts

The frontend regression suite covers:

- interleaved vector environments remain independent;
- the selected environment exposes only its current episode;
- terminal records delimit the following auto-reset episode;
- counter rollback creates a conservative episode boundary;
- switching environments recomputes price and PnL from the selected episode's own initial portfolio value;
- the old cross-episode `+1,025.00 USDT` result is not displayed when the correct current-episode value is `+25.00 USDT`.

The initial RED head was `d3828f650cb6e269d93c8291f3293e68a4faa91b`. CI run `29953082159` failed at `Verify Studio frontend`, while both compatibility jobs and the training image passed. This demonstrated the regression before implementation.

## 5. Code-head verification

The implementation candidate was verified at exact head `3c5a94ce45a41ed6a21178b3b028b33d57013121` by CI run `29953930696`.

Results:

```text
Studio Vitest: 9 files passed, 28 tests passed
Studio TypeScript: passed
Studio production build: passed
Studio fixed viewport: passed
Workflow security: passed
Ruff and format: passed
Mypy: passed
Import architecture: passed
Dead-code report: passed
Recovery and structured Serving smoke: passed
Pytest: 1163 passed, 2 skipped, 11 warnings
Coverage: 83.16% (required 80%)
Critical branch-coverage ratchet: passed
CLI smoke: passed
Ubuntu compatibility: passed
Windows compatibility: passed
Training image build, identity capture, and non-root probe: passed
```

Evidence artifacts from run `29953930696`:

```text
studio-layout-diagnostics: 8543251935
sha256:1d4e8981412d589a04a18a1bd71312deb2ae73bc165694fbd48074485da30231

static-diagnostics: 8543259913
sha256:4bb6c579dfeb285601e795fe168b195be1eff3d02de65dc9805c516b03516051

architecture-diagnostics: 8543260540
sha256:eac6cabe61bdcd3ef394f6294b195fdac844285803a577fe89f302127954d15c

pytest-diagnostics: 8543294825
sha256:80f9b739b005b58cea5c1624d7613c546bd45b06853b9f86d262e2b542f6f2f6

training-image-evidence: 8543261770
sha256:cb673d900a461a5e6e61d0d1b26c7302b5ebbb319962b45f2d48adbed02a8be4
```

## 6. Architecture result

`AUD-STUDIO-001` is remediated for the maintained Live Training presentation path by environment and current-episode isolation.

The change does not add an authoritative episode ID to the persisted telemetry schema. The browser therefore uses explicit terminal state and conservative monotonic-counter rollback as its episode-boundary contract. A future schema revision may add a producer-issued episode identity, but that is not required to prevent the confirmed path concatenation.

The repository remains a research system. This remediation does not establish profitability, release eligibility, production readiness, authenticated account access, or direct exchange order routing. Production status remains `NO-GO`.
