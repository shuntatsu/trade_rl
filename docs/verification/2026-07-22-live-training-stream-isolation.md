# Live Training Stream Isolation Verification

## Scope

This remediation addresses `AUD-STUDIO-001`: a seed-level telemetry stream can interleave multiple vector environments and multiple auto-reset episodes, while Live Training previously displayed the entire buffer as one continuous market, equity, PnL, baseline, drawdown, and event trajectory.

The implementation remains an exploratory diagnostics feature. It does not change observations, actions, rewards, transitions, training selection, sealed evaluation, promotion, release, serving, or exchange execution.

## Dependency

The implementation branch starts from PR #83 exact head:

```text
b2ac0df43c2b254653bdcd8089d23f37a28c70d9
```

## Implemented boundary

### Telemetry record compatibility

`TrainingTelemetryRecord` now has an additive nullable field:

```text
episode_id: int | None
```

The JSONL schema identifier remains `training_telemetry_v1` because no existing field changed meaning, legacy records without the property remain readable, and older readers can ignore the additive property.

Strict parsing accepts:

- a non-negative integer for new records;
- `None` when normalized from a legacy record that omitted the property.

Strict parsing rejects:

- negative integers;
- booleans;
- other invalid representations.

### Sampler lifecycle

`TrainingTelemetrySampler` now:

- allocates a stream-local episode ID independently for every vector environment;
- preserves the ID until the environment terminates or truncates;
- allocates a new ID on the first record after reset;
- starts allocation after the existing stream's last sequence to avoid collisions on resume;
- clears cached previous weights and previous close only after the terminal record is appended.

The terminal record therefore retains its own prior state, while the first record of the next episode cannot inherit a synthetic candle open or weight delta from the previous episode.

### Studio and frontend contracts

Studio exposes nullable `episodeId`. Browser runtime guards require the field to be either a non-negative integer or `null`; a missing browser response property fails strict validation because the Studio API normalizes legacy records to `null`.

### Explicit and legacy tracks

`deriveTelemetryTracks()` creates independent trajectories.

Explicit records are grouped by:

```text
(environmentId, episodeId)
```

Legacy records are grouped independently per environment and split conservatively after:

- `episode_end`, termination, or truncation;
- environment-step rollback;
- market-index rollback.

Explicit records and legacy records are never placed in one track.

### Live Training selection

Live Training now presents Environment and Episode selectors. The selected track alone supplies:

- current and latest record;
- market replay candles;
- replay cursor bounds;
- initial portfolio value;
- replay PnL;
- baseline delta;
- equity series;
- baseline series;
- drawdown series;
- recent events;
- jump and latest controls.

The UI displays selected-record count separately from total seed-buffer count.

### Chart defense

`MarketReplayChart` independently derives tracks from its input and renders only:

- the track containing the cursor sequence; or
- the newest track when there is no cursor match.

This prevents a future caller from restoring false continuity by passing a mixed record array.

## TDD evidence

### Python record and sampler RED

Expected failing contracts were committed before production implementation.

- RED workflow run: `29922737924`
- artifact: `8530429919`
- digest: `sha256:7a04ec087bf6b1a657cdbd1c36d539b6bfef2c65a8fed7a80a1f80e647afbf61`

The failures reproduced:

- missing `episode_id` record contract;
- no distinct episode identity for vector environments;
- no rotation after a terminal transition;
- visual fallback state bridging into the next episode.

### Frontend track RED

The standard CI run on the test-only head stopped in Studio verification because `telemetryTracks` did not yet exist.

- test-only head: `5552d73cb17e2911a6a4e3c14b6c8f2a40c6d03b`
- standard CI failure was expected before implementation.

### Core GREEN

The reviewed core patch was applied in a disposable Actions workspace. It was pushed only after all checks succeeded.

- workflow run: `29923782136`
- result: success
- Ruff: success
- Mypy: success
- record and sampler focused tests: success
- existing telemetry and Studio regressions: success
- track Vitest contracts: success
- TypeScript typecheck: success

Temporary patch and workflow files were removed after the verified implementation commit.

### Chart and page RED

Chart and page contracts were then added before changing those components.

- RED head: `65f577f147f584e204f4ca5a2e1a471f52574b46`
- standard CI run: `29924168247`
- Ubuntu compatibility: success
- Windows compatibility: success
- Core frontend verification: expected failure
- diagnostics artifact: `8531215809`
- digest: `sha256:48f74bd61b482f30ec10b2127def5a5ea87b17f1bd102df927665b26ac42422f`

The RED contracts required:

- cursor-track-only chart rendering;
- Environment and Episode selectors;
- newest-track default selection;
- environment changes selecting that environment's newest episode;
- episode-specific current price and PnL;
- episode-specific event filtering;
- selected/total record counts.

### UI GREEN

The reviewed UI patch was applied in a disposable Actions workspace. It was pushed only after all checks succeeded.

- GREEN head before bot push: `d891672b067bd64e228df8f64319d67bff45e4bf`
- workflow run: `29925071218`
- focused UI artifact: `8531581346`
- focused digest: `sha256:752b9d0bdfc41aa84686369be92a763fb4a0e55b2c34e483034cbca3781cda6a`
- full-suite artifact: `8531584397`
- full-suite digest: `sha256:7fed2ad02e54cef06d89ac1533362df49d667eadae413033fd13d09b1513acf4`
- focused track/chart/page tests: success
- complete Vitest suite: success
- TypeScript typecheck: success
- production build: success

One initial page assertion queried the selected price before the cursor-reset effect settled. The production selection was already correct. The test was stabilized by waiting for the same final selector, price, and PnL state; no product requirement was weakened.

## API and guard compatibility evidence

Additional contracts verify:

- an explicit JSONL record becomes `episodeId: 5` through the Studio API;
- a legacy JSONL record without `episode_id` becomes `episodeId: null`;
- frontend guards accept explicit and normalized legacy identities;
- frontend guards reject missing, negative, and boolean identities.

## Cleanup-head repository verification

Implementation cleanup head:

```text
0884bca1ee99f50f39fe236554bcfe7a5f311199
```

Workflows:

- CI run `29925411744`: success
- PostgreSQL Catalog run `29925411010`: success

Results:

- `1204 passed, 2 skipped, 11 warnings`
- total coverage: `83.47%`
- total branch coverage: `70.39%`
- Ruff: success
- format check: success
- Mypy: success
- import architecture: success
- dead-code report: success
- recovery and structured serving smoke: success
- critical branch coverage: success
- CLI smoke: success
- fixed viewport verification: success
- Ubuntu compatibility: success
- Windows compatibility: success
- training image and non-root probe: success
- PostgreSQL Compose, readiness, migration, tests, and shutdown: success

Pytest artifact:

- ID: `8531806043`
- digest: `sha256:8375534fa35e4de21211fce0f5bec76bc21d94cb03062163fdf8f7159bfc5c4d`

## Coverage ratchet

Measured branch coverage after the new contracts:

```text
trade_rl/rl/training_telemetry.py: 55 / 70 = 78.57%
trade_rl/telemetry/indexed_training.py: 76 / 110 = 69.09%
```

The new sampler lifecycle file is now protected at `78.5%`. The existing indexed telemetry threshold remains `69.0%`.

An initial attempted indexed threshold of `72.7%` failed because `num_partial_branches` was incorrectly treated as the missing-branch count. The checker uses `covered_branches / num_branches`; the configuration was corrected without changing tests or production code.

Corrected ratchet head:

```text
770b03ac65b8f8275e3342468c3f6b2dc6640530
```

Workflows:

- CI run `29926395873`: success
- PostgreSQL Catalog run `29926394283`: success
- critical coverage: success
- sampler lifecycle: `78.57% >= 78.5%`
- indexed telemetry: `69.09% >= 69.0%`

Pytest artifact:

- ID: `8532221818`
- digest: `sha256:3ce17fb18480f545b9f2f6266df87a8a66dcb770837ee34058f7e47a072c1ae1`

## Linux and Windows focused verification

Focused workflow run:

```text
29926758198
```

Both Ubuntu and Windows executed:

- record episode identity tests;
- sampler lifecycle tests;
- Studio explicit/legacy API test;
- frontend guard tests;
- explicit and legacy track tests;
- chart cursor-track isolation test;
- Live Training environment/episode isolation test.

Ubuntu:

- result: success
- artifact: `8532304157`
- digest: `sha256:cf20e33e77261af660981bc0b7fd52d106aeb8d25148ce2e2cb7a29d0a17550c`

Windows:

- result: success
- artifact: `8532309841`
- digest: `sha256:8fc2069d08772b30a4061564bd659c282e3678e60f1eccd5cd36dfd0d2b310c6`

The temporary cross-platform workflow was removed after artifact capture.

## Effective branch-specific files

Relative to PR #83 head, the remediation changes only:

- telemetry record and sampler contracts;
- Studio telemetry response mapping;
- frontend telemetry type and guard;
- pure track derivation;
- chart and Live Training selection;
- focused Python and frontend tests;
- coverage ratchet;
- design, plan, and verification documentation.

No temporary workflow or patch script remains in the effective final diff.

## Safety boundary

- Production remains `NO-GO`.
- No direct exchange routing was added.
- Telemetry remains exploratory diagnostics only.
- `training_telemetry_v1` remains the JSONL schema identifier.
- Legacy evidence is read and conservatively segmented; it is not rewritten.
- Seed-level storage, global sequence, sidecar index, stream generation, and process locking are unchanged.
- No training selection, sealed evaluation, promotion, release, serving, profitability, or execution contract changed.
- PR #84 remains Draft and unmerged.
