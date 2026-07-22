# Live Training Stream Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Live Training from presenting interleaved vector environments or reset episodes as one continuous market, equity, PnL, baseline, or drawdown trajectory.

**Architecture:** Add a backward-compatible nullable episode identity to telemetry records, make the sampler allocate explicit stream-local episode IDs, and derive isolated frontend tracks. `LiveTrainingPage` and `MarketReplayChart` render one track only; legacy records are segmented conservatively from existing boundaries.

**Tech Stack:** Python 3.12, dataclasses, Pydantic, NumPy, pytest, React, TypeScript, Vitest, Testing Library, GitHub Actions.

## Global Constraints

- Keep JSONL schema version `training_telemetry_v1`; `episode_id` is additive and nullable for legacy records.
- New sampler records must always contain a non-negative episode ID.
- Keep seed-level files, global sequence semantics, index schema, stream generation, process locking, and cursor reset behavior unchanged.
- Never join explicit and legacy records into one track.
- All market, equity, PnL, baseline, drawdown, event, and cursor calculations must use one selected track.
- Production remains `NO-GO`; telemetry remains exploratory diagnostics only.
- No direct exchange routing, selection, sealed-evaluation, promotion, release, or profitability behavior changes.
- The PR remains Draft and is not merged without an explicit request.

---

## File responsibility map

- `trade_rl/telemetry/training.py`: backward-compatible record contract and JSON validation.
- `trade_rl/rl/training_telemetry.py`: stream-local episode allocation and cache reset at terminal boundaries.
- `trade_rl/studio/telemetry.py`: API response mapping for nullable episode identity.
- `studio/src/data/types.ts`: browser telemetry type contract.
- `studio/src/live/telemetryGuards.ts`: strict runtime validation for episode identity.
- `studio/src/live/telemetryTracks.ts`: pure grouping and selection logic for explicit and legacy tracks.
- `studio/src/live/MarketReplayChart.tsx`: defensive single-track rendering.
- `studio/src/pages/LiveTrainingPage.tsx`: environment/episode selection and selected-track metric derivation.
- `tests/telemetry/test_training.py`: record compatibility tests.
- `tests/integrations/test_training_telemetry.py`: sampler lifecycle tests.
- `tests/studio/test_telemetry_api.py`: Studio response compatibility tests.
- `studio/src/live/telemetryTracks.test.ts`: track derivation tests.
- `studio/src/live/useTrainingTelemetry.test.tsx`: fixture compatibility only if required by strict TypeScript construction.
- `studio/src/pages/LiveTrainingPage.test.tsx`: selector and metric isolation tests.
- `studio/src/live/MarketReplayChart.test.tsx`: mixed-input defense tests.
- `docs/verification/2026-07-22-live-training-stream-isolation.md`: exact-head TDD and CI evidence.

---

### Task 1: Add the backward-compatible episode record contract

**Files:**
- Modify: `tests/telemetry/test_training.py`
- Modify: `trade_rl/telemetry/training.py`

**Interfaces:**
- Produces: `TrainingTelemetryRecord.episode_id: int | None`
- Produces JSON key: `episode_id`
- Legacy JSON without the key resolves to `None`.

- [ ] **Step 1: Write failing record tests**

Add explicit, legacy, and invalid-ID cases:

```python
def test_record_episode_id_round_trip_and_legacy_compatibility() -> None:
    explicit = TrainingTelemetryRecord(**{**record(1).__dict__, "episode_id": 9})
    assert TrainingTelemetryRecord.from_json_dict(explicit.to_json_dict()).episode_id == 9

    legacy = record(2).to_json_dict()
    legacy.pop("episode_id", None)
    assert TrainingTelemetryRecord.from_json_dict(legacy).episode_id is None


@pytest.mark.parametrize("value", (-1, True))
def test_record_rejects_invalid_episode_id(value: object) -> None:
    payload = record(1).to_json_dict()
    payload["episode_id"] = value
    with pytest.raises(ValueError, match="episode_id"):
        TrainingTelemetryRecord.from_json_dict(payload)
```

- [ ] **Step 2: Run the focused RED test**

Run:

```bash
pytest tests/telemetry/test_training.py -k episode_id -q
```

Expected: FAIL because the dataclass and parser do not expose `episode_id`.

- [ ] **Step 3: Implement the nullable field**

In `TrainingTelemetryRecord`, append the defaulted field before `schema_version`:

```python
episode_id: int | None = None
schema_version: str = TELEMETRY_SCHEMA_VERSION
```

Validate it in `__post_init__`:

```python
if self.episode_id is not None and (
    isinstance(self.episode_id, bool)
    or not isinstance(self.episode_id, int)
    or self.episode_id < 0
):
    raise ValueError("episode_id is invalid")
```

Emit it in `to_json_dict()` next to environment identity:

```python
"environment_id": self.environment_id,
"episode_id": self.episode_id,
```

Parse it with `_optional_int(raw, "episode_id")`.

- [ ] **Step 4: Run focused and telemetry regression tests**

Run:

```bash
pytest tests/telemetry/test_training.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/telemetry/training.py tests/telemetry/test_training.py
git commit -m "feat: add telemetry episode identity"
```

---

### Task 2: Allocate episode IDs in the sampler and clear boundary caches

**Files:**
- Modify: `tests/integrations/test_training_telemetry.py`
- Modify: `trade_rl/rl/training_telemetry.py`

**Interfaces:**
- Consumes: `TrainingTelemetryRecord.episode_id`
- Produces: every newly emitted sampler record has `episode_id: int`
- Maintains: `_episode_ids: dict[int, int]`, `_next_episode_id: int`

- [ ] **Step 1: Write failing sampler lifecycle tests**

Add a test that emits two environments, terminates environment zero, and emits its next episode:

```python
def test_sampler_assigns_distinct_episode_ids_and_rotates_after_done(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    sampler = TrainingTelemetrySampler(path, seed=3, sample_every=1)

    assert sampler.consume(
        global_step=2,
        actions=np.asarray([[0.1], [0.2]], dtype=np.float32),
        rewards=np.asarray([0.0, 0.0], dtype=np.float32),
        dones=np.asarray([False, False]),
        infos=(info(1), info(1)),
    ) == 2
    assert sampler.consume(
        global_step=4,
        actions=np.asarray([[0.0], [0.2]], dtype=np.float32),
        rewards=np.asarray([0.0, 0.0], dtype=np.float32),
        dones=np.asarray([True, False]),
        infos=(info(2, terminated=True), info(2)),
    ) == 2
    assert sampler.consume(
        global_step=6,
        actions=np.asarray([[0.3], [0.2]], dtype=np.float32),
        rewards=np.asarray([0.0, 0.0], dtype=np.float32),
        dones=np.asarray([False, False]),
        infos=(info(0), info(3)),
    ) == 2
    sampler.close()

    items = read_training_telemetry(path, limit=20).items
    env_zero = [item for item in items if item.environment_id == 0]
    env_one = [item for item in items if item.environment_id == 1]
    assert env_zero[0].episode_id == env_zero[1].episode_id
    assert env_zero[2].episode_id not in (None, env_zero[1].episode_id)
    assert len({item.episode_id for item in env_one}) == 1
    assert env_zero[0].episode_id != env_one[0].episode_id
```

Add a boundary-cache test using records without exact OHLC on the new episode and assert that the first next-episode open is not inherited from the previous close and `weights_before` does not inherit prior weights.

- [ ] **Step 2: Run the sampler RED tests**

Run:

```bash
pytest tests/integrations/test_training_telemetry.py -k "episode or boundary" -q
```

Expected: FAIL because records have no sampler-provided episode ID and caches survive done.

- [ ] **Step 3: Implement allocation and cleanup**

In `__init__`:

```python
self._episode_ids: dict[int, int] = {}
self._next_episode_id = self.sequence + 1
```

Add:

```python
def _episode_id(self, environment_id: int) -> int:
    current = self._episode_ids.get(environment_id)
    if current is not None:
        return current
    assigned = self._next_episode_id
    self._next_episode_id += 1
    self._episode_ids[environment_id] = assigned
    return assigned


def _finish_episode(self, environment_id: int) -> None:
    self._episode_ids.pop(environment_id, None)
    self._previous_weights.pop(environment_id, None)
    self._previous_close.pop(environment_id, None)
```

Resolve `episode_id` before constructing the record and pass it to `TrainingTelemetryRecord`. After a successful append, call `_finish_episode(environment_id)` when `done`, `hybrid_terminated`, or `TimeLimit.truncated` is true.

Do not clear state before the terminal record is built; the terminal transition must retain its own prior state.

- [ ] **Step 4: Run sampler and telemetry regressions**

Run:

```bash
pytest tests/integrations/test_training_telemetry.py tests/telemetry/test_training.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/training_telemetry.py tests/integrations/test_training_telemetry.py
git commit -m "fix: isolate telemetry episode lifecycle"
```

---

### Task 3: Expose episode identity through Studio and browser contracts

**Files:**
- Modify: `trade_rl/studio/telemetry.py`
- Modify: `tests/studio/test_telemetry_api.py`
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/live/telemetryGuards.ts`
- Modify any existing telemetry object literals reported by TypeScript or Vitest.

**Interfaces:**
- Produces Python API field: `episode_id: int | None`
- Produces browser field: `episodeId: number | null`

- [ ] **Step 1: Write failing Studio API tests**

Create one new record with `episode_id=5`, one legacy JSON line without `episode_id`, and assert response serialization returns `episodeId: 5` and `episodeId: null` respectively while seed, generation, and cursor fields remain unchanged.

- [ ] **Step 2: Run the Studio RED tests**

Run:

```bash
pytest tests/studio/test_telemetry_api.py -k episode -q
```

Expected: FAIL because `TelemetryRecordResponse` does not expose the field.

- [ ] **Step 3: Implement Python and TypeScript contracts**

Add to `TelemetryRecordResponse`:

```python
episode_id: int | None = Field(default=None, ge=0)
```

Add to `TrainingTelemetryRecord` in `studio/src/data/types.ts`:

```typescript
episodeId: number | null
```

Add to `isTrainingTelemetryRecord`:

```typescript
&& (value.episodeId === null || isNonNegativeInteger(value.episodeId))
```

Update strict test fixtures so every current record explicitly uses either a non-negative integer or `null`.

- [ ] **Step 4: Run contract regressions**

Run:

```bash
pytest tests/studio/test_telemetry_api.py tests/telemetry/test_training.py -q
npm --prefix studio test -- --run
npm --prefix studio run typecheck
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/studio/telemetry.py tests/studio/test_telemetry_api.py studio/src/data/types.ts studio/src/live/telemetryGuards.ts studio/src
git commit -m "feat: expose telemetry episode identity"
```

---

### Task 4: Derive explicit and legacy telemetry tracks

**Files:**
- Create: `studio/src/live/telemetryTracks.ts`
- Create: `studio/src/live/telemetryTracks.test.ts`

**Interfaces:**
- Produces:

```typescript
export interface TelemetryTrack {
  key: string
  environmentId: number
  episodeId: number | null
  legacyOrdinal: number | null
  records: TrainingTelemetryRecord[]
  firstSequence: number
  lastSequence: number
  ended: boolean
  inferred: boolean
}

export function deriveTelemetryTracks(records: TrainingTelemetryRecord[]): TelemetryTrack[]
export function selectTelemetryTrack(
  records: TrainingTelemetryRecord[],
  cursorSequence: number | null,
): TelemetryTrack | null
```

- [ ] **Step 1: Write failing utility tests**

Cover:

```typescript
it('keeps interleaved explicit environments and episodes independent', () => { /* env0/ep1, env1/ep2, env0/ep1, env0/ep3 */ })
it('starts a legacy track after episode_end', () => { /* same env, null episode IDs */ })
it('starts a legacy track after environment step rollback', () => { /* 8 -> 0 */ })
it('starts a legacy track after market index rollback', () => { /* 500 -> 100 */ })
it('never merges explicit and legacy records', () => { /* same environment */ })
it('selects the cursor track or newest track', () => { /* mixed input */ })
```

Use a local complete record factory with `episodeId` override support.

- [ ] **Step 2: Run utility RED tests**

Run:

```bash
npm --prefix studio test -- --run src/live/telemetryTracks.test.ts
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement pure track derivation**

Sort a copy by sequence. Group explicit records by `explicit:${environmentId}:${episodeId}`. Maintain per-environment legacy state containing ordinal, previous record, and active key. Start a new legacy key when the previous event ended, environment step decreases, or market index decreases.

Finalize tracks sorted by `lastSequence`, and set `ended` when the track's final record is terminal, truncated, or `episode_end`.

`selectTelemetryTrack` must choose the track containing `cursorSequence`; otherwise return the track with the greatest `lastSequence`.

- [ ] **Step 4: Run utility tests and typecheck**

Run:

```bash
npm --prefix studio test -- --run src/live/telemetryTracks.test.ts
npm --prefix studio run typecheck
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/src/live/telemetryTracks.ts studio/src/live/telemetryTracks.test.ts
git commit -m "feat: derive isolated telemetry tracks"
```

---

### Task 5: Make the chart fail closed to one track

**Files:**
- Modify: `studio/src/live/MarketReplayChart.tsx`
- Create: `studio/src/live/MarketReplayChart.test.tsx`

**Interfaces:**
- Consumes: `selectTelemetryTrack(records, cursorSequence)`
- Preserves props: `records`, `cursorSequence`, `compressed`

- [ ] **Step 1: Write a failing mixed-input chart test**

Render interleaved records with distinct symbols/prices and a cursor on environment zero. Assert the SVG accessible label and rendered step labels reflect only environment zero's track, and no candle group belonging only to environment one is present. Add `data-sequence` and `data-track-key` attributes to candle groups for deterministic assertions.

- [ ] **Step 2: Run the chart RED test**

Run:

```bash
npm --prefix studio test -- --run src/live/MarketReplayChart.test.tsx
```

Expected: FAIL because the chart renders every close-bearing record.

- [ ] **Step 3: Implement defensive selection**

At component entry:

```typescript
const track = selectTelemetryTrack(records, cursorSequence)
const trackRecords = track?.records ?? []
const selected = (compressed
  ? trackRecords.filter((record) => record.eventType !== 'rollout')
  : trackRecords)
  .filter((record) => record.close !== null)
  .slice(-96)
```

Use the selected track for labels and attach deterministic data attributes to each candle group.

- [ ] **Step 4: Run chart and utility tests**

Run:

```bash
npm --prefix studio test -- --run src/live/MarketReplayChart.test.tsx src/live/telemetryTracks.test.ts
npm --prefix studio run typecheck
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/src/live/MarketReplayChart.tsx studio/src/live/MarketReplayChart.test.tsx
git commit -m "fix: constrain market replay to one track"
```

---

### Task 6: Bind Live Training controls and metrics to the selected track

**Files:**
- Modify: `studio/src/pages/LiveTrainingPage.tsx`
- Modify: `studio/src/pages/LiveTrainingPage.test.tsx`
- Modify: `studio/src/liveTraining.css` only for selector layout needed by the existing fixed viewport.

**Interfaces:**
- Consumes: `deriveTelemetryTracks(records)`
- Maintains state: `selectedEnvironmentId: number | null`, `selectedTrackKey: string | null`
- Produces: one `selectedRecords` array used by every exploratory trajectory calculation.

- [ ] **Step 1: Write failing page isolation tests**

Use two explicit environments with materially different prices and portfolio values plus two episodes for one environment. Assert:

1. Environment and Episode selectors are present.
2. Initial selection chooses the newest track.
3. Changing Environment chooses that environment's newest episode.
4. Changing Episode changes the displayed current price and replay PnL.
5. Replay PnL equals active portfolio value minus the selected episode's first portfolio value, not the first value in the seed buffer.
6. Recent events exclude other environments/episodes.
7. The buffered-count text distinguishes selected records from total records.

- [ ] **Step 2: Run page RED tests**

Run:

```bash
npm --prefix studio test -- --run src/pages/LiveTrainingPage.test.tsx
```

Expected: FAIL because the page has no environment/episode selectors and uses all seed records.

- [ ] **Step 3: Implement selected-track state and derivations**

Add:

```typescript
const tracks = useMemo(() => deriveTelemetryTracks(telemetry.records), [telemetry.records])
const environments = useMemo(
  () => [...new Set(tracks.map((track) => track.environmentId))].sort((a, b) => a - b),
  [tracks],
)
```

Retain valid selection; otherwise choose the newest track. On environment changes, select that environment's newest track. Derive:

```typescript
const selectedTrack = tracks.find((track) => track.key === selectedTrackKey) ?? null
const selectedRecords = selectedTrack?.records ?? []
```

Replace every trajectory use of `telemetry.records` with `selectedRecords`, including cursor length, active/latest record, first portfolio value, event filtering, sparkline arrays, jump bounds, reset/latest buttons, and `MarketReplayChart` props.

Add Environment and Episode selectors near Seed. Label explicit tracks as `Episode {episodeId}` and legacy tracks as `Legacy inferred {legacyOrdinal}`. Show `selectedRecords.length / telemetry.records.length records`.

- [ ] **Step 4: Run full frontend verification**

Run:

```bash
npm --prefix studio test -- --run
npm --prefix studio run typecheck
npm --prefix studio run build
```

Expected: PASS, including fixed-viewport tests.

- [ ] **Step 5: Commit**

```bash
git add studio/src/pages/LiveTrainingPage.tsx studio/src/pages/LiveTrainingPage.test.tsx studio/src/liveTraining.css
git commit -m "fix: isolate live training trajectory selection"
```

---

### Task 7: Run repository-wide verification and record evidence

**Files:**
- Modify: `pyproject.toml` only if measured critical coverage can be raised without lowering another threshold.
- Create: `docs/verification/2026-07-22-live-training-stream-isolation.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces exact-head evidence and final PR summary.

- [ ] **Step 1: Run Python static and focused checks**

```bash
ruff check .
ruff format --check .
mypy .
pytest tests/telemetry/test_training.py tests/integrations/test_training_telemetry.py tests/studio/test_telemetry_api.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full Python suite and coverage checker**

```bash
pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json -q
python scripts/check_critical_coverage.py coverage.json
```

Expected: PASS with no existing threshold reduction.

If the measured `trade_rl/telemetry/training.py` or another already-ratcheted critical file supports a higher floor, raise only to the one-decimal value not exceeding the measured branch percentage, then rerun both commands.

- [ ] **Step 3: Run complete frontend checks**

```bash
npm --prefix studio test -- --run
npm --prefix studio run typecheck
npm --prefix studio run build
```

Expected: PASS.

- [ ] **Step 4: Run Linux and Windows focused telemetry tests in GitHub Actions**

Execute the record, sampler, Studio API, track, chart, and page tests on both supported operating systems. Preserve artifact IDs and SHA-256 digests.

- [ ] **Step 5: Run final exact-head workflows**

Require success for:

- standard CI;
- Ubuntu compatibility;
- Windows compatibility;
- training image and packaged non-root probe;
- PostgreSQL Catalog Compose, readiness, migrations, tests, and shutdown.

- [ ] **Step 6: Write verification evidence**

Record:

- RED heads/runs/artifacts/digests;
- GREEN focused heads/runs;
- final head;
- full test count and warnings;
- total and branch coverage;
- critical coverage result;
- frontend test/typecheck/build status;
- Linux and Windows results;
- PostgreSQL result;
- unchanged schema and safety boundaries;
- disposition of any unrelated nondeterministic failure, with an identical-head rerun rather than a hidden tolerance change.

- [ ] **Step 7: Commit evidence**

```bash
git add pyproject.toml docs/verification/2026-07-22-live-training-stream-isolation.md
git commit -m "docs: verify live training stream isolation"
```

- [ ] **Step 8: Update the Draft PR**

The PR body must identify the dependency on PR #83, enumerate the effective branch-specific files, include exact-head evidence, retain `NO-GO`, and state that it is unmerged.
