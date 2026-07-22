# Live Training Stream Isolation Design

## Status

Approved continuation of the post-merge architecture remediation roadmap. This design addresses `AUD-STUDIO-001` only.

## Problem

A seed-scoped telemetry JSONL stream currently interleaves records from every vector environment under one global sequence. The stream carries `environment_id`, but it does not carry episode identity. Studio buffers the full seed stream and calculates market replay, equity, PnL, baseline comparison, drawdown, recent events, and replay cursor state across every buffered record.

That can make two independent environments, or an environment before and after auto-reset, appear to be one continuous market and portfolio trajectory. The defect is limited to exploratory diagnostics; telemetry remains excluded from model selection, sealed evaluation, release, and execution evidence.

## Goals

1. Every newly emitted telemetry record identifies one episode explicitly.
2. Studio displays exactly one environment/episode trajectory at a time.
3. Market, equity, PnL, baseline, drawdown, events, and cursor state use the same selected trajectory.
4. Existing `training_telemetry_v1` artifacts remain readable.
5. Legacy records without episode identity are split conservatively and are never joined across a detected reset boundary.
6. The chart remains safe even if a future caller accidentally supplies mixed records.
7. Seed-level JSONL storage, global sequence semantics, stream generation, and process-lock behavior remain unchanged.

## Non-goals

- Splitting telemetry into one file per environment or episode.
- Using telemetry as selection, promotion, sealed-evaluation, profitability, or release evidence.
- Changing training actions, rewards, observations, environment transitions, or execution behavior.
- Repairing malformed or truncated telemetry.
- Reconstructing an exact historical episode identity when a legacy artifact does not contain enough information.

## Approaches considered

### A. Frontend-only segmentation

Use `environment_id`, `episode_end`, and local step rollback in the browser without changing records.

Advantages:

- Smallest backend change.
- Existing files need no compatibility handling.

Rejected because:

- A page can start mid-episode, so identity remains contextual rather than explicit.
- New records would continue to lack a durable episode boundary.
- Other consumers could still misinterpret the same stream.

### B. Add explicit episode identity and isolate displayed tracks

Keep the seed-level JSONL and global sequence, add a backward-compatible nullable `episode_id`, emit it for all new records, and derive isolated frontend tracks.

Advantages:

- Correct durable identity for new telemetry.
- Existing artifacts remain readable.
- No additional files, discovery paths, or generation domains.
- Small enough for an independent remediation PR.

Selected.

### C. One telemetry file per environment/episode

Make file identity enforce track identity.

Rejected because:

- It multiplies writer, lock, index, discovery, status, generation, and retention responsibilities.
- It is unnecessary to correct the display defect.

## Record contract

`TrainingTelemetryRecord` gains:

```text
episode_id: int | None
```

The field is additive and defaults to `None`, so existing positional construction and legacy `training_telemetry_v1` JSON remain readable. New sampler records always provide a non-negative integer.

The schema version remains `training_telemetry_v1` because:

- existing readers ignore unknown JSON properties;
- the new reader accepts the field as optional;
- no existing field changes meaning or representation;
- the file remains a single append-only stream with the same sequence contract.

Studio exposes the field as nullable `episodeId`. Frontend guards accept `null` for legacy records and require a non-negative integer otherwise.

## Episode identity allocation

`TrainingTelemetrySampler` owns a current episode ID per vector environment and a stream-local next ID counter.

- The next ID starts at `training_telemetry_status(path).last_sequence + 1`.
- When an environment is first observed without an active episode, the sampler assigns the next ID and increments the counter.
- Every emitted record for that environment uses the active ID.
- After a terminal or truncated transition is emitted, the active ID is removed.
- The next transition for that environment starts a new episode with a new ID.

Using a stream-local monotonic allocator avoids collisions when training resumes, even when model `global_step` restarts. IDs do not claim semantic equivalence to environment-internal episode counters; they identify telemetry trajectory segments.

On episode end, sampler caches that can bridge visual state are cleared for that environment:

- previous weights;
- previous close.

This prevents fallback OHLC or weight deltas from connecting the first record of the next episode to the terminal state of the prior episode.

## Track derivation

A new frontend utility derives `TelemetryTrack` objects from sequence-sorted records.

A track contains:

- stable key;
- environment ID;
- explicit episode ID or `null` for legacy data;
- inferred legacy ordinal when needed;
- ordered records;
- first and last sequence;
- ended state;
- whether identity was inferred.

### Explicit records

Records with `episodeId !== null` are grouped by `(environmentId, episodeId)`. Interleaving from other environments does not split or join the track.

### Legacy records

Records with `episodeId === null` are grouped independently per environment. A new inferred segment begins when any of these are true:

1. no active legacy segment exists for the environment;
2. the previous record for the environment ended an episode;
3. `environmentStep` decreases;
4. both market indices are present and `marketIndex` decreases.

The buffer start is treated as a conservative segment start. This may split one real legacy episode, but it cannot falsely connect two detected episodes.

Explicit and legacy records never share a track.

## Studio selection behavior

`LiveTrainingPage` derives tracks from the buffered seed records and presents Environment and Episode selectors.

Selection rules:

1. retain the selected track while it remains available;
2. otherwise select the newest track by last sequence;
3. when the environment changes, select that environment's newest track;
4. label inferred legacy tracks clearly;
5. show selected-record count separately from total buffered count.

The following values are calculated only from the selected track:

- active and latest record;
- first portfolio value;
- replay PnL;
- baseline delta;
- equity sparkline;
- baseline sparkline;
- drawdown sparkline;
- recent events;
- replay cursor and transport bounds;
- market chart records.

Changing environment or episode resets the replay cursor to the selected track's latest record in live mode and to its first record in buffered mode.

## Chart defense

`MarketReplayChart` also derives tracks internally. If it receives mixed records, it selects:

- the track containing `cursorSequence`, when present;
- otherwise the newest track.

Only that track is rendered. This defense prevents a future caller from reintroducing false continuity even if `LiveTrainingPage` filtering regresses.

## API and compatibility

The Studio telemetry endpoint continues returning the seed stream and global sequence cursor. It adds only `episodeId` to each item.

No new query filter is required in this PR. The browser's bounded 2,048-record buffer remains the source for interactive track selection. This scope corrects misleading continuity without adding an index-level environment/episode catalog.

Legacy artifacts:

- parse with `episode_id=None`;
- are segmented conservatively in Studio;
- never become release or evaluation evidence;
- are not rewritten.

New artifacts:

- carry explicit episode identity;
- remain readable by old consumers that ignore unknown fields.

## Failure handling

- Invalid negative or boolean episode IDs fail strict parsing.
- Missing episode IDs remain valid legacy records.
- Sampler writer failures retain existing fail-closed disabling behavior.
- If track derivation receives unsorted input, it sorts by sequence before grouping.
- Duplicate sequence handling remains owned by the existing telemetry reader and hook merge logic.
- An empty selected track renders the existing data-waiting state rather than falling back to mixed records.

## Test strategy

### Python record tests

- explicit episode ID JSON round-trip;
- missing legacy episode ID parses as `None`;
- negative and boolean IDs fail.

### Sampler tests

- two vector environments receive distinct episode IDs in one consume call;
- records from one environment retain the same ID until done;
- the next transition after done receives a new ID;
- previous close and weights do not bridge the episode boundary;
- sequence continuation after resume remains unchanged.

### Studio API tests

- `episodeId` is present for new records;
- legacy records expose `episodeId: null`;
- seed identity and generation/reset contracts remain unchanged.

### Frontend utility tests

- interleaved explicit environments form independent tracks;
- explicit episodes from one environment stay independent;
- legacy `episode_end` creates a boundary;
- legacy environment-step rollback creates a boundary;
- legacy market-index rollback creates a boundary;
- explicit and legacy records never merge.

### Component tests

- Environment and Episode selectors choose one track;
- PnL and metric arrays reset to the selected track's first portfolio value;
- recent events contain only the selected track;
- chart rendering ignores mixed records outside the cursor track;
- existing seed, generation reset, connection, replay transport, and viewport tests remain green.

## Verification

The final exact head must pass:

```text
ruff check .
ruff format --check .
mypy .
pytest -q
python scripts/check_critical_coverage.py coverage.json
npm --prefix studio test -- --run
npm --prefix studio run typecheck
npm --prefix studio run build
```

GitHub Actions must pass the standard exact-head CI, Ubuntu and Windows compatibility, training-image probe, and PostgreSQL Catalog workflow.

## Safety boundary

- Production remains `NO-GO`.
- No direct exchange routing is added.
- Telemetry remains exploratory diagnostics only.
- No selected-final, promotion, release, or sealed-evaluation contract changes.
- The PR remains Draft and is not merged without an explicit request.
