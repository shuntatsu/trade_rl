# Generation-Bound Telemetry Polling Design

## Context

Indexed training telemetry is now process-safe, but the polling cursor is still only a sequence number. The sidecar index detects file replacement or truncation and rebuilds itself, while Studio clients continue sending the old `after_sequence` value. If a new stream at the same path restarts sequence numbering, the client can receive empty pages indefinitely or mix records from different stream generations.

Polling also performs unnecessary durable index writes when the stream has not changed, and `read_indexed_training_telemetry()` keeps the exclusive process lock while parsing the returned page. Long page parsing can therefore delay training appends.

## Goals

1. Bind every status and event page to one stable opaque stream generation.
2. Detect replacement, truncation, or index rebuild without relying on sequence rollback heuristics.
3. Never return records from a new generation against an old generation cursor.
4. Reset Studio buffers explicitly before records from a replacement stream are accepted.
5. Avoid rewriting and `fsync`-ing an unchanged sidecar index.
6. Hold the process lock only while refreshing index metadata and capturing a consistent file snapshot.
7. Preserve training telemetry record schema v1, existing sequence semantics within one generation, and the current 2,000-record limit.

## Non-goals

- Replacing sequence pagination with byte cursors.
- Persisting browser cursor state across page reloads.
- Repairing malformed, truncated, or manually edited telemetry files.
- Changing telemetry sampling frequency, training callbacks, job ownership, or artifact publication.
- Supporting direct exchange execution or changing production `NO-GO` status.

## Chosen approach

Use a persisted opaque generation identifier in the telemetry sidecar index.

The generation is created whenever the index must be rebuilt from stream identity rather than incrementally extended. It remains stable while complete records are appended to the same stream. Index deletion, stream replacement, inode change, or truncation creates a new generation. A conservative generation change after index loss is acceptable because it fails closed and forces a clean client replay.

A sequence-only heuristic was rejected because a replacement stream may have the same or a larger last sequence. A complete byte-cursor API replacement was rejected because it would create unnecessary migration scope for the current Studio API.

## Sidecar index contract

The internal index schema moves from `training_telemetry_index_v1` to `training_telemetry_index_v2` and adds:

- `generation`: canonical lower-case UUID string.

Loading an old or malformed index returns `None`, causing a full rebuild and a new generation. The telemetry JSONL schema remains unchanged.

`_refresh_index_unlocked()` returns an explicit refresh result containing:

- the refreshed index or `None`;
- whether durable index state changed;
- snapshot file size;
- an open binary snapshot handle whose identity was verified against the index when a page read requires one.

The index is written only when it was created, rebuilt, or advanced through at least one complete appended line. A no-growth poll does not create a temporary file, call `fsync`, replace the index, or sync the directory.

## Snapshot and lock boundary

For event reads:

1. Acquire the per-stream process lock.
2. Load or rebuild the index and scan only bytes after `indexed_size`.
3. Open the telemetry file, verify its device and inode against the index, and capture `snapshot_size = indexed_size`.
4. Validate the caller's expected generation.
5. Release the process lock while keeping the verified snapshot handle open.
6. Parse only bytes up to `snapshot_size`.
7. Close the snapshot handle.

Appending after step 5 is allowed. The page never reads beyond its captured size, so new records appear on the next poll. Replacement after step 5 cannot change the already-open snapshot identity. Windows replacement is naturally blocked while the handle is open; POSIX retains the opened inode.

Status polling completes entirely under the short metadata lock and returns index metadata without parsing the page body.

## Public telemetry contracts

`TrainingTelemetryStatus` adds the following final field with a default of `None` so existing positional construction remains compatible:

- `stream_generation: str | None = None`.

`TrainingTelemetryPage` adds the following final fields with defaults so existing five-argument positional construction remains compatible:

- `stream_generation: str | None = None`;
- `reset_required: bool = False`.

`read_training_telemetry()` adds an optional keyword-only `expected_generation: str | None = None`.

Behavior:

- Missing stream: generation is `None`, page is empty, and `reset_required` is false.
- Initial request without an expected generation: return the current generation and normal records.
- Matching expected generation: return normal records.
- Mismatched expected generation: return no records, `next_sequence=0`, the current generation, and `reset_required=true`.
- A non-UUID expected generation is rejected as invalid input rather than treated as a legitimate old cursor.

No records are returned on a generation mismatch. The caller must reset before replaying the new stream.

## Studio API contract

`TelemetryStatusResponse` adds `streamGeneration`.

`TelemetryEventsResponse` adds:

- `streamGeneration`;
- `resetRequired`.

The events endpoint accepts optional `stream_generation` and passes it as `expected_generation` to the telemetry reader. The query value is validated as a canonical UUID string.

Studio preserves the existing camel-case response convention.

## Frontend behavior

`useTrainingTelemetry()` stores both sequence and generation refs.

Polling behavior:

1. Send the current generation with the event request when known.
2. If `resetRequired` is true, clear buffered records, set sequence to zero, adopt the returned generation, and immediately issue one replacement-generation poll.
3. If no generation is stored yet, adopt the page generation before accepting records.
4. Reject a page whose generation unexpectedly differs without `resetRequired`.
5. Compare the status and event page generations returned by the parallel requests. If both are non-null and differ, discard the responses and retry once rather than publishing a mixed snapshot.
6. On job or seed change, clear generation, sequence, records, and status together.

A single refresh performs at most one automatic reset or generation-race retry to prevent loops if the stream changes continuously.

## Error handling

- Invalid sidecar generation values invalidate the index and trigger a safe rebuild.
- File identity drift during snapshot capture raises a runtime telemetry identity error; Studio converts it to `artifact_invalid`.
- A stream that changes again during the automatic retry leaves the UI delayed/offline with an explicit error rather than combining generations.
- Incomplete trailing records remain excluded from `indexed_size`; append continues to fail closed until the tail is resolved externally.

## Testing

### RED contracts

1. Replace a stream after the client has cursor sequence 100; an old-generation request must currently return an empty normal page rather than an explicit reset.
2. Delete the index while preserving the JSONL file; the generation must change and invalidate the old cursor.
3. Instrument `_write_index`; repeated no-growth status/events polls must currently rewrite the index.
4. Block record parsing in a reader thread; a writer append must currently remain blocked because parsing holds the process lock.
5. Frontend receives `resetRequired`; current hook must fail to clear records and replay from zero.
6. Status and Events return different generations during one parallel refresh; the current hook must fail to discard the mixed snapshot.

### GREEN contracts

- Generation remains stable across normal appends.
- Replacement, truncation, or index loss changes generation.
- Old generation returns no records and requests reset.
- Reset replay returns only the new generation.
- No-growth polls perform zero durable index writes.
- Writer append completes while an already-snapshotted page is being parsed.
- Status and Events generation races are retried once without publishing either response.
- Linux and Windows compatibility tests pass.
- Existing strict parsing, duplicate seed, process-concurrency, Studio, and integration tests remain green.
- Add a deterministic large-stream test proving a near-tail page parses at most one checkpoint stride plus the requested page, rather than the whole file.

## Compatibility and migration

The sidecar index is a cache and may be rebuilt, so the v1-to-v2 transition requires no migration command. Existing telemetry JSONL files remain readable. Python and Studio response models gain additive fields with backward-compatible defaults where direct construction exists. Existing internal call sites that do not pass an expected generation retain initial-request behavior.

## Acceptance criteria

- Old cursor plus changed generation cannot return telemetry records.
- Studio never combines two generations in one in-memory record buffer.
- A status/events race cannot publish a mixed generation snapshot.
- Repeated unchanged polls do not rewrite the index.
- Page parsing does not hold the append serialization lock.
- All exact-head CI, PostgreSQL, Ubuntu/Windows compatibility, frontend tests/build, and critical coverage checks pass without lowering existing thresholds.
