# TensorBoard Incremental Studio Reader Design

## Problem

Trade RL Studio polls training metrics every two seconds. Each refresh currently calls both the status and scalar endpoints. `StudioTrainingMetricsReader` creates a new TensorBoard `EventAccumulator` for every event file on every endpoint call, reloads every file, reads every allowlisted scalar, and reconstructs the complete merged series before applying the client cursor.

For an unchanged active run this means the same append-only event files are fully revisited twice per polling cycle. The browser already merges points incrementally, but the backend does not preserve any incremental state.

## Considered approaches

### A. Increase the browser polling interval

This reduces load but makes the live training view less responsive and still performs duplicate full reads whenever polling occurs.

### B. Make the browser skip the status request

This removes one backend call but weakens seed, generation, availability, and reset handling. The scalar endpoint would still recreate all accumulators and reread all event files.

### C. Add a bounded reader-level incremental snapshot cache

Keep secure source discovery and generation checks on every request, but retain validated per-event-file accumulators and parsed allowlisted points in the long-lived `StudioTrainingMetricsReader`. Reload only files whose fingerprint changed and reuse the merged immutable snapshot for the immediately following status/scalar request. This is selected.

## Decision

`StudioTrainingMetricsReader` will own a thread-safe bounded LRU cache. One cache entry represents a validated seed source generation and contains:

- the secure source identity;
- one file fingerprint per event file;
- one reusable TensorBoard accumulator per event file;
- the validated allowlisted scalar points parsed from each file;
- the merged immutable scalar snapshot.

A file fingerprint contains size, nanosecond modification time, device, and inode identity where available.

## Refresh behavior

1. Secure path and symlink validation continues through `_sources()` on every request.
2. The source generation continues to depend only on the event-file and run-directory set, so ordinary appends do not force a browser reset.
3. If the generation and every fingerprint are unchanged, return the cached merged snapshot without calling `Reload()` or reading scalar tags.
4. If an existing file grew with the same stable file identity, reuse its accumulator and call `Reload()` only for that file.
5. If a file shrank, was replaced, or has no stable identity, construct a fresh accumulator for that file.
6. Parse and validate only changed files; reuse per-file parsed points for unchanged files.
7. Rebuild the merged snapshot only after every changed file reload and validation succeeds.
8. On any reload or parse failure, discard the affected source cache and raise the existing fail-closed `ArtifactInvalid` error. No partial snapshot is published.
9. When the event-file set changes, the generation changes and a new source snapshot is built. Superseded entries for the same member root are removed.
10. Cache access is serialized with a reentrant lock because synchronous FastAPI endpoints may execute concurrently in a thread pool.

## Cache bounds

The reader retains at most 32 source snapshots by default. The maximum is constructor-configurable for focused tests. Least-recently-used entries are evicted first.

## Dependency injection

The reader constructor accepts an optional private-facing accumulator factory. Production uses TensorBoard's `EventAccumulator`; tests use deterministic fake accumulators to prove reload counts and replacement behavior without depending on TensorBoard internals.

## Compatibility

- API response schemas and endpoint signatures remain unchanged.
- The browser keeps its two-second polling interval, cursors, generation identity, and reset behavior.
- Scalar allowlisting, finite-value validation, latest-wall-time conflict resolution, sorting, limits, and symlink protections remain unchanged.
- Appended points remain visible without changing generation.
- Resumed run directories and newly created event files still change generation and request a client reset.
- No training, evaluation, Serving, or production-release behavior changes.

## Testing

Regression tests will prove:

- `status()` followed by `scalars()` on unchanged files performs one initial reload only;
- an append performs one additional reload and exposes the new point;
- unchanged event files are not reparsed when another file changes;
- truncation or replacement creates a fresh accumulator;
- a failed reload invalidates the cache and does not publish a partial snapshot;
- concurrent status/scalar reads share one refresh;
- cache entries are bounded and LRU-evicted;
- existing generation, cursor, resume, allowlist, and symlink tests remain green.
