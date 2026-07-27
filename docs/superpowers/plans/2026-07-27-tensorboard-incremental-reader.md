# TensorBoard Incremental Studio Reader Implementation Plan

## Goal

Stop unchanged TensorBoard event files from being reloaded and reparsed twice every two-second Studio polling cycle while preserving all security, generation, cursor, and response contracts.

## Task 1: Lock unchanged and append behavior

Add focused tests with injected fake accumulators proving:

- one initial `Reload()` serves both `status()` and `scalars()`;
- repeated unchanged requests perform no additional reload or scalar extraction;
- appending to an event file performs exactly one additional reload;
- the appended scalar becomes visible with the existing stable generation.

Run the focused tests and confirm RED because the reader has no persistent cache or accumulator factory.

## Task 2: Lock replacement, failure, and cache-bound behavior

Add tests proving:

- file truncation or replacement creates a fresh accumulator;
- a reload failure removes the affected cache and the next successful request uses a fresh accumulator;
- changing one event file does not reparse unchanged files;
- the configured LRU limit evicts the least recently used source;
- concurrent status/scalar requests serialize one refresh.

## Task 3: Introduce typed cache contracts

Add private dataclasses and protocols for fingerprints, per-file snapshots, source snapshots, and accumulators. Add a reentrant lock and bounded ordered cache to `StudioTrainingMetricsReader`.

Keep the production constructor compatible with `StudioTrainingMetricsReader(settings)`.

## Task 4: Implement incremental refresh

Replace the static full `_load()` path with an instance refresh path that:

- validates the source and fingerprints;
- reuses unchanged file snapshots;
- incrementally reloads append-compatible files;
- recreates replaced or truncated files;
- merges only after all changed files validate;
- publishes the cache atomically;
- discards failed entries.

## Task 5: Preserve endpoint and browser contracts

Run existing Studio training-metrics unit and API tests. Confirm stable append generation, resumed-directory reset, cursor paging, allowlisting, malformed-value handling, and symlink rejection remain unchanged. No frontend polling change is required.

## Task 6: Full verification and integration

Run Ruff, format, Mypy, import architecture, dead-code, Recovery/Serving smoke, Ubuntu and Windows compatibility, Training image, PostgreSQL, full Pytest and coverage, critical branch coverage, and CLI smoke at the exact PR head. Update the PR evidence and squash merge only when every required check is green.
