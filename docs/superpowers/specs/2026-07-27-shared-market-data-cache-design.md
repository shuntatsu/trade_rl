# Shared Market Data Cache Design

## Goal

Keep Binance public market archives outside training-run storage, reuse them across containers and generations, and fail before GPU work when required archives are missing.

## Architecture

Docker owns three independent named-volume boundaries:

- `trade-rl-market-archives`: immutable-by-URL Binance Vision ZIP payload cache.
- `trade-rl-training-runs`: generation-scoped run artifacts, checkpoints, logs, and summaries.
- `trade-rl-teacher-cache`: reusable teacher artifacts.

A one-shot `market-data-sync` service mounts the market archive volume read-write. The `trainer` service mounts the same volume read-only. The canonical host-side training launcher always runs the sync service first and then starts the trainer without dependencies.

The sync service plans the exact Binance Vision URLs required by the maintained research interval, inspects the content-addressed URL cache, and downloads only missing archives. Existing cache files are not requested again. The configured research end remains the source of truth; advancing that end causes only newly required URLs to be downloaded.

## Training startup contract

The training image starts through `training_bootstrap.py` rather than calling the full-research entrypoint directly. Bootstrap performs a read-only cache completeness check before CUDA preflight or training. If any required archive is absent or empty, startup fails with a command that runs the sync service.

The trainer never repairs the archive cache. This preserves the ownership boundary and makes an accidental direct trainer invocation fail closed instead of silently reaching the network.

## URL coverage

The cache plan includes:

- Kline archives for every maintained symbol and native timeframe.
- Completed monthly funding-rate archives for USDⓈ-M symbols.

The trailing incomplete funding month is intentionally not required from Vision because the existing dataset path obtains that interval through the public REST funding endpoint. Exchange metadata remains governed by the selected metadata mode and is not treated as an immutable Vision archive.

## Reports and evidence

Each sync emits a deterministic JSON report containing the requested range, planned URL count, already-cached count, downloaded count, and cache root. The report is written into the archive volume and printed to stdout. Training bootstrap reports the same planned/cached counts during its read-only check.

## Failure handling

- Missing or empty cache payload: fail before CUDA preflight.
- Failed download: sync exits non-zero; trainer is not started by the canonical launcher.
- Cache path outside the configured root or non-Vision URL: reject.
- Duplicate planned URLs: deduplicate while preserving deterministic ordering.

## Testing

Tests cover deterministic URL planning, completed funding-month selection, missing-only synchronization, empty-file rejection, bootstrap ordering, launcher command ordering, and Compose volume access modes.
