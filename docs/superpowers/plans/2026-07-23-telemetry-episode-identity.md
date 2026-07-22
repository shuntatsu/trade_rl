# Telemetry Episode Identity Implementation Plan

Date: 2026-07-23

## Scope

Implement the design in `docs/superpowers/specs/2026-07-23-telemetry-episode-identity-design.md` as one independent PR from current `main`.

## Task 1 — Reproduce the missing boundary

Add a frontend regression in `studio/src/live/telemetryStreams.test.ts` where:

- one environment emits records for two explicit episode IDs;
- terminal flags are absent;
- `environmentStep` and `marketIndex` remain monotonic;
- the expected current episode contains only the newest explicit ID.

Verify the focused test fails because the current implementation joins both episodes.

## Task 2 — Add the record contract

Update `trade_rl/telemetry/training.py`:

- add nullable `episode_id`;
- validate non-negative integer semantics;
- serialize it in JSON;
- accept missing or null values as legacy records;
- reject boolean, negative, and non-integer explicit values.

Add focused record-contract tests.

## Task 3 — Allocate identity in the producer

Update `trade_rl/rl/training_telemetry.py`:

- maintain an active episode ID per vector environment;
- allocate a new ID for a new environment episode;
- emit the ID on every retained record;
- rotate only after writing a terminal or truncated record;
- clear previous-close and previous-weight fallback state at rotation.

Add integration tests for interleaved environments, stable identity, terminal rotation, and fallback-state reset.

## Task 4 — Expose Studio and browser contracts

Update:

- `trade_rl/studio/telemetry.py`;
- `studio/src/data/types.ts`;
- `studio/src/live/telemetryGuards.ts`;
- affected fixtures.

Require strict explicit values while normalizing legacy omission to `null`.

## Task 5 — Prefer explicit identity in current UI boundary

Update the existing `currentEnvironmentEpisode()` implementation rather than importing the older alternate page/track implementation:

- latest explicit ID selects exact matching records;
- latest legacy record uses the existing heuristic;
- existing environment and current-episode UI remains unchanged.

Keep PR #85 behavior and tests intact.

## Task 6 — Add measured coverage and documentation

Measure `trade_rl/rl/training_telemetry.py` branch coverage and add only the supported threshold to the current `pyproject.toml`; do not overwrite later telemetry-generation or environment-runtime ratchets.

Record RED/GREEN evidence, exact commit SHA, workflow runs, test totals, coverage, and artifact digests.

## Verification

Run exact-head CI covering:

- Studio Vitest, TypeScript, production build, and fixed viewport;
- Ruff and format;
- Mypy;
- Import Linter and architecture tests;
- Serving smoke;
- complete Pytest and critical branch coverage;
- CLI smoke;
- Ubuntu and Windows compatibility;
- complete training image and non-root probe;
- PostgreSQL Compose, migration, and integration tests.

Production remains `NO-GO`.
