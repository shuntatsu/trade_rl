# Telemetry Episode Identity Design

Date: 2026-07-23

## Problem

Live Training already isolates vector environments and conservatively detects the current episode from terminal records or counter rollback. That browser-side boundary prevents the confirmed visualization defect, but it is still inferential. An auto-reset environment can begin a new episode while `environment_step` and `market_index` remain monotonic and no retained terminal record is available. In that case, the current fallback can join two episodes.

## Goal

Add a producer-issued, nullable episode identity to training telemetry without changing the existing JSONL schema identifier or replacing the current Live Training UI.

## Invariants

- `schema_version` remains `training_telemetry_v1`.
- `episode_id` is additive and nullable so historical records remain readable.
- A sampler assigns one non-negative episode ID per vector environment.
- Records from the same active environment episode retain the same ID.
- A terminal or truncated transition is written with its current ID; the next record for that environment receives a new ID.
- Episode allocation is stream-local and collision-free for records emitted after sampler startup.
- Producer state for previous close and previous weights is cleared when the episode ends.
- Studio normalizes the field to `episodeId` and validates explicit values strictly.
- Live Training prefers an explicit ID when the latest selected-environment record has one.
- Legacy records with no ID continue to use the existing terminal/rollback boundary.
- Explicit and legacy records are never silently combined into one authoritative episode when the latest record is explicit.

## Data contract

`TrainingTelemetryRecord` gains:

```text
episode_id: int | None = None
```

JSON gains:

```json
{"episode_id": 12}
```

or, for legacy/unknown identity:

```json
{"episode_id": null}
```

The field does not alter sequence, stream generation, sparse-index, process-lock, cursor reset, or evidence-file identity contracts.

## Producer allocation

`TrainingTelemetrySampler` owns:

- a mapping from `environment_id` to active `episode_id`;
- a monotonically increasing next-ID counter initialized above the existing stream sequence.

The sequence-derived start avoids reusing an ID after appending to a pre-existing stream without scanning all historical optional IDs. The ID is not claimed to be globally unique outside one telemetry stream.

## Browser selection

For one selected environment:

1. Sort records by global telemetry sequence.
2. Inspect the latest record.
3. When its `episodeId` is non-null, return only records with that exact ID.
4. When it is null, retain the current conservative terminal, environment-step rollback, and market-index rollback logic.

This keeps legacy compatibility while ensuring explicit producer identity takes precedence.

## Non-goals

- No alternate Live Training page or selector design.
- No rewrite of historical JSONL evidence.
- No change to training, checkpoint selection, sealed evaluation, Serving, release, or execution.
- No production-readiness claim.

Production remains `NO-GO`.
