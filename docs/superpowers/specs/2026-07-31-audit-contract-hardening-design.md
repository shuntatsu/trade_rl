# Audit Contract Hardening Design

## Goal

Close the remaining execution-evidence, stage-resume, artifact-publication, and deserialization gaps found in the July 31 audit without weakening the current fail-closed research workflow.

## Scope

This change covers four independent but related boundaries:

1. execution promotion evidence and signed selection authorization;
2. symbol-triplet stage completion and cursor advancement;
3. checkpoint, replay-buffer, and TorchScript deserialization;
4. atomic pointer publication for research and serving artifacts.

The execution-cost identity, multi-asset isolated-margin rejection, dead helper removal, and microsecond run IDs from the earlier audit PR are retained and reapplied to current `main`.

## Execution evidence v3

`ExecutionEvidence` must bind the actual immutable order-event artifact, not only a claimed count. Version 3 adds:

- `order_event_artifact_digest`;
- `order_event_artifact_size_bytes`;
- `order_event_schema`;
- `order_event_count`;
- `terminal_book_digest`;
- `terminal_order_book_digest`.

Promotion validation receives the event artifact path, verifies that it is a regular non-symlink file, checks its size and SHA-256 digest, parses the canonical event artifact, recomputes the event count, and checks terminal state digests. A positive event count remains required.

The signed `SelectionProposal` binds the complete execution-evidence digest. Selected-final training rejects evidence whose digest differs from the authorized proposal.

## Stage completion chain

`SymbolTripletTrainingCursor` version 2 stores `last_completion_digest` in addition to the last stage ID. A stage after index zero can only be built from a completion whose digest exactly matches the cursor anchor.

Cursor advancement uses compare-and-swap semantics. The commit operation receives the expected persisted cursor digest, obtains an exclusive lock for the cursor path, reloads the cursor under the lock, and refuses stale updates. Completion publication is exclusive. The cursor is then atomically replaced. The lock is released only after directory durability is requested.

This design prevents two workers from successfully committing the same stage and prevents a compatible but substituted completion from becoming the next transfer source.

## Verified deserialization

All unsafe deserializers consume a private verified copy:

- Stable-Baselines3 policy checkpoints;
- replay buffers;
- single-member structured TorchScript exports;
- structured TorchScript ensemble members.

The source is opened with no-follow semantics where supported and must be a regular file. Bytes are copied into a private temporary directory, fsynced, hashed, and deserialized only from the verified copy. The original repository or bundle path is never reopened by the deserializer.

## Publication protocol

Atomic pointer writes return a commit-state result distinguishing:

- failure before the pointer replacement;
- replacement completed but directory durability failed;
- fully durable success.

A run is rolled back to staging only when pointer replacement definitely did not occur. If replacement occurred and only fsync failed, the run remains published and the operation raises an explicit durability error. Readers continue to validate the pointer target and fail closed if it is unavailable.

Temporary files are process-unique, exclusively created, and removed in `finally` blocks. The same primitive is shared by the artifact store and serving registry.

## Compatibility

Intentional fail-closed changes:

- execution-promotion evidence v1 and v2 are rejected;
- selection proposals without the execution-evidence digest are rejected;
- symbol-triplet cursors v1 are rejected;
- multi-asset isolated margin is rejected;
- symlink and non-regular deserialization inputs are rejected.

Exploratory training may create non-promotable execution evidence locally, but selected-final training requires the complete v3 event artifact chain.

## Testing

Regression tests cover:

- forged order-event counts and event-artifact substitution;
- proposal/evidence digest mismatch;
- stale and concurrent stage cursor commits;
- substituted previous completions;
- pointer failure before replace and fsync failure after replace;
- source replacement between verification and TorchScript deserialization;
- multi-asset isolated margin;
- execution-policy digest changes for every economic field;
- lazy-export attribute existence.

The final gate is the complete repository CI matrix, including Ruff, formatting, Mypy, Import Linter, dead-code report, full pytest with branch coverage, critical-coverage checks, platform compatibility, container probes, and PostgreSQL integration tests.
