# Telemetry Process Concurrency Hardening Plan

**Goal:** Make indexed telemetry append, index refresh, page read, and status read
safe across cooperating Linux and Windows processes without changing public
schemas or Studio APIs.

**Architecture:** A private cross-platform OS file lock protects one append or one
index/read snapshot. The indexed writer uses append-only binary writes and
process-visible sequence validation. Index writes use unique durable temporary
files and atomic replacement.

## Constraints

- Preserve public telemetry imports and JSON/index schema versions.
- Preserve the writer constructor and context-manager methods.
- Preserve duplicate-sequence and closed-writer error behavior.
- Do not lock for the writer lifetime.
- Do not silently truncate incomplete evidence.
- Use spawn-based multiprocessing tests.
- Keep Windows compatibility green.

## Task 1: Commit RED concurrency contracts

Create `tests/telemetry/test_indexed_process_concurrency.py`.

Tests:

- two already-open writers race sequence `1`; exactly one must succeed;
- final JSONL contains one valid record and indexed status reports one record;
- concurrent status/page readers complete without process errors while a writer
  appends ordered records;
- incomplete trailing bytes reject a subsequent append;
- index JSON remains valid and identity-bound.

Run the focused tests before production changes and record the failing workflow
artifact.

## Task 2: Implement cross-platform process lock

Modify `trade_rl/telemetry/indexed_training.py`.

Add:

- lock sidecar path helper;
- private context manager using `fcntl.flock` or `msvcrt.locking`;
- symlink rejection and reliable release;
- unlocked internal refresh helper to avoid nested lock acquisition.

Add focused tests for lock release after exceptions and repeated acquisition.

## Task 3: Implement process-safe writer

Replace the inherited no-op indexed writer with an API-compatible implementation
that:

- validates `flush_every`;
- owns a thread lock and append-only binary descriptor;
- refreshes latest index under the process lock on every append;
- rejects duplicate/regressing sequence across processes;
- rejects incomplete trailing records;
- writes the complete encoded line with a partial-write loop;
- uses `fsync` at `flush_every`, explicit `flush`, and `close`.

Run writer, training telemetry, and integration tests.

## Task 4: Make index refresh and reads transactional

Refactor index refresh into `_refresh_index_unlocked()` and a locked wrapper.

- unique PID/token index temporary;
- write/flush/fsync before `os.replace`;
- cleanup temporary in `finally`;
- optional POSIX parent-directory fsync;
- page read holds lock and stops at snapshot `indexed_size`;
- status holds lock through refresh and size capture.

Run process concurrency, indexed telemetry, and Studio telemetry tests.

## Task 5: Full verification and coverage ratchet

Run exact-head:

- Ruff and format;
- Mypy;
- Import Linter and dead-code report;
- full Pytest with branch coverage;
- telemetry and Studio integrations;
- Ubuntu/Windows compatibility;
- training image;
- PostgreSQL Catalog.

Measure branch coverage for `trade_rl/telemetry/indexed_training.py`. Add or raise a
critical-coverage threshold only to the measured safe value and never reduce an
existing threshold.

Create
`docs/verification/2026-07-22-telemetry-process-concurrency.md` with RED/GREEN,
workflow, artifact, digest, test-count, and coverage evidence.

Create a Draft PR based on PR #80 exact head while targeting `main` for exact-head
workflow execution. Do not merge.