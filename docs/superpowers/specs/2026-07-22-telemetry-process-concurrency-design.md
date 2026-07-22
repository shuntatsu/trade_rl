# Telemetry Process Concurrency Hardening Design

## Context

`IndexedTrainingTelemetryWriter` currently inherits a writer whose lock is a
`threading.Lock`. Two processes can therefore open the same JSONL stream, read
the same `last_sequence`, and both append an identical or out-of-order sequence.
The resulting stream is syntactically valid but violates the monotonic identity
contract.

Index refresh has a second race. Every reader writes the same temporary path,
`.<name>.index.json.tmp`, before replacing the index. Concurrent readers can
truncate, replace, or remove that shared temporary file while another reader is
using it.

The live-training UI reads telemetry while training writes it, so locking the
stream for the entire writer lifetime is not acceptable. The lock boundary must
be one append or one consistent read/index-refresh transaction.

## Approaches considered

### A. Add a third-party file-lock package

A package such as `filelock` or `portalocker` would reduce implementation code,
but it introduces a runtime dependency solely for a small OS primitive and would
need a lock-file compatibility review.

### B. Lifetime writer lock

Acquire an exclusive process lock in the writer constructor and release it on
close. This prevents competing writers, but also blocks live readers for the
entire training run.

### C. Cross-platform OS lock per operation

Use a stable sidecar lock file and native advisory locking:

- `fcntl.flock` on POSIX;
- `msvcrt.locking` on Windows.

Append, index refresh, indexed read, and status operations hold the same exclusive
lock only for their short critical section. This is the selected approach.

## Selected architecture

### `TelemetryProcessLock`

A private context manager in `indexed_training.py` owns a sidecar path:

`<telemetry-name>.lock`

It opens the lock file in binary read/write mode and obtains an OS-level exclusive
lock. OS locks are automatically released when a process exits or the handle is
closed, so crashed processes do not leave a logically held lock.

The implementation must:

- work on Linux and Windows compatibility jobs;
- reject a lock path that is a symbolic link;
- create the parent directory before opening the lock file;
- release the lock in `finally` paths;
- avoid nested acquisition on the same telemetry path.

### Process-safe writer

`IndexedTrainingTelemetryWriter` remains API-compatible with the base writer but
owns an append-only binary file descriptor rather than a buffered text handle.

For every append it:

1. acquires the thread lock;
2. serializes and validates the record before touching the file;
3. acquires the process lock;
4. refreshes the index from the latest visible file state;
5. rejects an incomplete trailing record;
6. compares the proposed sequence with the process-visible last sequence;
7. appends one complete UTF-8 JSON line through the append-only descriptor;
8. releases the process lock.

This ensures two processes racing to append the same sequence cannot both
succeed. One succeeds; the other observes the new sequence and raises the
existing monotonic-sequence `ValueError`.

`flush_every` becomes the durability cadence for `fsync`; every `os.write` is
immediately visible to cooperating processes, while the configured interval
controls durable synchronization. Explicit `flush()` and `close()` perform
`fsync`.

### Incomplete-tail fail-closed rule

A process crash can interrupt a low-level write and leave bytes without a final
newline. Appending another JSON record after those bytes would turn both records
into one malformed line.

While holding the process lock, the writer compares the refreshed
`indexed_size` with the current file size. If the file contains an incomplete
trailing record, append fails with a `RuntimeError`. The writer never silently
truncates research evidence. Repair remains an explicit operator action.

### Consistent indexed reads

`read_indexed_training_telemetry()` holds the process lock across:

- index refresh;
- checkpoint selection;
- JSONL scan.

The scan stops at the refreshed `indexed_size`, creating a consistent snapshot.
Records appended after the lock is released are returned by the next read rather
than being mixed with index metadata from an earlier snapshot.

`indexed_training_telemetry_status()` similarly holds the lock through refresh
and final size capture.

### Atomic index replacement

Index persistence uses a process-unique temporary path containing the PID and a
random token. It writes, flushes, and `fsync`s the temporary file before
`os.replace`. The temporary is removed in a `finally` block. On POSIX, the parent
directory is synchronized after replacement when supported.

The process lock is still mandatory; unique temporary paths protect cleanup and
crash recovery, not index lost-update semantics.

## Compatibility constraints

The change must preserve:

- telemetry JSONL schema and record serialization;
- public writer constructor, `append`, `flush`, `close`, and context-manager API;
- strict monotonic sequence error text;
- indexed page and status dataclasses;
- index schema version and checkpoint stride;
- handling of malformed complete lines and sequence gaps;
- live Studio telemetry endpoints;
- Linux and Windows compatibility.

No database, broker, direct-exchange, or production-readiness behavior changes.

## Error handling

- Duplicate or regressing sequence: existing `ValueError`.
- Append after close: existing `RuntimeError`.
- Incomplete trailing record: explicit `RuntimeError`.
- Lock path symbolic link: explicit `RuntimeError`.
- Invalid or stale index: rebuild under the process lock.
- Index temporary write failure: propagate the original exception and remove the
  process-unique temporary where possible.

## Testing strategy

1. Commit process-concurrency tests before production changes.
2. Use `multiprocessing.get_context("spawn")` so tests exercise Windows-compatible
   process semantics on every platform.
3. Start two writers before a shared event, then race the same sequence. Assert
   exactly one append succeeds and the stream contains exactly one record.
4. Run multiple processes repeatedly reading status and pages while a writer
   appends; assert no process error, no duplicate sequence, valid index JSON, and
   complete final status.
5. Verify incomplete-tail fail-closed behavior.
6. Verify explicit flush/close durability and append-after-close behavior.
7. Run existing telemetry, Studio API, training integration, Windows/Ubuntu
   compatibility, full suite, and critical coverage on the exact head.
8. Add or raise a telemetry branch-coverage ratchet only from measured coverage;
   do not reduce any existing threshold.

## Pull-request structure

This is a stacked Draft PR based on PR #80 exact head
`88f7e486b6db56fdc16ab89db5a77ef599c8f48f`. It targets `main` so repository
workflows run. After PR #79 and PR #80 merge, its effective diff will reduce to
the telemetry hardening files.