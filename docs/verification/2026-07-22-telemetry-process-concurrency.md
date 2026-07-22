# Telemetry Process Concurrency Verification

Date: 2026-07-22

Branch: `agent/harden-telemetry-process-concurrency-20260722`

Pull request: #81

Dependency base: PR #80 exact head
`88f7e486b6db56fdc16ab89db5a77ef599c8f48f`

## Scope

This change hardens indexed training telemetry for cooperating Linux and Windows
processes while preserving the JSONL schema, index schema, public writer API,
indexed page/status contracts, and Studio telemetry endpoints.

Implemented boundaries:

- per-stream sidecar OS lock using `fcntl.flock` on POSIX and `msvcrt.locking`
  on Windows;
- one append or one read/index snapshot per lock acquisition rather than a
  writer-lifetime lock;
- process-visible sequence validation before every append;
- append-only binary writes with a partial-write loop;
- fail-closed rejection of incomplete trailing records;
- fail-closed rejection when an open writer descriptor no longer identifies the
  current telemetry path;
- process-unique index temporary files, file `fsync`, atomic replacement, and
  best-effort parent-directory synchronization;
- indexed reads bounded by the refreshed `indexed_size` snapshot;
- explicit `flush()` and `close()` durability with configurable append cadence.

No evidence repair, truncation, database migration, direct exchange routing, or
production-readiness change is included.

## TDD RED evidence

### Process sequence, tail, and index races

RED head:

`d74e639f9d0835b3a35fe00a912b725d095e73ee`

Workflow run `29911288413` failed as intended on both Linux and Windows.

Linux artifact:

- artifact: `8525980884`
- digest:
  `sha256:054442d9b61ba1edb8a16b4872df696dc5951a1f6096ffc8eb53141e0e22e3f8`

Windows artifact:

- artifact: `8525984305`
- digest:
  `sha256:79f115ef017487868897507f32fdbc452626475ffe4689b431f9af0834e0ceae`

The valid RED run demonstrated:

1. two already-open processes both accepted sequence `1`;
2. the writer appended after incomplete trailing JSON bytes instead of failing
   closed;
3. concurrent readers raced on the shared fixed `.tmp` index path, producing
   file-not-found/replacement failures.

The earlier readiness-queue test-fixture defect was corrected before this RED run;
these failures were produced by the existing production implementation.

### Open-writer stream identity drift

RED head:

`b63f4615075f469780e556d79ab29f6da13dafbe`

Workflow run `29912033107` failed as intended because a writer whose path was
atomically replaced continued writing to its old inode and did not raise.

- artifact: `8526267296`
- digest:
  `sha256:63741401c5a48721f424165dd2d6d537fb9bff4d1e30ada8459881c7405effa8`

The production fix compares `fstat(writer_fd)` with `stat(telemetry_path)` while
holding the process lock and raises `RuntimeError` when the identities differ.

## Focused GREEN evidence

### Cross-platform process locking

Workflow run `29911808844` passed on Linux and Windows.

Linux artifact:

- artifact: `8526200257`
- digest:
  `sha256:fb9c56d0e53d239dd97dbfb00f1c550642e38bfab0f775db691f0cf0b30de7e5`

Windows artifact:

- artifact: `8526200535`
- digest:
  `sha256:928b57b58e889641cee7b6c6df1658eb8b0d95caff43b91357d9a73c346f71c6`

The run verified Ruff, formatting, repository-wide Mypy on Linux, native
`msvcrt` locking on Windows, spawn-based duplicate-writer races, incomplete-tail
rejection, concurrent page/status reads, existing telemetry contracts, training
integration, and Studio telemetry API tests.

### Stream identity fix

Workflow run `29912184255`, job `88897470828`, passed source transformation,
Ruff/format, repository-wide Mypy, process-concurrency tests, existing telemetry
contracts, training integration, Studio API tests, commit, and push.

No test tolerance or schema version was changed.

## Cleanup-head full verification

Cleanup head:

`b203f0816b45aa26c7940961bbf929fb9c0b424c`

GitHub Actions CI run `29912311501`: success.

- exact-head checkout: passed;
- Studio tests, TypeScript check, production build, and fixed viewport: passed;
- workflow-security validation: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1188 passed, 2 skipped, 11 warnings`;
- total coverage: `83.43%`;
- total branch coverage: `70.34%`;
- critical branch coverage: passed;
- CLI smoke: passed;
- Ubuntu and Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

Pytest diagnostics:

- artifact: `8526483274`
- digest:
  `sha256:22463e164a4499afa62a9a39ffd18660b93d46c68b4d63be26d5a48fd720ef4f`

PostgreSQL Catalog run `29912311550`: success.

## Coverage ratchet

Measured `trade_rl/telemetry/indexed_training.py` branch coverage:

- covered branches: 64;
- total branches: 94;
- observed: `68.0851%`;
- configured minimum: `68.0%`.

No existing critical-coverage threshold was reduced.

Ratchet head:

`7ff3e98511a9a3438277bee442587f6f55f89a6f`

GitHub Actions CI run `29912659864`: success.

- full Pytest: `1188 passed, 2 skipped, 11 warnings`;
- total coverage: `83.43%`;
- total branch coverage: `70.34%`;
- indexed telemetry branch ratchet: `68.09% >= 68.0%`;
- all static, architecture, serving, CLI, compatibility, and training-image jobs:
  passed.

Ratchet-head Pytest diagnostics:

- artifact: `8526594309`
- digest:
  `sha256:3450bc292a9da0a55b71168b6290a24bc574cfe92f86a672279825ea8cf0c19f`

PostgreSQL Catalog run `29912659868`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- migration: passed;
- unit and integration tests: passed;
- cleanup: passed.

## Compatibility and review

Comparison from PR #80 exact head to the cleanup head was limited to four files:

- the design document;
- the implementation plan;
- spawn-based process-concurrency tests;
- `trade_rl/telemetry/indexed_training.py`.

The coverage ratchet and this verification document are the only additional final
files.

Review conclusions:

- public telemetry exports and schemas are unchanged;
- `flush_every`, `append`, `flush`, `close`, and context-manager usage remain
  source compatible;
- the existing strictly-increasing sequence error text is preserved;
- live readers are blocked only for short append/read/index transactions;
- crashed processes release advisory locks through handle closure;
- index temporary names no longer collide across processes;
- incomplete evidence is never truncated or automatically repaired;
- an open descriptor cannot silently write to a replaced telemetry inode;
- no repository caller relies on the base writer's former private text handle;
- no temporary verification workflow or patch script remains in the intended
  final diff.

No critical or important review issue remained at this checkpoint.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- PR #81 remains Draft and is not merged.
