# Telemetry Process Concurrency Verification

Date: 2026-07-23

Merged pull request: #95

Merge commit: `e9888e792cf4b1e3477f62987834ec4eb221e39e`

Verified implementation head: `0e22094485c0b530b64da3a0c70f96b5410b2c66`

## Scope

This change hardens indexed training telemetry for cooperating Linux and Windows
processes while preserving the JSONL schema, sparse-index schema, public writer
API, indexed page/status contracts, and Studio telemetry endpoints.

Implemented boundaries:

- per-stream sidecar OS lock using `fcntl.flock` on POSIX and `msvcrt.locking`
  on Windows;
- one append or one read/index snapshot per lock acquisition rather than a
  writer-lifetime lock;
- latest process-visible sequence validation before every append;
- exactly-one-writer behavior when processes race to append the same sequence;
- append-only binary writes with a partial-write loop;
- fail-closed rejection of incomplete trailing records;
- fail-closed rejection when an open writer descriptor no longer identifies the
  current telemetry path;
- process-unique index temporary files, file `fsync`, atomic replacement, and
  best-effort parent-directory synchronization;
- indexed reads bounded by one locked refreshed-index and JSONL snapshot;
- explicit `flush()` and `close()` durability with configurable append cadence.

No evidence repair, truncation, database migration, direct exchange routing, or
production-readiness change is included.

## TDD RED evidence

### Process sequence, tail, and index races

RED head: `d74e639f9d0835b3a35fe00a912b725d095e73ee`

Workflow run `29911288413` failed as intended on Linux and Windows.

Linux artifact:

- artifact: `8525980884`
- digest:
  `sha256:054442d9b61ba1edb8a16b4872df696dc5951a1f6096ffc8eb53141e0e22e3f8`

Windows artifact:

- artifact: `8525984305`
- digest:
  `sha256:79f115ef017487868897507f32fdbc452626475ffe4689b431f9af0834e0ceae`

The valid RED run demonstrated:

1. two already-open processes could both accept sequence `1`;
2. the writer appended after incomplete trailing JSON bytes instead of failing
   closed;
3. concurrent readers raced on the shared fixed `.tmp` index path.

### Open-writer stream identity drift

RED head: `b63f4615075f469780e556d79ab29f6da13dafbe`

Workflow run `29912033107` failed as intended because a writer whose path was
atomically replaced continued writing to its old inode and did not raise.

- artifact: `8526267296`
- digest:
  `sha256:63741401c5a48721f424165dd2d6d537fb9bff4d1e30ada8459881c7405effa8`

The production fix compares `fstat(writer_fd)` with `stat(telemetry_path)` while
holding the process lock and raises when the identities differ.

## Focused GREEN evidence

Workflow run `29911808844` passed the cross-platform locking contracts.

Linux artifact:

- artifact: `8526200257`
- digest:
  `sha256:fb9c56d0e53d239dd97dbfb00f1c550642e38bfab0f775db691f0cf0b30de7e5`

Windows artifact:

- artifact: `8526200535`
- digest:
  `sha256:928b57b58e889641cee7b6c6df1658eb8b0d95caff43b91357d9a73c346f71c6`

The run verified Ruff, formatting, repository-wide Mypy, native Windows locking,
spawn-based duplicate-writer races, incomplete-tail rejection, concurrent
page/status reads, existing telemetry contracts, training integration, and Studio
telemetry API tests.

Workflow run `29912184255`, job `88897470828`, verified the stream-identity fix
with source transformation, Ruff/format, Mypy, process-concurrency regressions,
existing telemetry contracts, training integration, and Studio API tests.

No test tolerance or schema version was changed.

## Clean current-main reconstruction

The original PR #81 was stacked on the unsquashed PR #80 history. After its
architecture dependencies were squash-merged independently, PR #95 recreated only
the telemetry-concurrency delta directly from current `main` at
`6cf23b98698f5d53ec40629dd723efc4bd4cfbb6`.

The effective diff contained exactly six files:

- `trade_rl/telemetry/indexed_training.py`;
- `tests/telemetry/test_indexed_process_concurrency.py`;
- one measured telemetry coverage-threshold entry in `pyproject.toml`;
- design, implementation-plan, and verification documentation.

No environment-runtime or PR #79 implementation file was repeated. No temporary
workflow, generated patch, or migration remained in the merged diff.

## Exact-head verification

GitHub Actions CI run `29956385683`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed viewport: passed;
- workflow-security validation: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1197 passed, 2 skipped, 11 warnings`;
- total coverage: `83.43%`;
- total branch coverage: `70.34%`;
- critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility and spawn multiprocessing regressions: passed;
- Windows compatibility and spawn multiprocessing regressions: passed;
- complete training-image build and packaged non-root runtime probe: passed.

Exact-head artifacts:

- Pytest diagnostics: `8544270458`, digest
  `sha256:eb3af56b36c057495d96bdba85f9be894cac3dd96e18bdaf1ddcc4407b3bdf84`
- architecture diagnostics: `8544227996`, digest
  `sha256:141c5b3f3bf4b4443a73754899d79effb96c21dc0bd8989125f37606722fd81b`
- static diagnostics: `8544227415`, digest
  `sha256:095c032b677c6af423add78e5093813bd7ae7d7f71bbe9bb38ef708f5f1858ea`
- training-image evidence: `8544221968`, digest
  `sha256:b06b24a42458cf6be0a00ad2dec46671761c02eeb314cb28c1e7f8e8b3e304ef`
- Studio layout diagnostics: `8544216788`, digest
  `sha256:03f7e919f7b5169aeaf7ac2c84ba122015ee764e04c3d7a7e093db5c1e530932`
- Windows compatibility: `8544216083`, digest
  `sha256:92e00bdd1ea56592e4bafa87c6662b4603097cc5d0393b8aae8a012803d9ee01`
- Ubuntu compatibility: `8544208737`, digest
  `sha256:d3d7cd79eb2bae8bab4e6219ebccf9ae9555ea9f766b21443f6d48b1480d5bed`

PostgreSQL Catalog run `29956385705`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

## Coverage ratchet

Measured `trade_rl/telemetry/indexed_training.py` branch coverage:

- covered branches: 64;
- total branches: 94;
- observed: `68.0851%`;
- configured minimum: `68.0%`.

No existing critical-coverage threshold was reduced.

## Compatibility and architecture review

- public telemetry exports and schemas are unchanged;
- `flush_every`, `append`, `flush`, `close`, and context-manager usage remain
  source compatible;
- the existing strictly-increasing sequence error contract is preserved;
- live readers are blocked only for short append/read/index transactions;
- crashed processes release advisory locks through handle closure;
- index temporary names cannot collide across cooperating processes;
- incomplete evidence is never truncated or automatically repaired;
- an open descriptor cannot silently write to a replaced telemetry inode;
- read/index refresh is observed as one locked process-visible snapshot;
- no repository caller relies on the base writer's former private text handle;
- no unresolved critical or important review issue remained before merge.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- JSONL and sparse-index schemas remain unchanged;
- this verification records corruption prevention and regression evidence, not
  production authorization.
