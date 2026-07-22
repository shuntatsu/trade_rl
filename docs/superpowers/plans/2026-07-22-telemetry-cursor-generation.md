# Generation-Bound Telemetry Polling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind Studio telemetry cursors to an explicit stream generation, prevent mixed-generation replay, and shorten polling lock duration without weakening process safety.

**Architecture:** Persist a UUID generation in sidecar index v2. Refresh index metadata under the existing cross-process lock, capture an identity-verified file snapshot, then parse the bounded page after releasing the lock. Propagate generation/reset state through Python telemetry contracts, Studio API models, TypeScript guards, API client, and the polling hook.

**Tech Stack:** Python 3.12, dataclasses, pathlib, UUID, pytest, FastAPI/Pydantic, React 19, TypeScript, Vitest, Testing Library, Ruff, Mypy, Import Linter, GitHub Actions.

## Global Constraints

- Preserve telemetry JSONL schema `training_telemetry_v1`.
- Replace only the internal sidecar schema with `training_telemetry_index_v2`.
- Keep sequence pagination and the maximum page limit of 2,000 records.
- New Python dataclass fields must be final fields with defaults so existing positional constructors remain valid.
- A generation mismatch must return zero records and `reset_required=True`.
- A no-growth poll must perform zero sidecar writes and zero `fsync` calls through `_write_index()`.
- Page-body parsing must occur outside the append serialization lock.
- Existing critical coverage thresholds may not be reduced.
- Production remains `NO-GO`; direct exchange routing is out of scope.

---

## File map

- Modify `trade_rl/telemetry/training.py`: additive page/status generation fields.
- Modify `trade_rl/telemetry/indexed_training.py`: index v2, generation validation, change-aware refresh, snapshot capture, generation-aware reads.
- Modify `tests/telemetry/test_training.py`: deterministic generation, no-op poll, lock duration, and near-tail scaling contracts.
- Modify `tests/telemetry/test_indexed_process_concurrency.py`: preserve generation and process-safety regression coverage where needed.
- Modify `trade_rl/studio/telemetry.py`: expose generation/reset in Studio response models and reader methods.
- Modify `trade_rl/studio/api.py`: accept the expected generation query parameter.
- Modify `tests/studio/test_telemetry_api.py`: API generation/reset and invalid-query contracts.
- Modify `studio/src/data/types.ts`: TypeScript response fields.
- Modify `studio/src/live/telemetryGuards.ts`: strict UUID and reset validation.
- Modify `studio/src/api/studioApi.ts`: send `stream_generation`.
- Modify `studio/src/live/useTrainingTelemetry.ts`: generation-bound polling and one retry.
- Create `studio/src/live/useTrainingTelemetry.test.tsx`: hook-level reset and race tests.
- Modify frontend mocks that implement `StudioApi`, including `studio/src/pages/LiveTrainingPage.test.tsx` and `studio/src/pages/RuntimePages.test.tsx`.
- Modify `pyproject.toml`: add a non-regressing telemetry polling coverage group after measuring final coverage.
- Create `docs/verification/2026-07-22-telemetry-cursor-generation.md`: exact RED/GREEN and CI evidence.

---

### Task 1: Commit Python generation RED contracts

**Files:**
- Modify: `tests/telemetry/test_training.py`

**Interfaces:**
- Consumes: existing `TrainingTelemetryWriter`, `read_training_telemetry()`, and `training_telemetry_status()` package exports.
- Produces: failing contracts for `stream_generation`, `expected_generation`, and `reset_required`.

- [ ] **Step 1: Add generation helpers and failing tests**

Add `import os` and the following tests after `test_index_rebuilds_after_stream_replacement`:

```python
def _required_generation(value: str | None) -> str:
    assert value is not None
    return value


def test_generation_remains_stable_across_normal_append(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
    first = training_telemetry_status(path)

    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(2))
    second = training_telemetry_status(path)

    assert _required_generation(first.stream_generation) == second.stream_generation


def test_old_generation_requests_reset_after_stream_replacement(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))
    old_generation = _required_generation(
        training_telemetry_status(path).stream_generation
    )

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        json.dumps(record(1).to_json_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(replacement, path)

    page = read_training_telemetry(
        path,
        after_sequence=2,
        limit=10,
        expected_generation=old_generation,
    )

    assert page.items == ()
    assert page.next_sequence == 0
    assert page.reset_required is True
    assert page.stream_generation not in (None, old_generation)


def test_index_loss_rotates_generation_and_invalidates_cursor(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
    first = training_telemetry_status(path)
    old_generation = _required_generation(first.stream_generation)
    path.with_name(f"{path.name}.index.json").unlink()

    page = read_training_telemetry(
        path,
        after_sequence=1,
        limit=10,
        expected_generation=old_generation,
    )

    assert page.items == ()
    assert page.reset_required is True
    assert page.stream_generation not in (None, old_generation)


def test_expected_generation_requires_canonical_uuid(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))

    with pytest.raises(ValueError, match="expected_generation"):
        read_training_telemetry(
            path,
            after_sequence=0,
            limit=10,
            expected_generation="not-a-generation",
        )
```

- [ ] **Step 2: Run the RED tests**

Run:

```bash
uv run pytest -q \
  tests/telemetry/test_training.py::test_generation_remains_stable_across_normal_append \
  tests/telemetry/test_training.py::test_old_generation_requests_reset_after_stream_replacement \
  tests/telemetry/test_training.py::test_index_loss_rotates_generation_and_invalidates_cursor \
  tests/telemetry/test_training.py::test_expected_generation_requires_canonical_uuid
```

Expected: FAIL because status/page generation fields and `expected_generation` do not exist.

- [ ] **Step 3: Record RED evidence**

Save the pytest output as an Actions artifact or a committed verification note containing the exact failing head, run ID, artifact ID, and SHA-256 digest.

- [ ] **Step 4: Commit the tests**

```bash
git add tests/telemetry/test_training.py
git commit -m "test: bind telemetry cursor to stream generation"
```

---

### Task 2: Commit low-contention polling RED contracts

**Files:**
- Modify: `tests/telemetry/test_training.py`

**Interfaces:**
- Consumes: private module hooks `_write_index` and `_parse_record` only inside tests.
- Produces: deterministic tests proving unchanged polls do not write, page parsing releases the process lock, and near-tail reads are bounded.

- [ ] **Step 1: Add imports**

Add:

```python
import threading

from trade_rl.telemetry import indexed_training as indexed
```

- [ ] **Step 2: Add no-growth write test**

```python
def test_no_growth_polls_do_not_rewrite_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        for sequence in range(1, 70):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 69

    writes: list[int] = []

    def fail_on_write(_path: Path, _index: object) -> None:
        writes.append(1)

    monkeypatch.setattr(indexed, "_write_index", fail_on_write)

    status = training_telemetry_status(path)
    page = read_training_telemetry(path, after_sequence=64, limit=10)

    assert status.last_sequence == 69
    assert [item.sequence for item in page.items] == [65, 66, 67, 68, 69]
    assert writes == []
```

- [ ] **Step 3: Add lock-release test**

```python
def test_page_parsing_does_not_hold_append_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        for sequence in range(1, 131):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 130

    original_parse = indexed._parse_record
    parse_started = threading.Event()
    release_parse = threading.Event()
    reader_done = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []

    def blocking_parse(raw_line: bytes):
        if not parse_started.is_set():
            parse_started.set()
            if not release_parse.wait(timeout=10.0):
                raise TimeoutError("page parse was not released")
        return original_parse(raw_line)

    monkeypatch.setattr(indexed, "_parse_record", blocking_parse)

    def read_page() -> None:
        try:
            read_training_telemetry(path, after_sequence=64, limit=20)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            reader_done.set()

    def append_record() -> None:
        try:
            with TrainingTelemetryWriter(path, flush_every=1) as writer:
                writer.append(record(131))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            writer_done.set()

    reader = threading.Thread(target=read_page)
    reader.start()
    assert parse_started.wait(timeout=10.0)

    writer = threading.Thread(target=append_record)
    writer.start()
    assert writer_done.wait(timeout=2.0), "writer remained blocked by page parsing"

    release_parse.set()
    reader.join(timeout=10.0)
    writer.join(timeout=10.0)

    assert reader_done.is_set()
    assert errors == []
    assert training_telemetry_status(path).last_sequence == 131
```

- [ ] **Step 4: Add deterministic near-tail scaling test**

```python
def test_near_tail_page_parses_at_most_one_checkpoint_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=256) as writer:
        for sequence in range(1, 4_097):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 4_096

    original_parse = indexed._parse_record
    parsed = 0

    def counted_parse(raw_line: bytes):
        nonlocal parsed
        parsed += 1
        return original_parse(raw_line)

    monkeypatch.setattr(indexed, "_parse_record", counted_parse)
    page = read_training_telemetry(path, after_sequence=4_080, limit=20)

    assert [item.sequence for item in page.items] == list(range(4_081, 4_097))
    assert parsed <= 80
```

- [ ] **Step 5: Run the RED tests**

Run:

```bash
uv run pytest -q \
  tests/telemetry/test_training.py::test_no_growth_polls_do_not_rewrite_index \
  tests/telemetry/test_training.py::test_page_parsing_does_not_hold_append_process_lock \
  tests/telemetry/test_training.py::test_near_tail_page_parses_at_most_one_checkpoint_stride
```

Expected: the no-growth write test and lock-release test FAIL against the current implementation. The near-tail bound may already pass and is retained as a non-regression contract.

- [ ] **Step 6: Commit the tests**

```bash
git add tests/telemetry/test_training.py
git commit -m "test: constrain telemetry polling critical section"
```

---

### Task 3: Implement sidecar generation and snapshot reads

**Files:**
- Modify: `trade_rl/telemetry/training.py`
- Modify: `trade_rl/telemetry/indexed_training.py`
- Test: `tests/telemetry/test_training.py`
- Test: `tests/telemetry/test_indexed_process_concurrency.py`

**Interfaces:**
- Produces: `TrainingTelemetryPage.stream_generation`, `TrainingTelemetryPage.reset_required`, `TrainingTelemetryStatus.stream_generation`, and `read_indexed_training_telemetry(..., expected_generation=None)`.
- Preserves: `TrainingTelemetryWriter`, page size limits, strict record parsing, sequence gap behavior, and process lock path.

- [ ] **Step 1: Add backward-compatible dataclass fields**

In `trade_rl/telemetry/training.py`, change the dataclasses to:

```python
@dataclass(frozen=True, slots=True)
class TrainingTelemetryPage:
    items: tuple[TrainingTelemetryRecord, ...]
    next_sequence: int
    truncated: bool
    malformed_lines: int
    sequence_gaps: tuple[tuple[int, int], ...]
    stream_generation: str | None = None
    reset_required: bool = False


@dataclass(frozen=True, slots=True)
class TrainingTelemetryStatus:
    available: bool
    record_count: int
    last_sequence: int
    malformed_lines: int
    size_bytes: int
    stream_generation: str | None = None
```

- [ ] **Step 2: Add index v2 generation parsing**

In `trade_rl/telemetry/indexed_training.py`:

```python
from uuid import UUID, uuid4

_INDEX_SCHEMA = "training_telemetry_index_v2"


def _canonical_generation(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"telemetry index {field_name} is invalid")
    try:
        resolved = str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"telemetry index {field_name} is invalid") from error
    if resolved != value:
        raise ValueError(f"telemetry index {field_name} is invalid")
    return resolved
```

Add `generation: str` to `_TelemetryIndex`, include it in `to_json_dict()`, and require it in `_load_index()`.

Create new indexes with:

```python
_TelemetryIndex(
    device=int(stat.st_dev),
    inode=int(stat.st_ino),
    generation=str(uuid4()),
)
```

- [ ] **Step 3: Make index refresh change-aware**

Add:

```python
@dataclass(frozen=True, slots=True)
class _IndexRefresh:
    index: _TelemetryIndex | None
    changed: bool
```

Refactor `_refresh_index_unlocked()` to return `_IndexRefresh`. Set `changed=True` when:

- no valid index existed;
- stream device/inode changed;
- stream size shrank below `indexed_size`;
- at least one complete line advanced `indexed_size`;
- malformed/blank complete appended lines changed index counters or size.

Call `_write_index()` only when `changed` is true. Do not mutate `last_scan_start` or write the index on a no-growth poll.

- [ ] **Step 4: Add verified snapshot capture**

Add:

```python
@dataclass(slots=True)
class _TelemetrySnapshot:
    handle: BinaryIO
    size: int
    index: _TelemetryIndex

    def close(self) -> None:
        self.handle.close()


def _open_snapshot_unlocked(path: Path, index: _TelemetryIndex) -> _TelemetrySnapshot:
    handle = path.open("rb")
    try:
        stat = os.fstat(handle.fileno())
        if (int(stat.st_dev), int(stat.st_ino)) != (index.device, index.inode):
            raise RuntimeError("telemetry stream identity changed")
        if stat.st_size < index.indexed_size:
            raise RuntimeError("telemetry stream size regressed")
        return _TelemetrySnapshot(handle=handle, size=index.indexed_size, index=index)
    except BaseException:
        handle.close()
        raise
```

- [ ] **Step 5: Implement generation-aware page reads**

Change the signature to:

```python
def read_indexed_training_telemetry(
    path: Path,
    *,
    after_sequence: int = 0,
    limit: int = 512,
    expected_generation: str | None = None,
) -> _training.TrainingTelemetryPage:
```

Validate a non-null expected generation with `UUID` canonicalization and a `ValueError` message containing `expected_generation`.

Under `_telemetry_process_lock`:

1. refresh the index;
2. return the missing-stream page when no index exists;
3. return an empty reset page when the expected generation differs;
4. capture `_TelemetrySnapshot` and compute the checkpoint offset.

Release the lock before parsing. Parse only until `snapshot.size`, close the snapshot in `finally`, and return the index generation with `reset_required=False`.

Mismatch return:

```python
return _training.TrainingTelemetryPage(
    items=(),
    next_sequence=0,
    truncated=False,
    malformed_lines=index.malformed_lines,
    sequence_gaps=tuple(index.sequence_gaps),
    stream_generation=index.generation,
    reset_required=True,
)
```

- [ ] **Step 6: Add generation to status and writer refresh usage**

Update `indexed_training_telemetry_status()` to use `_IndexRefresh.index` and return `stream_generation=index.generation`.

Update writer initialization and append to read `.index` from the refresh result. Preserve descriptor/path identity checks and incomplete-tail rejection.

- [ ] **Step 7: Run focused Python verification**

```bash
uv run ruff check trade_rl/telemetry tests/telemetry
uv run ruff format --check trade_rl/telemetry tests/telemetry
uv run mypy .
uv run pytest -q \
  tests/telemetry/test_training.py \
  tests/telemetry/test_indexed_process_concurrency.py \
  tests/integrations/test_training_telemetry.py
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit**

```bash
git add \
  trade_rl/telemetry/training.py \
  trade_rl/telemetry/indexed_training.py \
  tests/telemetry/test_training.py \
  tests/telemetry/test_indexed_process_concurrency.py
git commit -m "fix: bind telemetry pages to stream generation"
```

---

### Task 4: Add Studio backend generation contract

**Files:**
- Modify: `trade_rl/studio/telemetry.py`
- Modify: `trade_rl/studio/api.py`
- Modify: `tests/studio/test_telemetry_api.py`

**Interfaces:**
- Consumes: Python telemetry page/status generation fields.
- Produces: camel-case `streamGeneration`, `resetRequired`, and optional `stream_generation` query input.

- [ ] **Step 1: Write failing Studio API tests**

Add tests that assert:

```python
status_payload = client.get(
    f"/api/studio/jobs/{job_id}/telemetry/status?seed=7"
).json()
assert UUID(status_payload["streamGeneration"])

old_generation = status_payload["streamGeneration"]
# Replace the JSONL stream with a new sequence-1 stream and remove/rebuild its index.
reset_payload = client.get(
    f"/api/studio/jobs/{job_id}/telemetry/events",
    params={
        "seed": 7,
        "after_sequence": 10,
        "limit": 10,
        "stream_generation": old_generation,
    },
).json()
assert reset_payload["items"] == []
assert reset_payload["nextSequence"] == 0
assert reset_payload["resetRequired"] is True
assert reset_payload["streamGeneration"] != old_generation
```

Add an invalid query test:

```python
response = client.get(
    f"/api/studio/jobs/{job_id}/telemetry/events",
    params={"stream_generation": "not-a-uuid"},
)
assert response.status_code == 422
```

- [ ] **Step 2: Run RED Studio tests**

```bash
uv run pytest -q tests/studio/test_telemetry_api.py
```

Expected: FAIL because response fields and query input do not exist.

- [ ] **Step 3: Implement response fields**

In `trade_rl/studio/telemetry.py`:

```python
class TelemetryStatusResponse(StudioModel):
    ...
    stream_generation: str | None = None


class TelemetryEventsResponse(StudioModel):
    ...
    stream_generation: str | None = None
    reset_required: bool = False
```

Add `stream_generation: str | None = None` to `StudioTelemetryReader.events()` and pass it as `expected_generation`.

Map page/status generation fields into the response models for both available and unavailable paths.

- [ ] **Step 4: Implement API query validation**

In `trade_rl/studio/api.py`, add:

```python
stream_generation: str | None = Query(
    default=None,
    pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
),
```

Pass it to `telemetry_reader.events()`.

- [ ] **Step 5: Verify and commit**

```bash
uv run ruff check trade_rl/studio tests/studio/test_telemetry_api.py
uv run ruff format --check trade_rl/studio tests/studio/test_telemetry_api.py
uv run mypy .
uv run pytest -q tests/studio/test_telemetry_api.py tests/telemetry tests/integrations/test_training_telemetry.py
git add trade_rl/studio/telemetry.py trade_rl/studio/api.py tests/studio/test_telemetry_api.py
git commit -m "feat: expose telemetry stream generation in Studio"
```

---

### Task 5: Make frontend polling generation-safe

**Files:**
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/live/telemetryGuards.ts`
- Modify: `studio/src/api/studioApi.ts`
- Modify: `studio/src/live/useTrainingTelemetry.ts`
- Create: `studio/src/live/useTrainingTelemetry.test.tsx`
- Modify: `studio/src/pages/LiveTrainingPage.test.tsx`
- Modify: `studio/src/pages/RuntimePages.test.tsx`

**Interfaces:**
- Consumes: `streamGeneration` and `resetRequired` API fields.
- Produces: generation-bound `loadTelemetryEvents()` calls and a hook that performs at most one reset/race retry.

- [ ] **Step 1: Add failing guard and hook tests**

Create `studio/src/live/useTrainingTelemetry.test.tsx` using `renderHook`, `act`, and `waitFor`.

Test reset behavior with an API mock whose first event response is:

```typescript
{
  seed: 7,
  items: [],
  nextSequence: 0,
  truncated: false,
  malformedLines: 0,
  sequenceGaps: [],
  streamGeneration: '22222222-2222-4222-8222-222222222222',
  resetRequired: true,
}
```

and whose retry returns one sequence-1 record for the new generation. Assert:

```typescript
await waitFor(() => expect(result.current.records.map((item) => item.sequence)).toEqual([1]))
expect(api.loadTelemetryEvents).toHaveBeenNthCalledWith(
  1,
  'job-live',
  0,
  512,
  7,
  null,
)
expect(api.loadTelemetryEvents).toHaveBeenNthCalledWith(
  2,
  'job-live',
  0,
  512,
  7,
  '22222222-2222-4222-8222-222222222222',
)
```

Add a generation-race test where status returns generation A and events returns generation B. The retry returns matching generation B. Assert no generation-A records are ever published and exactly two event calls occur.

Add guard tests in the same file or a focused guard test block proving malformed generation strings and `resetRequired` non-booleans are rejected.

- [ ] **Step 2: Run frontend RED**

```bash
cd studio
npm test -- --run src/live/useTrainingTelemetry.test.tsx
```

Expected: FAIL because fields/signatures/retry behavior do not exist.

- [ ] **Step 3: Extend TypeScript contracts and guards**

In `studio/src/data/types.ts`:

```typescript
export interface TelemetryStatusResponse {
  ...
  streamGeneration: string | null
}

export interface TelemetryEventsResponse {
  ...
  streamGeneration: string | null
  resetRequired: boolean
}
```

In `telemetryGuards.ts`, add:

```typescript
const generationPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const isNullableGeneration = (value: unknown): value is string | null =>
  value === null || (typeof value === 'string' && generationPattern.test(value))
```

Require the generation field in status/events and require `resetRequired` to be boolean.

- [ ] **Step 4: Extend the API client signature**

Change `loadTelemetryEvents()` and `StudioApi.loadTelemetryEvents` to accept:

```typescript
(
  jobId: string,
  afterSequence?: number,
  limit?: number,
  seed?: number | null,
  streamGeneration?: string | null,
) => Promise<TelemetryEventsResponse>
```

The exported function may retain `fetcher` as its final sixth parameter. Add `stream_generation` to `URLSearchParams` only when non-null.

Update all StudioApi test mocks to the new signature and add generation/reset fields to response fixtures.

- [ ] **Step 5: Implement one-retry hook behavior**

Add:

```typescript
const generation = useRef<string | null>(null)
```

Refactor refresh into an inner function `loadSnapshot(allowRetry: boolean)`.

Rules:

- call events with `generation.current`;
- if `page.resetRequired`, clear records/status, set sequence to zero, adopt `page.streamGeneration`, and retry once;
- if status/page generations are both non-null and differ, clear state, set generation to null, and retry once;
- if either condition repeats when `allowRetry` is false, throw an explicit generation-change error;
- before accepting records, require the page generation to equal the stored generation;
- reset generation to null in the job/seed effect cleanup path.

- [ ] **Step 6: Run frontend verification**

```bash
cd studio
npm test -- --run
npm run typecheck
npm run build
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit**

```bash
git add \
  studio/src/data/types.ts \
  studio/src/live/telemetryGuards.ts \
  studio/src/api/studioApi.ts \
  studio/src/live/useTrainingTelemetry.ts \
  studio/src/live/useTrainingTelemetry.test.tsx \
  studio/src/pages/LiveTrainingPage.test.tsx \
  studio/src/pages/RuntimePages.test.tsx
git commit -m "fix: reset live telemetry on stream generation change"
```

---

### Task 6: Verify process safety, coverage, and evidence

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-22-telemetry-cursor-generation.md`

**Interfaces:**
- Consumes: completed backend/frontend implementation.
- Produces: a non-regressing branch coverage floor and exact-head verification evidence.

- [ ] **Step 1: Run focused cross-platform process tests**

Run locally or in an exact-head matrix:

```bash
uv run pytest -q \
  tests/telemetry/test_training.py \
  tests/telemetry/test_indexed_process_concurrency.py \
  tests/integrations/test_training_telemetry.py \
  tests/studio/test_telemetry_api.py
```

Run the same focused set on Ubuntu and Windows. Expected: all pass, including native `fcntl` and `msvcrt` paths.

- [ ] **Step 2: Run the full repository verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json
cd studio && npm test -- --run && npm run typecheck && npm run build
```

Expected: every command exits 0 and overall coverage remains at least 80%.

- [ ] **Step 3: Compute the telemetry polling coverage floor**

Run:

```bash
python - <<'PY'
import json
from math import floor

coverage = json.load(open("coverage.json", encoding="utf-8"))
summary = coverage["files"]["trade_rl/telemetry/indexed_training.py"]["summary"]
covered = summary["covered_branches"]
total = summary["num_branches"]
percent = 100.0 if total == 0 else covered / total * 100.0
minimum = floor(percent * 10.0) / 10.0
print(f"covered={covered} total={total} percent={percent:.4f} minimum={minimum:.1f}")
PY
```

Set a new group in `pyproject.toml`:

```toml
[tool.trade_rl.critical_coverage.groups.telemetry_polling]
minimum = <the printed minimum>
paths = [
    "trade_rl/telemetry/indexed_training.py",
]
```

Do not modify or lower any existing group.

- [ ] **Step 4: Re-run the full CI command after adding the ratchet**

Expected: critical branch coverage passes with the measured floor.

- [ ] **Step 5: Write verification evidence**

Create `docs/verification/2026-07-22-telemetry-cursor-generation.md` containing:

- design and plan paths;
- RED commit SHAs and failure summaries;
- GREEN commit SHA;
- Ubuntu and Windows run IDs;
- final exact-head SHA;
- full test counts, skipped tests, warnings, total coverage, total branch coverage, and telemetry branch coverage;
- pytest artifact ID and SHA-256 digest;
- PostgreSQL run ID and each successful step;
- confirmation that telemetry JSON schema remains v1, index schema is v2, and production remains `NO-GO`.

- [ ] **Step 6: Commit final evidence**

```bash
git add pyproject.toml docs/verification/2026-07-22-telemetry-cursor-generation.md
git commit -m "docs: verify generation-bound telemetry polling"
```

- [ ] **Step 7: Verify the exact final head once more**

Fetch workflow runs for the final documentation commit. Do not claim completion until normal CI and PostgreSQL Catalog both report `completed/success` for that exact SHA.
