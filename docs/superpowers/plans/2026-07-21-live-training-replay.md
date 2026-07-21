# Live Training Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a working training-rollout telemetry path from Stable-Baselines3 to a chart-first Trade RL Studio Live Training page.

**Architecture:** A sampled SB3 callback reduces environment `info` dictionaries into append-only JSONL telemetry under each seed artifact directory. Studio exposes cursor-based status/events endpoints scoped through known jobs. React polls those endpoints, buffers records independently from playback, and renders the approved dark chart-first replay UI.

**Tech Stack:** Python 3.12, Pydantic, FastAPI, Stable-Baselines3 callbacks, React 19, TypeScript, Vite, Vitest, Testing Library, SVG, CSS Grid.

## Global Constraints

- Keep direct exchange execution `NO-GO`.
- Keep the browser document free of vertical page scrolling.
- Do not change environment actions, rewards, reset behavior or artifact identity.
- Telemetry failure must not stop training.
- Invalid or missing telemetry must be shown explicitly, never interpolated as factual data.
- Use strict TypeScript and runtime response guards.
- Do not add a charting dependency; use accessible code-native SVG.

---

### Task 1: Telemetry record and append-only store

**Files:**
- Create: `trade_rl/telemetry/__init__.py`
- Create: `trade_rl/telemetry/training.py`
- Test: `tests/telemetry/test_training.py`

**Interfaces:**
- Produces: `TrainingTelemetryRecord`, `TrainingTelemetryStatus`, `TrainingTelemetryPage`, `TrainingTelemetryWriter.append(record)`, `read_training_telemetry(path, after_sequence, limit)`.

- [ ] **Step 1: Write failing store tests**

```python
from pathlib import Path
from trade_rl.telemetry.training import TrainingTelemetryRecord, TrainingTelemetryWriter, read_training_telemetry


def test_writer_and_cursor_reader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    writer = TrainingTelemetryWriter(path, flush_every=1)
    writer.append(TrainingTelemetryRecord.sample(sequence=1))
    writer.append(TrainingTelemetryRecord.sample(sequence=2))
    writer.close()
    page = read_training_telemetry(path, after_sequence=1, limit=10)
    assert [item.sequence for item in page.items] == [2]
    assert page.next_sequence == 2
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/telemetry/test_training.py -q`
Expected: import failure because the telemetry package does not exist.

- [ ] **Step 3: Implement immutable JSON-native records and JSONL append/read**

The reader validates monotonic sequence values, emits an explicit `gap` record for malformed lines and caps `limit` to 2,000. The writer uses line-buffered UTF-8 output and never rewrites prior records.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `uv run pytest tests/telemetry/test_training.py -q`
Expected: all telemetry store tests pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/telemetry tests/telemetry/test_training.py
git commit -m "feat: add append-only training telemetry"
```

### Task 2: SB3 rollout telemetry callback

**Files:**
- Create: `trade_rl/integrations/training_telemetry.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/integrations/test_training_telemetry.py`

**Interfaces:**
- Consumes: `TrainingTelemetryWriter`.
- Produces: `build_training_telemetry_callback(output_path, seed, sample_every=32)` and `compose_callbacks(*callbacks)`.

- [ ] **Step 1: Write failing callback tests**

Create a fake SB3 callback context with vectorized `infos`, actions and rewards. Assert normal HOLD-like records are sampled, position/risk/terminal records bypass sampling, and serialization contains no NumPy objects.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/integrations/test_training_telemetry.py -q`
Expected: module import failure.

- [ ] **Step 3: Implement reducer and callback**

Reduce `ResidualMarketEnv.step()` info fields including `submitted_target`, `executed_target`, `hybrid_execution.book.weights`, `portfolio_value_after`, `reward_total_scaled`, `drawdown_after`, `interval_cost`, `hybrid_risk.reasons`, `emergency_deleverage`, termination and current market index. Read OHLC and timestamp from the unwrapped environment dataset when available. Catch writer exceptions, close the writer at training end and return `True` from all callback hooks.

- [ ] **Step 4: Compose with checkpoint callback**

Modify `StableBaselines3Backend.train()` so `model.learn()` receives both callbacks. Put telemetry under `output_path.parent / "telemetry" / "training-telemetry.jsonl"`.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/integrations/test_training_telemetry.py tests/integrations/test_sb3_training.py -q`
Expected: callback and existing training tests pass.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/integrations tests/integrations/test_training_telemetry.py
git commit -m "feat: emit sampled rollout telemetry"
```

### Task 3: Studio telemetry contracts and API

**Files:**
- Modify: `trade_rl/studio/contracts.py`
- Create: `trade_rl/studio/telemetry.py`
- Modify: `trade_rl/studio/api.py`
- Test: `tests/studio/test_telemetry_api.py`

**Interfaces:**
- Produces: `TelemetryStatusResponse`, `TelemetryEventsResponse`, `StudioTelemetryReader.status(job)`, `StudioTelemetryReader.events(job, after_sequence, limit)`.

- [ ] **Step 1: Write failing API tests**

Construct a temporary `StudioSettings`, persisted `JobSummary` and telemetry JSONL. Assert status returns record count/last sequence, cursor requests return only later records, unknown jobs return 404 and artifact-root escapes are rejected.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/studio/test_telemetry_api.py -q`
Expected: endpoint 404 or missing contract imports.

- [ ] **Step 3: Implement reader and endpoints**

Add:

```text
GET /api/studio/jobs/{job_id}/telemetry/status
GET /api/studio/jobs/{job_id}/telemetry/events?after_sequence=0&limit=512
```

Resolve only through `JobSupervisor.get_job()`, join the declared artifact root/run staging locations safely, and return an unavailable status when the stream has not been created yet.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/studio/test_telemetry_api.py tests/studio/test_api.py -q`
Expected: telemetry and existing Studio API tests pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/studio tests/studio/test_telemetry_api.py
git commit -m "feat: expose training telemetry through Studio"
```

### Task 4: Frontend telemetry contract and polling hook

**Files:**
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/api/guards.ts`
- Modify: `studio/src/api/studioApi.ts`
- Create: `studio/src/live/useTrainingTelemetry.ts`
- Test: `studio/src/live/useTrainingTelemetry.test.tsx`

**Interfaces:**
- Produces: `TrainingTelemetryRecord`, `TelemetryStatusResponse`, `TelemetryEventsResponse`, `StudioApi.loadTelemetryStatus(jobId)`, `StudioApi.loadTelemetryEvents(jobId, afterSequence, limit)`, `useTrainingTelemetry(jobId, api)`.

- [ ] **Step 1: Write failing guard/API/hook tests**

Assert invalid numbers and sequence regressions are rejected, paused playback still appends remote records, and a job ID change clears the buffer/cursor.

- [ ] **Step 2: Verify RED**

Run: `npm test --prefix studio -- --run src/live/useTrainingTelemetry.test.tsx`
Expected: missing hook/types.

- [ ] **Step 3: Implement types, guards, API and polling hook**

Poll every 1,000 ms while a job is running, resume from the accepted server sequence, cap retained items to 2,048 and expose connection state, records, latest sequence and refresh.

- [ ] **Step 4: Verify GREEN**

Run: `npm test --prefix studio -- --run src/live/useTrainingTelemetry.test.tsx src/api/studioApi.test.ts`
Expected: focused frontend tests pass.

- [ ] **Step 5: Commit**

```bash
git add studio/src/data/types.ts studio/src/api studio/src/live
git commit -m "feat: add live telemetry client"
```

### Task 5: Chart-first Live Training workspace

**Files:**
- Create: `studio/src/pages/LiveTrainingPage.tsx`
- Create: `studio/src/live/MarketReplayChart.tsx`
- Create: `studio/src/live/ReplayControls.tsx`
- Create: `studio/src/live/AgentStatePanel.tsx`
- Create: `studio/src/live/MetricStrip.tsx`
- Create: `studio/src/live/EventTimeline.tsx`
- Modify: `studio/src/components/Sidebar.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/styles.css`
- Test: `studio/src/pages/LiveTrainingPage.test.tsx`
- Modify: `studio/src/App.test.tsx`

**Interfaces:**
- Consumes: `useTrainingTelemetry(jobId, api)` and `JobSummary`.
- Produces: the `live` workspace and accessible SVG replay chart.

- [ ] **Step 1: Write failing page/navigation tests**

Assert Live Training appears after Run Center, renders `NO-GO`, defaults to buffered replay and candle timeline, switches modes, pauses without discarding records, jumps the cursor and synchronizes event selection with the chart.

- [ ] **Step 2: Verify RED**

Run: `npm test --prefix studio -- --run src/pages/LiveTrainingPage.test.tsx src/App.test.tsx`
Expected: page/workspace missing.

- [ ] **Step 3: Implement the approved Concept B composition**

Use the existing shell. Render a dominant SVG candle/price chart, teal replay cursor, green/red markers, compact navigator, current-agent rail, four metric cards and event table. Include empty/offline/gap states and `research only` copy.

- [ ] **Step 4: Add design tokens and fixed-viewport layout**

Extend existing navy/charcoal tokens with cyan active states, green gains, red losses and amber risk. Bound all panel bodies and preserve document-level no-scroll behavior at 1536×1024 and 1440×900.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
npm test --prefix studio -- --run src/pages/LiveTrainingPage.test.tsx src/App.test.tsx
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add studio/src
git commit -m "feat: add chart-first live training replay"
```

### Task 6: Documentation and full verification

**Files:**
- Modify: `studio/README.md`
- Modify: `README.md`

**Interfaces:**
- Produces: local launch and telemetry behavior documentation.

- [ ] **Step 1: Document usage and limitations**

Document starting Studio, launching a training job, opening Live Training, buffered/live mode behavior, telemetry storage location and the research-only boundary.

- [ ] **Step 2: Run repository verification**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Expected: all checks pass with coverage at or above the repository threshold.

- [ ] **Step 3: Commit**

```bash
git add README.md studio/README.md
git commit -m "docs: explain live training replay"
```
