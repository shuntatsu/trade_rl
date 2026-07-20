# Trade RL Studio Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing no-scroll React shell to a local FastAPI runtime that safely catalogs real dataset/run artifacts and starts observable exploratory training jobs.

**Architecture:** Add `trade_rl.studio` above the workflow layer. The API reads only validated canonical artifacts, resolves every user-supplied path beneath configured local roots, and delegates training to the existing `trade-rl train run` command in a child process. The React app keeps one typed API adapter and replaces placeholder Data Lab, Experiments, and Run Center workspaces with bounded, no-document-scroll views.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic v2, subprocess-based local jobs, React 19, Vite 8, strict TypeScript, Vitest, Testing Library.

## Global Constraints

- Browser document scrolling remains disabled; large lists use bounded panes and paging.
- The runtime binds to `127.0.0.1` by default and exposes no exchange-order endpoint.
- All runs remain `NO-GO`; only exploratory `train run` jobs may be started.
- Dataset and run summaries come from validated canonical artifacts, never from unverified filenames alone.
- User paths must be relative to configured roots and must not escape them.
- TypeScript never reimplements training, validation, or artifact identity logic.
- Job state is persisted under `var/studio/jobs` and logs remain available after failure.

---

### Task 1: Runtime contracts, settings, and artifact catalog

**Files:**
- Create: `trade_rl/studio/__init__.py`
- Create: `trade_rl/studio/settings.py`
- Create: `trade_rl/studio/contracts.py`
- Create: `trade_rl/studio/catalog.py`
- Test: `tests/studio/test_catalog.py`
- Modify: `pyproject.toml`
- Modify: `.importlinter`

**Interfaces:**
- Produces: `StudioSettings.from_environment(project_root: Path | None)`, `StudioCatalog.list_datasets()`, `StudioCatalog.list_runs()`, and `StudioCatalog.overview(jobs)`.
- Dataset summaries expose `id`, `name`, `relative_path`, symbols, bar/feature counts, cadence, UTC range, validation state, and validation error.
- Run summaries expose `id`, `relative_path`, run kind, algorithm, dataset ID, created/completed timestamps, file count, production status, validation state, and optional walk-forward metrics.

- [ ] Write tests that publish a canonical dataset, build a canonical run manifest, add one corrupt directory, and assert that valid and invalid records are separated without path escape.
- [ ] Run `pytest -q tests/studio/test_catalog.py` and verify failure because `trade_rl.studio.catalog` does not exist.
- [ ] Implement settings, Pydantic response contracts, safe path resolution, dataset validation, run validation, and optional `walk-forward.json` metric extraction.
- [ ] Run the catalog tests and verify they pass.

### Task 2: Persistent local training job supervisor

**Files:**
- Create: `trade_rl/studio/jobs.py`
- Test: `tests/studio/test_jobs.py`

**Interfaces:**
- Produces: `TrainingJobRequest`, `JobSupervisor.submit_training(request)`, `list_jobs()`, `get_job(job_id)`, `cancel(job_id)`, and `tail_log(job_id, limit)`.
- Worker command is a fixed argument vector invoking `trade_rl.cli.main(["train", "run", ...])`; no shell command strings are accepted.
- States are `queued`, `running`, `succeeded`, `failed`, `cancelling`, and `cancelled`.

- [ ] Write tests using an injected process factory to verify command arguments, persisted JSON, successful/failed reconciliation, duplicate run rejection, cancellation, and bounded log tailing.
- [ ] Run `pytest -q tests/studio/test_jobs.py` and verify RED.
- [ ] Implement atomic job records, root-contained path resolution, subprocess spawning, poll-based reconciliation, process termination, and log retention.
- [ ] Run the job tests and verify GREEN.

### Task 3: FastAPI application and local CLI entrypoint

**Files:**
- Create: `trade_rl/studio/api.py`
- Create: `trade_rl/studio/cli.py`
- Test: `tests/studio/test_api.py`
- Test: `tests/studio/test_cli.py`
- Modify: `trade_rl/cli/__init__.py`

**Interfaces:**
- Produces endpoints `GET /api/studio/overview`, `GET /api/studio/datasets`, `GET /api/studio/runs`, `GET /api/studio/jobs`, `GET /api/studio/jobs/{id}`, `GET /api/studio/jobs/{id}/log`, `POST /api/studio/jobs/training`, and `POST /api/studio/jobs/{id}/cancel`.
- Produces command `trade-rl studio start --host 127.0.0.1 --port 8765 --project-root .`.

- [ ] Write TestClient tests for validated catalog payloads, job submission, unknown IDs, cancellation, local-only CORS behavior, and `NO-GO` status.
- [ ] Write CLI tests that monkeypatch Uvicorn and assert default local binding and explicit rejection of non-loopback hosts unless `--allow-remote` is supplied.
- [ ] Run the focused tests and verify RED.
- [ ] Implement dependency-injected app construction and lazy CLI dispatch so importing `trade_rl.cli` does not import FastAPI.
- [ ] Run focused tests and verify GREEN.

### Task 4: Typed frontend runtime adapter and real workspaces

**Files:**
- Modify: `studio/src/data/types.ts`
- Modify: `studio/src/api/studioApi.ts`
- Modify: `studio/src/api/studioApi.test.ts`
- Create: `studio/src/pages/DataLabPage.tsx`
- Create: `studio/src/pages/ExperimentsPage.tsx`
- Create: `studio/src/pages/RunCenterPage.tsx`
- Create: `studio/src/pages/RuntimePages.test.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/styles.css`

**Interfaces:**
- Produces `loadDatasets`, `loadRuns`, `loadJobs`, `submitTrainingJob`, `cancelJob`, and `loadJobLog`.
- Data Lab shows validated/invalid datasets in a paged split view.
- Experiments selects an existing config and dataset, validates the run ID client-side, and submits an exploratory job.
- Run Center refreshes jobs, shows persisted status, supports cancellation, and displays a bounded log pane.

- [ ] Add failing adapter and interaction tests for API payload validation, page rendering, job submission, selection, cancellation, and empty/error states.
- [ ] Run `npm test -- --run src/api/studioApi.test.ts src/pages/RuntimePages.test.tsx` and verify RED.
- [ ] Implement typed adapters and compact no-scroll pages; keep dashboard demo fallback only when the API is unavailable.
- [ ] Run frontend tests and verify GREEN.

### Task 5: Startup integration, documentation, and complete verification

**Files:**
- Modify: `studio/vite.config.ts`
- Modify: `studio/README.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Delete: `.github/workflows/studio-source-export.yml`

**Interfaces:**
- Vite dev server proxies `/api` to `http://127.0.0.1:8765`.
- CI installs the `studio` Python extra, runs studio Python tests, installs frontend dependencies, and runs tests/typecheck/build.

- [ ] Add Vite proxy and exact two-terminal startup instructions.
- [ ] Run `pytest -q tests/studio`, `ruff check trade_rl/studio tests/studio`, `mypy trade_rl/studio`, and `lint-imports`.
- [ ] Run `npm test -- --run`, `npm run typecheck`, `npm run build`, and `npm run check:layout` from `studio/`.
- [ ] Start the API and built frontend against temporary real artifacts, verify the Data Lab → Experiments → Run Center path, and capture a final screenshot.
- [ ] Remove the temporary source-export workflow and confirm the branch diff contains only product code, tests, CI, and documentation.
