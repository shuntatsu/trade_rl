# BC Teacher Rollout Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Oracle BC teacher-rollout process and environment-construction overhead without changing episode data, ordering, shard identity, or failure semantics.

**Architecture:** Keep the existing serial collector as the behavioral authority. Partition pending episode items into deterministic contiguous chunks containing at most eight episodes while still creating at least one task per available worker when enough episodes exist. Each process/thread task creates one environment, resets it independently for every episode in its chunk, returns episode datasets in contract order, and closes the environment once; the fork pool retains `maxtasksperchild=1` so memory isolation remains bounded.

**Tech Stack:** Python 3.12, NumPy, `multiprocessing.Pool`, `ThreadPoolExecutor`, pytest.

## Global Constraints

- Do not change teacher targets, observation schema, artifact schema, shard path, or episode ordering.
- Do not add a user-facing configuration knob; use one internal maximum of eight episodes per worker task.
- Preserve the serial path when `max_workers == 1` or the batch contains one episode.
- Preserve `maxtasksperchild=1` for the fork path.
- Keep Windows/non-fork behavior equivalent to Linux/fork behavior.
- Follow TDD: record the failing regression before production changes.

---

### Task 1: Define the environment-reuse contract

**Files:**
- Create: `tests/learning/test_episode_teacher_parallel_chunking.py`

**Interfaces:**
- Consumes: `collect_episode_teacher_rollout_parallel(environment_factory, batch, teacher_config_digest, max_workers)`.
- Produces: a regression proving eight episodes with two workers create exactly two environments on the non-fork path while preserving episode/action order.

- [ ] **Step 1: Write a deterministic fake environment and eight-episode Oracle batch.**
- [ ] **Step 2: Force the ThreadPool path by replacing `mp.get_all_start_methods()` with `("spawn",)`.**
- [ ] **Step 3: Assert the parallel dataset contains all episodes in order and the factory is called twice, not eight times.**
- [ ] **Step 4: Run the focused test in CI and verify it fails because the current implementation creates one environment per episode.**
- [ ] **Step 5: Commit the RED test as `test(bc): require chunked teacher rollout workers`.**

### Task 2: Collect bounded episode chunks per worker task

**Files:**
- Modify: `trade_rl/learning/episode_teacher_artifact.py`
- Test: `tests/learning/test_episode_teacher_parallel_chunking.py`

**Interfaces:**
- Produces: `_chunk_episode_items(items, worker_count)` and `_collect_isolated_episode_chunk(...)` internal helpers.
- Preserves: `_collect_isolated_episode(...)` compatibility wrapper and public `collect_episode_teacher_rollout_parallel(...)` signature.

- [ ] **Step 1: Add `_MAX_EPISODES_PER_TEACHER_TASK = 8`.**
- [ ] **Step 2: Partition items using `chunk_size = min(8, ceil(item_count / worker_count))`.**
- [ ] **Step 3: Build one environment per chunk, collect each episode through the existing serial collector, and close once in `finally`.**
- [ ] **Step 4: Change fork and thread executors to submit chunks and persist each returned episode individually.**
- [ ] **Step 5: Keep fork `maxtasksperchild=1`, deterministic item order, and existing shard-resume behavior.**
- [ ] **Step 6: Run the focused test and existing episode teacher integration tests.**
- [ ] **Step 7: Commit as `perf(bc): batch teacher episodes per worker task`.**

### Task 3: Exact-head verification and PR completion

**Files:**
- Modify: PR description only.

**Interfaces:**
- Consumes: final branch head and GitHub Actions results.
- Produces: a Ready-for-review PR with exact verification evidence and explicit remaining risks.

- [ ] **Step 1: Run Ruff, Ruff format, MyPy, import-linter, dead-code, full pytest/coverage, Windows/Linux compatibility, and training-image checks on the same head.**
- [ ] **Step 2: Confirm the PR has no unresolved review threads and is not behind `main`.**
- [ ] **Step 3: Review the final diff for unrelated files, debug code, artifact changes, and accidental public API changes.**
- [ ] **Step 4: Update the PR body with RED/GREEN evidence, exact head, test counts, and the remaining lack of end-to-end GPU wall-time measurement.**
- [ ] **Step 5: Mark the PR Ready for review; do not merge it into `main` without explicit user authorization.**
