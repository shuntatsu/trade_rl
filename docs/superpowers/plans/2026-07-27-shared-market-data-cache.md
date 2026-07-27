# Shared Market Data Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate persistent Binance Vision archive volume, incremental missing-only synchronization, and a fail-closed training startup cache check.

**Architecture:** A package-level cache planner owns deterministic URL planning, cache inspection, and synchronization. A one-shot sync container writes the archive volume, while the trainer reads it only after bootstrap verifies complete coverage. A host launcher runs sync then trainer for every canonical training invocation.

**Tech Stack:** Python 3.12, Docker Compose, pytest, existing Binance integration.

## Global Constraints

- Keep PostgreSQL metadata-only; do not store numerical market payloads in PostgreSQL.
- Keep the maintained fixed research interval authoritative and reproducible.
- Download only missing Binance Vision archives.
- Trainer mounts market archives read-only and must not repair missing data.
- Fail before CUDA preflight when archive coverage is incomplete.

---

### Task 1: Deterministic Binance Vision cache planner

**Files:**
- Create: `trade_rl/integrations/binance_cache.py`
- Test: `tests/integrations/test_binance_cache.py`

**Interfaces:**
- Produces: `BinanceVisionCachePlan`, `BinanceVisionCacheReport`, `plan_binance_vision_cache`, `inspect_binance_vision_cache`, `sync_binance_vision_cache`.

- [ ] Write tests for deterministic kline/funding URL planning, completed funding-month selection, missing-only synchronization, and empty cached file rejection.
- [ ] Run the focused tests and confirm they fail because the module does not exist.
- [ ] Implement immutable plan/report dataclasses and cache operations using the existing Binance URL planners and `BinancePublicTransport` request/cache contract.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the planner and tests.

### Task 2: Sync CLI and fail-closed training bootstrap

**Files:**
- Create: `examples/binance-multitimeframe/sync_market_data.py`
- Create: `examples/binance-multitimeframe/training_bootstrap.py`
- Test: `tests/examples/test_market_data_sync.py`
- Test: `tests/examples/test_training_bootstrap.py`

**Interfaces:**
- Consumes: cache planner from Task 1 and maintained pipeline constants.
- Produces: sync JSON report and `run_bootstrap(cache_root, full_entrypoint)`.

- [ ] Write tests proving sync uses maintained symbols/timeframes/range and bootstrap checks cache before invoking the full entrypoint.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement the sync CLI, report persistence, and read-only bootstrap check with actionable failure text.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit CLI/bootstrap and tests.

### Task 3: Separate Docker volume ownership and canonical launcher

**Files:**
- Modify: `compose.training.yaml`
- Modify: `Dockerfile.training`
- Create: `scripts/run_docker_training.py`
- Test: `tests/test_training_compose_contract.py`
- Test: `tests/scripts/test_run_docker_training.py`

**Interfaces:**
- Produces: `market-data-sync` service, RO trainer archive mount, separate runs/teacher volumes, and sequential sync/trainer launcher.

- [ ] Write tests that parse the Compose contract and assert RW sync versus RO trainer mounts plus launcher command ordering.
- [ ] Run focused tests and confirm expected failures.
- [ ] Update Compose volumes/services, switch the image CMD to bootstrap, and implement the host launcher.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit Docker and launcher changes.

### Task 4: Operations documentation and regression verification

**Files:**
- Modify: `docs/operations/docker-gpu-full-training.md`
- Modify: `README.md`

**Interfaces:**
- Documents: canonical launcher, manual sync, direct-trainer fail-closed behavior, volume backup/removal, and fixed-range incremental semantics.

- [ ] Update operations documentation and README commands.
- [ ] Run Ruff on changed Python files.
- [ ] Run focused pytest suites for cache, bootstrap, launcher, Compose contract, existing Binance cache tests, and full-run entrypoint tests.
- [ ] Run broader relevant integration/example tests when available.
- [ ] Review the final diff for accidental network access from trainer and payload storage in PostgreSQL.
- [ ] Commit documentation and verification adjustments.
