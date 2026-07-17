# Binance Metadata Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unlock the maintained 226-feature Docker GPU research workflow with an identity-bound frozen Binance execution snapshot while preserving strict signed history and adding conservative execution sensitivity.

**Architecture:** Capture current exchange metadata as immutable raw evidence, resolve all metadata through an explicit mode, and inject canonical evidence into existing dataset identity metadata. After selection, replay declared stricter execution scenarios with unchanged policies and normalizers rather than modifying returns.

**Tech Stack:** Python 3.12, dataclasses/StrEnum, urllib-based Binance transport, NumPy, pytest, Ruff, MyPy, Docker Compose, Stable-Baselines3/PyTorch.

## Global Constraints

- Never represent a current snapshot as historical or Binance-authenticated evidence.
- Keep `historical_signed` verification and coverage fail-closed.
- Preserve the Docker named volume and Binance Vision cache.
- Use TDD: each production behavior starts with a test observed failing for the missing behavior.
- Keep production status `NO-GO`; this change authorizes research execution only.

---

### Task 1: Identity-bound metadata evidence

**Files:**
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/integrations/binance.py`
- Test: `tests/data/test_market_builder.py`
- Test: `tests/integrations/test_binance.py`

**Interfaces:**
- Produces: `MarketDatasetBuilder.build(..., identity_provenance: Mapping[str, object] | None = None)`.
- Produces: `build_binance_market_dataset(..., metadata_evidence: Mapping[str, object] | None = None)`.

- [ ] Add a test proving different metadata mode/digest payloads yield different dataset IDs while canonical mapping order yields the same ID.
- [ ] Run the focused test and confirm it fails because `identity_provenance` is unsupported.
- [ ] Add the keyword argument, canonicalize it into the identity payload under `metadata_evidence`, and pass it through the Binance builder.
- [ ] Run focused builder/integration tests and Ruff/MyPy for the edited modules.

### Task 2: Exact exchange-information snapshot

**Files:**
- Modify: `trade_rl/integrations/binance.py`
- Test: `tests/integrations/test_binance.py`

**Interfaces:**
- Produces: immutable `BinanceExchangeInfoSnapshot(payload, raw_payload, source_uri, retrieved_at, raw_payload_sha256)`.
- Produces: `BinancePublicTransport.load_exchange_information_snapshot(*, market, mode=AUTO, clock=lambda: datetime.now(UTC))`.
- Existing `load_exchange_information()` delegates and retains its return contract.

- [ ] Add tests with mocked exact bytes proving one request, official URI, aware UTC capture time, raw byte preservation, SHA-256 correctness, parsed payload, and compatibility delegation.
- [ ] Run the tests and confirm missing snapshot API failures.
- [ ] Implement the immutable snapshot and one-fetch transport path; reject Vision mode and invalid object JSON.
- [ ] Run focused transport tests, Ruff, and MyPy.

### Task 3: Runner metadata modes and evidence artifacts

**Files:**
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `compose.training.yaml`
- Modify: `Dockerfile.training`
- Test: `tests/examples/test_binance_multitimeframe_full_assets.py`
- Test: `tests/examples/test_docker_training_assets.py`

**Interfaces:**
- Produces: CLI `--metadata-mode {historical_signed,frozen_snapshot,conservative_static}` with default `frozen_snapshot` for the Docker workflow.
- Produces: a canonical resolution containing metadata, optional effective-dated histories, evidence payload, and optional raw bytes.
- Consumes: Task 1 identity provenance and Task 2 snapshot API.

- [ ] Add failing tests for strict history preservation, frozen snapshot disclosure and raw-byte output, unknown/missing filters, one resolution reused by dataset A/B, and Docker mode wiring.
- [ ] Confirm tests fail for the absent mode dispatcher and CLI option.
- [ ] Implement mode parsing and resolution. Historical mode calls the existing signed loader unchanged. Frozen mode extracts positive static values and writes raw/canonical evidence before dataset construction. Conservative mode requires an explicit versioned static payload and positive stress factors.
- [ ] Pass canonical evidence into both dataset builds and assert repeat IDs/artifact digests match.
- [ ] Publish mode, source, digest, as-of, coverage, authentication, point-in-time status, policy version, and limitations in `exchange-info.json`, dataset result, and summary.
- [ ] Run focused runner/Docker tests, Ruff, and MyPy.

### Task 4: Conservative closed-loop execution replay

**Files:**
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Test: `tests/simulation/test_execution.py`
- Test: `tests/workflows/test_market_walk_forward.py`
- Test: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: versioned scenarios nominal, each-rule 2x, joint 2x, and joint 5x.
- Produces: immutable `execution-sensitivity.json` plus digest and summary reference.
- Consumes: selected and baseline policy artifacts without invoking a training backend.

- [ ] Add failing executor tests for multiplicative floors, zero-rule rejection, and sensitivity-only adverse tick rounding.
- [ ] Implement an evaluation-only overlay without changing nominal execution behavior or dataset arrays.
- [ ] Add failing walk-forward tests proving scenarios are plan-digest-bound, excluded from selection, replay selected and baseline, extend sealed access evidence, and publish complete immutable artifacts.
- [ ] Implement deterministic closed-loop OOS replay and metrics for return, drawdown, uplift, turnover, costs, trades, and rule burden percentiles.
- [ ] Require joint 2x positive selected return, nonnegative baseline uplift, and drawdown at most 20%; keep joint 5x report-only.
- [ ] Run focused simulation/workflow/runner tests, Ruff, and MyPy.

### Task 5: Documentation, full verification, and GPU launch

**Files:**
- Modify: `docs/BINANCE.md`
- Modify: `docs/operations/docker-gpu-full-training.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: operator commands and immutable Docker-volume evidence.

- [ ] Document all modes, limitations, snapshot artifacts, sensitivity gates, and strict-history opt-in without weakening signed-history claims.
- [ ] Run `uv run ruff check .`, `uv run mypy trade_rl`, and full pytest with coverage threshold 70%.
- [ ] Build a clean-provenance image from the committed source and run CUDA preflight.
- [ ] Start a unique detached `frozen_snapshot` generation using `/workspace/var/cache/binance-vision` and record container ID, image provenance, snapshot digest, dataset ID, feature count 226, four CUDA rollout environments, and policy parameter count.
- [ ] Monitor through walk-forward, selected training, confirmation, and conservative sensitivity; verify the final OOS return, baseline uplift, drawdown, and research gate from immutable artifacts.

