# Nautilus Historical Engine Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Execute factual single-instrument Stage A historical intervals through the pinned NautilusTrader `BacktestEngine`, bind actual position/funding evidence, and expose the candidate through an opt-in RL dual-shadow runtime without changing legacy authority.

**Architecture:** Stage A workflow identity remains separate from Nautilus. The integration runner consumes framework-neutral target intervals, reuses `TargetExposureController` and the existing order adapter, observes actual fills, snapshots actual positions, and settles funding only after candidate/factual position agreement. A fresh-process full-prefix runner remains the deterministic reference implementation. RL execution uses a spawned episode worker which owns one pinned `BacktestEngine` and advances it with `run(streaming=True)` using only each new interval; the parent process contains no Nautilus engine state. The RL layer depends only on a framework-neutral observer protocol; Nautilus remains in `integrations`.

**Tech Stack:** Python 3.12, `nautilus_trader==1.230.0`, SB3 2.3.2, pytest, Ruff, MyPy, Import Linter, GitHub Actions `Nautilus Capability`.

## Global Constraints

- Maintained instrument: `BTCUSDT-PERP.BINANCE` / dataset symbol `BTCUSDT` only.
- Nautilus runtime: exactly `1.230.0`.
- OMS: `NETTING`; account type: `MARGIN`.
- Production remains `NO-GO`.
- Stage A actions are single-instrument target exposures in `[-1, 1]`.
- Target activation is causal: the first interval open quote must be observed before reconciliation.
- Sign reversals reduce to flat first and open the opposite side only after terminal fill evidence.
- Target-to-quantity conversion reuses `TargetExposureController`.
- Funding remains integration-boundary settlement; native Python-engine funding is not claimed.
- Fill-level equity is not synthesized.
- Synthetic fixtures may prove contracts and measure CI microbenchmark overhead, but they are not representative real historical promotion evidence.

### Task 1: Generic historical BacktestEngine target runner

- [x] RED: opening/flat round trip contract.
- [x] RED: safe sign-flip contract.
- [x] RED: actual candidate position snapshots at factual boundaries.
- [x] GREEN: causal projected OHLC quote replay through the pinned `BacktestEngine`.
- [x] GREEN: deferred sign-flip re-plan after the reducing fill.
- [x] GREEN: actual `OrderFilled` events canonicalized with the existing trace adapter.
- [x] Verify in an isolated-process `Nautilus Capability` step.

Implemented in `trade_rl/integrations/nautilus/historical_execution.py` with tests in `tests/integrations/test_nautilus_historical_execution.py`.

### Task 2: Stage A historical execution bridge and funding settlement

- [x] Require single-value Stage A actions.
- [x] Use factual `equity_before` as the decision-equity anchor.
- [x] Settle candidate funding from actual candidate position snapshots and factual mark/rate/multiplier inputs.
- [x] Fail closed when candidate and factual funding quantities disagree.
- [x] Keep funding records separate from fill signatures.
- [x] Verify in an isolated exact-wheel Nautilus step.

Implemented in `trade_rl/workflows/stage_a_nautilus_historical_replay.py` with tests in `tests/workflows/test_stage_a_nautilus_historical_execution.py`.

### Task 3: Fresh-process historical execution reference runtime

- [x] RED: repeated replay from one parent must not reuse the process-global Nautilus kernel.
- [x] GREEN: JSON request/result boundary and one fresh Python child per full-prefix replay.
- [x] Verify different worker PIDs with deterministic identical execution output.
- [x] Retain this runner as the deterministic reference for streaming differential tests.

Implemented in `trade_rl/integrations/nautilus/historical_subprocess.py`.

### Task 4: Persistent pinned-runtime streaming worker

- [x] Prove exact `1.230.0` state continuity across successive `run(streaming=True)` batches with one `BacktestEngine`, `clear_data()` between batches, and final `end()`.
- [x] Add a spawned persistent worker that owns one child PID and one `BacktestEngine` while the parent remains free of Nautilus engine state.
- [x] Send only the newly executed target interval on each worker call.
- [x] Preserve safe target reconciliation and cumulative canonical execution output.
- [x] Require exact cumulative parity against the fresh-process full-prefix reference for round trip, safe sign reversal, and same-side target changes.
- [x] Add deterministic close/join/terminate handling and fail-closed child message validation.

Implemented in `trade_rl/integrations/nautilus/historical_streaming.py` with exact-wheel tests in `tests/integrations/test_nautilus_streaming_execution.py` and `tests/integrations/test_nautilus_streaming_worker.py`.

### Task 5: Opt-in RL dual-shadow integration

- [x] Add a framework-neutral observer protocol in the RL execution layer.
- [x] Observe authoritative hybrid execution only; preserve legacy result and reward authority.
- [x] Add `NautilusEnvironmentDualShadow` in `integrations` using one persistent streaming child per episode.
- [x] Add an opt-in `ExecutionDualShadowResidualMarketEnv` wrapper without modifying the base environment contract.
- [x] Include candidate runtime identity in the wrapper environment digest.
- [x] Reset candidate state from the actual initial book and replace any previous episode worker.
- [x] Propagate environment `close()` to the candidate runtime so spawned children do not leak.
- [x] Fail closed for non-maintained dataset symbols.

### Task 6: Minimal RL training and differential proof

- [x] Run actual three-step SB3 PPO training with the streaming dual-shadow wrapper.
- [x] Run actual three-step Lagrangian PPO training with the streaming dual-shadow wrapper.
- [x] Require the policy artifact and exact three timesteps.
- [x] Execute both in `Nautilus Capability` with `train-sb3` installed.
- [x] Persist Stage A structural/funding differential evidence without claiming fill-price equivalence.
- [x] Define and test exact cost-neutral economic normalization in settlement minor units without allowing it to override structural/funding mismatch.

### Task 7: Performance evidence and authority-promotion work

- [x] Record an observational legacy-versus-streaming CPU PPO throughput artifact on a deterministic synthetic BTCUSDT fixture without using timing as a flaky CI pass/fail threshold.
- [x] Keep that artifact explicitly non-authorizing with `performance_approved=false`.
- [x] Verified eight-step CI observation: legacy about `11.47 step/s`, streaming dual-shadow about `1.275 step/s`, elapsed slowdown about `8.99x`.
- [ ] Benchmark memory behavior and broader representative workloads.
- [ ] Define and review an explicit performance-approval threshold before `performance_approved` may become true.
- [x] Run differential dual-shadow replay on persisted representative **real** BTCUSDT historical windows.
- [x] Evaluate and persist structural, funding, and cost-neutral economic evidence on those representative windows.
- [x] Connect persisted promotion evidence to walk-forward, selected-final, sealed-test, export, and Studio reporting without silently changing the fail-closed authority default.
- [ ] Keep production `NO-GO` until reconciliation, secrets, kill switch, live controls, and authorization are implemented.

The checked-in representative CI evidence uses factual Binance USDⓈ-M BTCUSDT 15-minute windows selected at time quantiles `0.1`, `0.5`, and `0.9`, with quote-notional volume and factual funding-boundary evidence. These bounded CI fixtures prove the differential/evidence pipeline deterministically; they do not by themselves authorize production authority or replace a reviewed representative catalog run and signed authorization/confirmation artifacts.

### Final verification gate

The final PR head must satisfy all of the following after the last implementation/documentation change. These boxes are gate conditions rather than a mutable completion record; verification is reported from the exact final Git commit so marking the checklist itself does not create a new unverified head.

- [ ] Ruff and format.
- [ ] MyPy and Import Linter.
- [ ] Dead-code report.
- [ ] Full pytest/coverage and critical branch coverage.
- [ ] Ubuntu/Windows compatibility and training image.
- [ ] Exact-wheel `Nautilus Capability`, including streaming parity, PPO/Lagrangian smoke, throughput artifact, conformance, and deterministic digest.
- [ ] `PostgreSQL Catalog`.
- [ ] Same final PR head for all applicable checks.
