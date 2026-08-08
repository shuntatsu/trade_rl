# Nautilus Historical Engine Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Execute factual single-instrument Stage A historical intervals through the pinned NautilusTrader `BacktestEngine`, bind actual position/funding evidence, and expose the candidate through an opt-in RL dual-shadow runtime without changing legacy authority.

**Architecture:** Stage A workflow identity remains separate from Nautilus. The integration runner consumes framework-neutral target intervals, reuses `TargetExposureController` and the existing order adapter, observes actual fills, snapshots actual positions, and settles funding only after candidate/factual position agreement. Repeated RL use crosses a fresh-process JSON boundary because pinned `1.230.0` has process-global kernel state. The RL layer depends only on a framework-neutral observer protocol; Nautilus remains in `integrations`.

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

### Task 3: Fresh-process historical execution runtime

- [x] RED: repeated replay from one parent must not reuse the process-global Nautilus kernel.
- [x] GREEN: JSON request/result boundary and one fresh Python child per replay.
- [x] Verify different worker PIDs with deterministic identical execution output.

Implemented in `trade_rl/integrations/nautilus/historical_subprocess.py`.

### Task 4: Opt-in RL dual-shadow integration

- [x] Add a framework-neutral observer protocol in the RL execution layer.
- [x] Observe authoritative hybrid execution only; preserve legacy result and reward authority.
- [x] Add `NautilusEnvironmentDualShadow` in `integrations`, replaying the target prefix in fresh children.
- [x] Add an opt-in `ExecutionDualShadowResidualMarketEnv` wrapper without modifying the base environment contract.
- [x] Include candidate runtime identity in the wrapper environment digest.
- [x] Reset candidate state from the actual initial book.
- [x] Fail closed for non-maintained dataset symbols.

### Task 5: Minimal RL training proof

- [x] Run actual three-step SB3 PPO training with the dual-shadow wrapper.
- [x] Run actual three-step Lagrangian PPO training with the dual-shadow wrapper.
- [x] Require the policy artifact and exact three timesteps.
- [x] Execute both in `Nautilus Capability` with `train-sb3` installed.

### Task 6: Remaining authority-promotion work

- [ ] Run differential dual-shadow replay on persisted representative BTCUSDT historical windows.
- [ ] Define the historical economic comparison contract for the Nautilus one-tick L1 spread versus the accelerated legacy cost model without weakening structural/funding checks.
- [ ] Replace or optimize prefix replay only after an exact pinned-runtime streaming/persistent-worker lifecycle test succeeds.
- [ ] Benchmark memory and throughput against the accelerated legacy backend.
- [ ] Persist full differential evidence and connect it to walk-forward, selected-final, sealed-test, export, and Studio reporting.
- [ ] Keep production `NO-GO` until reconciliation, secrets, kill switch, live controls, and authorization are implemented.

### Final verification gate

- [ ] Ruff and format.
- [ ] MyPy and Import Linter.
- [ ] Full pytest/coverage and critical branch coverage.
- [ ] Ubuntu/Windows compatibility and training image.
- [ ] Exact-wheel `Nautilus Capability` including PPO/Lagrangian smoke.
- [ ] `PostgreSQL Catalog`.
- [ ] Same final PR head for all checks.