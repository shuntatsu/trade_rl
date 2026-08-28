# Causal Alpha V8 Exposure State Machine Implementation Plan

**Spec:** `docs/implementation-plans/specs/2026-08-26-causal-alpha-v8-exposure-state-machine-design.md`

1. Add V8 candidate, target config, and target-path contracts with exact digest
   binding and no tunable thresholds beyond the reused V6 contract.
2. TDD the robust exposure utility and state machine: entry confirmation,
   continuation, unsupported exit, no direct flip, cadence, liquidity, and risk
   behavior.
3. Add V8 replay, attribution, Signal, Selection, Admission, artifact store,
   checkpoint, runner, and pipeline contracts. Reuse numerical V7 science only
   through explicit V8 bindings.
4. Add immutable replay-leaf persistence and fail-closed resume reconstruction.
5. Assemble the DB-backed real-data stage entry and authored config without
   changing symbols, partitions, reward, cost, or gate values.
6. Run focused tests RED/GREEN, all V8 tests, Ruff, Mypy on Linux, provenance
   validation, and image smoke checks.
7. Build an exact-HEAD Docker image and execute Signal -> Selection -> Admission
   with durable per-replay checkpoints.
8. Monitor per-episode gross/net wealth, per-symbol wealth, transition
   attribution, turnover, and execution cost. Repair implementation defects only.
9. If admitted, bind the selected teacher to BC/RL and finish final learned-policy
   selection. Otherwise preserve the rejection and start the next evidence-led
   architecture iteration without opening the holdout.
10. Produce `report/causal-alpha-v8-<terminal>-20260826.md` with branch, commits,
    image ID/labels, artifact roots/digests, gate outcomes, and economic results.

