# Trade RL

Trade RL is being rebuilt as a lean research system for evaluating independent long/short trading strategies under causal market data, realistic execution, hard risk constraints, and unused-data evaluation.

The current design source of truth is:

- `docs/trade_rl_lean_redesign_20260908.md`

## Current direction

The repository is removing the accumulated U-series / Causal Alpha generation stack and the mandatory `teacher -> admission -> BC -> RL` pipeline.

The retained core is intentionally small:

1. point-in-time market data and deterministic features
2. one fill-based execution and accounting ledger
3. hard safety and execution feasibility
4. a replaceable strategy interface
5. shared walk-forward, unseen-data, and execution-stress evaluation

Three strategy families will be compared on the same core:

- simple rules
- forecast + controller
- teacher-free PPO

Unsupported families are removed instead of being kept as permanent compatibility layers.

## Status

- Research redesign: active
- Legacy generation cleanup: in progress
- Profitability claim: none
- Production/live order routing: not authorized

Software correctness, backtest evidence, generalization evidence, and production authorization are separate states.

## Development principle

Keep only mechanisms that are required to answer the research question correctly. Do not weaken causal timing, execution accounting, hard risk, or unused-data evaluation to make a strategy pass.
