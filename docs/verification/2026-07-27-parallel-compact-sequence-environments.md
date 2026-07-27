# Parallel Compact Sequence Environments Verification

Date: 2026-07-27

## Scope

This verification covers H4 of the training performance-hardening sequence: parallel CPU environment stepping for multi-environment structured-sequence PPO without transferring full sequence tensors through process IPC.

## Verified software behavior

- `vector_environment_mode` is validated and included in the training configuration digest.
- The default `auto` mode preserves the previous in-process behavior.
- Explicit `subprocess` mode selects spawn-based workers.
- Structured-sequence subprocess workers suppress `SequencePolicyPlane` materialization and emit only current structured components plus `decision_index`.
- No `sequence_*` array crosses worker IPC.
- The parent process batches all current worker indices into one sequence reconstruction call.
- Compact terminal observations are batched and rehydrated before PPO or Cost Critic time-limit bootstrap sees them.
- The parent wrapper exposes the original full observation space.
- The maintained direct and walk-forward full CUDA configurations explicitly request subprocess mode.
- Architecture evidence distinguishes `direct`, `in_process`, `subprocess`, and `subprocess_compact_sequence`.
- Reward, action, execution, risk, episode sampling, observation-contract digest, PPO, Cost Critic, Lagrangian, checkpoint, Serving, and production semantics remain unchanged.

## TDD and review evidence

The initial RED contract stopped because the compact parent wrapper, compact observation APIs, materialization context, vector-mode setting, and backend selection did not exist.

The first GREEN focused run exposed two compatibility issues:

1. a test `VecEnv` did not implement the required `render_mode` attribute response;
2. the first `auto` implementation changed existing non-sequence multi-environment behavior.

The test fake was corrected and `auto` was restored to the exact prior in-process behavior. Only explicit `subprocess` requests activate the new path.

The final focused H4 suite passed 118 tests. It covers compact observation assembly, suppressed worker policy-plane materialization, batched current and terminal rehydration, backend mode selection, existing sequence observations, SB3 training, Cost Critic, Lagrangian, checkpoints, and maintained full configurations.

A separate real-process smoke starts two actual `SubprocVecEnv` workers with `spawn`, exchanges compact observations, rehydrates reset and step observations in the parent, and rehydrates terminal observations after worker auto-reset.

The first repository-wide run then passed 1,955 tests and failed only the existing environment-constructor decomposition budget because two transport-state assignments increased `ResidualMarketEnv.__init__` from 150 to 152 lines. The transport initialization was moved into a dedicated helper. The architecture budget, real spawn smoke, observation contract, Ruff, format, and Mypy all passed after that correction.

## Remaining evidence boundary

The corrected exact-head repository-wide CI must pass before merge. GitHub-hosted CI does not provide the target RTX 4070 Ti SUPER, so this change does not claim a numeric speedup. H1 through H4 must be benchmarked under identical data, seed, model, and training configuration on the target GPU before performance acceptance.

Production status remains `NO-GO`.
