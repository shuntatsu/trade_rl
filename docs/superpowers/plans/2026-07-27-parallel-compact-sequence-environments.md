# Parallel Compact Sequence Environments Implementation Plan

> **For agentic workers:** Use test-driven development and systematic debugging. Do not add production code before observing the RED contracts.

**Goal:** Parallelize CPU-heavy environment stepping for multi-environment sequence PPO without sending large sequence tensors through process IPC or constructing a full sequence policy plane in every worker.

**Architecture:** Add an identity-bound `vector_environment_mode`. The maintained full CUDA configurations use `subprocess`. A parent probe retains the complete sequence contract and read-only `SequencePolicyPlane`. Spawned workers construct the same economic environment and full observation contract, but suppress policy-plane materialization and emit only current structured state plus `decision_index`. A parent `VecEnvWrapper` batches those indices, restores all sequence channels from the probe-bound reconstructor, and also restores compact terminal observations before SB3 sees them.

**Tech stack:** Python 3.12, Stable-Baselines3 2.3.2, Gymnasium 0.29.1, NumPy, pytest.

## Global constraints

- Preserve reward, action, execution, risk, episode sampling, observation-contract digest, PPO, Cost Critic, Lagrangian, checkpoint, Serving, and production semantics.
- `vector_environment_mode` is validated and included in the training digest.
- `auto` preserves the previous behavior: direct for one environment, subprocess for non-sequence multi-env, and in-process vectorization for sequence multi-env.
- `in_process` explicitly selects `DummyVecEnv` for multi-env training.
- `subprocess` selects spawn-based workers. For sequence policies it must use compact worker observations and parent rehydration.
- Workers must not materialize `SequencePolicyPlane` and must not call sequence builders while compact transport is enabled.
- Worker observations contain exactly current structured components and `decision_index`; no `sequence_*` keys cross IPC.
- Parent rehydration batches all current environment indices into one reconstructor call.
- `info["terminal_observation"]` values are batched and rehydrated before PPO time-limit bootstrap or Cost Critic bootstrap.
- The parent wrapper exposes the original full observation space.
- Spawn remains explicit for CUDA/process safety and Windows-compatible semantics.
- Existing synchronous/in-process behavior remains available and is the default for non-maintained configurations.
- Maintained direct and walk-forward full CUDA configurations request subprocess mode.
- Architecture evidence records the effective mode as `direct`, `in_process`, `subprocess`, or `subprocess_compact_sequence`.
- No numeric speedup is claimed until H1–H4 are measured under identical target-GPU conditions.
- Production remains `NO-GO`.

## Tasks

### Task 1: RED configuration identity

- Add tests for `vector_environment_mode` validation and digest identity.
- Assert maintained full configurations request `subprocess`.

### Task 2: RED compact observation contract

- Add a compact assembler test proving current components and `decision_index` match the full observation while the sequence plane is not queried.
- Add environment transport-mode tests proving the exposed worker observation space excludes all `sequence_*` keys and can be restored to the full space.
- Add a construction-context test proving workers suppress sequence policy-plane materialization.

### Task 3: RED parent rehydration

- Add pure tests for batched current-observation restoration.
- Add terminal-observation tests proving multiple terminal observations use one batch reconstruction and preserve unvectorized component shapes.
- Add a wrapper test proving reset and step outputs expose the full observation space.

### Task 4: RED backend selection

- Prove sequence `subprocess` mode uses spawn workers through the compact factory and parent wrapper.
- Prove `auto` retains in-process sequence behavior.
- Prove architecture evidence records the effective vector mode.

### Task 5: GREEN implementation

- Extend `ResidualTrainingConfig` and maintained JSON configurations.
- Add sequence-policy-plane materialization context control.
- Add compact observation assembly and the environment runtime transport toggle.
- Add `ParallelSequenceVecEnv`, batched rehydration helpers, and compact worker factory.
- Wire the backend and checkpoint-independent architecture evidence.

### Task 6: Verification

- Run focused environment, observation, vectorization, SB3, Cost Critic, Lagrangian, checkpoint, and full-config tests.
- Run Ruff, format, Mypy, import architecture, full pytest/coverage, critical coverage, Ubuntu, Windows, training-image, recovery/Serving, and CLI gates.
- Record exact-head evidence without claiming target-hardware speedup.
