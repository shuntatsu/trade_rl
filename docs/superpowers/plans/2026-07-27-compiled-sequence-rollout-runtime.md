# Compiled Sequence Rollout Runtime Implementation Plan

> **For agentic workers:** Use test-driven development and execute this plan task-by-task. No production code is added before the RED contracts are observed.

**Goal:** Reduce sequence-policy CUDA overhead with an explicitly identity-bound runtime that compiles only the maintained sequence feature extractor and transfers reconstructed rollout sequences through pinned CPU staging with non-blocking CUDA copies.

**Architecture:** Add three training fields: `sequence_compile`, `sequence_compile_mode`, and `sequence_transfer_mode`. The fields are part of the training digest and are inactive unless the sequence encoder is enabled. The compact rollout buffer retains its existing synchronous path and adds an opt-in `pinned_non_blocking` path only for reconstructed sequence tensors. The SB3 adapter applies `nn.Module.compile()` in place to `SequenceAssetFeatureExtractor` after model construction or checkpoint load and records the exact runtime contract in `model-architecture.json`.

**Tech stack:** Python 3.12, PyTorch 2.3.1, Stable-Baselines3 2.3.2, NumPy, pytest.

## Global constraints

- Preserve reward, action, PPO, Cost Critic, Lagrangian, checkpoint, Serving, and production semantics.
- Runtime settings must be included in `ResidualTrainingConfig.digest_payload()`.
- Non-default sequence runtime settings are invalid when `sequence_encoder=False`.
- `sequence_compile=True` must fail closed unless the resolved model device is CUDA and the maintained feature extractor exposes an in-place `compile()` method.
- Compile only the top-level sequence feature extractor; do not compile the SB3 algorithm object, optimizer loop, environment, callbacks, or checkpoint code.
- Use `fullgraph=False` and `dynamic=False`; allow only maintained compile modes.
- Keep the synchronous transfer path as the default and compatibility fallback.
- The pinned path is explicit and uses pinned CPU staging followed by `.to(cuda, non_blocking=True)`.
- Reconstructed sequence tensors remain materialized once per rollout and reused across PPO minibatches.
- Apply the same runtime settings after checkpoint load.
- Record requested/effective compile and transfer settings in architecture evidence.
- Enable the runtime only in the maintained CUDA full-training configurations.
- Do not claim numeric speedup until H1/H2/H3 are measured under identical target-GPU conditions.
- Production remains `NO-GO`.

## Tasks

### Task 1: RED configuration and identity contracts

- Add tests proving the three fields are accepted only for sequence policies.
- Prove each field changes the training digest.
- Reject unsupported compile and transfer modes.
- Assert the maintained direct and walk-forward full configurations enable `reduce-overhead` compile and `pinned_non_blocking` transfer.

### Task 2: RED transfer contract

- Add a fake-CUDA unit test proving the pinned path allocates pinned CPU staging, copies the reconstructed NumPy tensor into it, and calls `.to(..., non_blocking=True)`.
- Preserve the synchronous path and one-materialization-per-rollout behavior.

### Task 3: RED compile contract

- Add a fake-module test proving in-place compile targets only `policy.features_extractor` with `mode`, `fullgraph=False`, and `dynamic=False`.
- Prove disabled compile does not touch the module.
- Prove CPU or missing compile support fails closed when compile is requested.

### Task 4: GREEN implementation

- Extend `ResidualTrainingConfig` validation and digest payload.
- Extend `IndexBackedDictRolloutBuffer` constructor, binding, and sequence conversion.
- Pass transfer settings through fresh construction and checkpoint resume.
- Add the SB3 sequence-runtime configurator and architecture evidence.
- Enable the maintained full CUDA configurations.

### Task 5: Verification

- Run focused runtime, config, compact-buffer, SB3, checkpoint, and full-config tests.
- Run Ruff, format, Mypy, import architecture, full pytest/coverage, critical coverage, Ubuntu, Windows, training-image, recovery/Serving, and CLI gates.
- Record exact-head evidence without claiming target-GPU speedup.
