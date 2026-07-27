# Compiled Sequence Rollout Runtime Verification

Date: 2026-07-27

## Scope

This verification covers H3 of the training performance-hardening sequence: an identity-bound sequence runtime for in-place feature-extractor compilation and explicit pinned, non-blocking CUDA transfer of reconstructed rollout sequences.

## Verified software behavior

- `sequence_compile`, `sequence_compile_mode`, and `sequence_transfer_mode` are validated and included in the training configuration digest.
- Non-default runtime settings are rejected when the sequence encoder is disabled.
- Requested compilation fails closed unless the resolved model device is CUDA and the feature extractor supports in-place compilation.
- Compilation targets only `policy.features_extractor` with `fullgraph=False` and `dynamic=False`.
- The synchronous sequence transfer path remains the default compatibility path.
- The accelerated path stages reconstructed NumPy tensors in pinned CPU memory and requests non-blocking transfer to CUDA.
- Reconstructed sequence tensors are still materialized once per rollout and reused across PPO minibatches.
- Checkpoint resume rebinds both the sequence reconstructor and the requested transfer mode.
- Runtime settings are written to `model-architecture.json`.
- Legacy or test-created rollout buffers without the new transfer attribute fall back to synchronous transfer.
- The maintained direct and walk-forward full CUDA configurations request `reduce-overhead` compilation and `pinned_non_blocking` transfer.

## TDD evidence

The initial RED run failed seven contracts because the configuration fields, full-config entries, transfer helper, and compile configurator did not exist.

The first GREEN run exposed one backward-compatibility defect: buffers reconstructed without calling the new constructor lacked `sequence_transfer_mode`. The existing compact-rollout performance test reproduced the failure. The final implementation defaults that legacy boundary to `synchronous`.

## Focused verification

The final focused run passed 82 tests, covering:

- sequence runtime configuration and digest identity;
- pinned and synchronous transfer contracts;
- compile target and CUDA fail-closed behavior;
- compact rollout reconstruction and performance evidence;
- SB3 training and active architecture;
- checkpoint behavior;
- maintained full-training assets.

Ruff check, Ruff format, and Mypy passed for the changed runtime modules.

## Remaining evidence boundary

Repository-wide exact-head CI must pass before merge. GitHub-hosted CI does not provide the target RTX 4070 Ti SUPER, so this change does not claim a numeric CUDA speedup or prove that `reduce-overhead` is the optimal compile mode for the full production-sized run. H1/H2/H3 must be measured under identical data, seed, model, and training configuration on the target GPU before performance acceptance.

Production status remains `NO-GO`.
