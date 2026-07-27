# Cost Critic Shared Feature Cache Verification

Date: 2026-07-27

## Scope

This verification covers the H2 performance-hardening change that removes duplicate policy feature-extractor execution from Cost Critic PPO and Lagrangian PPO without changing reward-policy, value, cost-learning, checkpoint, or constrained-optimization semantics.

## Verified software behavior

- Rollout Cost Critic values reuse the exact detached feature tensor produced by the policy forward.
- SB3 2.3.2 value bootstrap remains numerically identical while its value-side features are reused for Cost Critic bootstrap values.
- One post-PPO full-rollout feature cache is built per Cost Critic update.
- Cost Critic epochs, minibatches, and diagnostics index the same device-local cache.
- Cache rows match fresh evaluation-mode features with `rtol=0` and `atol=0`.
- Ordinary PPO versus Cost Critic PPO policy-state parity remains exact.
- Lagrangian multiplier freezing, actor advantage composition, dual updates, rollout evidence, and checkpoint identity remain unchanged.
- A cache-build failure restores policy training mode, Cost Critic training mode, and Torch RNG state.
- The cache is local to one update and is not serialized.

## TDD evidence

The initial RED contract observed 17 visible `policy.extract_features` calls for a four-step rollout with one PPO epoch and three Cost Critic epochs. The maintained GREEN contract observes six visible calls: four rollout policy forwards, one PPO minibatch evaluation, and one post-PPO cache build. SB3 `predict_values()` invokes `BaseModel.extract_features()` directly, so value-bootstrap extraction is validated separately through zero-tolerance value and feature equality tests.

A second RED contract proved that cache-build failure could leave training modes and Torch RNG mutated. The final implementation moves cache construction inside the guarded region and restores state on failure.

## Focused verification

The following groups passed on the implementation branch before this record was added:

- Cost Critic shared-feature cache regression tests;
- Cost Critic PPO behavior and checkpoint tests;
- Lagrangian PPO behavior, checkpoint, and round-trip tests;
- SB3 Cost Critic and Lagrangian backend tests;
- training-performance instrumentation tests;
- Ruff check and format check;
- Mypy for `trade_rl/integrations/cost_critic_ppo.py`.

## Remaining evidence boundary

The exact-head repository-wide CI run for the final documented head must pass before merge. No numeric CUDA speedup is claimed until H1 and H2 are measured under identical data, seed, model, and training configuration on the target RTX 4070 Ti SUPER.

Production status remains `NO-GO`.
