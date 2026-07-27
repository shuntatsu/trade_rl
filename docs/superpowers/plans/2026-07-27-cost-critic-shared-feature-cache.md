# Cost Critic Shared Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:executing-plans. Every production change follows an observed RED failure.

**Goal:** Remove repeated policy feature-extractor execution from Cost Critic PPO and Lagrangian PPO while preserving exact reward-policy updates, Cost Critic inputs, diagnostics, checkpoint identity, and constrained-optimization semantics.

**Architecture:** During rollout collection, capture the exact feature tensor returned by the policy's normal `extract_features` path and pass a detached view to the Cost Critic instead of invoking the extractor again. Stable-Baselines3 2.3.2 value bootstrap bypasses that overridable method, so terminal and final value bootstrap explicitly reproduce the maintained `BaseModel.extract_features` → value MLP → value head path while sharing its exact value features. After the reward actor/value update, materialize one detached full-rollout feature tensor in policy evaluation mode. Every Cost Critic minibatch and diagnostic pass uses indexed rows from that immutable tensor. The cache exists only inside one Cost Critic update and is never serialized.

**Tech Stack:** Python 3.12, PyTorch 2.3.1, Stable-Baselines3 2.3.2, NumPy, pytest.

## Global constraints

- Preserve exact ordinary PPO policy-state parity with `rtol=0` and `atol=0`.
- Preserve the current Cost Critic feature contract: when `policy.extract_features()` returns a tuple, use the second tensor exactly as before; otherwise use the returned tensor.
- Do not approximate, cast, clone through CPU, or change dtype/device/order before the Cost Critic.
- Capture exactly one feature-extractor result per normal policy forward. Missing or ambiguous captures fail closed.
- Preserve the exact SB3 2.3.2 `predict_values()` computation for value bootstrap.
- Restore the exact prior instance attribute state even when the policy operation raises.
- The rollout feature cache is built only after the PPO actor/value update, with the policy in evaluation mode and under `torch.no_grad()`.
- Cost Critic optimization must not backpropagate into the policy feature extractor.
- Diagnostics must reuse the same post-PPO feature cache as Cost Critic optimization.
- Lagrangian PPO inherits the optimization without changing multiplier freezing, actor advantage composition, dual updates, or rollout evidence.
- Do not claim speedup until H1 evidence is populated on the target RTX 4070 Ti SUPER.
- Production remains `NO-GO`.

---

### Task 1: Lock rollout capture and restoration contracts

**Files:**
- Create: `tests/integrations/test_cost_critic_feature_cache.py`

- [ ] Write a RED test that runs one four-step Cost Critic PPO update and counts `policy.extract_features` calls. With one PPO epoch and any number of Cost Critic epochs, the optimized contract is exactly six visible calls: four rollout policy forwards, one PPO minibatch evaluation, and one full-rollout Cost Critic cache build. SB3 value bootstrap is tested separately because `predict_values()` directly invokes `BaseModel.extract_features()`.
- [ ] Write a RED test for `_run_policy_with_cost_features`: the captured tensor equals the policy's own feature result, is detached, and the exact prior instance-level `extract_features` binding is restored.
- [ ] Write a RED exception test proving restoration when the wrapped policy operation raises after feature extraction.
- [ ] Write a zero-tolerance test that `_predict_values_with_cost_features` matches SB3 `predict_values()` and yields the same Cost Critic feature tensor.
- [ ] Run the focused test and record the expected RED failure caused by missing shared-feature APIs or excessive extraction calls.

### Task 2: Lock cache identity and minibatch indexing

**Files:**
- Modify: `tests/integrations/test_cost_critic_feature_cache.py`

- [ ] Write a RED test that builds the post-PPO full-rollout cache, selects non-contiguous indices, and compares each cached row against fresh evaluation-mode `_cost_features` output with zero tolerance.
- [ ] Require one full-rollout extractor call regardless of `cost_n_epochs` and `cost_batch_size`.
- [ ] Verify the cache tensor is detached, finite, on the model device, and has exactly `n_steps * n_envs` rows.
- [ ] Verify diagnostics consume the supplied cache without another policy feature-extractor call.

### Task 3: Implement rollout and value-bootstrap feature sharing

**Files:**
- Modify: `trade_rl/integrations/cost_critic_ppo.py`

- [ ] Add one private feature-selection helper that preserves the existing tuple/tensor semantics.
- [ ] Add `_run_policy_with_cost_features(operation)` that temporarily wraps `policy.extract_features`, captures exactly one selected feature tensor, restores the exact prior attribute layout in `finally`, and returns `(operation_result, detached_features)`.
- [ ] Use that helper for normal rollout policy forward.
- [ ] Add `_predict_values_with_cost_features(observations)` that exactly mirrors SB3 2.3.2 value prediction and returns both values and detached value features.
- [ ] Use the explicit value helper for time-limit terminal bootstrap and final rollout bootstrap.
- [ ] Feed the shared detached tensors directly into `self.cost_critic` for cost-value prediction.
- [ ] Keep all action, value, log-probability, reward bootstrap, and cost-rollout storage behavior unchanged.

### Task 4: Implement one post-PPO rollout cache

**Files:**
- Modify: `trade_rl/integrations/cost_critic_ppo.py`

- [ ] Add `_build_cost_feature_cache()` over canonical flat rollout indices under policy evaluation mode and `torch.no_grad()`.
- [ ] Add `_cached_cost_features(cache, indices)` using device-local `torch.long` indices and `index_select`.
- [ ] Build the cache once at the start of `_train_cost_critic`, after PPO or Lagrangian actor/value training has completed.
- [ ] Replace every Cost Critic minibatch extractor call with indexed cache rows.
- [ ] Pass the same cache to diagnostic construction and prohibit a second extractor pass.
- [ ] Keep the cache local to the update and absent from save/load and checkpoint identity.

### Task 5: Preserve exact constrained-training behavior

**Files:**
- Modify: `tests/integrations/test_cost_critic_ppo.py`
- Modify: `tests/integrations/test_lagrangian_ppo.py` if required

- [ ] Re-run ordinary PPO versus Cost Critic PPO exact policy-state parity.
- [ ] Re-run zero-multiplier Lagrangian PPO parity.
- [ ] Re-run Cost Critic save/load, optimizer-state, diagnostics, rare-event support, and vector-environment tests.
- [ ] Verify actor/value and Cost Critic parameters remain finite and update counts remain unchanged.

### Task 6: Verification and evidence

**Files:**
- Create after verification: `docs/verification/2026-07-27-cost-critic-shared-feature-cache.md`

- [ ] Run Ruff, format, Mypy, import architecture, dead-code, focused constrained-PPO tests, full pytest/coverage, critical coverage, Ubuntu, Windows, training-image, recovery/Serving, and CLI gates.
- [ ] Record exact-head CI and test counts.
- [ ] State only the software result: duplicate extractor calls are removed and exact semantic gates pass.
- [ ] Do not claim a numeric CUDA speedup until a representative H1/H2 pair is measured on identical 4070 Ti SUPER hardware, data, seed, model, and training configuration.
