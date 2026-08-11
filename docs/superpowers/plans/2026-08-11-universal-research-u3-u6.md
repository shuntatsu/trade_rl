# Universal Research U3-U6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the universal single-instrument research path from 206-channel target-local data through symbol-balanced BC/critic warm start, architecture ablation/zero-shot evidence, and a fail-closed full-research runner documented in `START.md`.

**Architecture:** Keep the maintained single-instrument policy contract (`INSTRUMENT` and one scalar target-weight action) while routing each episode to a concrete training symbol. Fit feature statistics only from the train-symbol partition, preassemble train-only multi-symbol teacher evidence, and inject Universal pretraining into the existing SB3 backend through an explicit hook so the legacy single-dataset BC path remains unchanged. U5/U6 consume immutable manifests and evidence rather than reaching back into training data implicitly.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3, PyTorch, PostgreSQL/psycopg, Ruff, mypy, pytest, GitHub Actions.

## Global Constraints

- Policy-facing symbol contract is exactly `("INSTRUMENT",)` with action name `("target_weight:INSTRUMENT",)` and action shape `(1,)`.
- Universal market feature profile is 206 target-local channels; the existing 226-channel extended profile remains backward compatible.
- Validation/test symbols must never enter normalization fitting, teacher generation, BC batches, critic warm-start batches, or PPO training preprocessing.
- Symbol-balanced normalization and BC must give train symbols equal contribution independent of raw history length.
- Instrument identity must not be exposed as ticker ID, hash, or one-hot; only continuous point-in-time descriptors are allowed.
- Research success is separate from software success. Software completion must not claim zero-shot research success without actual sealed-test evidence.
- Temporary GitHub Actions workflows used for remote TDD are deleted before completion.
- Do not merge to `main`, force-push, or delete source branches without explicit user permission.

---

### Task 1: Close U3 data and runtime contracts

**Files:**
- Modify: `trade_rl/integrations/postgres_market_dataset.py`
- Modify: `trade_rl/rl/sequence_normalization.py`
- Create/Modify: `trade_rl/rl/universal_instrument_context.py`
- Modify: `trade_rl/rl/universal_single_instrument_env.py`
- Modify: `trade_rl/rl/policies.py`
- Modify: `trade_rl/integrations/sb3_model_assembly.py`
- Test: `tests/integrations/test_postgres_market_dataset.py`
- Test: `tests/rl/test_universal_training_environment.py`
- Test: `tests/integrations/test_universal_sb3_model_assembly.py`

**Interfaces:**
- Consumes: `binance_universal_feature_specs(...)`, `SymbolBalancedStandardNormalizer`, `EpisodeRoutedSingleInstrumentEnv`.
- Produces: 206-channel PostgreSQL datasets, availability-aware normalization, 9-channel `instrument_context`, and an SB3-compatible one-action Universal sequence observation contract.

- [x] **Step 1: Write regression tests for explicit 206 feature selection and legacy 226 fallback.**
- [x] **Step 2: Verify the new tests fail against the 226-only PostgreSQL path.**
- [x] **Step 3: Add `feature_specs` support to PostgreSQL materialization without changing the default 226 path.**
- [x] **Step 4: Add availability-aware normalization so stored missing zeros never enter fitted statistics and transform back to zero.**
- [x] **Step 5: Add causal continuous instrument descriptors and expose them as `instrument_context`.**
- [x] **Step 6: Fuse `instrument_context` into sequence asset tokens without adding identity features.**
- [x] **Step 7: Route Universal sequence PPO through full Dict rollout storage instead of the single-dataset index-backed buffer.**
- [x] **Step 8: Run Ruff, mypy, and focused U3/runtime tests and commit only after GREEN.**

### Task 2: Close U4 symbol-balanced BC and critic warm start

**Files:**
- Modify: `trade_rl/learning/behavior_cloning.py`
- Modify: `trade_rl/integrations/behavior_cloning.py`
- Modify: `trade_rl/integrations/universal_behavior_cloning.py`
- Create: `trade_rl/integrations/universal_pretraining.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/integrations/test_universal_behavior_cloning.py`
- Test: `tests/integrations/test_universal_pretraining_bundle.py`
- Test: `tests/integrations/test_sb3_universal_pretraining_hook.py`
- Test: `tests/integrations/test_universal_critic_warm_start.py`

**Interfaces:**
- Consumes: `SupervisedPolicyDataset`, `BehaviorCloningSplit`, per-symbol critic return-to-go vectors.
- Produces: `UniversalPretrainingBundle`, `build_universal_pretraining_hook(bundle)`, and immutable Universal BC/critic evidence.

- [x] **Step 1: Write a failing regression test proving Universal BC must supply a deterministic train-only batch provider.**
- [x] **Step 2: Extend BC core with an optional `training_batch_provider` while preserving legacy shuffling when absent.**
- [x] **Step 3: Implement `SymbolBalancedBatchSampler.epoch_batches()` so every batch has equal symbol contribution and every train sample is covered each epoch.**
- [x] **Step 4: Add the SB3 Universal pretraining hook boundary after model construction and before PPO learning; bypass the legacy single-dataset teacher path only when the hook is configured.**
- [ ] **Step 5: Make `combine_symbol_teachers()` close exactly over each symbol's train/validation/purged split and concatenate critic targets in the same sample order.**
- [ ] **Step 6: Run symbol-balanced BC, enforce the BC reconstruction gate, then run critic-only and conservative joint warm start on exactly `bundle.split.train_indices`.**
- [ ] **Step 7: Verify `actor_max_abs_drift_critic_only == 0.0`, persist evidence digests, and run Ruff/format/mypy/focused pytest.**

Exact verification command:

```bash
uv run ruff format --check trade_rl/integrations/universal_pretraining.py trade_rl/integrations/universal_behavior_cloning.py tests/integrations/test_universal_pretraining_bundle.py
uv run ruff check trade_rl/integrations/universal_pretraining.py trade_rl/integrations/universal_behavior_cloning.py tests/integrations/test_universal_pretraining_bundle.py
uv run mypy trade_rl/integrations/universal_pretraining.py trade_rl/integrations/universal_behavior_cloning.py
uv run pytest tests/integrations/test_universal_pretraining_bundle.py tests/integrations/test_universal_behavior_cloning.py tests/integrations/test_sb3_universal_pretraining_hook.py tests/integrations/test_universal_critic_warm_start.py -q
```

### Task 3: Build the real-data Universal training assembly

**Files:**
- Create: `trade_rl/workflows/universal_training.py`
- Create: `scripts/run_universal_training.py`
- Modify: `trade_rl/rl/sequence_normalization.py` or a focused Universal adapter module if dataset rebinding is required.
- Test: `tests/workflows/test_universal_training.py`

**Interfaces:**
- Consumes: `UniversalInstrumentArtifactBundle`, PostgreSQL indicator cache, `TrainingRunConfig`, train-symbol partition, 206 feature profile, shared normalizer, per-symbol `ResidualMarketEnv` factories.
- Produces: one routed Universal environment factory, one `UniversalPretrainingBundle`, one `StableBaselines3Backend` configured with the Universal pretraining hook, and member policy artifacts.

- [ ] **Step 1: Write a failing test that assembles two toy train symbols and proves the runner never opens a validation/test symbol during normalization, teacher generation, or PPO environment construction.**
- [ ] **Step 2: Add a shared-normalizer rebinding contract that preserves statistics while binding each single-symbol dataset/layout identity explicitly.**
- [ ] **Step 3: For every train symbol, build a 206-feature single-symbol dataset and Universal observation wrapper using the same shared statistics and descriptor schema.**
- [ ] **Step 4: Generate Oracle episode contracts/targets per train symbol and collect full Universal Dict observations, including sequence tensors and `instrument_context`, rather than compact `decision_index` observations that require one fixed dataset.**
- [ ] **Step 5: Build per-symbol `BehaviorCloningSplit` values, finite-horizon critic return-to-go vectors, and combine them with `combine_symbol_teachers()`.**
- [ ] **Step 6: Construct the routed PPO environment over train symbols only and inject `build_universal_pretraining_hook(bundle)` into `StableBaselines3Backend`.**
- [ ] **Step 7: Add `scripts/run_universal_training.py` with explicit paths for the Universal instrument artifact root, training config, PostgreSQL URL, output root, and architecture candidate.**
- [ ] **Step 8: Run Ruff, mypy, and `tests/workflows/test_universal_training.py` plus the U2/U3/U4 focused regression suites.**

### Task 4: Make U5 architecture ablation and zero-shot evaluation executable

**Files:**
- Modify: `trade_rl/rl/universal_architecture.py`
- Modify: `trade_rl/evaluation/universal_zero_shot.py`
- Create/Modify: `trade_rl/workflows/universal_architecture_ablation.py`
- Create: `scripts/run_universal_architecture_ablation.py`
- Test: `tests/rl/test_universal_research_u3_u6.py`
- Test: `tests/workflows/test_universal_architecture_ablation.py`

**Interfaces:**
- Consumes: fixed split/normalizer/BC/budget/seed/cost identities and candidate enum `U_SMALL_DIRECT`, `U_MEDIUM_DIRECT`, `U_MEDIUM_GATE`, `U_LARGE_DIRECT`.
- Produces: candidate-by-symbol-by-fold-by-seed paired evidence, deterministic bootstrap summaries, zero-shot gate result, and selected candidate identity.

- [ ] **Step 1: Write a failing contract test that rejects candidate comparisons when any fixed experiment identity differs.**
- [ ] **Step 2: Train/evaluate all four candidates through the same Universal training assembly and record candidate-specific architecture identity only.**
- [ ] **Step 3: Evaluate seen validation and unseen zero-shot symbols without allowing unseen symbols into training preprocessing.**
- [ ] **Step 4: Compute deterministic moving/block bootstrap evidence and require complete paired baseline evidence for every candidate/symbol/fold/seed row.**
- [ ] **Step 5: Fail closed when independent-symbol support is insufficient for pooled summaries; never manufacture random-effects evidence.**
- [ ] **Step 6: Run focused U5 tests, Ruff, and mypy.**

### Task 5: Build U6 full-research integration and START.md entry point

**Files:**
- Create: `scripts/run_universal_full_research.py`
- Modify: `trade_rl/workflows/universal_research.py`
- Modify: `START.md`
- Test: `tests/workflows/test_universal_full_research.py`
- Test: `tests/test_start_universal_full_training.py`

**Interfaces:**
- Consumes: frozen Universal instrument artifacts, selected U5 architecture evidence, training config, PostgreSQL URL, seed manifest, cost identity, normalizer identity, BC/critic evidence, paired baseline evidence.
- Produces: a fail-closed U6 research run directory and a command in `START.md` that invokes the real runner rather than a documentation-only pseudo path.

- [ ] **Step 1: Write a failing runner test that rejects missing or mismatched universe, seed, feature, normalizer, BC, cost, observation, or paired-baseline identities.**
- [ ] **Step 2: Orchestrate U3 data/normalization, U4 pretraining, selected U5 architecture training, baseline/control evaluation, robustness evidence, and U6 manifests from one command.**
- [ ] **Step 3: Make resume reuse only artifacts whose complete manifest/digest closure matches the requested run.**
- [ ] **Step 4: Add the exact `uv run python scripts/run_universal_full_research.py ...` invocation to `START.md`, including required PostgreSQL and artifact/config inputs.**
- [ ] **Step 5: Add a docs contract test that parses the documented command and confirms the referenced script/options exist.**
- [ ] **Step 6: Run focused U6/docs tests, Ruff, and mypy.**

### Task 6: Architecture review, self-review, cleanup, and final verification

**Files:**
- Review all files changed by PR #393.
- Delete all `.github/workflows/*temporary*.yml` files introduced solely for remote TDD.
- Update PR #393 body with What/Why/design/non-goals/tests/risks.

**Interfaces:**
- Consumes: completed U3-U6 implementation and tests.
- Produces: clean final branch head with no temporary workflows and verifiable CI evidence.

- [ ] **Step 1: Run architecture checks and inspect dependency direction, duplicated Universal abstractions, train/test leakage paths, fail-open behavior, and legacy 226 compatibility.**
- [ ] **Step 2: Fix every architecture finding with a regression test, then rerun the nearest suite. Repeat until a fresh architecture pass finds no actionable issue.**
- [ ] **Step 3: Perform a reviewer-style diff pass for naming, complexity, dead code, security, numerical precision, time/range boundaries, artifact identity, and unnecessary files. Fix every actionable finding and rerun the nearest suite.**
- [ ] **Step 4: Delete temporary workflows and verify the final tree contains none of them.**
- [ ] **Step 5: Run repository verification on the exact final head: Ruff check, Ruff format check, mypy, import/architecture checks, full pytest, frontend checks if CI requires them, build/training-image checks, and PostgreSQL catalog checks.**
- [ ] **Step 6: Fetch all GitHub Actions runs for the exact final head and require all required checks to succeed before calling the branch complete.**
- [ ] **Step 7: Compare every source branch against the consolidation branch and confirm each unique legitimate diff is either present or intentionally superseded; do not delete branches without user permission.**
- [ ] **Step 8: Update PR #393 as Draft/Ready according to verification state. Do not merge to `main`.**
