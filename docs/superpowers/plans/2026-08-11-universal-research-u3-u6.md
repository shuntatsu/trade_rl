# Universal Research U3-U6 Implementation Plan

> **Execution mode:** Superpowers TDD + executing-plans. This file now records the implemented path rather than the earlier provisional file layout.

**Goal:** Complete the universal single-instrument research path from the 206-channel target-local data contract through symbol-balanced BC/critic warm start, architecture ablation/zero-shot evidence, and fail-closed full-research integration documented in `START.md`.

**Architecture:** Maintain one identity-free `INSTRUMENT` policy with one scalar target-weight action. Training routes episodes across a frozen train-symbol partition; validation/test symbols never enter fit, Oracle teacher, BC, critic warm-start, or PPO preprocessing. U5/U6 consume immutable manifests and checkpoint/evidence identities. Software completion never implies zero-shot research success.

## Global Constraints

- Policy-facing symbol/action remain exactly `("INSTRUMENT",)` / `("target_weight:INSTRUMENT",)` with action shape `(1,)`.
- Universal market feature profile is 206 target-local channels plus 9 continuous point-in-time instrument descriptors; legacy 226-channel extended mode remains supported.
- Validation/test symbols are excluded from normalization, teacher generation, BC, critic warm-start, and PPO training preprocessing.
- Symbol-balanced normalization and BC give each train symbol equal contribution independent of history length.
- Ticker ID/hash/one-hot identity features remain prohibited.
- Research success is gated by real sealed unseen-symbol paired evidence; training artifacts intentionally report `research_success=false` until that evidence exists.
- Temporary GitHub Actions/TDD patch assets are removed before final verification.
- Do not merge `main`, force-push, or delete source branches without explicit user permission.

---

### Task 1: Close U3 data and runtime contracts

- [x] Add explicit 206-feature PostgreSQL materialization while preserving the legacy 226 path.
- [x] Fit availability-aware, train-only, symbol-balanced statistics and preserve missing/passthrough semantics.
- [x] Add the 9-channel causal continuous `instrument_context` contract without discrete identity leakage.
- [x] Bind shared statistics to concrete single-symbol datasets with exact schema/action/training identities.
- [x] Publish immutable train-symbol dataset artifacts and lazy-load at most one concrete child environment per worker.
- [x] Preserve subprocess worker indices through the compact sequence environment path.
- [x] Unify Universal feature-schema identity so normalizer/workflow artifacts cannot disagree.
- [x] Run focused Ruff/mypy/pytest contracts to GREEN.

Primary implementation surfaces include `trade_rl/data/universal_features.py`, `trade_rl/workflows/universal_training.py`, `trade_rl/workflows/universal_training_runner.py`, `trade_rl/rl/universal_instrument_context.py`, and the Universal routed environment modules.

### Task 2: Close U4 symbol-balanced BC and critic warm start

- [x] Keep legacy BC shuffling unchanged unless a Universal symbol-balanced provider is configured.
- [x] Implement deterministic symbol-balanced Universal BC batches and exact train/validation/purged closure.
- [x] Collect full identity-free Dict observations through the generic `INSTRUMENT` teacher surface.
- [x] Generate paired per-symbol Oracle episodes with one fixed BC seed and explicit train-fold stop.
- [x] Add a shared Oracle teacher-config identity guard before reusing targets across candidates/algorithms.
- [x] Run symbol-balanced BC, BC admission gates, critic-only warm start, and conservative joint warm start on train indices only.
- [x] Verify the critic-only phase leaves actor drift exactly zero and persist BC/critic evidence identities.
- [x] Connect the Universal pretraining hook into `StableBaselines3Backend` before PPO learning.

Primary implementation surfaces include `trade_rl/integrations/universal_behavior_cloning.py`, `trade_rl/integrations/universal_pretraining.py`, `trade_rl/integrations/universal_critic_warm_start.py`, `trade_rl/workflows/universal_teacher_runtime.py`, and `trade_rl/integrations/sb3_runtime.py`.

### Task 3: Build the real-data Universal training assembly

- [x] Materialize only the frozen train-symbol partition from PostgreSQL and reject scope/order mismatches.
- [x] Fit one shared Universal normalizer and bind it exactly to each single-symbol dataset.
- [x] Publish immutable per-symbol train datasets for spawn-safe lazy worker loading.
- [x] Bound Oracle episode sampling by both train start and train stop to prevent temporal leakage.
- [x] Build Oracle batches once per train symbol, close each concrete child, and verify dataset identity.
- [x] Build generic per-symbol teacher observations and combine them into one Universal pretraining bundle.
- [x] Rebind each training configuration into an immutable `UniversalTrainingRuntime` and routed environment identity.
- [x] Train all configured member seeds through the real SB3 backend and persist `universal-training.json` with `research_success=false`.
- [x] Publish a final `CheckpointManifest` for the exact completed policy, not merely an intermediate checkpoint.
- [x] Expose the maintained executable path through `scripts/run_universal_full_research.py` plus an explicit `module:function` runtime-factory seam carrying PostgreSQL/artifact/fold/digest context.

### Task 4: Make U5 architecture ablation and zero-shot evaluation executable

- [x] Maintain exactly four candidates: U-Small Direct, U-Medium Direct, U-Medium Gate, U-Large Direct.
- [x] Project architecture fields onto a common base training configuration and reject non-architecture condition drift.
- [x] Share only execution/risk-identical Oracle targets; rebind training contract, generic teacher, BC/critic artifacts, and SB3 environment per candidate.
- [x] Train all four candidates in deterministic enum order with identical split/normalizer/BC/budget/seed/cost identities.
- [x] Convert completed training runs to Stage A candidates using exact final checkpoint-manifest digests and policy-architecture identity.
- [x] Reuse the existing `StageAZeroShotEvaluationPlan` / `StageAZeroShotEvaluationOrchestrator` / production evaluator for candidate × symbol × fold × seed paired evidence.
- [x] Keep validation selection separate from the one-use sealed test and fail closed on incomplete paired evidence or insufficient pooled-symbol support.
- [x] Run focused U5 Ruff/mypy/pytest contracts to GREEN.

Primary orchestration: `trade_rl/workflows/universal_stage_a_training.py` and `trade_rl/workflows/universal_stage_a.py`.

### Task 5: Build U6 full-research integration and START.md entry point

- [x] Project only the U5-selected architecture onto PPO, Lagrangian PPO, and Discounted Lagrangian PPO configurations.
- [x] Require PPO gamma=1, Lagrangian gamma=1, and Discounted Lagrangian `0 < gamma < 1`; permit `discount_half_life_hours` only as the equivalent gamma parameterization.
- [x] Reject unexpected comparison-condition drift and require identical seed closure.
- [x] Share Oracle target batches once while rebinding algorithm-specific teacher/pretraining/training identities.
- [x] Train all three algorithms and create the complete required pair closure while leaving `completed_pairs` empty until economic evidence is produced.
- [x] Reuse `UniversalResearchManifest` / existing full-research state machine for evidence-gated transitions and resume identity closure.
- [x] Add strict executable entrypoint `scripts/run_universal_full_research.py` with `module:function` runtime-factory loading and explicit PostgreSQL/artifact/fold/digest context.
- [x] Add canonical U6 comparison configs with fixed BC seed and enabled critic warm start: `universal-u6-ppo.json`, `universal-u6-lagrangian.json`, `universal-u6-discounted.json`.
- [x] Document the exact command and `research_success=false` / sealed-test `NO-GO` boundary in `START.md`.
- [x] Run focused U6/docs/config Ruff/mypy/pytest contracts and CLI help smoke to GREEN.

Primary implementation: `trade_rl/workflows/universal_full_research_training.py`, `trade_rl/workflows/universal_full_research_entrypoint.py`, `scripts/run_universal_full_research.py`.

### Task 6: Architecture review, self-review, cleanup, and final verification

- [ ] **Step 1: Run a fresh architecture review of the complete `main...integration/universal-research-u3-u6` diff for dependency direction, duplicated Universal abstractions, train/test leakage, fail-open paths, process/memory boundaries, final-checkpoint correctness, and legacy 226 compatibility.**
- [ ] **Step 2: Fix every actionable architecture finding with a regression test and rerun the nearest suite; repeat until a fresh pass has no actionable finding.**
- [ ] **Step 3: Perform a reviewer-style self-review for naming, complexity, dead code, security/secrets, numerical/time/range boundaries, artifact identity, configuration drift, docs accuracy, and unnecessary files; fix and retest every actionable finding.**
- [ ] **Step 4: Delete every `.github/workflows/*temporary*.yml` and `.github/temporary/*` asset introduced solely for remote TDD, and verify none remain.**
- [ ] **Step 5: Run exact-final-head repository verification: Ruff check/format, mypy, import-linter, vulture, full pytest/coverage, frontend test/typecheck/build/viewport checks, workflow security, package/UV checks, compatibility matrix, training-image build, and PostgreSQL catalog checks.**
- [ ] **Step 6: Fetch all required GitHub Actions results for the exact final head and require success before calling the branch complete.**
- [ ] **Step 7: Compare every source-branch head against the consolidation branch and confirm each legitimate unique diff is present or intentionally superseded; do not delete source branches without permission.**
- [ ] **Step 8: Update PR #393 What/Why/design/non-goals/tests/risks and set Draft/Ready according to exact-head verification state. Do not merge `main`.**
