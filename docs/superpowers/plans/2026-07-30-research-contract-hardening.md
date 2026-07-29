# Research Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the maintained dataset, Oracle BC, publication, cache, policy telemetry and documentation paths fail closed under one consistent research contract.

**Architecture:** Introduce one data-layer economic-semantics constructor consumed by both Vision and PostgreSQL. Strengthen framework-neutral BC evaluation with deterministic bootstrap evidence, move maintained preset validation ahead of artifact publication, add raw-cache sidecars, and extend existing telemetry/documentation contracts without changing the hierarchical actor checkpoint layout.

**Tech Stack:** Python 3.12, NumPy, pytest, Stable-Baselines3 integration boundary, GitHub Actions, Import Linter.

## Global Constraints

- Preserve `training_run_config_v3` as the only maintained training schema.
- Preserve `hierarchical_gate_target_v1` checkpoint compatibility while describing the continuous scalar as change intensity.
- Preserve deterministic content identities and fail closed on missing evidence.
- Do not add exchange account authentication or order submission.
- Keep runtime and training imports separated from offline signers.

---

### Task 1: Common economic semantics

**Files:**
- Create: `trade_rl/data/economic_semantics.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/integrations/postgres_market_dataset.py`
- Test: `tests/data/test_economic_semantics.py`
- Test: `tests/integrations/test_postgres_market_dataset.py`

**Interfaces:**
- Produces: `MarketEconomicSemantics` and `build_market_economic_semantics(...)`.
- Consumes: canonical `InstrumentContract`, timestamps, source row/tradability/availability arrays and close/mark/index observations.

- [ ] Write tests proving inactive periods, point-in-time execution rules, fees, spread, participation, borrow and side permissions are explicit and immutable.
- [ ] Write a PostgreSQL/Vision parity test over every economic array.
- [ ] Run focused tests and verify RED because no shared constructor exists and PostgreSQL relies on defaults.
- [ ] Implement the frozen contract and route both builders through it.
- [ ] Run focused tests and verify GREEN.
- [ ] Commit `feat: unify market economic semantics`.

### Task 2: Oracle BC causal gate

**Files:**
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `trade_rl/learning/episode_oracle_bc.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: maintained target-weight training JSON profiles
- Test: `tests/learning/test_learning_evaluation.py`
- Test: `tests/learning/test_episode_teacher_integration.py`
- Test: `tests/rl/test_training_config_active_fields.py`
- Test: `tests/examples/test_target_weight_constrained_growth_profiles.py`

**Interfaces:**
- Produces: deterministic one-sided bootstrap upper regret bound in holdout evidence.
- Adds config fields: `behavior_cloning_causal_holdout_bootstrap_resamples` and `behavior_cloning_causal_holdout_confidence_level`.

- [ ] Write tests for insufficient 30-trade support, zero required relative improvement, missing/non-finite confidence evidence and upper-bound failure.
- [ ] Verify RED.
- [ ] Implement deterministic bootstrap evidence and mandatory gate metric.
- [ ] Set maintained profiles to positive relative improvement, 30 trades, 2,000 resamples and 95% confidence.
- [ ] Verify GREEN.
- [ ] Commit `feat: strengthen oracle BC causal gate`.

### Task 3: Validate before publication

**Files:**
- Modify: `examples/binance-multitimeframe/full_research_pipeline.py`
- Test: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: `validate_maintained_dataset_preset(dataset, *, use_postgres)`.

- [ ] Write a test that replaces publication with a failing sentinel and supplies a wrong-preset dataset.
- [ ] Verify RED because publication currently happens first.
- [ ] Extract validation and call it before `publish_market_dataset_artifact`.
- [ ] Verify GREEN.
- [ ] Commit `fix: validate dataset before publication`.

### Task 4: Content-attested Vision cache

**Files:**
- Modify: `trade_rl/integrations/binance.py`
- Test: `tests/integrations/test_binance.py`
- Modify: `docs/BINANCE.md`

**Interfaces:**
- Produces adjacent `<digest>.json` cache evidence using schema `binance_vision_raw_cache_v1`.

- [ ] Write tests for sidecar creation, cache reuse, byte tampering, sidecar tampering and payload-only legacy entries.
- [ ] Verify RED.
- [ ] Implement canonical atomic sidecar writes and read-time SHA-256/size/URL verification.
- [ ] Verify GREEN.
- [ ] Commit `feat: attest Binance Vision cache content`.

### Task 5: Change-intensity telemetry

**Files:**
- Modify: hierarchical policy diagnostics/telemetry modules located by existing actor probes.
- Modify: environment/training info compaction only where required.
- Test: corresponding policy and training telemetry tests.

**Interfaces:**
- Preserve serialized `hierarchical_gate_target_v1` fields.
- Add metrics for `change_intensity`, sampled action, post-mask action, submitted target and effective filled weight.

- [ ] Write tests distinguishing deterministic composed action from exploration-sampled and downstream effective actions.
- [ ] Verify RED.
- [ ] Add stage-specific telemetry without changing policy outputs or checkpoint identity.
- [ ] Verify GREEN.
- [ ] Commit `feat: report effective hierarchical actions`.

### Task 6: Documentation and audit contracts

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/BINANCE.md`
- Modify: `examples/quickstart/training.json`
- Test: `tests/test_current_documentation_contract.py`
- Test: `tests/workflows/test_training_run_config.py`
- Test: workflow security and architecture contract tests as needed.

**Interfaces:**
- Documents `training_run_config_v3`, `structured_policy_export_v2`, change-intensity semantics and explicit offline signer modules.

- [ ] Write documentation contract tests that reject v2/v1 stale schema text and implicit Quickstart reward defaults.
- [ ] Verify RED.
- [ ] Update all maintained documents and pin Quickstart hybrid reward fields explicitly.
- [ ] Verify GREEN.
- [ ] Commit `docs: align maintained research contracts`.

### Task 7: Complete verification and publication

- [ ] Run focused tests after each task.
- [ ] Run full pytest, Ruff lint/format, MyPy, Import Linter, workflow security, documentation contracts, serving smoke and package smoke.
- [ ] Inspect the complete diff for unrelated changes.
- [ ] Push the branch and open a draft PR against `main`.
- [ ] Require all hosted CI checks to pass; do not claim new CUDA execution unless the main-only GPU workflow runs.
- [ ] Merge by squash only after exact-head verification, then remove the feature branch.