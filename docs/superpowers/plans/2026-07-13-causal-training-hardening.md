# Causal Training Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make episode endings, execution timing, PPO training identity, nested Walk-Forward evaluation, AUM scaling, release gates, and serving activation causal, reproducible, and fail-closed.

**Architecture:** Keep market dynamics in the environment/executor, training hyperparameters in one immutable configuration, fold chronology in typed workflow requests, and deployment identity in content-addressed manifests. Explicit evaluation liquidation is separated from training truncation. Policy, release, and serving artifacts bind observation, environment, AUM, and final evaluation identities.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 PPO, NumPy, dataclasses, pytest, Ruff, Mypy, Import Linter.

## Global Constraints

- Training time limits must remain bootstrap-compatible truncations.
- Explicit liquidation must be terminal and fail closed when positions remain.
- Policy inputs must not expose synthetic episode progress or future tradability.
- Next-open capacity must use information known at decision time.
- Initial capital must be explicit and identity-bound.
- Outer-OOS ranges may not reach training or selection adapters.
- Production activation requires a release identity by default.
- Production status remains NO-GO until real-data mandatory gates pass.

---

### Task 1: Correct terminal semantics and policy inputs

- [x] Add failing tests for truncation without liquidation, terminal explicit liquidation, current tradability, and removal of synthetic episode progress.
- [x] Confirm RED on the pre-change environment.
- [x] Implement the causal environment contracts.
- [x] Confirm focused and full tests GREEN.

### Task 2: Make execution capacity causal

- [x] Add failing tests showing next-bar total volume is unavailable at decision time.
- [x] Use the last completed bar's volume for next-open capacity.
- [x] Keep actual next-bar tradability as transition dynamics.
- [x] Confirm execution regressions GREEN.

### Task 3: Make PPO work and compute identity explicit

- [x] Add tests for full PPO configuration, rollout-rounded work, device reporting, and real SB3 save/load/predict.
- [x] Bind training configuration and observation schema into policy identity.
- [x] Record requested and actual timesteps and resolved device.
- [x] Confirm real SB3 smoke and full CI GREEN.

### Task 4: Execute nested Walk-Forward and bind final evaluation

- [x] Add concrete range-scoped fold training/evaluation contracts.
- [x] Implement deterministic selection, baseline fallback, sealed outer-OOS execution, and independent stitching.
- [x] Bind Gate decisions to dataset, selected policy when applicable, and final evaluation identity.
- [x] Preserve selection and gate evaluation identities in ReleaseManifest.

### Task 5: Make AUM and environment identity first-class

- [x] Remove the silent one-unit initial-capital default.
- [x] Hash timing, trend, risk, execution, reward, schema, alpha, dataset, and AUM into the environment digest.
- [x] Propagate environment digest and AUM through every ensemble member and policy manifest.
- [x] Reject inconsistent seed identities.

### Task 6: Harden serving and activation

- [x] Add serving bundle schema v2 with observation schema, input size, environment digest, and AUM.
- [x] Reject runtime observation-size/schema mismatches.
- [x] Require release identity by default in Runtime and Registry.
- [x] Keep unreleased activation behind explicit `allow_unreleased=True` research mode.

### Task 7: Verify

- [x] Run Ruff.
- [x] Run Ruff format check.
- [x] Run Mypy.
- [x] Run Import Linter.
- [x] Run full pytest with branch coverage.
- [x] Run CLI smoke test.
- [x] Update README, Research Status, design specs, plan, and pull request description.
