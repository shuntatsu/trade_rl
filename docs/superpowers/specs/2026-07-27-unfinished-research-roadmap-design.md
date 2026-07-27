# Unfinished Research Roadmap Design

Date: 2026-07-27
Status: approved planning scope
Base: integrated `main` after PRs #188, #190, #193, and #194

## Purpose

Define one dependency-ordered roadmap for every currently unfinished item that remains relevant after the branch and pull-request consolidation. This document distinguishes unfinished implementation from unfinished empirical validation and prevents old, superseded branches from being reintroduced.

The repository remains a research system. Production status stays `NO-GO`; direct exchange routing remains out of scope.

## Completed foundations

The roadmap treats the following as complete software foundations and does not plan to rebuild them:

- conservative stateful OHLCV execution with persistent orders, latency, partial fills, time in force, gap handling, and evidence;
- causal multi-timeframe observations, nested walk-forward, sealed-test ledger, immutable artifacts, and Serving identity checks;
- Causal Scenario Evaluator C1 and train-only/past-only scenario library C2;
- action-path diagnostics and seven independent constraint costs;
- Cost Critics and corrected Lagrangian PPO semantics;
- optional TensorBoard scalar logging, Studio training diagnostics, learning-rate schedules, and full-horizon checkpoint placement;
- PostgreSQL metadata catalog and persistent sealed-test reservations;
- paper-reconciliation artifact validation and release-attestation verification capability.

## Unfinished work inventory

### Track 1 — Causal Scenario C3

Implement the missing walk-forward comparison and Phase A entry gate. C3 must compare Scenario Oracle, Trend, deterministic PPO mean, random candidates, and compatible Perfect-Information evidence on identical fold/range/environment identities. It must produce paired inference, ranking/calibration evidence, execution evidence, robustness evidence, and a machine-readable gate result.

This is the first active research track because the user fixed the sequence as C first, then A only if C passes.

### Track 2 — Constrained PPO PR D

PRs A, B, and corrected C are complete. PR D remains unfinished:

- maintained constrained-growth experiment profiles;
- gamma/GAE ablations;
- unconstrained-versus-constrained walk-forward comparison;
- constraint-aware selection reporting;
- nominal and adverse execution evaluation;
- empirical budget and multiplier evidence without tuning on sealed data.

### Track 3 — Real GPU and schedule evidence

The software supports constant, linear, and cosine schedules and TensorBoard diagnostics, but the empirical comparison is unfinished. Run controlled experiments on the RTX 4070 Ti SUPER without shrinking the maintained model:

- `sequence_d_model=336`;
- 8 heads, 2 layers;
- actor `[384, 256, 128]`;
- value `[512, 384, 256]`;
- PPO batch size 128;
- 10 PPO epochs;
- 524,288 timesteps per seed;
- seeds 0, 1, 2;
- six independently reset folds for formal validation when sufficient data exists.

Only the schedule changes inside the schedule experiment. Throughput, peak memory, optimization stability, checkpoint behavior, and walk-forward outcomes are evidence; chart smoothness is not a selection criterion.

### Track 4 — Conditional Phase A teacherization

Phase A is prohibited unless C3 passes every declared gate. After passage, build a separate value/regret-based teacher artifact from frozen C3 semantics. The initial student is value-weighted behavior cloning into the maintained residual policy followed by PPO fine-tuning. Perfect-information actions and realized query futures never become labels.

Phase A compares pure PPO and C3-value-BC→PPO under identical folds, seeds, architecture, execution, and timesteps.

### Track 5 — Post-foundation model improvements

These remain independent ablations and must not be bundled together. The order follows the user's priority:

1. state-dependent exploration;
2. PopArt for reward and separate cost critics;
3. regime-balanced sampling and execution domain randomization;
4. residual critic adapters and VALUE/RISK tokens;
5. auxiliary quantile and hazard heads;
6. self-supervised encoder pretraining;
7. recurrent decision memory;
8. large-model capacity comparison.

Each ablation must preserve an unmodified control profile and must earn continuation through checkpoint and selection evidence before the next item begins.

### Track 6 — Real paper evidence and release gates

The validator exists, but no maintained real paper run has cleared the gate. Add an external-log ingestion workflow that does not submit orders, collect one identity-bound paper interval, reconcile terminal orders/fills/accounting, calibrate simulator assumptions, obtain signed fresh confirmation, and generate release-attestation evidence only when every gate passes.

A failed paper run remains valid failed evidence and does not trigger parameter tuning against the same interval.

### Track 7 — Phase B inference-time MPC

Phase B is inference-time Scenario MPC and remains last. It begins only if:

- C3 passes;
- Phase A is completed and compared;
- a material residual `ScenarioOracle - student` approximation gap remains;
- latency and paper-only operational budgets are predeclared.

Version one applies the already specified one-step residual deviation and zero-residual continuation. It does not introduce a multi-step scenario tree. It remains paper-only until deterministic replay, bounded latency, fallback behavior, and fresh paper evidence pass.

## Dependency graph

```text
Integrated main
  ├─ Track 1: C3 software + selection gate
  │    └─ PASS only → Track 4: Phase A
  │                    └─ material approximation gap → Track 7: Phase B
  ├─ Track 2: constrained PPO PR D
  │    └─ Track 3: controlled GPU/schedule and main-scale evidence
  ├─ Track 5: model ablations, one at a time after trustworthy diagnostics
  └─ Track 6: paper evidence after one candidate is frozen
```

Track 1 and Track 2 may be implemented independently, but Phase A cannot start before C3 passage. Formal selected-final training and paper evidence wait until the research candidate and configuration are frozen.

## Global gates

Every implementation PR must pass:

- Ruff and repository-pinned formatting;
- repository-wide MyPy;
- import architecture and dead-code checks;
- focused statement and branch coverage for new modules;
- complete Pytest and critical coverage ratchets;
- Ubuntu and Windows compatibility;
- complete training-image build and non-root runtime probe;
- PostgreSQL Catalog workflow when catalog, dependencies, configuration, or workflow paths change;
- exact-head verification with no temporary workflows or diagnostic files in the final diff.

Every empirical comparison must bind:

- source commit and lockfile;
- dataset and fold ranges;
- environment, action, observation, reward, execution, and constraint identities;
- architecture and optimizer configuration;
- seeds, requested and observed timesteps;
- AUM and sensitivity scenario;
- selected checkpoints and deterministic ensemble identity;
- complete evidence digests.

## Production boundary

Production remains `NO-GO` until all repository-declared conditions pass, including maintained GPU verification, at least 180 OOS days, a strictly positive paired block-bootstrap lower bound against the baseline, conservative execution evidence, signed fresh confirmation, and a real paper-reconciliation artifact. This roadmap does not add authenticated exchange access, direct order submission, secrets management for a venue, or operational live-trading authorization.
