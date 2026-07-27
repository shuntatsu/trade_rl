# Constrained PPO PR D and GPU Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish constrained-growth PPO Phase 1 by adding experiment profiles, constraint-aware walk-forward evaluation, and controlled 4070 Ti SUPER evidence for schedules and main-scale training.

**Architecture:** PR D does not change the corrected Lagrangian actor. It adds typed experiment profiles and reporting above the existing `lagrangian_ppo`, Cost Critics, diagnostics, evidence, checkpoint identity, and walk-forward framework. A separate evidence runner executes constant/linear/cosine schedules on identical software and data identities.

**Tech Stack:** Python 3.12, Stable-Baselines3 2.3.2, PyTorch 2.3.1, NumPy, existing `LagrangianPPO`, market walk-forward workflow, execution sensitivity, TensorBoard event files, deterministic JSON artifacts, Docker CUDA runner.

## Global Constraints

- Preserve corrected raw-unit actor composition and final-only advantage normalization.
- Preserve exact ordinary-PPO behavior when all multipliers are zero.
- Preserve frozen multipliers for a rollout and denominator-aware EMA.
- Do not tune budgets, multipliers, gamma, GAE, or schedules on sealed evidence.
- Do not combine capacity changes with PR D.
- Use identical fold ranges, seeds, AUM, execution, and timesteps for constrained/unconstrained comparisons.
- Required adverse execution includes `joint_2x`.
- Production remains `NO-GO`.

---

### Task 1: Add canonical constrained-growth experiment profiles

**Files:**
- Create: `examples/binance-multitimeframe/training-constrained-growth.json`
- Create: `examples/binance-multitimeframe/walk-forward-constrained-growth.json`
- Create: `examples/binance-multitimeframe/training-constrained-growth-gae097.json`
- Create: `examples/binance-multitimeframe/training-constrained-growth-discounted.json`
- Test: `tests/examples/test_constrained_growth_profiles.py`

**Interfaces:**
- Produces four immutable configuration identities: canonical constrained, GAE 0.97 ablation, discounted gamma 0.9995 ablation, and the existing unconstrained growth control.

- [ ] **Step 1: Write RED profile-closure tests**

Assert the canonical profile uses:

```json
{
  "algorithm": "lagrangian_ppo",
  "gamma": 1.0,
  "gae_lambda": 0.95,
  "seeds": [0, 1, 2],
  "timesteps": 524288,
  "batch_size": 128,
  "n_epochs": 10,
  "device": "cuda"
}
```

Assert the only canonical/GAE-ablation difference is `gae_lambda=0.97`, and the only canonical/discounted-ablation difference is `gamma=0.9995`. Assert all model, action, reward, execution, AUM, constraint, schedule, and seed fields match.

- [ ] **Step 2: Require complete seven-cost configuration**

The profile must name costs in canonical order and specify budget, dual learning rate, EMA beta, initial multiplier, upper cap, warmup, update interval, minimum completed episodes, cost gamma, and cost GAE lambda for each cost. Unknown, missing, reordered, or duplicate costs fail closed.

- [ ] **Step 3: Bind adverse scenarios and reporting requirements**

The walk-forward profile must require nominal and `joint_2x`; optional diagnostics may include individual fee/spread/impact/funding/borrow stresses but cannot substitute for `joint_2x`.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/examples/test_constrained_growth_profiles.py -q
```

Commit: `feat: add constrained growth experiment profiles`.

### Task 2: Add constraint-aware fold and selection summaries

**Files:**
- Create: `trade_rl/evaluation/constrained_policy_report.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Test: `tests/evaluation/test_constrained_policy_report.py`
- Test: `tests/workflows/test_constrained_walk_forward.py`

**Interfaces:**
- Produces `ConstraintFoldSummary`, `ConstraintAggregateSummary`, `ConstrainedPolicyEligibility`, and `build_constrained_policy_report`.
- Consumes existing Lagrangian rollout evidence, evaluation returns, execution sensitivity, seed members, and deterministic ensemble results.

- [ ] **Step 1: Write RED schema tests**

For every cost store:

```text
unit
aggregation
budget
mean
worst_seed
worst_fold
completed_episode_denominator
censored_episode_count
raw_estimate
ema_estimate
multiplier_mean
multiplier_max
upper_cap_fraction
lower_bound_fraction
```

Also store reward/penalty L2 ratio, cost-critic explained variance/loss, raw-to-filled distortion, drawdown, return, turnover, and economic costs.

- [ ] **Step 2: Implement eligibility rules**

A candidate is ineligible when any required constraint violates its selection budget under nominal or `joint_2x`, when a rare-event denominator is below the configured minimum, when diagnostics are non-finite, or when the deterministic ensemble identity differs from the evaluated members.

A lower-bound multiplier is not saturation. Only upper-cap occupancy contributes to `upper_cap_fraction`.

- [ ] **Step 3: Preserve ordinary PPO reporting**

Ordinary PPO reports no multiplier or Cost Critic artifacts. Comparison code represents constraint fields as explicitly absent, not zero-filled fake evidence.

- [ ] **Step 4: Add pooled and fold-local inference**

Do not concatenate fold equity curves. Report per-fold return/drawdown and aggregate distributions. Paired performance differences use matching fold/day identities and the maintained moving-block bootstrap.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_constrained_policy_report.py tests/workflows/test_constrained_walk_forward.py -q
```

Commit: `feat: report constrained policy walk-forward evidence`.

### Task 3: Add PR D comparison artifact and CLI

**Files:**
- Create: `trade_rl/evaluation/constrained_experiment_artifact.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/evaluation/test_constrained_experiment_artifact.py`
- Test: `tests/cli/test_constrained_experiment_cli.py`

**Interfaces:**
- Produces a deterministic comparison across:
  - unconstrained growth PPO;
  - canonical constrained PPO;
  - constrained GAE 0.97;
  - objective-misaligned discounted gamma 0.9995.

- [ ] **Step 1: Write RED comparison-identity tests**

Reject comparisons unless dataset, fold ranges, seeds, AUM, architecture, reward, action, observation, execution, schedule, and timesteps are equal. Permit gamma/GAE differences only in the predeclared ablation fields.

- [ ] **Step 2: Implement explicit objective-alignment labeling**

The gamma 0.9995 result must be labeled `objective_misaligned_ablation=true`. It cannot be selected solely because its return score is better.

- [ ] **Step 3: Implement deterministic artifact closure**

Store raw fold/seed arrays, aggregate summaries, eligibility, paired intervals, config digests, source commit, and complete file hashes. Loading recomputes comparison metrics and rejects extra files or mutated summaries.

- [ ] **Step 4: Add CLI output**

The command returns one JSON line containing artifact path, digest, eligible candidates, selected research candidate or baseline fallback, and `production_status="NO-GO"`.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_constrained_experiment_artifact.py tests/cli/test_constrained_experiment_cli.py -q
```

Commit: `feat: add constrained PPO PR D comparison`.

### Task 4: Add a controlled schedule evidence contract

**Files:**
- Create: `trade_rl/evaluation/training_schedule_evidence.py`
- Create: `tools/run_schedule_comparison.py`
- Test: `tests/evaluation/test_training_schedule_evidence.py`
- Test: `tests/tools/test_run_schedule_comparison.py`

**Interfaces:**
- Produces `TrainingScheduleRunEvidence`, `TrainingScheduleComparisonEvidence`, and a deterministic runner manifest.

- [ ] **Step 1: Write RED fixed-field tests**

The comparison accepts exactly:

```text
constant
linear(final_ratio=0.1)
cosine(final_ratio=0.1)
```

All non-schedule fields must be identical. Reject a comparison if model size, seeds, timesteps, PPO parameters, data ranges, constraints, AUM, or execution differs.

- [ ] **Step 2: Record runtime and optimization evidence**

For each run store:

```text
requested/observed timesteps
wall-clock seconds
environment steps per second
rollout seconds
optimization seconds
peak CUDA allocated bytes
peak CUDA reserved bytes
OOM status
checkpoint requested/observed steps
resume identity
learning-rate trace digest
approx KL
clip fraction
entropy loss
value loss
explained variance
reward and constraint summaries
```

TensorBoard event files are diagnostic sources; the final evidence is normalized into immutable finite arrays and digested.

- [ ] **Step 3: Define schedule comparison rules**

Selection requires software validity, finite metrics, no OOM, complete requested timesteps, and the same walk-forward eligibility gates. Optimization smoothness or lower loss alone cannot select a schedule.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_training_schedule_evidence.py tests/tools/test_run_schedule_comparison.py -q
```

Commit: `feat: add controlled schedule evidence`.

### Task 5: Add 4070 Ti SUPER generation controls

**Files:**
- Modify: `.github/workflows/control-binance-frozen-226.yml`
- Modify: `.github/workflows/monitor-binance-frozen-226.yml`
- Modify: `compose.training.yaml`
- Modify: `docs/operations/docker-gpu-full-training.md`
- Test: `tests/workflows/test_gpu_training_workflows.py`
- Test: `tests/examples/test_full_run_entrypoint.py`

**Interfaces:**
- Adds explicit `experiment_kind` values `schedule-comparison`, `constrained-pr-d`, and `selected-final`.

- [ ] **Step 1: Add RED workflow-security tests**

Require exact-head checkout, pinned actions, read-minimal default permissions, protected environment for privileged runs, one supervised container, immutable image/source/lock identity, non-empty CUDA device, generation heartbeat, OOM capture, and artifact retention before cleanup.

- [ ] **Step 2: Prevent model shrinking**

The runner validates the full model contract:

```text
d_model=336
heads=8
layers=2
actor=[384,256,128]
value=[512,384,256]
max_parameters=12000000
batch_size=128
n_epochs=10
```

A mismatch aborts before training.

- [ ] **Step 3: Require complete three-seed execution**

A schedule candidate is incomplete unless seeds 0, 1, and 2 finish or produce preserved failed evidence. Do not silently average only successful seeds.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/workflows/test_gpu_training_workflows.py tests/examples/test_full_run_entrypoint.py -q
```

Commit: `ci: add full-scale constrained and schedule GPU runs`.

### Task 6: Execute PR D and schedule experiments

**Files:**
- Create after execution: `docs/verification/2026-07-27-constrained-ppo-pr-d.md`
- Create after execution: `docs/verification/2026-07-27-4070ti-schedule-comparison.md`

- [ ] **Step 1: Freeze experiment manifests before CUDA execution**

Persist exact config, dataset, ranges, constraints, schedule candidates, seeds, AUM, source commit, Docker image, and gate thresholds. Sign/authorize where the maintained workflow requires it.

- [ ] **Step 2: Run CPU smoke on all candidates**

Use a small synthetic/fixture dataset to prove end-to-end artifact creation and fail-closed behavior. Smoke results are not performance evidence.

- [ ] **Step 3: Run the 4070 Ti SUPER experiments**

Execute 524,288 timesteps for every seed and candidate. Preserve generation artifacts and failed runs. Do not reuse a failed generation as clean evidence.

- [ ] **Step 4: Run six-fold formal comparison**

Use at least 180 selection/OOS days. Evaluate nominal and `joint_2x`, constrained eligibility, paired return differences, drawdown, turnover, and costs.

- [ ] **Step 5: Record an honest result**

The verification documents must state one of:

```text
candidate eligible and selected for further research
baseline fallback selected
software valid but empirical gate failed
run incomplete with preserved evidence
```

They must not convert a failed empirical gate into a software success claim.

### Task 7: Complete exact-head software verification

- [ ] **Step 1: Run repository gates**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
uv run python tools/check_critical_coverage.py
```

- [ ] **Step 2: Run platform and container gates**

Require Ubuntu, Windows, training-image/non-root runtime, and PostgreSQL Catalog success on the exact final head.

- [ ] **Step 3: Remove temporary material**

The final diff must contain no temporary workflow, raw log, event file, generated model, dataset payload, patch transfer, or secret material.

- [ ] **Step 4: Commit final verification**

Commit: `docs: verify constrained PPO PR D and GPU evidence`.
