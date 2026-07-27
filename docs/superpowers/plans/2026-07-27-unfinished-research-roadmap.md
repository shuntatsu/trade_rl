# Unfinished Research Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every still-relevant unfinished software and empirical work item from the approved Trade RL research direction without mixing gated phases or weakening the production `NO-GO` boundary.

**Architecture:** Work is divided into seven dependency-ordered tracks. C3 and constrained-PPO PR D are the two immediate software tracks. Real GPU evidence follows frozen software. Phase A is conditional on C3 passage, model improvements are isolated ablations, paper evidence follows candidate freeze, and inference-time MPC remains last.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.3.1, Stable-Baselines3 2.3.2, PostgreSQL metadata catalog, deterministic JSON/NPZ artifacts, React/Vite/TypeScript Studio, Docker GPU workflows, Pytest, Ruff, MyPy, import-linter.

## Global Constraints

- Preserve the maintained residual action contract; do not revive the superseded direct-target experiment as the primary route.
- Preserve exact all-cost net log growth as the scalar reward.
- Keep the seven constraint costs outside the reward.
- Keep Causal Scenario artifacts outside training, Serving, promotion, release, and execution until the declared gate authorizes the next phase.
- Use six independently reset folds and at least 180 selection/OOS days for formal gates.
- Use seeds `0`, `1`, and `2`; do not describe 18 validation models as one production ensemble.
- Preserve the 4070 Ti SUPER full model and obtain speed through implementation, not by shrinking capacity.
- Do not open sealed evidence to tune budgets, multipliers, schedules, architecture, or teacher thresholds.
- Do not add direct exchange routing.
- Production remains `NO-GO` unless every maintained release condition passes.

---

## Execution order

### Milestone 1: Complete C3 software and evidence gate

Detailed plan: `docs/superpowers/plans/2026-07-27-causal-scenario-c3-walk-forward.md`.

Deliverables:

- C3 fold runner and persisted-before-replay candidate decisions;
- Scenario Oracle / Trend / deterministic PPO / random comparator / compatible Perfect-Information reports;
- paired growth, ranking, calibration, regret, cost, turnover, drawdown, neighbor, and robustness evidence;
- machine-readable Phase A gate;
- explicit `not_comparable` perfect-information result when dominance assumptions do not hold;
- sealed-ledger support without automatic outer opening.

Merge gate:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

### Milestone 2: Complete constrained PPO PR D

Detailed plan: `docs/superpowers/plans/2026-07-27-constrained-ppo-pr-d-and-gpu-evidence.md`.

Deliverables:

- canonical constrained-growth profile;
- three gamma/GAE profiles from the approved design;
- constraint-aware walk-forward summaries and eligibility;
- ordinary growth PPO versus Lagrangian PPO comparison under identical data, folds, seeds, AUM, and execution;
- required nominal and `joint_2x` evidence;
- budget, multiplier, saturation, penalty/reward ratio, cost-critic, and action-distortion reports.

The sealed test may evaluate a frozen winner once. It may not select budgets or dual hyperparameters.

### Milestone 3: Run controlled GPU and schedule evidence

The implementation in PR #194 is not itself empirical proof. Run:

1. constant schedule control;
2. linear schedule with final ratio `0.1`;
3. cosine schedule with final ratio `0.1`.

Everything else remains fixed:

```text
sequence_d_model=336
sequence_attention_heads=8
sequence_layers=2
policy_net_arch=[384,256,128]
value_net_arch=[512,384,256]
batch_size=128
n_epochs=10
timesteps=524288
seeds=[0,1,2]
device=cuda
```

Record wall time, environment steps/s, rollout and update time, peak allocated/reserved VRAM, OOM state, checkpoint steps, resume identity, KL, clip fraction, entropy, value loss, explained variance, constraint metrics, and selection outcomes. Select no schedule from TensorBoard appearance alone.

### Milestone 4: Freeze the research candidate

A candidate may be frozen only after C3 and/or PR D evidence identifies an eligible route. Create one immutable selection proposal and authorization. The proposal binds exact candidate, schedule, constraints, architecture, folds, seeds, AUM, execution scenario, and source commit.

No later task may silently modify those fields. A changed field creates a new experiment plan and cannot reuse the prior sealed reservation.

### Milestone 5: Conditional Phase A

Detailed plan: `docs/superpowers/plans/2026-07-27-phase-a-and-model-evolution.md`.

Start only when the C3 gate artifact says `passed=true`. Build value/regret teacher data exclusively from causal Scenario Oracle predictions under the frozen C3 configuration. Train value-weighted BC into the maintained residual actor, then PPO fine-tune the same model.

Compare pure PPO and BC→PPO with identical folds, seeds, timesteps, execution, and model size. A failed C3 gate leaves this milestone prohibited, not merely postponed.

### Milestone 6: Independent model ablations

Execute in this fixed order, one PR and one empirical comparison at a time:

1. state-dependent exploration;
2. PopArt for reward and separate cost critics;
3. regime-balanced sampling;
4. execution domain randomization;
5. residual critic adapters and VALUE/RISK tokens;
6. quantile and hazard auxiliary heads;
7. self-supervised encoder pretraining;
8. recurrent decision memory;
9. large-model capacity comparison.

Each ablation has an unchanged control and must pass software gates before empirical comparison. Do not combine adjacent items to save runs; attribution is a requirement.

### Milestone 7: Real paper evidence and release gate

Detailed plan: `docs/superpowers/plans/2026-07-27-paper-evidence-and-phase-b-mpc.md`.

Use an external paper venue or independently generated order/fill logs; the repository remains read-only with respect to exchange accounts. Import logs, validate identity, create `paper_reconciliation_evidence_v1`, compare simulator assumptions with paper outcomes, and preserve failed evidence.

Fresh confirmation uses a completely unused interval chosen and sealed before ingestion. Release attestation is created only after the paper report, confirmation, conservative execution evidence, and all research gates pass.

### Milestone 8: Phase B inference-time MPC

Begin last and only when Phase A leaves a material, statistically supported `ScenarioOracle - student` gap. Implement paper-only one-step Scenario MPC with bounded latency, deterministic fallback to the maintained policy, and complete decision evidence. Multi-step scenario trees remain out of scope.

## Branch and PR policy

Create one branch per independently reviewable deliverable:

```text
agent/causal-scenario-c3
agent/constrained-ppo-pr-d
agent/gpu-schedule-evidence-contract
agent/phase-a-value-teacher
agent/state-dependent-exploration
agent/popart-cost-critics
agent/regime-balanced-sampling
agent/execution-domain-randomization
agent/value-risk-residual-adapters
agent/quantile-hazard-heads
agent/ssl-sequence-pretraining
agent/recurrent-policy-memory
agent/large-model-ablation
agent/paper-evidence-ingestion
agent/inference-time-scenario-mpc
```

Do not stack unrelated branches. Dependency branches may be stacked only when the base PR is independently green and the child PR body explicitly declares the dependency.

## Evidence completion checklist

- [ ] C3 implementation merged.
- [ ] C3 six-fold/180-day selection evidence generated.
- [ ] Phase A gate recorded as pass or fail.
- [ ] Constrained PPO PR D merged.
- [ ] Constrained/unconstrained and gamma/GAE comparisons completed.
- [ ] Constant/linear/cosine schedule comparison completed on 4070 Ti SUPER.
- [ ] Candidate, architecture, schedule, budgets, AUM, and execution profile frozen.
- [ ] Phase A implemented only after a passing C3 gate.
- [ ] Model ablations executed independently in the fixed order.
- [ ] Fresh unused confirmation interval evaluated once.
- [ ] Real paper order/fill/accounting reconciliation completed.
- [ ] Release attestation created only if all gates pass.
- [ ] Phase B evaluated last and paper-only before any release consideration.
- [ ] Direct exchange routing remains unimplemented.
