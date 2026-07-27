# Causal Scenario C3 Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the missing evaluation-only C3 walk-forward comparison and produce the exact machine-readable gate that determines whether Phase A teacherization is allowed.

**Architecture:** C3 adds three focused modules to `trade_rl.workflows.causal_scenario`: a realized comparison runner, a deterministic report artifact, and a gate evaluator. It reuses C1 values, C2 frozen libraries, the maintained fold/range contracts, deterministic PPO mean actions, stateful execution, paired inference, Perfect-Information Bound, and sealed-test ledger. Candidate selection is persisted before realized replay; no C3 result enters policy fitting or Serving.

**Tech Stack:** Python 3.12, NumPy, immutable dataclasses, existing causal-scenario C1/C2 APIs, `MarketDatasetView`, stateful execution, paired moving-block bootstrap, canonical JSON/NPZ artifacts, Pytest.

## Global Constraints

- Run at least six independently reset folds covering at least 180 selection days for the formal Phase A gate.
- Use exactly the same dataset, environment, action, observation, execution, AUM, initial weights, and range identities for compared policies.
- Persist the predicted candidate selection before any realized query-future replay.
- Never feed realized ranking, return, regret, or future bars back into C1/C2 selection.
- Mark Perfect-Information evidence `not_comparable` unless the documented feasible-set dominance conditions are proven.
- Do not import C3 from maintained training, Serving, promotion, release, or execution packages.
- Production remains `NO-GO`.

---

### Task 1: Define C3 configuration and per-query comparison contracts

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_contracts.py`
- Test: `tests/evaluation/test_causal_scenario_c3_contracts.py`

**Interfaces:**
- Produces `CausalScenarioC3Config`, `ComparedPolicyKind`, `PersistedScenarioDecision`, `RealizedCandidateOutcome`, `CausalScenarioQueryComparison`, and `PerfectInformationComparisonStatus`.
- Consumes C1 `CausalQuerySnapshot`, `CausalScenarioValueResult`, C2 `FrozenCausalScenarioLibrary`, and maintained identity digests.

- [ ] **Step 1: Write RED tests for immutable configuration closure**

Test the exact defaults:

```python
config = CausalScenarioC3Config(
    horizon_decisions=96,
    scenario_count=64,
    random_comparator_count=8,
    bootstrap_block_days=7,
    ranking_tolerance=1e-8,
    required_folds=6,
    required_selection_days=180,
)
assert config.policy_order == (
    "trend",
    "scenario_oracle",
    "ppo_mean",
    "random_candidate",
    "perfect_information",
)
```

Reject booleans passed as integers, non-positive counts, a horizon/scenario mismatch with the C1/C2 identities, non-finite tolerances, duplicate policy names, and unknown schema versions.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_contracts.py -q
```

Expected: import failure for `c3_contracts`.

- [ ] **Step 3: Implement immutable contracts**

`PersistedScenarioDecision` must bind:

```text
dataset_id
fold_id
query_index
query_timestamp_ns
causal_state_digest
scenario_library_digest
scenario_selection_digest
candidate_set_digest
value_artifact_digest
selected_candidate_digest
selected_raw_residual
selected_submitted_target
created_before_realized_replay=true
```

All arrays are copied, finite, C-contiguous, and read-only. Recompute every digest in `__post_init__` and reject mismatches.

- [ ] **Step 4: Add reconstruction tests**

Recompute selected candidate, tie set, score, regret, and selection digest from stored C1 evidence. Tampering with one score, candidate coordinate, scenario anchor, or query identity must fail closed.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_contracts.py -q
uv run ruff check trade_rl/workflows/causal_scenario/c3_contracts.py tests/evaluation/test_causal_scenario_c3_contracts.py
uv run mypy trade_rl/workflows/causal_scenario/c3_contracts.py
```

Commit: `feat: define causal scenario C3 contracts`.

### Task 2: Persist decisions before realized replay

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_decisions.py`
- Create: `trade_rl/workflows/causal_scenario/c3_decision_artifact.py`
- Test: `tests/evaluation/test_causal_scenario_c3_decisions.py`
- Test: `tests/evaluation/test_causal_scenario_c3_decision_artifact.py`

**Interfaces:**
- Produces `build_persisted_scenario_decision`, `write_c3_decision_artifact`, and `load_c3_decision_artifact`.
- The realized replay runner accepts only a loaded decision artifact, never an in-memory unpersisted result.

- [ ] **Step 1: Write RED chronology tests**

Create a fake realized-replay adapter that records whether a decision artifact path exists. Assert replay raises until the artifact has been atomically published and reloaded.

- [ ] **Step 2: Implement deterministic artifact closure**

Write:

```text
<decision-digest>/
  decision.json
  arrays.npz
```

`decision.json` lists exact files, sizes, SHA-256 values, schema, and all scalar identities. `arrays.npz` contains selected raw residual, submitted target, score vector, regret vector, and tie indices. Loading rejects extra files, symlinks, missing files, dtype/shape drift, and digest mismatch.

- [ ] **Step 3: Add duplicate/idempotency behavior**

Publishing identical content to the same digest returns the existing artifact after full validation. Conflicting content at an existing path raises and never overwrites.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_decisions.py tests/evaluation/test_causal_scenario_c3_decision_artifact.py -q
```

Commit: `feat: persist C3 decisions before replay`.

### Task 3: Implement same-period realized comparison

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_runner.py`
- Modify: `trade_rl/evaluation/perfect_information_bound.py`
- Test: `tests/evaluation/test_causal_scenario_c3_runner.py`
- Test: `tests/evaluation/test_causal_scenario_c3_perfect_information.py`

**Interfaces:**
- Produces `run_c3_query_comparison(...) -> CausalScenarioQueryComparison`.
- Consumes a loaded decision artifact, a causal query snapshot, a realized-range capability, deterministic PPO mean action provider, Trend zero-residual provider, seeded random comparator provider, stateful rollout engine, and optional Perfect-Information evaluator.

- [ ] **Step 1: Write RED identity-equivalence tests**

For every comparator, assert exact equality of:

```text
dataset_id
environment_digest
action_spec_digest
observation_schema_digest
execution_policy_digest
risk_digest
initial_state_digest
query_index
realized_stop_index
AUM
```

A single mismatch must abort the whole query comparison.

- [ ] **Step 2: Write RED policy semantics tests**

- Trend uses zero residual at every decision.
- Scenario Oracle uses the persisted selected residual once, then zero residual.
- PPO uses deterministic mean residual once, then zero residual.
- Each random comparator chooses from the already persisted candidate set using a counter-based seed derived from query/config digest.
- All policies terminate and liquidate under the same finite-horizon contract.

- [ ] **Step 3: Implement independent state cloning**

Every policy/candidate replay receives a fresh clone of account, pending-order, book, reward, drawdown, risk, and random state. Add a mutation sentinel test proving one replay cannot change another.

- [ ] **Step 4: Implement realized ranking and calibration fields**

Store gross log return, baseline-relative advantage, filled turnover, fees, spread, impact, funding, borrow, fill count, pending-order events, termination reason, maximum drawdown, and terminal equity for every candidate. Recompute top-one regret and Spearman correlation from complete finite candidate rankings.

- [ ] **Step 5: Implement compatible Perfect-Information comparison**

Return `comparable` only when period, initial weights, return matrix, exposure limits, AUM, and relaxation-containment evidence match. Otherwise return `not_comparable` with a finite enumerated reason and no asserted gap.

- [ ] **Step 6: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_runner.py tests/evaluation/test_causal_scenario_c3_perfect_information.py -q
```

Commit: `feat: compare C3 policies on realized periods`.

### Task 4: Add fold and aggregate C3 reports

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_report.py`
- Create: `trade_rl/workflows/causal_scenario/c3_report_artifact.py`
- Test: `tests/evaluation/test_causal_scenario_c3_report.py`
- Test: `tests/evaluation/test_causal_scenario_c3_report_artifact.py`

**Interfaces:**
- Produces `CausalScenarioFoldReport`, `CausalScenarioAggregateReport`, `build_c3_fold_report`, `build_c3_aggregate_report`, writer, and loader.

- [ ] **Step 1: Write RED paired-inference tests**

Use daily paired log-growth differences:

```python
delta = daily_log_growth["scenario_oracle"] - daily_log_growth["trend"]
```

Use the maintained paired moving-block bootstrap with a predeclared seven-day block. Store mean, 95% interval, p-value, effective days, and fold identity. Never concatenate fold equity curves.

- [ ] **Step 2: Add ranking/calibration aggregation**

Report predicted-versus-realized Spearman correlation, top-one regret, random-ranking difference, score-bucket predicted mean/CVaR versus realized mean/downside, neighbor distance quantiles, anchor concentration, and effective historical coverage.

- [ ] **Step 3: Add execution and robustness aggregation**

Report nominal and maintained adverse scenarios separately. Include turnover, each economic cost component, fill ratio, pending/cancel/replace counts, termination distribution, maximum drawdown, and Scenario-Oracle uplift.

- [ ] **Step 4: Implement deterministic report artifacts**

Use exact JSON/NPZ closure. Recompute all aggregate statistics during load and reject reports whose stored values differ from raw fold/query arrays.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_report.py tests/evaluation/test_causal_scenario_c3_report_artifact.py -q
```

Commit: `feat: build causal scenario C3 reports`.

### Task 5: Implement the Phase A entry gate

**Files:**
- Create: `trade_rl/workflows/causal_scenario/c3_gate.py`
- Test: `tests/evaluation/test_causal_scenario_c3_gate.py`

**Interfaces:**
- Produces `PhaseAEntryGateEvidence` and `evaluate_phase_a_entry_gate(report) -> PhaseAEntryGateEvidence`.

- [ ] **Step 1: Encode all nine approved gate conditions**

The gate passes only when:

1. no leakage, identity, replay, artifact, or determinism failure exists;
2. at least six folds cover at least 180 selection days;
3. at least four folds have positive Scenario-Oracle uplift over Trend;
4. aggregate paired 95% lower bound is strictly positive;
5. worst-fold drawdown is at most 20% and at most two percentage points worse than Trend;
6. realized regret beats the random comparator with positive paired 95% lower margin;
7. aggregate Spearman correlation and its lower bound are strictly positive;
8. every asserted Perfect-Information comparison is compatible and ordered within tolerance;
9. nominal and required adverse scenarios pass cost, turnover, drawdown, and uplift limits.

- [ ] **Step 2: Add fail-closed support tests**

Insufficient folds, missing days, absent comparator, non-finite intervals, unverified Perfect-Information evidence, or missing adverse scenario must produce `passed=false` with one reason per failed condition.

- [ ] **Step 3: Ensure the gate is pure and immutable**

The function reads only the frozen report and has no dataset, filesystem, environment, or model access. The gate digest binds all inputs and threshold values.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_causal_scenario_c3_gate.py -q
```

Commit: `feat: add Phase A causal evidence gate`.

### Task 6: Integrate C3 with walk-forward and CLI without training dependencies

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/cli/extended.py`
- Modify: `trade_rl/evaluation/__init__.py`
- Create: `tests/workflows/test_causal_scenario_c3_walk_forward.py`
- Create: `tests/cli/test_causal_scenario_c3_cli.py`
- Create: `tests/architecture/test_causal_scenario_c3_boundary.py`
- Create: `docs/verification/2026-07-27-causal-scenario-c3.md`

**Interfaces:**
- Add an explicit evaluation command that consumes an already published walk-forward run and frozen C2 libraries.
- It must not mutate the training run, checkpoint choices, or selection evidence.

- [ ] **Step 1: Add RED architecture tests**

Reject imports of `trade_rl.workflows.causal_scenario.c3_*` from `trade_rl.rl`, `trade_rl.serving`, `trade_rl.release`, `trade_rl.promotion`, and direct execution packages.

- [ ] **Step 2: Add RED CLI lifecycle tests**

The command must:

```text
validate source run and fold identities
load frozen C2 library
create and persist query decisions
run realized comparison
publish fold and aggregate reports
publish gate evidence
return machine-readable JSON
```

Re-running with identical inputs is idempotent; changing source/config requires a new output identity.

- [ ] **Step 3: Implement bounded integration**

Do not add C3 to ordinary `train run`. Integrate through evaluation-only workflow adapters and stage-scoped dataset capabilities.

- [ ] **Step 4: Run complete verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
uv run python tools/check_critical_coverage.py
```

Run Ubuntu, Windows, training-image, and PostgreSQL workflows on the exact head. Record run IDs, test count, total/branch coverage, focused module coverage, commit SHA, and artifact digests in the verification document.

- [ ] **Step 5: Commit**

Commit: `feat: integrate causal scenario C3 walk-forward evaluation`.
