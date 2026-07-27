# Paper Evidence and Phase B MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce real identity-bound paper evidence for one frozen research candidate and, only after the required earlier gates, evaluate inference-time Scenario MPC as the final Phase B.

**Architecture:** Paper validation remains external-log based and read-only with respect to exchange accounts. The repository ingests normalized order/fill/account snapshots, validates exact identity and chronology, creates the existing reconciliation artifact, and compares observed execution with the conservative simulator. Phase B adds a paper-only decision adapter that may replace one policy residual with the bounded C1/C2 scenario decision and always has a deterministic policy fallback.

**Tech Stack:** Python 3.12, existing Serving state snapshots, `paper_reconciliation_evidence_v1`, Ed25519 fresh confirmation and release attestation, stateful execution, C1/C2/C3 artifacts, deterministic JSON/NPZ evidence, Pytest.

## Global Constraints

- Do not add authenticated exchange access, API-key entry, order submission, cancellation, replacement, or direct broker control.
- A paper source is external and supplies normalized observations/order/fill/account logs.
- Freeze candidate, configuration, AUM, execution assumptions, collection interval, tolerances, and log schemas before collection.
- Preserve failed paper evidence; do not overwrite or relabel it as clean.
- Fresh confirmation uses a completely unused interval selected and sealed before data ingestion.
- Phase B starts only after C3 passes, Phase A is completed, and a material Scenario-Oracle-to-student gap remains.
- Phase B version one is one-step residual deviation followed by zero residual, not multi-step tree MPC.
- Production remains `NO-GO` unless all maintained gates pass.

---

### Task 1: Define an external paper-log ingestion contract

**Files:**
- Create: `trade_rl/evaluation/paper_log_ingestion.py`
- Create: `trade_rl/evaluation/paper_log_artifact.py`
- Test: `tests/evaluation/test_paper_log_ingestion.py`
- Test: `tests/evaluation/test_paper_log_artifact.py`

**Interfaces:**
- Produces `PaperCollectionPlan`, `NormalizedPaperOrder`, `NormalizedPaperFill`, `NormalizedPaperAccountSnapshot`, `PaperLogArtifact`, writer, and loader.
- Consumes only caller-supplied normalized records and immutable selected-final identities.

- [ ] **Step 1: Write RED schema-closure tests**

`PaperCollectionPlan` binds:

```text
dataset_id
environment_digest
policy_digest
training_run_digest
serving_bundle_digest
execution_policy_digest
AUM
collection_start
collection_stop
order_schema
fill_schema
account_schema
position_notional_tolerance
cash_tolerance
equity_tolerance
created_at
```

Reject collection intervals that begin before plan creation, overlap train/checkpoint/selection/sealed/fresh ranges, or lack monotonic UTC timestamps.

- [ ] **Step 2: Define normalized record closure**

Orders store external ID, strategy decision identity, symbol, side, type, quantity, limit/stop values, creation/eligibility/terminal timestamps, time in force, replacement/cancel linkage, and terminal status. Fills store order ID, fill ID, timestamp, quantity, price, fee, fee currency, liquidity role when available, and source sequence. Account snapshots store cash, positions, marks, equity, and source sequence.

Reject duplicate external IDs, duplicate fill IDs, unknown orders, negative quantities/costs, non-finite values, time reversal, and snapshots that cannot be ordered deterministically.

- [ ] **Step 3: Implement exact-file artifact publication**

Write canonical metadata plus immutable arrays/tables. Bind raw source-file digests but do not store credentials or unredacted private account metadata. Extra files, symlinks, schema drift, and identity mismatch fail closed.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_paper_log_ingestion.py tests/evaluation/test_paper_log_artifact.py -q
```

Commit: `feat: ingest normalized paper execution logs`.

### Task 2: Create real paper reconciliation evidence

**Files:**
- Modify: `trade_rl/evaluation/paper_reconciliation.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/evaluation/test_real_paper_reconciliation.py`
- Test: `tests/cli/test_real_paper_reconciliation_cli.py`

**Interfaces:**
- Extends the existing reconciliation builder to consume `PaperLogArtifact` while preserving the current request/API contract.

- [ ] **Step 1: Write RED end-to-end pass and fail fixtures**

Pass fixture: every terminal strategy order has matching fills/status, no unknown/duplicate fills, no open terminal orders, and all position/cash/equity differences are within the predeclared tolerance capped at `1e-6`.

Fail fixtures cover missing terminal order, unmatched fill, duplicate fill, unknown order, open order, position difference, cash difference, equity difference, collection-identity mismatch, and log chronology violation.

- [ ] **Step 2: Recompute pass state internally**

Caller-supplied `passed` remains forbidden. The report derives all counts, maxima, and pass/fail reasons from the loaded immutable logs.

- [ ] **Step 3: Bind fresh-confirmation chronology**

Reconciliation creation must be no later than confirmation signing, and the confirmation must bind the exact reconciliation digest and collection interval. A stale or differently scoped confirmation fails.

- [ ] **Step 4: Add machine-readable CLI output**

Return report path, digest, pass state, failure reasons, interval, bundle/policy/run identities, and `production_status="NO-GO"`.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_real_paper_reconciliation.py tests/cli/test_real_paper_reconciliation_cli.py -q
```

Commit: `feat: reconcile real paper execution evidence`.

### Task 3: Calibrate conservative simulator assumptions against paper logs

**Files:**
- Create: `trade_rl/evaluation/execution_calibration.py`
- Create: `trade_rl/evaluation/execution_calibration_artifact.py`
- Test: `tests/evaluation/test_execution_calibration.py`
- Test: `tests/evaluation/test_execution_calibration_artifact.py`

**Interfaces:**
- Produces diagnostics only: `ExecutionCalibrationReport` and artifact writer/loader.
- Compares matched simulated and paper orders/fills without mutating the frozen selected-final configuration.

- [ ] **Step 1: Define matched evidence metrics**

Report:

```text
eligibility latency error
fill latency error
fill quantity ratio
fill price slippage difference
fee difference
cancel/replace disagreement
partial-fill disagreement
terminal-status disagreement
position/cash/equity path error
```

Stratify by symbol, order type, volatility, spread, liquidity, and requested participation. Preserve unmatched records as explicit counts.

- [ ] **Step 2: Prohibit post-hoc release tuning**

Calibration from the current paper interval is diagnostic. Any changed execution model/config creates a new experiment generation and requires new walk-forward, fresh confirmation, and paper evidence. It cannot retroactively make the current report pass.

- [ ] **Step 3: Add optional next-generation proposal**

The report may emit a bounded `execution_calibration_proposal_v1` containing observed quantiles and recommended predeclared ranges. The proposal is not an active configuration and has no Serving effect.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_execution_calibration.py tests/evaluation/test_execution_calibration_artifact.py -q
```

Commit: `feat: compare simulated and paper execution`.

### Task 4: Add fresh unused confirmation workflow

**Files:**
- Create: `trade_rl/evaluation/fresh_interval_plan.py`
- Modify: `examples/binance-multitimeframe/recheck_confirmation.py`
- Modify: `trade_rl/workflows/full_research_state.py`
- Test: `tests/evaluation/test_fresh_interval_plan.py`
- Test: `tests/examples/test_recheck_confirmation.py`

**Interfaces:**
- Produces `FreshIntervalPlan` and a one-shot evaluation capability.

- [ ] **Step 1: Define the deterministic unused-window rule**

Select the first complete contiguous interval satisfying all conditions:

```text
start >= experiment_plan.frozen_at
start >= previous_outer_stop
no overlap with train/checkpoint/selection/test/paper-tuning ranges
duration >= configured_confirmation_days
all required market/execution metadata complete
```

Persist the interval plan before loading interval rows. The plan digest enters the sealed ledger key.

- [ ] **Step 2: Enforce one-shot access**

A second open for the same experiment/dataset/interval identity fails across processes through the PostgreSQL sealed ledger. Filesystem-only mode cannot claim durable cross-process uniqueness.

- [ ] **Step 3: Preserve fail evidence**

A failed confirmation produces an immutable failed report and blocks release. It cannot be reopened for parameter changes.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_fresh_interval_plan.py tests/examples/test_recheck_confirmation.py -q
```

Commit: `feat: seal fresh confirmation intervals`.

### Task 5: Complete release-attestation gating

**Files:**
- Modify: `trade_rl/release/attestation.py`
- Modify: `trade_rl/serving/package.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/release/test_selected_final_release_gate.py`
- Test: `tests/serving/test_paper_reconciliation_packaging.py`

**Interfaces:**
- Release eligibility consumes a frozen selection authorization, selected-final run, walk-forward gate evidence, conservative execution evidence, fresh confirmation, paper reconciliation, source/dependency provenance, and approver signature.

- [ ] **Step 1: Write RED complete-chain tests**

Reject packaging when any required digest is missing, mismatched, expired, unsigned, signed by an unknown key, created in the wrong chronology, or associated with a different bundle/policy/dataset/environment/run.

- [ ] **Step 2: Preserve bundle/private-material separation**

The serving bundle contains verified public evidence copies and identities, not private signing keys or authorization secrets. Detached attestation remains separate.

- [ ] **Step 3: Keep production status honest**

Passing packaging/attestation creates an eligible artifact; it does not implement direct exchange routing. Status text must distinguish `release-eligible paper serving` from `live exchange trading`.

- [ ] **Step 4: Run GREEN and commit**

Commit: `feat: require complete paper release evidence`.

### Task 6: Define Phase B authorization and latency budget

**Files:**
- Create: `trade_rl/evaluation/phase_b_authorization.py`
- Test: `tests/evaluation/test_phase_b_authorization.py`

**Interfaces:**
- Produces `PhaseBInferenceAuthorization` only from C3, Phase A, and paper evidence.

- [ ] **Step 1: Encode entry requirements**

Authorization requires:

```text
C3 gate passed
Phase A comparison completed
BC/student candidate eligible
ScenarioOracle - student approximation gap lower bound > configured minimum
paper reconciliation available for the student candidate
predeclared max decision latency
predeclared timeout/fallback behavior
paper-only purpose
```

- [ ] **Step 2: Reject automatic production promotion**

Authorization purpose is exactly `paper_inference_mpc`. It cannot satisfy release or direct execution authorization.

- [ ] **Step 3: Run GREEN and commit**

```bash
uv run pytest tests/evaluation/test_phase_b_authorization.py -q
```

Commit: `feat: gate inference time scenario MPC`.

### Task 7: Implement paper-only one-step Scenario MPC

**Files:**
- Create: `trade_rl/inference/scenario_mpc.py`
- Create: `trade_rl/inference/scenario_mpc_evidence.py`
- Modify: `trade_rl/serving/runtime.py`
- Test: `tests/inference/test_scenario_mpc.py`
- Test: `tests/inference/test_scenario_mpc_evidence.py`
- Test: `tests/serving/test_scenario_mpc_paper_runtime.py`

**Interfaces:**
- Produces `ScenarioMPCConfig`, `ScenarioMPCDecision`, and `ScenarioMPCPaperAdapter`.
- Consumes current causal Serving state, frozen C2 library, C1 evaluator, authorized config, and maintained policy fallback.

- [ ] **Step 1: Write RED semantics tests**

On each decision:

```text
build complete causal query snapshot
select 64 strictly historical train scenarios
evaluate bounded candidate set
select one raw residual
return that residual for the current decision only
use maintained policy as fallback on timeout/failure
```

The evaluator never reads realized future rows and never mutates the persistent library.

- [ ] **Step 2: Enforce bounded latency and deterministic timeout**

Use monotonic time. When the predeclared deadline is reached, discard incomplete MPC work and return the already computed maintained-policy mean action. Record timeout/failure reason and fallback digest.

- [ ] **Step 3: Preserve action/risk/execution contracts**

The selected residual still passes through the canonical composer, emergency controls, portfolio feasibility, no-trade logic, and execution. MPC cannot bypass hard risk or tradability constraints.

- [ ] **Step 4: Record complete decision evidence**

Store query/library/config identities, candidate scores, selected action, policy fallback action, elapsed time, timeout/failure, projected/submitted target, and deterministic digest. Evidence remains excluded from training and selection for the same paper interval.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/inference/test_scenario_mpc.py tests/inference/test_scenario_mpc_evidence.py tests/serving/test_scenario_mpc_paper_runtime.py -q
```

Commit: `feat: add paper only one step scenario MPC`.

### Task 8: Compare student policy and Phase B on a new paper interval

**Files:**
- Create: `trade_rl/evaluation/phase_b_paper_comparison.py`
- Test: `tests/evaluation/test_phase_b_paper_comparison.py`
- Create after execution: `docs/verification/2026-07-27-paper-evidence-and-phase-b.md`

- [ ] **Step 1: Freeze a new comparison interval**

The Phase B interval must not be the same interval used to authorize MPC or calibrate execution. Freeze policy, MPC config, latency budget, fallback, C2 library, AUM, and metrics before collection.

- [ ] **Step 2: Use paired paper decisions**

Compare maintained student action and MPC action on the same causal state. Only one route may be designated as acted paper intent; the other remains a shadow comparator. Keep order/fill/account evidence separated by route identity.

- [ ] **Step 3: Evaluate value, latency, and reliability**

Report paired net growth, drawdown, turnover, economic costs, action distortion, timeout rate, fallback rate, evaluator failures, scenario-neighbor quality, and realized regret/ranking diagnostics.

- [ ] **Step 4: Define honest disposition**

Possible results are:

```text
MPC rejected: no material value or latency/reliability failure
MPC retained for further paper research
student retained as simpler route
comparison incomplete with preserved evidence
```

No result authorizes direct exchange routing.

### Task 9: Run final exact-head verification

- [ ] **Step 1: Run repository gates**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
uv run python tools/check_critical_coverage.py
```

- [ ] **Step 2: Run platform/container/catalog gates**

Require Ubuntu, Windows, training-image/non-root runtime, and PostgreSQL Catalog success where changed paths apply.

- [ ] **Step 3: Verify no private material is committed**

The final branch contains no account identifier, credential, private key, raw secret-bearing venue export, private authorization payload, or unredacted personal account metadata.

- [ ] **Step 4: Commit final verification**

Commit: `docs: verify paper evidence and Phase B boundaries`.
