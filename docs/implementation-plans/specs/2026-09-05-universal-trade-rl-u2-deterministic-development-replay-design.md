# Universal Trade RL U2 Deterministic Development Replay Design

## 1. Status and scope

This specification defines Task 7C-1 for Universal Trade RL U2 V1.

The task is **synthetic-only**. It establishes a deterministic Development replay contract without opening or evaluating real Development or Admission artifacts.

Production status remains `NO-GO`.

### Non-goals

- no real Development numeric evaluation
- no real PPO checkpoint evaluation
- no Development Selection decision
- no bootstrap decision
- no Admission access or authorization
- no Production eligibility decision
- no hyperparameter, threshold, normalizer, feature, action-scale, Risk, Execution, or accounting change
- no redefinition of gross economic metrics in this task

## 2. Objective

Build one fail-closed U2 replay boundary that can deterministically execute the fixed U1 policy/runtime semantics over one exact preregistered U2 evaluation scope and emit immutable raw evidence suitable for later metric aggregation.

The replay boundary must support four policy variants on the same scope:

- `candidate`: deterministic mean action from the supplied policy model
- `cash`: constant normalized action `0.0`
- `constant_long`: constant normalized action `+1.0`
- `constant_short`: constant normalized action `-1.0`

Each policy variant runs in a fresh U1 environment instance. Mutable environment, BookState, execution, pending-order, and action state must never be shared across variants.

## 3. Why the generic walk-forward evaluator is not reused

`trade_rl.workflows.walk_forward_evaluation.evaluate_range_evidence()` reconstructs a generic `ResidualMarketEnv` and configures terminal liquidation for range evaluation.

U1/U2 V1 instead freezes:

- `UniversalTradeEnvironment`
- external time-limit truncation
- fixed 720-hour episode horizon
- cash-only reset
- no finite-horizon observation
- no terminal liquidation

Reconstructing a generic evaluation environment would therefore alter the frozen U1 economic/runtime contract. Task 7C-1 must drive a verified `UniversalTradeEnvironment` directly.

Existing generic metric helpers may be reused later only when their input semantics match U2 evidence exactly.

## 4. Authoritative inputs

The replay boundary consumes only content-addressed, preregistered identity inputs plus synthetic test fixtures:

- `UniversalTradeRLUniverseManifest`
- `UniversalTradeRLU2TimePartition`
- `UniversalTradeRLU2Contract`
- `UniversalTradeRLU1Contract`
- frozen `UniversalTradePolicyContract`
- frozen `UniversalTradeSequenceNormalizer`
- a supplied U2 Development evaluation closure
- a source artifact locator/loader already constrained by Task 7B
- one exact scope identity
- one policy variant
- one deterministic policy model for `candidate`, or no model for fixed baselines
- one U1 environment factory that builds a fresh environment from the verified scope dataset

Real Development/Admission source material remains outside this implementation task.

## 5. Canonical closure rule

The caller-supplied Development evaluation closure is not authoritative by itself.

Before numeric source loading or environment construction, the runner must reconstruct the canonical closure from the authoritative manifest, U2 time partition, and U2 contract using the maintained closure builder. The supplied closure must match that canonical closure exactly, including its content digest and complete scope set.

This prevents a self-consistent but incomplete closure from silently omitting one B/C/D tile or one required symbol.

A mismatch is a hard contract error and must occur before the numeric loader is invoked.

## 6. Scope contract

One replay call evaluates exactly one canonical scope.

A scope binds at minimum:

- U0 manifest digest
- U1 contract digest
- U2 training contract digest
- concrete symbol
- symbol role
- cell identity
- window digest
- episode start timestamp
- episode stop timestamp
- exact source/common-view dataset identity
- execution/risk identity through the U1 contract
- policy variant
- candidate checkpoint identity when the variant is `candidate`

The scope must exist exactly once in the canonical closure.

Task 7C-1 must fail closed on unknown, duplicated, missing, or identity-drifted scopes.

## 7. Dataset rule

Task 7C-1 consumes only the Task 7B verified Train/Development common-view dataset for the selected scope.

The dataset must:

- contain exactly one concrete symbol
- preserve the canonical source dataset digest lineage
- cover the complete common view needed for U1 sequence history and the target evaluation window
- never contain or load Admission data
- use the frozen U1 feature order and policy contract
- use the frozen U1 normalizer without refitting

The replay runner does not implement a second independent dataset loader. It delegates source verification/materialization to the Task 7B boundary.

## 8. Episode alignment

For a target evaluation interval `[S, E)` expressed in decision-bar timestamps, the U1 environment reset starts at the immediately preceding state timestamp `S - one decision step`.

The frozen U1 episode is 720 hours at a 15-minute decision cadence:

```text
720 h * 4 decisions/h = 2880 decisions
```

For a normal complete U2 tile, replay must therefore observe exactly 2880 policy decisions and finish with the environment positioned at `E`.

The runner must verify the reset `start_index` and `end_index` against the scope timestamps rather than trusting arithmetic alone.

No caller may override `episode_hours`, `episode_bars`, initial state, or terminal accounting behavior.

## 9. Policy variants

### 9.1 Candidate

The candidate model is evaluated with deterministic inference only:

```python
model.predict(observation, deterministic=True)
```

The returned action must pass the frozen U1 strict scalar normalized-target-exposure contract. Stochastic evaluation is not allowed in Selection evidence.

### 9.2 Cash

Cash uses normalized action:

```text
0.0
```

It is the primary comparator required by the U2 design.

### 9.3 Constant long / short

Diagnostic static baselines use normalized actions:

```text
constant_long  = +1.0
constant_short = -1.0
```

They run through the same U1 Risk, Execution, Accounting, signal delay, and episode semantics as the candidate.

They are diagnostic context and do not replace cash as the primary comparator.

## 10. Fresh-environment isolation

Every `(scope, variant)` replay owns a fresh `UniversalTradeEnvironment` and fresh mutable runtime state.

The following may be shared because they are immutable/content-addressed:

- `MarketDataset` arrays if their implementation remains read-only
- frozen normalizer
- U1/U2 contracts
- policy contract

The following must not be shared across variants:

- `UniversalTradeEnvironment`
- `UniversalTradeMarketEnv`
- hybrid/shadow `BookState`
- pending target/order state
- executor mutable random state
- action diagnostics
- episode/runtime state

This requirement is verified by integration tests, not assumed from factory structure.

## 11. Normal completion versus economic termination

Normal scope completion under U1 V1 must satisfy:

- exact expected decision count
- final `current_index` equals the scope stop position
- `terminated == False`
- `truncated == True`
- terminal accounting mode is mark-to-market
- no terminal liquidation

Economic termination before the scope stop is not silently converted into a normal replay. It is retained as explicit evidence with its termination reason and shorter realized path, and the replay result marks the scope as not normally completed.

Infrastructure/code errors propagate as errors and must not be converted into economic evidence.

## 12. Raw replay evidence

Task 7C-1 emits raw immutable replay evidence, not the final U2 Selection gate decision.

The evidence records at minimum:

### Identity

- schema version
- canonical evaluation closure digest
- U0 manifest digest
- U1 contract digest
- U2 contract digest
- scope digest / scope identity
- concrete symbol and role
- cell/window identity
- dataset identity
- policy variant
- candidate checkpoint digest where applicable

### Episode/runtime

- reset/start/end indices or timestamps
- observed decision count
- normal-completion flag
- terminated/truncated flags
- termination reason if any
- terminal accounting mode

### Economic/accounting

- initial capital
- final net portfolio value
- net wealth ratio
- realized net log-return series or equivalent exact per-decision wealth-growth evidence
- maximum drawdown
- total turnover
- total execution cost
- funding PnL
- borrow cost
- trade/fill count
- rebalance count

### Exposure/action/execution diagnostics

- deterministic normalized action trace
- realized exposure trace sufficient to derive mean/p95 absolute exposure later
- target-change count
- submitted/executed change evidence sufficient for later counts
- sign-flip evidence
- hard-risk violation evidence
- unexplained execution rejection evidence
- meaningful-execution evidence inputs

Evidence must be content-addressable. Digest computation must include every field that can affect later Selection metrics.

## 13. Gross metric boundary

The U2 design requires gross log growth / gross wealth and gross-to-net retention, but the current maintained accounting object is directly authoritative for realized after-cost net wealth.

Task 7C-1 does **not** invent a gross definition by adding back selected cost fields.

Before Task 7C-2 computes gross metrics, a separate normative amendment must define exactly which accounting components are removed from net performance and how the resulting gross path is reconciled per decision.

Until that amendment is fixed, raw cost/funding/borrow/execution evidence is retained without claiming a gross return series.

## 14. Failure modes and fail-closed behavior

Task 7C-1 must reject or explicitly surface at least:

- supplied closure differs from canonical closure
- missing required scope/tile
- duplicated scope identity
- unknown scope identity
- wrong symbol or symbol role
- wrong cell/window bounds
- wrong common-view dataset identity
- Admission source access attempt
- U1 contract drift
- normalizer drift or refit
- policy-contract drift
- environment factory returning shared mutable env instances
- off-by-one reset alignment
- unexpected episode end index
- wrong decision count on normal completion
- normal tile ending as `terminated=True`
- terminal liquidation on a normal time-limit completion
- stochastic candidate inference
- malformed/non-finite/out-of-range action
- variant-specific Risk/Execution/runtime drift
- infrastructure exceptions being mislabeled as economic termination

## 15. Test oracle

Correctness is determined from observable contracts, not merely successful execution.

Required oracles include:

- canonical closure digest and exact scope equality
- loader call audit: zero calls before closure validation failure
- reset `start_index` / `end_index`
- environment `current_index`
- exact decision count
- `terminated` / `truncated`
- terminal accounting mode and liquidation evidence
- BookState initial/final portfolio values
- reconciliation of net wealth with realized per-step net log growth
- environment/U1/normalizer/policy digests
- action trace and deterministic predict calls
- mutable object identity across variant runs
- execution diagnostics and economic termination reason

## 16. Required test layers

### Unit

- replay request/result/evidence identity validation
- canonical closure equality and scope lookup
- policy variant validation
- evidence digest stability and drift sensitivity

### Integration

Using synthetic source artifacts and an actual `UniversalTradeEnvironment`:

- one complete 720-hour cash replay
- deterministic candidate replay
- constant long/short replay
- same-scope identity across all variants
- fresh mutable environment state across variants
- exact 2880-decision external truncation
- no terminal liquidation on normal completion

### Falsification / regression

- omit one canonical scope from a supplied closure and prove rejection before numeric loading
- wrong common-view identity
- wrong U1 runtime contract
- non-deterministic/mock model contract violation
- malformed action
- early economic termination is retained as evidence, not reported as normal completion
- factory object reuse is rejected
- Admission locator/loader remains unopened

### Repository quality checks

- focused U2 tests
- Ruff
- format check
- Mypy
- import architecture check
- dead-code/static checks required by CI
- full tests and coverage
- compatibility/training-image gates when triggered

## 17. Quality gate

Task 7C-1 is not complete unless all of the following hold on the same final commit:

1. canonical closure is independently reconstructed before numeric replay
2. exact scope identity is verified
3. real U1 environment semantics are used without runtime/economic mutation
4. deterministic candidate and fixed cash/long/short variants are supported
5. variant mutable state is isolated
6. normal completion is exact 2880-decision external truncation with no terminal liquidation
7. economic early termination is explicit evidence
8. raw evidence is immutable and content-addressed
9. no gross metric claim is made without the separate gross-definition amendment
10. no real Development or Admission numeric source is opened by this task
11. focused, integration, falsification, static, and full repository checks required by the change pass on the exact final HEAD
12. residual risks and unverified items are documented

Passing tests alone is not sufficient evidence of completion.

## 18. Follow-up boundary

After Task 7C-1 passes its quality gate:

- Task 7C-2 may define and compute the preregistered per-scope/cell metrics, after the gross-accounting amendment is frozen.
- Task 7D may implement seed/cell aggregation and moving-block bootstrap.
- Task 8 may implement the Development Selection AND rule.
- Admission remains sealed until Development Selection passes and separate authorization is created.
