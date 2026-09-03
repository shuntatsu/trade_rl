# Universal Trade RL U2 Base PPO / Selection Implementation Plan

> Source of truth: `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`
>
> Production: **NO-GO**  
> Real U2 training: **NO-GO until real production-candidate U0/U1 freeze + fresh stack verification**

## 0. Quality contract

### Objective

U1を一切緩めず、Train symbols × FIT timeだけで3-seed Base PPOを学習し、DevelopmentのB/C/D cellとseed robustnessを固定gateで評価できるcontent-addressed U2 runtimeを実装する。

### Non-goals

- real training runの開始
- Admission open
- Production promotion
- algorithm / architecture / checkpoint / seed sweep
- BC / teacher / Lagrangian
- reward/action/U1 semantics変更

### Acceptance Criteria

1. U2 time partitionがU0 metadataだけから決定論的にmaterializeされる。
2. U1 normalizer cutoffとFIT endが一致しない場合fail closed。
3. `RL_TRAINING` provenanceはTrain symbols × FIT cutoffだけ。
4. U2 training contract digestがU1 contractとPPO recipeを完全bindする。
5. BASE_TRAINING run identityの`model_config_digest`はU2 training contract digest。
6. routed training envはU1 observation/action/rewardをそのまま使い、context providerは両方None。
7. Train future / Development / Admission dataをfitへ要求できない。
8. PPO seed setはexact `(0,1,2)`、primary seed 0、final checkpoint only。
9. B/C1/C2/D1/D2のscope closureが決定論的。
10. cash / +1 / -1 baselineがcandidateとexact same scopesで評価される。
11. primary-seed core gateとD seed-robustness gateがspecどおりAND評価される。
12. AdmissionはDevelopment pass + exact authorization前にnumeric access不可。
13. artifactsはcanonical / atomic / fail-closed。
14. targeted + falsification + integration + compatibility + static + full suite + build + exact-head CIがpass。

### Invariants

- U1 `u1_contract.json` digestは変更しない。
- policy tensorへticker/role/dataset identityを追加しない。
- rewardはpure realized after-cost wealth growthのまま。
- actionはstrict scalar target exposureのまま。
- FIT後のoutcomeはtrainingへ入らない。
- Development/Admissionはfitへ入らない。
- intermediate checkpointはSelection candidateにならない。
- Development結果でseed 0を差し替えない。
- Production statusはNO-GO。

### Critical Failure Modes

- future-time leakage
- Development/Admission fit leakage
- wrong U1 contract training
- checkpoint/seed cherry-pick
- same-scopeでないbaseline comparison
- Development後のgate/hyperparameter mutation
- Admission pre-open
- resume lineage drift
- one losing symbol / one lucky seedをaggregateが隠す

### Test Oracle

- canonical artifact digests
- exact time/bar boundaries
- source-access audit
- router cycle
- U1 observation/action/reward digest equality
- checkpoint lineage/timestep count
- per-scope BookState wealth
- same-scope baseline IDs
- exact cell membership
- gate metrics and rejection reasons
- sealed Admission access count

### Required Test Layers

- Unit
- Property/Falsification
- Contract
- Integration
- Compatibility
- Static analysis / Type check / Import architecture
- Full suite / branch coverage / build
- Exact-head CI
- Independent/falsification review

### Quality Gate

実U2 trainingを許可するのは、ソフトウェアQuality Gateに加えてreal U0/U1 freezeとfresh strong stack verificationが両方完了した後だけ。

---

## Task 1 — Mark historical architecture semantics as superseded

**Files**

- Modify `docs/architecture/universal-single-instrument-zero-shot-design.md`
- Add/verify tests that maintained docs do not claim old finite-horizon/BC U6 semantics are U2 V1 source of truth if documentation tests exist.

### Step 1 — Add top-level supersession notice

State that U0/U1 maintained docs + U2 preregistration override historical sections about:

- finite-horizon termination/time-to-go
- action clipping
- baseline reset
- BC/Lagrangian prerequisites
- instrument/V4 context
- architecture sweep

Do not delete historical rationale; label it legacy/background.

### Step 2 — Verify no maintained U2 doc references old U6 configs as executable source of truth

```bash
rg "finite_horizon_termination|behavior cloning|instrument_context" docs/UNIVERSAL_TRADE_RL.md docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md
```

Expected: references are only explicit prohibitions/supersession notes.

---

## Task 2 — U2 time partition artifact — RED first

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_time_partition.py`
- Create `tests/workflows/test_universal_trade_rl_u2_time_partition.py`

### Step 1 — RED: exact 60/10/10/20 bar partition

Build a synthetic U0 manifest with >=600 days of common 15m bars. Assert:

- common interval is `max(first)` / `min(last)`
- FIT / D1 / D2 / Admission bar sets are disjoint and complete
- `SEEN_TIME_PROBE` is exactly final 1,440h inside FIT
- all boundaries are canonical decision indices/timestamps

### Step 2 — RED: minimum data and episode closure

Reject:

- common interval <600d
- any evaluation window with <2 complete 720h tiles
- malformed/non-monotone timestamps
- different 15m cadence

### Step 3 — RED: deterministic metadata-only behavior

Mutate numeric market values without changing U0 source metadata. Time partition digest must remain identical.

### Step 4 — GREEN: strict immutable codec

Implement:

- `UniversalTradeRLU2TimePartition`
- exact payload / digest / `from_payload`
- deterministic bar-boundary rounding toward older data
- explicit episode tile definitions

No dataset arrays are accepted by the partition builder.

### Step 5 — Verify

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_time_partition.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_time_partition.py tests/workflows/test_universal_trade_rl_u2_time_partition.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_time_partition.py tests/workflows/test_universal_trade_rl_u2_time_partition.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_time_partition.py
```

---

## Task 3 — U2 fixed training contract / identity — RED first

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_contract.py`
- Create `tests/workflows/test_universal_trade_rl_u2_contract.py`

### Step 1 — RED: bind exact U0/U1/time/provenance

Require:

```text
u0 manifest digest
u1 contract digest
u1 normalizer digest
u1 cutoff == fit_end
RL_TRAINING provenance digest
provenance cutoff == fit_end
time partition digest
```

Mismatch must fail before model/environment creation.

### Step 2 — RED: fixed PPO V1 recipe

Assert exact resolved values from preregistration, including:

- PPO
- 524288 timesteps
- seeds `(0,1,2)`
- primary seed 0
- 8 env / 128 steps / batch 256 / 10 epochs
- LR / schedule / gamma / GAE / KL / log std
- U_MEDIUM_DIRECT
- no BC/warm start/Lagrangian
- both context providers disabled

Every semantic field change must change contract digest.

### Step 3 — RED: run identity binding

Construct `UniversalTradeRLRunIdentity(stage=BASE_TRAINING, ...)` with:

```text
model_config_digest = u2_training_contract.digest
fit_provenance_digests = (rl_training_provenance.digest,)
```

Reject alternate model-config digest or extra fit provenance.

### Step 4 — GREEN: strict codec/builder

Prefer a dedicated U2 contract artifact over changing U0 run-identity schema.

### Step 5 — Verify

Run U2 contract + U1 contract + U0 run-identity/provenance tests together.

---

## Task 4 — Training access preflight — prove FIT-only / Train-only before arrays

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_preflight.py`
- Create `tests/workflows/test_universal_trade_rl_u2_preflight.py`

### Step 1 — RED: Development/Admission access is rejected before loader call

Use spies/counters around source loaders. Requesting a Development/Admission symbol in training scope must fail with loader call count 0.

### Step 2 — RED: Train bar after FIT end is rejected before numeric read

Provide a source artifact that has more rows than FIT. Preflight may inspect artifact identity/metadata but must pass a bounded FIT slice/contract to the numeric loader.

### Step 3 — RED: U1 normalizer generation mismatch

Reject:

- wrong U0 manifest
- wrong normalizer provenance
- wrong cutoff
- wrong U1 contract digest

### Step 4 — GREEN

Return a frozen `U2TrainingSourceClosure` containing only verified Train symbol dataset identities and FIT bounds.

---

## Task 5 — Reuse routed environment with U1 concrete env — RED first

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- Create `tests/rl/test_universal_trade_u2_environment.py`
- Reuse `EpisodeRoutedSingleInstrumentEnv` / `DeterministicBalancedInstrumentRouter` unless a narrowly scoped adapter is necessary.

### Step 1 — RED: concrete U1 parity

For every routed symbol:

- child is `UniversalTradeEnvironment`
- same U1 policy contract digest
- same frozen universal normalizer digest
- observation/action spaces identical
- action shape `(1,)`
- reward reconciliation unchanged

### Step 2 — RED: no contexts

Assert exact:

```python
environment._instrument_context_provider is None
environment._v4_context_provider is None
```

Prefer public contract fields if added; do not make production code depend on private test access.

### Step 3 — RED: complete balanced cycles

Across multiple env indices/cycles, each Train symbol appears exactly once before repeat.

### Step 4 — RED: no state crossing

Trade on symbol A, complete episode, route symbol B. B must start cash with zero pending/order state. Returning to cached A later must also reset cleanly.

### Step 5 — RED: FIT-only episode starts

Thousands of deterministic resets across seeds/env indices must never produce an episode stop after FIT end.

### Step 6 — GREEN

Implement only the adapter/factory needed to build U1 child environments from U2 source closure.

Do not add U2-specific observation channels.

---

## Task 6 — Fixed PPO assembly and anti-cherry-picking contract

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_training.py`
- Create `tests/workflows/test_universal_trade_rl_u2_training.py`
- Reuse maintained SB3 assembly where possible.

### Step 1 — RED: exact algorithm projection

Verify resolved SB3 PPO parameters match U2 contract exactly. No omitted library default may silently change identity.

### Step 2 — RED: no BC / no Lagrangian / no alternate architecture

Attempt to enable any forbidden component and expect fail-closed.

### Step 3 — RED: seed set and primary seed

Reject:

- missing seed
- extra seed
- reordered/duplicate seed if canonical order matters
- primary seed !=0

### Step 4 — RED: final checkpoint only

Training may save recovery checkpoints, but `selection_candidate_checkpoint()` must accept only exact final timestep `524288` for the configured seed.

Reject `32768`, `262144`, or arbitrary “best” checkpoint.

### Step 5 — RED: exact resume lineage

Resume must preserve:

- U2 contract digest
- seed
- model architecture
- U1 digest
- source closure
- actual timesteps

Wrong lineage fails before optimizer update.

### Step 6 — GREEN

Add the smallest U2 orchestration around maintained PPO backend.

---

## Task 7 — Deterministic B/C/D evaluation scopes and baselines

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_evaluation.py`
- Create `tests/workflows/test_universal_trade_rl_u2_evaluation.py`

### Step 1 — RED: exact cell assignment

Create synthetic Train/Development symbols and timestamps. Assert exact scopes for:

- A
- B
- C1/C2
- D1/D2

Admission scope constructor must require authorization and remain unavailable here.

### Step 2 — RED: non-overlap tile contract

Ensure evaluation episode outcomes are disjoint within each window and fully contained.

### Step 3 — RED: deterministic policy evaluation

Selection candidate must use deterministic action. Stochastic evaluation artifacts must carry `diagnostic_only=true` and cannot be consumed by Selection.

### Step 4 — RED: exact-scope cash/static baselines

Candidate, cash, +1, -1 records must share identical:

- symbol
- start/stop
- U1 runtime/execution/risk identity
- source dataset digest

Changing any scope field rejects paired comparison.

### Step 5 — GREEN

Produce canonical per-scope replay/evidence records using realized BookState economics.

---

## Task 8 — U2 Selection evidence and gates — RED first

**Files**

- Create `trade_rl/evaluation/universal_trade_rl_u2_selection.py`
- Create `tests/evaluation/test_universal_trade_rl_u2_selection.py`

### Step 1 — RED: symbol-balanced aggregation

Construct a case where one winner dominates several losers. Balanced average alone may be >1, but minimum-symbol gate must reject.

### Step 2 — RED: core seed-0 gates

One parameterized test per rejection reason:

- balanced gross <=1
- balanced net <=1
- median symbol <1
- minimum symbol <1
- positive scopes <0.5
- CVaR10 < -0.01
- turnover p95 >1/day
- one symbol with no meaningful execution
- hard-risk violation
- unexplained execution rejection
- net/gross log retention <0.5

### Step 3 — RED: all B/C/D cells are mandatory

Make exactly one of B,C1,C2,D1,D2 fail. Overall Selection must fail with selected checkpoint null.

### Step 4 — RED: seed lottery rejection

Seed 0 strong, seed 1 strong, seed 2 below cash. D robustness must fail; implementation must not choose seed 0 alone.

### Step 5 — RED: moving-block bootstrap

Use deterministic synthetic paired return series to prove:

- positive lower CI passes
- zero/negative lower CI fails
- bootstrap seed/resample rule is reproducible

### Step 6 — RED: primary checkpoint closure

Even if seed 1 has best Development wealth, selected checkpoint must remain seed 0 final if all gates pass.

### Step 7 — GREEN

Return canonical `U2DevelopmentSelectionEvidence` with `promotion_eligible=false` and `admission_eligible` only when every preregistered condition passes.

---

## Task 9 — Admission authorization contract, without opening Admission

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_admission_authorization.py`
- Create `tests/workflows/test_universal_trade_rl_u2_admission_authorization.py`

### Step 1 — RED: failed Development cannot authorize

### Step 2 — RED: authorization binds exact seed-0 final checkpoint

Reject seed 1/2 or intermediate seed-0 checkpoint.

### Step 3 — RED: binds U0/U1/U2/time/Selection identities

Any changed digest invalidates authorization.

### Step 4 — RED: creating authorization does not read Admission arrays

Use source-access spies; numeric Admission read count must remain 0.

### Step 5 — GREEN

Only create the signed/content-addressed authorization contract. Actual Admission evaluator remains a follow-up after Development pass.

---

## Task 10 — Artifact publication

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_runner.py`
- Create `tests/workflows/test_universal_trade_rl_u2_runner.py`

Canonical preregistration/runtime outputs should include at least:

```text
u2/
  u2_time_partition.json
  u2_training_contract.json
  base_training_identity.json
```

Later training/evaluation output directories bind back to these digests.

Apply U0/U1 publication rules:

- canonical JSON
- staging
- file fsync
- directory fsync where supported
- one directory publish
- byte-identical rerun only
- drift/extra/partial write fail closed

---

## Task 11 — Synthetic end-to-end falsification

**Files**

- Create `tests/integrations/test_universal_trade_rl_u2_pipeline.py`

Use small synthetic symbols and short *test-only* timestep counts through an explicitly non-production test builder; production U2 contract must still fix 524288.

Verify:

1. Train-only/FIT-only source closure.
2. routed U1 child envs.
3. PPO model assembly.
4. final-checkpoint identity.
5. deterministic B/C/D evaluation.
6. same-scope cash pairing.
7. failed seed robustness rejects Selection.
8. passing synthetic evidence authorizes seed 0 only.
9. Admission arrays remain unopened.

Do not make production values configurable merely to simplify this test; use a distinct test fixture/factory.

---

## Task 12 — Documentation / full verification / independent review

**Files**

- Update `docs/UNIVERSAL_TRADE_RL.md`
- Update this plan if implementation discovers a real contract issue; record reason before changing the quality bar.
- Add a U2 handoff/report only after verification.

### Targeted verification

```bash
uv run pytest -q \
  tests/workflows/test_universal_trade_rl_u2_time_partition.py \
  tests/workflows/test_universal_trade_rl_u2_contract.py \
  tests/workflows/test_universal_trade_rl_u2_preflight.py \
  tests/rl/test_universal_trade_u2_environment.py \
  tests/workflows/test_universal_trade_rl_u2_training.py \
  tests/workflows/test_universal_trade_rl_u2_evaluation.py \
  tests/evaluation/test_universal_trade_rl_u2_selection.py \
  tests/workflows/test_universal_trade_rl_u2_admission_authorization.py \
  tests/workflows/test_universal_trade_rl_u2_runner.py \
  tests/integrations/test_universal_trade_rl_u2_pipeline.py
```

Then:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
uv run pytest -q --cov=trade_rl --cov-branch
```

Also require package build, Ubuntu/Windows compatibility, training image/non-root runtime, applicable PostgreSQL/Nautilus gates, and exact-final-HEAD CI.

### Independent / falsification review checklist

Reconstruct from the preregistration, not the implementation:

- Can Development or Admission influence fit before loader access?
- Can any future outcome bar enter training?
- Can a concrete ticker reach policy tensor/model identity?
- Can an intermediate checkpoint be selected?
- Can seed 1/2 replace seed 0 after results?
- Can one winner symbol hide a loser?
- Can one lucky seed hide a failing seed?
- Can flat/cash behavior pass via equality?
- Can gross gains with cost-destroyed net results pass?
- Can overlapping evaluation episodes inflate sample count?
- Can baseline use easier scopes/accounting?
- Can Admission be opened before exact authorization?
- Can old U6 finite-horizon/BC/context defaults re-enter via config defaults?

Fix any discovered issue and rerun from targeted tests through full verification.

---

## Task 13 — Real execution authorization gate

Do **not** launch production-candidate U2 training as part of software implementation.

Before launch, separately verify:

1. final U0 head / PR state and fresh strong verification;
2. final U1 head / PR state and exact-head verification;
3. real U0 artifacts;
4. real U2 time partition;
5. real U1 normalizer/U1 contract with cutoff exactly FIT end;
6. real RL_TRAINING provenance;
7. U2 training contract materialization;
8. source git SHA / image / lockfile identity;
9. Admission access audit still zero;
10. no uncommitted/debug/temp workflow state.

Only then may the exact preregistered 3-seed PPO run start.

If any precondition fails, report `U2 TRAINING NO-GO`; do not silently repair or relax the preregistration.
