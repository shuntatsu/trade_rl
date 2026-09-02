# Universal Trade RL U2 Base Training / Development Gate Design

> Status: **DESIGN / Production NO-GO**
>
> U2 does not authorize Admission access, production deployment, profitability claims, or live trading. It defines the first fixed Base RL training experiment and the Development gate that may reject that experiment before Admission is opened.

## 1. Conclusion

U2 V1 is a **single-candidate, multi-seed Base RL experiment** over the U1 one-symbol Universal Trade environment.

The scientific question is not "which RL configuration backtests best?". It is:

> Can one fixed symbol-independent policy, trained only on U0 Train symbols and only before one global time cutoff, produce positive after-cost generalization on both unseen time and unseen symbols without selecting the best seed or best checkpoint?

U2 therefore freezes:

- one U0 universe generation;
- one global temporal partition;
- one U1 contract and one U1 normalizer fitted only through the U2 fit cutoff;
- one model/training configuration;
- one training budget;
- exactly eight deterministically derived training seeds;
- one final checkpoint per seed;
- fixed Development evaluation scopes and fixed baselines;
- fail-closed Development gates.

The primary Development question is **joint generalization**: unseen Development symbols evaluated in a future Development time interval that was excluded from both normalization fit and RL gradient updates.

## 2. Ordering Constraint: U2 Temporal Design Precedes Final U1 Freeze

U2 execution depends on a completed U1 Quality Gate, but the U2 temporal boundary must be frozen before the final U1 normalizer is fitted.

Required order:

```text
U0 universe generation freeze
  -> U2 temporal partition design freeze
  -> T_fit_end freeze
  -> U1 normalizer fit with knowledge_cutoff = T_fit_end
  -> U1 artifact / identity freeze
  -> U1 Quality Gate
  -> U2 Base RL training
  -> U2 Development evaluation
```

Reason: the U1 normalizer is learned statistical state. If it is fitted using Train observations after `T_fit_end`, future distribution information leaks into an otherwise time-OOS U2 evaluation even if no future return label is used.

This design therefore treats the equality below as a Critical invariant:

```text
U1 normalizer knowledge_cutoff
  == U2 RL_TRAINING knowledge_cutoff
  == T_fit_end
```

A U1 artifact whose normalizer cutoff exceeds `T_fit_end` is not valid for this U2 generation.

## 3. Objective

Build an auditable Base RL experiment that can answer, with fixed selection rules:

1. whether the policy learns anything economically useful beyond staying flat;
2. whether the result survives future time on seen Train symbols;
3. whether the result transfers to unseen Development symbols in already-known market time;
4. whether the result survives the hardest Development scope: unseen symbols in unseen future time;
5. whether the conclusion is robust across stochastic training seeds;
6. whether after-cost performance is not explained by one exceptional seed, symbol, or episode.

## 4. Non-goals

U2 V1 does not:

- search PPO/SAC/TD3 or other algorithm families;
- tune architecture, learning rate, entropy, discount, GAE, batch size, or other model hyperparameters using Development results;
- select the best seed;
- select the best intermediate checkpoint;
- fit or refit normalization on Development or Admission;
- fine-tune on Development or Admission;
- use Causal Alpha, Trend, BC, teacher actions, DAgger, anchored residual actions, or reward shaping;
- change U1 Observation / Action / Reward semantics;
- perform 1-minute execution-fidelity research;
- perform multi-asset portfolio allocation;
- open Admission;
- claim profitability, zero-shot production readiness, or Production GO.

Any later change to a scientific hyperparameter after observing Development results creates a **new U2 generation** and does not overwrite the rejected generation.

## 5. Maintained Architecture

U2 reuses existing maintained boundaries:

```text
U0 Universe / access firewall
        |
        v
U1 contract + normalizer + one-symbol environment
        |
        v
EpisodeRoutedSingleInstrumentEnv
        |
        v
existing framework adapter / PPO training path
        |
        v
one final checkpoint per seed
        |
        v
Development evaluator
```

No second Universal symbol router, execution engine, risk engine, accounting implementation, reward implementation, or normalizer implementation is introduced.

`ResidualMarketEnv` remains the sole Risk / Execution / Accounting authority under U1.

`EpisodeRoutedSingleInstrumentEnv` remains the symbol-routing authority. U2 V1 must instantiate it with:

```text
instrument_context_provider = None
v4_context_provider = None
```

so concrete ticker identity, instrument descriptors, and cross-market V4 context do not re-enter the U1 policy surface.

## 6. Algorithm Contract

U2 V1 uses the repository's maintained PPO/SB3 training path as the **single Base RL candidate**.

This is not a claim that PPO is optimal. It is chosen because U2 is testing the Universal policy/environment hypothesis, not conducting an algorithm tournament, and the repository already has maintained PPO training, checkpoint, resume, and capability infrastructure.

The complete PPO model/training configuration must be stored in one canonical `model_config_digest` before any Development evaluation begins.

No field that can materially change learning dynamics may be omitted from the digest. At minimum it binds:

- algorithm and implementation/version identity;
- policy architecture and feature extractor identity;
- optimizer and learning rate schedule;
- discount / GAE semantics;
- rollout length;
- batch size;
- epoch count;
- clipping semantics;
- entropy/value coefficients;
- gradient clipping;
- observation/action spaces and U1 contract digest;
- training budget;
- environment count and routing semantics;
- deterministic seed derivation schema.

The exact numeric hyperparameter values are a configuration freeze task for the implementation plan. U2 V1 permits one precommitted configuration only; Development cannot choose among multiple values.

## 7. Universe Contract

U2 consumes exactly one U0 materialized universe generation.

Required role semantics:

- `Train`: may contribute to normalization and RL training before `T_fit_end`;
- `Development`: evaluation only; never contributes to fit or gradient updates;
- `Admission`: inaccessible throughout U2 Base Training and Development Selection;
- `Excluded`: inaccessible.

Role membership, source dataset identities, and U0 materialization identity are immutable for one U2 generation.

If a symbol lacks sufficient history for the frozen temporal contract, do not shorten the evaluation period after seeing results. Either:

1. exclude it before the U0 generation is frozen; or
2. create a new U0/U2 generation.

## 8. Global Temporal Partition

### 8.1 Absolute timestamps, not per-symbol percentages

All eligible symbols use the same UTC boundaries:

```text
T_data_start < T_fit_end < T_dev_end <= T_data_end
```

The main intervals are:

```text
FIT = [T_data_start, T_fit_end]
DEV = (T_fit_end, T_dev_end]
```

A later sealed time interval may be reserved for post-Development research/Admission design, but U2 Development Selection itself does not open Admission.

Per-symbol 70/15/15 percentage splits are forbidden because listing dates differ and would place symbols in different market regimes.

### 8.2 Minimum intended duration

Production-candidate boundary selection should target, where available:

- FIT: at least 24 months;
- DEV: at least 12 months;
- later SEALED time: at least 12 months.

These are design targets, not permission to invent dates without inspecting the production source catalog. Final UTC timestamps are frozen only after catalog coverage is audited.

### 8.3 Complete episode semantics

U1 uses a fixed 720-hour episode horizon. Training and evaluation include only complete episodes whose required causal observation history and complete economic execution interval lie inside the authorized temporal scope.

No episode may cross from FIT into DEV.

Boundary fragments are excluded deterministically and their exclusion count is evidence.

## 9. Two-dimensional Generalization Matrix

Development evidence is separated into three scopes before aggregation.

```text
                       SYMBOL
                 Seen Train      Unseen Development
              +---------------+--------------------+
FIT time      | training only | G2 symbol-OOS      |
              +---------------+--------------------+
DEV time      | G1 time-OOS   | G3 joint-OOS       |
              +---------------+--------------------+
```

Definitions:

### G1 — Time generalization

```text
symbols = U0 Train
period  = DEV
```

Tests future-time behavior on symbols seen during training.

### G2 — Symbol generalization

```text
symbols = U0 Development
period  = FIT
```

Tests unseen symbols in a market period whose broad regime could overlap training.

### G3 — Joint generalization — PRIMARY

```text
symbols = U0 Development
period  = DEV
```

Tests unseen symbols in future time. G3 is the primary U2 generalization scope.

G1, G2, and G3 must remain separately reported. A strong G1 result cannot compensate for a failed G3 result.

## 10. Fit Firewall

Only this cell may update learned/statistical state:

```text
U0 Train x FIT
```

The following are prohibited inputs to all fit/update operations:

- Train x DEV;
- Development x FIT;
- Development x DEV;
- all Admission rows;
- all Excluded rows.

The rule applies to:

- feature normalization;
- RL gradient updates;
- optimizer state updates;
- architecture or hyperparameter choice;
- reward coefficient choice;
- calibration;
- population thresholds;
- early-stopping performance criteria;
- checkpoint selection.

## 11. U1 Dependency Contract

U2 requires a frozen U1 generation that passes its own Quality Gate.

At minimum U2 verifies:

- U1 contract digest;
- U1 normalizer digest;
- U1 normalizer provenance digest;
- U0 universe manifest digest;
- U0 materialization identity digest;
- U1 observation/state/action/reward digests;
- runtime/Risk/Execution identities bound by the final U1 contract;
- `production_status = NO-GO` remains unchanged.

Additionally:

```text
U1 normalizer.knowledge_cutoff_ns == T_fit_end
```

must hold exactly.

U2 must not accept a debug U1 environment with `normalizer=None` as a Base Training environment.

## 12. Seed Contract

### 12.1 Exactly eight training seeds

U2 V1 trains exactly eight stochastic runs.

The seeds are not manually chosen and are not chosen after observing returns.

Canonical derivation:

```text
seed_i = uint32(
  SHA256(
    u2_contract_digest
    || "universal_trade_rl_u2_seed_v1"
    || i
  )[0:8]
)

for i = 0..7
```

The implementation must specify exact byte/string serialization and collision handling. The resolved ordered seed vector is persisted in the U2 contract before training.

### 12.2 No best-seed selection

Every seed produces one final policy checkpoint. All eight are part of the scientific result.

Forbidden:

- selecting the best Development seed;
- discarding a weak seed as a failed training run when it completed validly;
- replacing a weak seed with an additional seed;
- changing the seed vector after any economic result is observed.

A technical execution failure may be rerun only under the exact same run identity and seed. A successfully completed but economically poor seed is evidence, not a retry condition.

## 13. Symbol Routing Contract

Use `DeterministicBalancedInstrumentRouter` without a second routing algorithm.

Invariant:

> In every complete routing cycle for each environment, each Train symbol appears exactly once.

The U2 training evidence records, per environment and seed:

- completed episode count;
- routing cycle;
- symbol episode counts;
- incomplete final cycle, if any.

Training budget should be expressed so the final intended budget ends on complete routing cycles where practical. If the framework budget cannot terminate exactly at a cycle boundary, the deterministic remainder is recorded and must not be chosen based on performance.

## 14. Training Budget / Checkpoint Contract

One training budget is fixed in the U2 model config before Development evaluation.

Intermediate checkpoints may exist only for:

- crash recovery;
- NaN/divergence diagnostics;
- learning-curve evidence;
- reproducible resume.

They are **not candidates**.

Development evaluates only the canonical final checkpoint for each seed.

Forbidden:

```text
20% / 40% / 60% / 80% / 100% checkpoints
  -> evaluate all
  -> choose best
```

If the fixed budget is later judged inadequate after observing Development, the next budget is a new U2 generation.

## 15. Baselines

At minimum evaluate these fixed baselines on every G1/G2/G3 scope under the same economic simulator contract:

1. `CASH_FLAT` — target exposure always zero;
2. `BUY_AND_HOLD_LONG` — long exposure under the same U1 Risk / Execution / cost contract;
3. `TREND_BASELINE` — existing maintained TrendStrategy used only as an external benchmark.

TrendStrategy must not enter the U1 policy observation, reset state, reward, or action composition.

Baseline comparison must use equivalent:

- timestamps and episode boundaries;
- fees;
- spread;
- impact/slippage;
- liquidity / participation constraints;
- funding;
- borrow cost;
- Risk / margin rules where semantically applicable.

A cheaper execution path for a baseline is not a valid comparison.

## 16. Economic Evidence Unit

The atomic evaluation record is:

```text
(seed, scope, symbol, episode)
```

For each record persist at minimum:

- initial/final wealth;
- after-cost net return / net log growth;
- gross return where available;
- maximum drawdown;
- turnover;
- execution cost;
- funding PnL;
- borrow cost;
- fill/rebalance/trade counts;
- termination reason;
- hard Risk violations;
- unexplained execution rejections;
- policy / environment / source identities.

Aggregate summaries must remain reconstructible from immutable leaf records.

## 17. Statistical Aggregation

U2 must not reduce eight seeds and multiple symbols to one unqualified mean.

Required evidence includes:

- per-seed metrics;
- per-symbol metrics;
- per-episode metrics;
- median across seeds;
- worst valid seed;
- positive-seed fraction;
- median across symbols;
- positive-symbol fraction;
- lower-tail / CVaR-style evidence;
- moving-block or other time-dependence-aware bootstrap interval for the primary after-cost excess-return statistic.

The existing seed robustness / block-bootstrap infrastructure should be reused where its contracts match, rather than implementing a second incompatible statistics stack.

All bootstrap seeds, block-size rules, resample counts, and aggregation formulas are precommitted in the U2 contract.

## 18. Development Gate

Development is a rejection gate for the single frozen U2 V1 candidate.

### 18.1 Structural Hard Gate

Any of the following rejects the generation before profitability interpretation:

- NaN/Inf policy output, reward, wealth, or required metric;
- U0/U1/U2 identity mismatch;
- source/provenance drift;
- observation/action/reward contract violation;
- unauthorized Development/Admission fit/update;
- unexplained execution rejection;
- hard Risk violation;
- insolvency;
- margin-call termination caused by a valid policy trajectory;
- missing required seed/symbol/episode evidence;
- retry/replacement of an economically poor but technically valid seed.

### 18.2 Economic Gate V1

The following are the proposed U2 V1 preregistered gates.

#### G1

```text
median seed after-cost excess net growth vs CASH_FLAT > 0
```

#### G2

```text
median seed after-cost excess net growth vs CASH_FLAT > 0
```

#### G3 — primary

All must hold:

```text
median seed after-cost excess net growth vs CASH_FLAT > 0
95% time-aware bootstrap lower bound of primary excess statistic > 0
positive seed count >= 6 / 8
median symbol after-cost net growth > 0
positive Development-symbol fraction >= 0.60
```

#### Trend secondary gate

```text
G3 median after-cost excess net growth vs TREND_BASELINE > 0
```

For U2 V1 this is a secondary gate, not permission to ignore a failed CASH/generalization gate.

### 18.3 Risk / concentration guardrails

Before final implementation, numeric thresholds for these fields must be frozen in the U2 contract from pre-Development rationale, not tuned to observed Development results:

- maximum per-episode drawdown;
- worst-symbol net wealth floor;
- lower-tail/CVaR loss limit;
- maximum cost-to-gross-profit ratio when gross profit is positive;
- maximum turnover / target-churn budget if required to prevent an execution-insensitive policy.

The implementation plan must identify each threshold's independent rationale. It may reuse already preregistered repository risk limits where those semantics match U2. It must not derive thresholds from U2 Development outcomes.

## 19. Selection Semantics

U2 V1 has exactly one candidate configuration.

Therefore Development produces only:

```text
ACCEPT_FOR_NEXT_RESEARCH_STAGE
or
REJECT_U2_GENERATION
```

It does not rank multiple model configurations.

If rejected, diagnostics may identify why, for example:

- no learned edge;
- seen-symbol only performance;
- future-time failure;
- symbol transfer failure;
- joint-OOS failure;
- seed instability;
- one-symbol concentration;
- cost collapse;
- downside/tail failure;
- excessive target churn.

Any scientific response that changes model/training semantics is a new generation with a new contract digest.

## 20. Admission Firewall

U2 Development Selection must not open U0 Admission.

Even if G1/G2/G3 pass, the result means only:

> the frozen Base RL candidate survived the preregistered Development gate.

It does not mean final zero-shot success.

Admission requires a later explicit authorization artifact bound to:

- U0 universe generation;
- exact frozen U1 identity;
- exact frozen U2 model config and seed vector;
- exact Development evidence digest;
- exact accepted candidate/policy-set identity;
- no post-Development refit or threshold change.

No normalization, gradient update, calibration, reward tuning, threshold tuning, seed selection, or checkpoint selection is permitted after Development acceptance and before Admission.

## 21. U2 Artifact / Identity Contract

The final implementation should produce canonical immutable artifacts analogous to U0/U1.

Minimum logical artifacts:

```text
u2_contract.json
u2_training_identity.json
seeds.json
training/<seed>/final-checkpoint + manifest
development/records/<scope>/<seed>/<symbol>/<episode>.json
development/summary.json
development/decision.json
```

`u2_contract.json` must bind at minimum:

- schema version;
- U0 universe manifest/materialization digests;
- U1 artifact / contract / normalizer digests;
- `T_data_start`, `T_fit_end`, `T_dev_end`;
- temporal-partition semantic version;
- complete-episode rule;
- model/training config digest;
- exact training budget;
- seed derivation schema and resolved ordered seed vector digest;
- symbol router digest/semantics;
- baseline identities;
- evaluation scope definitions;
- statistical aggregation identity;
- Development gate thresholds;
- software/code identity required for replay/resume acceptance;
- `production_status = NO-GO`.

The U0 `BASE_TRAINING` run identity should bind at least:

- U0 universe manifest digest;
- U2 model config digest;
- U1 FEATURE_NORMALIZATION provenance digest;
- U0 `RL_TRAINING` provenance digest.

The `RL_TRAINING` provenance cutoff must equal `T_fit_end`.

## 22. Resume / Retry Semantics

Training and evaluation may resume only when immutable identity matches.

A valid persisted final seed checkpoint must not be retrained simply because its Development performance is poor.

A valid persisted evaluation leaf must not be recomputed and replaced after the aggregate result is observed.

Crash between computation and durable publication may cause exact-scope recomputation only if the eventual durable record is required to match all immutable inputs and identities.

Corrupt, partial, unknown, or identity-drifted evidence fails closed; it is not silently repaired by overwriting final output.

## 23. Invariants

1. U0 Train/Development/Admission symbol roles remain disjoint.
2. Only Train x FIT can contribute to normalization or RL gradient state.
3. `U1.normalizer.knowledge_cutoff == RL_TRAINING.knowledge_cutoff == T_fit_end`.
4. No training episode crosses `T_fit_end`.
5. Development and Admission never update model/statistical state.
6. U1 Observation / Action / Reward semantics are unchanged.
7. U2 training requires a real frozen U1 normalizer; `normalizer=None` is not a valid Base Training surface.
8. Exactly one U2 V1 model/training configuration exists.
9. Exactly eight precommitted training seeds exist.
10. No best-seed selection.
11. Exactly one performance-eligible final checkpoint per seed.
12. No best-checkpoint selection.
13. Train symbols are episode-balanced by the maintained router.
14. G1/G2/G3 are evaluated and reported separately.
15. G3 is the primary Development generalization scope.
16. All economic comparisons are after cost.
17. Baselines use comparable economic execution semantics.
18. Aggregate evidence is reconstructible from immutable leaf records.
19. Development failure cannot be converted into success by changing gates in place.
20. Admission remains inaccessible throughout U2.
21. Passing U2 does not imply profitability or Production readiness.

## 24. Primary Failure Modes

### Critical

- U1 normalizer uses rows after `T_fit_end`;
- RL gradient or optimizer state uses any non-Train or post-cutoff observation;
- Development/Admission refit;
- best seed / best checkpoint chosen after Development;
- G3 definition changed after results;
- Admission accessed before authorization;
- identity drift or evidence overwrite;
- reward/accounting mismatch inherited from invalid U1 generation.

### High

- per-symbol percentage time splits create regime mismatch;
- one long-history symbol dominates training episodes;
- one seed or one symbol explains aggregate profit;
- gross profit disappears after cost;
- severe lower-tail loss despite positive average;
- excessive turnover/target churn creates execution-fragile results;
- evaluation baseline receives cheaper execution assumptions;
- incomplete episode boundary handling differs across symbols.

### Medium

- insufficient Development episode count for stable interval estimates;
- bootstrap configuration too weak for serial dependence;
- logging/checkpoint cadence adds large training overhead;
- final budget ends with a small symbol-routing imbalance.

## 25. Test Oracle

Correctness is not "training completed".

Required observable oracles include:

### Data / leakage

- exact authorized timestamp range for every fit sample;
- exact Train symbol set for every fit sample;
- normalizer cutoff equality;
- zero Development/Admission fit provenance;
- no episode crossing a temporal boundary.

### Training

- exact model config digest;
- exact resolved seed vector;
- exact per-seed training budget;
- exact routing counts/cycles;
- final checkpoint identity;
- optimizer/resume identity where applicable.

### Evaluation

- immutable `(seed, scope, symbol, episode)` leaf records;
- independent recomputation of after-cost wealth from accounting evidence;
- G1/G2/G3 scope membership;
- baseline parity;
- deterministic aggregate/gate recomputation from leaf evidence.

### Selection

- only final checkpoints evaluated for promotion eligibility;
- no omitted valid seed;
- no Development-driven retry;
- fixed gate thresholds before Development read;
- `development_decision` reproducible byte-for-byte from frozen inputs.

## 26. Required Test Layers

- Unit: temporal contract, seed derivation, identity codecs, gate arithmetic;
- Property: boundary timestamps, seed uniqueness/determinism, symbol routing balance;
- Integration: U0 -> U1 -> U2 training factory; U1 cutoff equality; Base Training provenance;
- Falsification: Development leakage, future-normalizer leakage, best-seed substitution, best-checkpoint substitution, missing leaf, tampered identity;
- Economic integration: fees/spread/impact/funding/borrow and margin behavior;
- Compatibility: existing U0/U1 and maintained PPO/universal router paths;
- Static Analysis: Ruff, format, MyPy, import architecture;
- Full suite;
- package build;
- exact-final-HEAD CI;
- independent/falsification review.

## 27. Acceptance Criteria

U2 V1 may be called software-complete only when all of the following are evidenced on one exact final HEAD:

1. One frozen U0 universe generation is bound.
2. One frozen temporal contract is bound.
3. Production-candidate absolute UTC boundaries are evidence-backed from source coverage.
4. U1 final normalizer cutoff equals `T_fit_end`.
5. U1 Quality Gate is complete with no unresolved substantive finding relevant to U2.
6. U2 accepts no `normalizer=None` Base Training environment.
7. Only Train x FIT can fit/update.
8. Exact fixed PPO configuration is canonical and identity-bound.
9. Exact eight-seed vector is deterministic and frozen before training.
10. Each seed receives the same fixed training budget.
11. Maintained balanced symbol router is used.
12. No intermediate checkpoint is performance-selected.
13. Exactly one final checkpoint per valid seed is promotion-eligible.
14. G1/G2/G3 scopes are immutable and independently auditable.
15. CASH, Buy-and-Hold, and Trend baselines are comparable after-cost evaluations.
16. Immutable leaf evaluation evidence exists for every required scope/seed/symbol/episode.
17. Aggregate summaries reproduce exactly from leaf evidence.
18. Structural Hard Gate is implemented fail-closed.
19. Economic gate calculations and thresholds are preregistered and tested.
20. Poor but valid seeds cannot be retried/replaced.
21. Development cannot trigger refit or in-generation threshold change.
22. Admission is inaccessible.
23. Targeted/Property/Integration/Falsification/Compatibility tests pass.
24. Ruff, format, MyPy, import architecture, full suite, package build pass.
25. Self-review and independent/falsification review find no unresolved substantive issue.
26. Required CI is green on the exact final HEAD.
27. Final report distinguishes software validity from economic acceptance.

## 28. Development Economic Acceptance

Software completion and economic acceptance are separate states.

### Software valid, economic reject

If all software Quality Gates pass but Development economic gates fail:

```text
U2 software = VALID
U2 generation = DEVELOPMENT_REJECTED
Admission = CLOSED
Production = NO-GO
```

The rejection is a successful scientific result and must remain durable.

### Software valid, Development accept

If all software and preregistered Development gates pass:

```text
U2 software = VALID
U2 generation = DEVELOPMENT_ACCEPTED
Admission = STILL CLOSED
Production = NO-GO
```

A separate later design/authorization is required to open Admission.

## 29. What U2 Can and Cannot Claim

If U2 passes Development, it can support the limited statement:

> Under one frozen U0/U1/U2 contract, the eight-seed Base PPO policy set showed preregistered positive after-cost Development generalization, including unseen Development symbols in the future DEV interval, without best-seed or best-checkpoint selection.

It still cannot claim:

- final zero-shot performance;
- robustness to an unopened Admission universe;
- live-market profitability;
- execution parity at 1-minute or tick fidelity;
- Production readiness;
- superiority to all alternative strategies or RL algorithms.

## 30. Handoff to Implementation Planning

Before writing the implementation plan, the remaining design-time values that must be resolved from evidence are:

1. production-candidate source catalog coverage;
2. exact `T_data_start`, `T_fit_end`, `T_dev_end` UTC timestamps;
3. exact fixed PPO hyperparameter configuration and training budget;
4. exact time-aware bootstrap configuration;
5. preregistered numeric drawdown/tail/cost/turnover guardrails.

These values must be resolved without reading U2 Development economic outcomes.

Implementation must then follow Red -> Green -> Refactor and preserve this contract. A test failure is not permission to weaken these gates.
