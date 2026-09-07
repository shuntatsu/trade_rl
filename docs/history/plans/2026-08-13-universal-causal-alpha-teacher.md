# Universal Causal Alpha Teacher Implementation Plan

> **Execution rule:** implement task-by-task with test-first RED -> minimal GREEN -> refactor -> focused verification. Do not weaken existing economic, leakage, architecture, or reproducibility gates to make the teacher pass.

**Goal:** Replace the rejected hindsight Oracle / fixed-trend Universal BC teacher with the approved deterministic train-only causal alpha teacher while preserving the existing episode-aligned BC/critic/SB3 interface and pure net-growth reward.

**Architecture:** Put framework-neutral causal labels, pooled NumPy ridge fitting, immutable model/controller contracts, and target conversion in `trade_rl.learning`. Put chronological episode partitioning, expanding-fit generation, candidate selection through production replay, evidence assembly, and Universal batch construction in `trade_rl.workflows`. Keep SB3/Torch out of workflow modules. Reuse `EpisodeOracleBatch` / `OracleEpisodeContract` as the maintained episode carrier, and reuse the existing Universal pretraining gate so a failed holdout still prevents critic warm-start and PPO updates.

**Primary contracts:** `MarketDataset` rows are bar closes and row `t` first executes at row `t+1`; Universal canonical data is 15m decisions with 720h episodes and one-decision signal delay; each of the nine train symbols reserves exactly its latest complete 720h episode; all signal fitting/scaling/config selection is train-only and must use labels fully realized strictly before the prediction episode start; validation/test symbols never enter fitting/selection/BC.

---

## Task 1: Add explicit teacher identity and pure causal signal primitives

**Files**
- Create: `trade_rl/learning/causal_alpha_teacher.py`
- Create: `tests/learning/test_causal_alpha_teacher.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `tests/rl/test_algorithm_configs.py`

**RED tests**
1. `ResidualTrainingConfig` accepts only the new explicit teacher token `causal_alpha_ridge` in addition to legacy diagnostic values; malformed values fail closed.
2. Forward-label helper proves a decision at close `t` starts economic labeling at the first executable bar and resolves exact 24h/72h endpoints on regular cadence; an unavailable full horizon is rejected.
3. Prefix scaler fits only rows whose label realization ends strictly before `knowledge_cutoff`; nonfinite rows are excluded, while feature-level unavailable entries use fitted-mean/standardized-zero semantics and constant or unavailable columns are recorded.
4. Pooled ridge over multiple symbol matrices is deterministic and byte-stable under canonical serialization; coefficient/config/data identity drift changes digest.

**Implementation**
- Define `CAUSAL_ALPHA_TEACHER_KIND = "causal_alpha_ridge"`.
- Add frozen dataclasses for label horizon, prefix scaling/model artifact, and pooled ridge configuration.
- Implement deterministic NumPy ridge with an explicit intercept (do not regularize intercept), finite/shape/rank checks, float64 solve, canonical float payload/digest.
- Keep label calculation gross; no fee/spread/impact subtraction here.

**Focused verification**
`uv run pytest tests/learning/test_causal_alpha_teacher.py tests/rl/test_algorithm_configs.py -q`

## Task 2: Implement the stateful target controller

**Files**
- Modify: `trade_rl/learning/causal_alpha_teacher.py`
- Modify: `tests/learning/test_causal_alpha_teacher.py`

**RED tests**
1. Entry threshold opens from cash; lower exit threshold returns toward cash.
2. A sign reversal requires the new-direction entry threshold.
3. There is no time/clock holding lock: a valid reversal may occur on the immediately following decision.
4. No-trade band suppresses small target changes.
5. Maximum target delta clips one decision change without exceeding configured exposure.
6. Non-zero episode initial weight is the controller's initial state.
7. 24h-only, 72h-only, and equal-horizon combination are deterministic and bounded by `tanh`.

**Implementation**
- Add immutable `CausalAlphaControllerConfig` with `exit_threshold < entry_threshold`, positive score scale, nonnegative no-trade band, positive max target delta, and maintained horizon combination enum.
- Implement one-step controller transition and path generation from explicit initial weights.
- Record submitted/suppressed/changed/sign-flip diagnostics alongside targets.

**Focused verification**
`uv run pytest tests/learning/test_causal_alpha_teacher.py -q`

## Task 3: Build deterministic chronological 720h episode partitions

**Files**
- Create: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Create: `tests/workflows/test_universal_causal_alpha_teacher.py`

**RED tests**
1. Complete episodes are non-overlapping and chronological within the fold and environment trainable range.
2. The latest complete episode is exactly the holdout; all prior complete episodes are selection/BC candidates.
3. In a nine-symbol synthetic partition, exactly nine holdouts are produced, one per train symbol.
4. Bindings outside `split="train"`, duplicate symbols, missing complete episodes, or dataset identity drift fail closed.
5. The explicit BC split marks only the latest episode as validation and never includes it in train indices.

**Implementation**
- Derive episode bars from the concrete environment's maintained 720h episode contract.
- Create chronological `OracleEpisodeContract` values with stable chronological `episode_index` and exact reset initial weights.
- Do not use uniform-with-replacement sampling for the causal teacher.
- Build an explicit `BehaviorCloningSplit` from collected episode provenance; do not approximate the holdout with `validation_fraction`.

**Focused verification**
`uv run pytest tests/workflows/test_universal_causal_alpha_teacher.py -q`

## Task 4: Expanding pooled fits and teacher target generation

**Files**
- Modify: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Modify: `trade_rl/learning/causal_alpha_teacher.py`
- Modify: `tests/workflows/test_universal_causal_alpha_teacher.py`
- Modify: `tests/learning/test_causal_alpha_teacher.py`

**RED tests**
1. For every prediction episode, all scaler/model fit labels have `label_end_index < episode.start`.
2. The 24h/72h fit sample counts and knowledge cutoffs are persisted.
3. Validation/test symbols cannot be passed to the fitter.
4. Prediction arrays/digests change on feature/model/cutoff/context identity drift.
5. The resulting batch target shapes exactly match `EpisodeOracleBatch` contracts and respect non-zero initial weights.

**Implementation**
- Extract selected target-local market features plus the maintained nine causal instrument descriptors at each eligible decision.
- Use expanding pooled fits across the nine train symbols only. Fit scaling statistics inside each allowed prefix.
- Fit the two horizons independently.
- Generate predictions and controller targets for each selection episode and each untouched holdout using a model whose labels end strictly before that episode start.
- Add immutable fit/prediction evidence payloads with canonical digests and code/config identity.

**Focused verification**
`uv run pytest tests/learning/test_causal_alpha_teacher.py tests/workflows/test_universal_causal_alpha_teacher.py -q`

## Task 5: Train-only grid selection through production execution replay

**Files**
- Modify: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Create: `tests/workflows/test_universal_causal_alpha_selection.py`

**RED tests**
1. Candidate ranking is lexicographic: lower-tail net return, mean net return, turnover/day, total execution cost.
2. A candidate with majority negative gross-return episodes is inadmissible.
3. A no-meaningful-trades candidate is inadmissible.
4. Risk-contract violation is inadmissible.
5. Latest holdout episode metrics are not available to candidate ranking (leakage sentinel test).
6. Complete grid metrics and selected config digest are deterministic.

**Implementation**
- Predeclare the bounded candidate grid in immutable config, with no data-derived grid expansion.
- Replay candidate target paths with `evaluate_episode_action_path` / maintained production environment execution rather than hand-coded cost math.
- Compute per-symbol/episode gross and net return, turnover, costs, trade counts, drawdown/risk evidence.
- Select only from earlier episodes; persist all candidate evidence before holdout evaluation.

**Focused verification**
`uv run pytest tests/workflows/test_universal_causal_alpha_selection.py -q`

## Task 6: Integrate causal batches into Universal U4/U5/U6 without changing policy surface

**Files**
- Modify: `trade_rl/workflows/universal_teacher_runtime.py`
- Modify: `trade_rl/workflows/universal_training_runner.py`
- Modify: `trade_rl/workflows/universal_stage_a_training.py`
- Modify: `trade_rl/workflows/universal_full_research_training.py`
- Modify: `trade_rl/integrations/universal_pretraining.py`
- Modify: `tests/workflows/test_universal_sb3_training_assembly.py`
- Modify: `tests/workflows/test_universal_full_research_training.py`
- Modify: `tests/workflows/test_universal_stage_a_training.py`
- Modify: `tests/integrations/test_universal_pretraining_bundle.py`

**RED tests**
1. `causal_alpha_ridge` routes to the causal package and never calls Oracle/trend builders.
2. U5 architecture candidates share one identical causal teacher package.
3. U6 PPO/Lagrangian/discounted share identical teacher actions/config digest; gamma only changes critic targets downstream.
4. Bundle contains exactly one causal validation episode per train symbol and train samples exclude all nine holdouts.
5. Validation/test symbols never enter fitting/selection.

**Implementation**
- Introduce an internal `UniversalTeacherPackage` carrying `batches` plus immutable teacher-selection/holdout evidence, while preserving `EpisodeOracleBatch` as the BC-facing carrier.
- Keep legacy Oracle/trend paths for diagnostics/compatibility but canonical U6 configs will point to `causal_alpha_ridge`.
- `assemble_universal_sb3_training_backend` accepts the shared package and builds the existing pretraining hook.
- U5 stops hard-coding `build_universal_oracle_batches`; share the configured teacher package exactly once.

**Focused verification**
`uv run pytest tests/workflows/test_universal_sb3_training_assembly.py tests/workflows/test_universal_full_research_training.py tests/workflows/test_universal_stage_a_training.py tests/integrations/test_universal_pretraining_bundle.py -q`

## Task 7: Fail-closed causal teacher admission evidence

**Files**
- Modify: `trade_rl/integrations/universal_pretraining.py`
- Modify: `trade_rl/learning/evaluation.py` only if a generic evidence type is required
- Create: `tests/integrations/test_universal_causal_teacher_admission.py`

**RED tests**
1. Selected teacher aggregate gross return must be non-negative.
2. Most train-symbol holdouts may not have negative gross return; one profitable symbol cannot carry the teacher.
3. The nine holdouts are evaluated exactly once after selection.
4. Teacher evidence is written before BC and has stable digest.
5. Teacher gate failure raises before `pretrain_universal_policy`, critic warm-start, or backend training can run.
6. Existing BC reconstruction/economic thresholds remain unchanged.

**Implementation**
- Add a causal-teacher pre-admission check at the start of the Universal pretraining hook.
- Persist selection grid, fit/cutoff evidence, per-symbol holdout teacher economics, aggregate evidence, and gate result.
- Do not repurpose Oracle regret as teacher admission; retain existing BC policy-vs-teacher economic gate after reconstruction.

**Focused verification**
`uv run pytest tests/integrations/test_universal_causal_teacher_admission.py tests/integrations/test_universal_pretraining_bundle.py -q`

## Task 8: Canonical configuration and documentation closure

**Files**
- Modify: `examples/binance-multitimeframe/universal-u6-ppo.json`
- Modify: `examples/binance-multitimeframe/universal-u6-lagrangian.json`
- Modify: `examples/binance-multitimeframe/universal-u6-discounted.json`
- Modify related canonical Universal configs only where they represent the same U6 path
- Modify: `docs/CONFIGURATION.md`
- Modify: `tests/workflows/test_universal_u6_example_configs.py`

**RED tests**
- Canonical U6 configs use `behavior_cloning_teacher="causal_alpha_ridge"` consistently across all three algorithms.
- Existing economic gate thresholds are unchanged.
- Reward remains pure net log growth with no added transaction-cost penalty.

**Focused verification**
`uv run pytest tests/workflows/test_universal_u6_example_configs.py tests/rl/test_algorithm_configs.py -q`

## Task 9: Architecture review, full verification, and real-data gate sequence

**Code verification on one exact head**
1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run mypy trade_rl`
4. `uv run lint-imports`
5. repository dead-code check
6. focused causal teacher test suite
7. full `uv run pytest` with existing coverage/critical-coverage checks
8. frontend/build/package identity checks through normal CI
9. PostgreSQL Catalog workflow

**Self-review**
- Re-read full diff for leakage, off-by-one label timing, cutoffs, split identity, duplicate cost accounting, train/validation/test scope, controller state, execution/risk replay, digest completeness, dependency direction, and dead temporary artifacts.
- Fix every issue found and rerun nearest tests, then full exact-head verification.

**Real-data execution, in approved order**
1. train-only counterfactual/economic diagnostic;
2. CUDA BC causal holdout admission;
3. deterministic reproduction of any passing admission;
4. PPO/Lagrangian/discounted three-update economic smoke;
5. only after all gates pass: three algorithms x three seeds x 524,288 timesteps plus existing audits/final report.

A failed gate is a research result, not a reason to lower thresholds. Preserve the failure generation and evidence before changing the next hypothesis.
