# Universal Trade RL U2 Base Training / Development Gate Design

> Status: **DESIGN / Production NO-GO**
>
> U2 does not authorize Admission access, production deployment, profitability claims, or live trading. It defines one preregistered Base RL experiment and the Development gate that may reject that experiment before Admission is opened.

## 1. Conclusion

U2 V1 is a **single-candidate, eight-seed PPO experiment** over the U1 one-symbol Universal Trade environment.

The scientific question is not "which RL configuration backtests best?". It is:

> Can one fixed symbol-independent policy, trained only on U0 Train symbols and only before one global time cutoff, produce positive after-cost generalization on both unseen time and unseen symbols without selecting the best seed or best checkpoint?

U2 therefore freezes:

- one U0 universe generation;
- one deterministic global temporal partition derived from frozen source identities;
- one U1 contract and one U1 normalizer fitted only through the U2 fit cutoff;
- one U1-specific sequence-policy architecture;
- one exact PPO configuration and training budget;
- exactly eight deterministically derived training seeds;
- one performance-eligible final checkpoint per seed;
- fixed G1/G2/G3 Development scopes;
- fixed Cash / Buy-and-Hold / Trend baselines;
- fixed statistical aggregation and Development gates.

The primary Development question is **G3 joint generalization**: unseen Development symbols evaluated in a future interval that was excluded from normalization fit and RL gradient updates.

## 2. Ordering Constraint: U2 Temporal Design Precedes Final U1 Freeze

U2 execution depends on a completed U1 Quality Gate, but the U2 temporal boundary must be frozen before the final U1 normalizer is fitted.

Required order:

```text
U0 universe generation freeze
  -> U2 temporal contract materialization
  -> T_fit_end freeze
  -> U1 normalizer fit with knowledge_cutoff = T_fit_end
  -> U1 artifact / identity freeze
  -> U1 Quality Gate
  -> U2 Base RL training
  -> U2 Development evaluation
```

The U1 normalizer is learned statistical state. Fitting it with Train observations after `T_fit_end` would leak future distribution information into a time-OOS U2 evaluation even when no future return label is used.

Critical equality:

```text
U1 normalizer knowledge_cutoff
  == U0 RL_TRAINING provenance knowledge_cutoff
  == U2 T_fit_end
```

Any mismatch rejects the U2 generation before training.

## 3. Objective

Build an auditable Base RL experiment that can answer:

1. whether a pure U1 policy learns economically useful behavior beyond Cash;
2. whether that behavior survives future time on seen Train symbols;
3. whether it transfers to unseen Development symbols in pre-cutoff time;
4. whether it survives unseen Development symbols in future time;
5. whether the result is robust across stochastic training seeds;
6. whether aggregate profit is not explained by one seed, symbol, or episode;
7. whether gross edge survives the maintained execution-cost path.

## 4. Non-goals

U2 V1 does not:

- compare PPO/SAC/TD3/TQC or other algorithm families;
- tune architecture or hyperparameters from Development results;
- use Behavior Cloning, teacher actions, Causal Alpha, DAgger, anchored residual actions, or Trend priors;
- select a best seed;
- select a best intermediate checkpoint;
- use Development or Admission for normalization, gradient updates, calibration, threshold fitting, or reward tuning;
- change U1 Observation / Action / Reward semantics;
- reintroduce instrument descriptors or V4 cross-market context;
- perform 1-minute execution-fidelity research;
- perform multi-asset portfolio allocation;
- open Admission;
- claim profitability, live-market validity, or Production GO.

A scientific change after Development is observed creates a **new U2 generation**. Rejected evidence is never overwritten.

## 5. Existing Boundaries Reused

U2 reuses:

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
U2 authorized-episode wrapper
        |
        v
maintained SB3 PPO infrastructure
        |
        v
8 final checkpoints
        |
        v
Development evaluator / gate
```

No second symbol router, Risk engine, execution engine, accounting implementation, reward implementation, or normalizer implementation is introduced.

`ResidualMarketEnv` remains the sole Risk / Execution / Accounting authority under U1.

`EpisodeRoutedSingleInstrumentEnv` remains the symbol-routing authority and must be instantiated with:

```text
instrument_context_provider = None
v4_context_provider = None
```

## 6. Discovered U1-to-PPO Boundary

The existing `hierarchical_sequence_v2` `SequenceAssetFeatureExtractor` is **not** the U2 policy encoder. It requires legacy structured keys including:

```text
current_snapshot
asset_state
global_state
active
current_weights
```

U1 intentionally replaces those legacy planes with one versioned `policy_state` vector. Reintroducing the legacy keys would undermine the U1 contract.

U2 therefore introduces one narrow policy-side adapter:

```text
UniversalTradeSequenceFeatureExtractor
```

It consumes only the exact U1 observation keys and reuses existing causal temporal/fusion primitives. It does not change the U1 environment.

## 7. U2 V1 Policy Architecture

### 7.1 Input surface

The extractor accepts exactly:

```text
sequence_15m_values / available / staleness
sequence_1h_values  / available / staleness
sequence_4h_values  / available / staleness
sequence_1d_values  / available / staleness
policy_state
```

No concrete symbol string, dataset ID, instrument descriptor, alpha output, Trend state, baseline/shadow state, remaining horizon, or Admission identity may enter the tensor path.

### 7.2 Timeframe encoding

For each maintained timeframe:

```text
input_t = concat(
  normalized_values,
  availability_float,
  log1p(staleness_hours)
)
```

The implementation reuses:

- `CausalTimeframeEncoder`;
- `CrossTimeframeFusion`;
- `sequence_encoder_widths("standard")`.

Fixed latent widths:

```text
15m = 192
1h  = 192
4h  = 160
1d  = 128
```

Fixed fusion configuration:

```text
d_model = 336
timeframe_attention_heads = 8
timeframe_attention_layers = 2
timeframe_ffn_multiplier = 3
timeframe_gate_bias = -2.0
sequence_dropout = 0.05
```

There is no cross-asset attention because U1 has one instrument slot.

### 7.3 Policy-state context

`policy_state` width and field order are taken from the frozen U1 state-layout digest, never hard-coded independently.

Two deterministic encoders are built from the same U1 state vector:

```text
context encoder:
  state_width -> 256 -> 336

global critic/context encoder:
  state_width -> 256 -> 128
```

with LayerNorm + SiLU after hidden/projection layers.

The 336-wide state context is supplied to `CrossTimeframeFusion` as the one-instrument context token.

### 7.4 Reuse maintained bounded PPO policy

U2 reuses `SharedPerAssetActorCriticPolicy` with:

```text
shared_actor_n_symbols = 1
shared_actor_d_model = 336
shared_actor_global_dim = 128
shared_actor_head = "shared_target_v1"
```

The feature extractor emits the maintained policy layout:

```text
[fused instrument token]
[pooled token]
[128-wide state global]
[distribution-active mask]
[current_weight]
```

For U2 V1:

- `distribution-active mask` is always `1.0`;
- U1 `asset_active` and `tradable` remain explicit fields inside `policy_state`;
- Risk/Execution, not the policy adapter, decide whether a requested target can execute;
- `current_weight` is copied from the exact U1 state field and is also present in the versioned policy-state vector by design.

The constant distribution mask prevents hidden policy-side zeroing of requests while retaining the maintained bounded action distribution implementation.

### 7.5 No hierarchical Gate head

U2 V1 does **not** use `hierarchical_gate_target_v1`.

The actor learns the scalar normalized target exposure directly. Turnover suppression belongs to learned state dependence and the maintained Risk / Execution contract, not a second hidden Gate semantic.

### 7.6 Strict bounded action invariant

`SharedPerAssetActorCriticPolicy` already uses the maintained squashed Gaussian distribution. U2 requires rollout and deterministic actions to be finite and inside `[-1, +1]` **before** the environment receives them.

External SB3 action-space clipping must be an identity operation. A test must prove that the value produced by the policy and the value received by the U1 strict action parser are equal within the exact float32 transport tolerance. U2 must not rely on external clipping to make invalid actions legal.

## 8. Exact PPO Configuration

U2 V1 has one preregistered configuration. It reuses the maintained Universal PPO resource profile where appropriate, removes BC/teacher-dependent behavior, and uses the U1-specific encoder above.

```text
algorithm = ppo
timesteps = 524288
gamma = 1.0
learning_rate = 0.00012
learning_rate_schedule = linear
learning_rate_final_ratio = 0.1
n_envs = 8
n_steps = 128
batch_size = 256
n_epochs = 10
gae_lambda = 0.95
clip_range = 0.2
normalize_advantage = true
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5
log_std_init = -0.5
target_kl = 0.02
use_sde = false
policy = maintained bounded MultiInput PPO policy
observation_encoder = universal_trade_sequence_v1
policy_actor_head = shared_target_v1
policy_net_arch = (384, 256, 128)
value_net_arch = (512, 384, 256)
sequence_tcn_capacity = standard
sequence_d_model = 336
sequence_timeframe_attention_heads = 8
sequence_timeframe_attention_layers = 2
sequence_timeframe_ffn_multiplier = 3
sequence_timeframe_gate_bias = -2.0
sequence_dropout = 0.05
sequence_compile = false
sequence_compile_mode = reduce-overhead
sequence_transfer_mode = pinned_non_blocking
vector_environment_mode = subprocess
max_policy_parameters = 12000000
max_rollout_buffer_bytes = 805306368
checkpoint_interval_steps = 32768
max_checkpoints = 8
tensorboard_enabled = true
tensorboard_log_interval = 1
behavior_cloning_epochs = 0
behavior_cloning_critic_warm_start_steps = 0
behavior_cloning_joint_warm_start_steps = 0
```

`log_std_init=-0.5` intentionally uses the maintained pure-PPO default instead of the very low exploration scale used by the BC-initialized Universal U6 profile.

Any change to these values is a new U2 model-config digest and therefore a new generation.

## 9. U0 Universe Contract

U2 consumes exactly one U0 materialized universe generation.

- `Train`: normalization + RL training only inside FIT;
- `Development`: evaluation only;
- `Admission`: sealed and inaccessible to U2 execution;
- `Excluded`: inaccessible.

Role membership and source identities are immutable for the generation.

## 10. Deterministic Temporal Partition

### 10.1 Episode unit

The unit is the U1 fixed horizon:

```text
E = 720 hours
```

Fixed counts:

```text
DEV_EPISODES = 12
SEALED_EPISODES = 12
MIN_TRAIN_FIT_EPISODES = 24
G2_FIT_EPISODES = 12
```

Therefore DEV and reserved SEALED time are each 8640 hours (360 days), aligned to complete U1 horizons.

### 10.2 Boundary derivation

Temporal boundaries are derived from the frozen U0 manifest/source identities, not chosen after economic inspection.

Let:

```text
T_sealed_end = minimum last_timestamp_ns across
               Train + Development + Admission source identities

T_dev_end = T_sealed_end - 12 * E
T_fit_end = T_dev_end    - 12 * E
```

All timestamps must lie on the maintained 15-minute clock. A non-aligned source identity fails materialization.

Only U0 source identity metadata may be consulted to derive these cutoffs. Admission price/feature arrays and economic outcomes remain unopened.

### 10.3 Exact period semantics

- FIT episode: final accounting timestamp `<= T_fit_end`;
- DEV episode: initial decision timestamp `>= T_fit_end` and final accounting timestamp `<= T_dev_end`;
- SEALED episode: initial decision timestamp `>= T_dev_end` and final accounting timestamp `<= T_sealed_end`.

Lookback observations may naturally reference earlier timestamps. No reward, execution, or next-state accounting may cross the period's final cutoff.

### 10.4 Coverage requirements

Before training:

- every Train symbol must expose at least 24 valid complete FIT episodes;
- every Train symbol must expose all 12 fixed DEV evaluation episodes for G1;
- every Development symbol must expose the fixed latest 12 FIT episodes used by G2;
- every Development symbol must expose all 12 DEV episodes used by G3;
- Admission source identity metadata must cover the reserved 12-episode SEALED interval.

If coverage fails, do not shorten periods in place. Change the universe before freeze or create a new U0/U2 generation.

### 10.5 Fixed evaluation episode grids

G1 and G3 use the same 12 non-overlapping DEV episode intervals anchored by `[T_fit_end, T_dev_end]`.

G2 uses the latest 12 non-overlapping FIT episode intervals ending at `T_fit_end`.

The episode grid is part of the temporal-contract digest.

## 11. Authorized Episode Planning API

U2 must not duplicate the maintained environment's episode-validity logic.

Before U2 training, U1 must expose a read-only episode-planning surface equivalent to:

```text
valid_episode_starts()
episode_end_index(start_index)
```

for the fixed U1 720h contract, backed by the maintained episode sampler.

This API changes no economics and exposes no policy input. U2 uses it to create immutable per-symbol authorized FIT start sets and fixed Development episode plans.

Training child environments are wrapped by a thin U2 authorized-episode wrapper that:

- samples only from the precomputed Train/FIT authorized starts;
- forbids caller override to a non-authorized start;
- records the chosen start/end and temporal-contract digest;
- delegates exactly once to the U1 child environment.

The existing symbol router is unchanged.

## 12. Two-dimensional Generalization Matrix

```text
                       SYMBOL
                 Seen Train      Unseen Development
              +---------------+--------------------+
FIT time      | training only | G2 symbol-OOS      |
              +---------------+--------------------+
DEV time      | G1 time-OOS   | G3 joint-OOS       |
              +---------------+--------------------+
```

### G1 — time-OOS

```text
symbols = U0 Train
period  = fixed 12 DEV episodes
```

### G2 — symbol-OOS

```text
symbols = U0 Development
period  = latest fixed 12 FIT episodes
```

### G3 — joint-OOS — PRIMARY

```text
symbols = U0 Development
period  = fixed 12 DEV episodes
```

G1/G2/G3 are always reported separately. Strong G1/G2 cannot compensate for failed G3.

## 13. Fit Firewall

Only:

```text
U0 Train x FIT
```

may update learned/statistical state.

Forbidden as fit/update input:

- Train x DEV;
- Development x FIT;
- Development x DEV;
- all Admission;
- all Excluded.

This covers:

- feature normalization;
- PPO gradients;
- optimizer state;
- architecture/hyperparameter selection;
- reward coefficients;
- calibration/threshold fitting;
- performance early stopping;
- checkpoint selection.

Training-only diagnostics may detect implementation failure, NaN, zero gradients, or resource failure. They may not compare economic candidate performance because U2 has only one candidate.

## 14. U1 Dependency Contract

U2 accepts only a frozen U1 generation that passes U1 Quality Gate.

Required checks include:

- U0 universe/materialization digests;
- U1 contract/artifact/normalizer/provenance digests;
- exact observation/action/reward/state-layout identities;
- U1 runtime/Risk/Execution identities;
- unavailable market values are zeroed on the policy-facing value plane regardless of raw placeholder;
- U1 normalizer is present; `normalizer=None` is forbidden for Base Training;
- `U1.normalizer.knowledge_cutoff_ns == T_fit_end`;
- `production_status = NO-GO`.

## 15. Seed Contract

### 15.1 No circular seed identity

Seeds are derived before the final U2 contract digest from a seed namespace that excludes the resolved seed vector.

```text
seed_namespace_digest = SHA256(canonical_json({
  schema_version: "universal_trade_rl_u2_seed_namespace_v1",
  universe_manifest_digest,
  u1_artifact_digest,
  temporal_contract_digest,
  model_config_digest,
  seed_count: 8
}))
```

For `i = 0..7`:

```text
seed_digest_i = SHA256(canonical_json({
  schema_version: "universal_trade_rl_u2_seed_v1",
  seed_namespace_digest,
  index: i
}))

seed_i = unsigned big-endian uint32(first 4 bytes of seed_digest_i)
```

If any collision occurs among the eight resolved seeds, materialization fails closed. It does not silently probe another seed.

The ordered seed vector is then bound into the final U2 contract digest.

### 15.2 Seed domains

Each resolved seed deterministically binds:

- SB3/PPO seed;
- Python/NumPy/Torch seed inputs where maintained APIs expose them;
- Universal router `run_seed`;
- child-environment episode sampling seed derivation;
- execution RNG seed derivation.

U2 does not claim bit-for-bit CUDA determinism unless separately verified. Seed robustness is part of the scientific evidence.

### 15.3 No best seed

Every technically valid completed seed is evidence.

Forbidden:

- selecting the best Development seed;
- discarding a poor seed;
- replacing a poor seed with an extra seed;
- changing the seed vector after any Development result.

A technical failure may rerun only the exact same seed/run identity.

## 16. Symbol Routing Contract

Use `DeterministicBalancedInstrumentRouter` unchanged.

Invariant:

> Every complete routing cycle per environment contains each Train symbol exactly once.

Persist per seed/environment:

- completed episode count;
- routing cycle/position evidence;
- per-symbol episode count;
- deterministic incomplete final cycle, if any.

No symbol receives higher routing probability because its history is longer.

## 17. Training Budget / Checkpoint Contract

The fixed budget is `524288` SB3 timesteps per seed.

Intermediate checkpoints at the fixed checkpoint cadence exist for:

- crash recovery;
- NaN/divergence diagnostics;
- learning curves;
- exact resume.

They are never Development candidates.

Only the canonical final checkpoint is performance-eligible.

If the budget is changed after Development, that is a new generation.

## 18. Baselines

Evaluate on every G1/G2/G3 leaf with the same economic authority:

1. `CASH_FLAT` — requested target always zero;
2. `BUY_AND_HOLD_LONG` — fixed long target under the same Risk / Execution path;
3. `TREND_BASELINE` — maintained TrendStrategy only as an external benchmark.

Baseline comparisons use identical:

- episode timestamps;
- source data;
- fees/spread/impact;
- liquidity and partial fills;
- funding/borrow;
- Risk/margin semantics;
- terminal accounting semantics.

Trend never enters U1 policy input, reset state, reward, or action composition.

## 19. Immutable Economic Leaf

Atomic record:

```text
(seed, scope, symbol, episode_index)
```

Persist:

- episode start/end timestamps;
- initial/final wealth;
- gross return;
- after-cost net return and net log growth;
- Cash/BuyHold/Trend corresponding metrics;
- drawdown;
- turnover and turnover/day;
- execution cost;
- funding PnL;
- borrow cost;
- requested/executed/filled target evidence;
- trade/rebalance/fill counts;
- termination reason;
- hard Risk violation count;
- execution rejection reasons;
- policy/environment/source/contract identities.

Aggregate evidence must be exactly reconstructible from leaf records.

## 20. Statistical Aggregation

### 20.1 Equal-symbol and seed-robust aggregation

Do not pool raw rows or treat repeated seeds on the same market episode as independent time samples.

For each scope:

1. compute each seed/symbol's compound episode net log growth;
2. retain all eight seed results;
3. report seed median and worst valid seed;
4. report per-symbol median across seeds;
5. equal-weight symbols in cross-symbol summaries.

### 20.2 G3 primary time series

For each of the 12 fixed DEV episodes `j`:

```text
leaf_excess(seed, symbol, j)
  = policy_net_log_growth
    - CASH_FLAT_net_log_growth

symbol_episode_excess(symbol, j)
  = median over 8 seeds of leaf_excess

primary_excess(j)
  = median over Development symbols of symbol_episode_excess
```

This yields exactly 12 ordered time observations. Seed repetitions are collapsed before the time-series significance calculation.

### 20.3 Bootstrap

Use the existing moving-block mean test with fixed parameters:

```text
n_bootstrap = 2000
block_size = 3 episodes
confidence interval = existing 2.5% / 97.5% implementation
bootstrap_seed = uint32(first 4 bytes of SHA256(
  seed_namespace_digest || "u2-development-bootstrap-v1"
))
```

The bootstrap configuration is identity-bound. Development results cannot change it.

## 21. Development Gate

Development is a rejection gate for one frozen configuration.

### 21.1 Structural Hard Gate

Any of the following invalidates the run/evidence before economic interpretation:

- NaN/Inf policy output, reward, wealth, or required metric;
- U0/U1/U2 identity mismatch;
- source/provenance drift;
- unauthorized fit/update;
- U1 contract violation;
- policy action outside `[-1,+1]` before environment transport;
- hidden external action clipping changing the policy action;
- missing required seed/symbol/episode evidence;
- unexplained execution rejection;
- hard Risk invariant violation;
- evidence overwrite/tamper;
- economically poor seed retry/replacement.

### 21.2 G1 gate

```text
median across 8 seed-level G1 excess net log growth vs CASH_FLAT > 0
```

### 21.3 G2 gate

```text
median across 8 seed-level G2 excess net log growth vs CASH_FLAT > 0
```

### 21.4 G3 primary gate

All conditions are required:

```text
median across 8 seed-level G3 excess net log growth vs CASH_FLAT > 0
mean(primary_excess[12 episodes]) > 0
moving-block bootstrap lower_ci(primary_excess) > 0
positive seed count >= 6 / 8
median Development-symbol absolute net log growth > 0
positive Development-symbol excess-vs-Cash fraction >= 0.60
minimum required G3 leaf net return >= -0.05
for every seed: mean G3 turnover_per_day <= 1.0
G3 economic termination count = 0
```

The `-5%` episode floor and `1.0x/day` turnover ceiling reuse established repository research guardrails rather than being fitted to U2 Development.

Economic termination includes drawdown-stop, minimum-equity, margin-call, execution-cost exhaustion, and insolvency. Such a result is economically rejected even when software behavior is correct.

### 21.5 Trend secondary gate

For Development acceptance also require:

```text
median across 8 seed-level G3 excess net log growth vs TREND_BASELINE > 0
```

No significance claim versus Trend is made at U2; the requirement only prevents promoting a Base RL generation whose median result is weaker than the maintained simple strategy.

### 21.6 Buy-and-Hold

Buy-and-Hold is always reported but is diagnostic, not a U2 pass condition because long-market beta is not the Universal RL objective.

## 22. Seed-level and Symbol-level Statistics

For each seed and scope:

1. sum non-overlapping episode log growth per symbol;
2. compute equal-symbol median excess vs Cash;
3. the seed is positive when this statistic is `> 0`.

For each symbol and scope:

1. sum its non-overlapping episode log growth per seed;
2. take median across eight seeds;
3. report absolute growth and excess vs every baseline.

The positive Development-symbol fraction uses excess vs Cash.

Worst seed, worst symbol, worst episode, and `minimum G3 leaf return` are always published even when the generation passes.

## 23. Selection Semantics

There is one U2 V1 candidate.

Development result is only:

```text
DEVELOPMENT_ACCEPTED
or
DEVELOPMENT_REJECTED
```

No ranking occurs.

Failure diagnostics may classify:

- no edge;
- future-time failure;
- symbol-transfer failure;
- joint-OOS failure;
- seed instability;
- symbol concentration;
- cost collapse;
- downside breach;
- excessive turnover;
- economic termination.

Changing policy/training/gate semantics creates a new generation.

## 24. Admission Firewall

U2 does not open Admission.

Even after Development acceptance:

```text
Admission = CLOSED
Production = NO-GO
```

A later authorization must bind:

- U0 universe generation;
- frozen U1 identity;
- frozen U2 contract/model config/seed vector;
- complete Development leaf/summary/decision digests;
- accepted eight-policy set identity;
- proof of no post-Development refit or threshold change.

Between Development acceptance and Admission, no normalization, gradient update, calibration, reward tuning, threshold tuning, seed selection, or checkpoint selection is permitted.

## 25. Artifact / Identity Contract

Logical artifacts:

```text
u2_temporal_contract.json
u2_contract.json
u2_training_identity.json
seeds.json
training/<seed>/final-checkpoint + manifest
development/records/<scope>/<seed>/<symbol>/<episode>.json
development/summary.json
development/decision.json
```

`u2_temporal_contract.json` binds:

- U0 universe/source identity digest;
- episode semantic `720h`;
- derivation schema;
- `T_fit_end`, `T_dev_end`, `T_sealed_end`;
- G1/G2/G3 fixed episode grids;
- per-symbol authorized FIT start-set digests;
- coverage evidence.

`u2_contract.json` binds at minimum:

- schema version;
- U0 universe/materialization identities;
- U1 artifact/contract/normalizer identities;
- temporal-contract digest;
- exact policy architecture digest;
- exact PPO model-config digest;
- exact training budget;
- seed namespace and ordered seed-vector digest;
- router semantics;
- baseline identities;
- statistical aggregation/bootstrap identity;
- exact Development thresholds;
- evaluator/gate code identity;
- `production_status = NO-GO`.

U0 `BASE_TRAINING` run identity binds:

- U0 universe manifest digest;
- U2 model config digest;
- U1 FEATURE_NORMALIZATION provenance digest;
- U0 `RL_TRAINING` provenance digest.

## 26. Resume / Retry

Resume is accepted only when all immutable identity matches.

A valid persisted final checkpoint is never retrained because performance is poor.

A valid Development leaf is never recomputed and replaced after aggregate results are observed.

Crash between computation and durable publication may recompute the exact same leaf only under the exact same immutable inputs. Partial/corrupt/identity-drifted final evidence fails closed.

## 27. Invariants

1. U0 symbol roles remain disjoint.
2. Only Train x FIT updates statistical/model state.
3. U1 normalizer cutoff = RL training cutoff = `T_fit_end`.
4. No training execution/reward crosses `T_fit_end`.
5. Development/Admission never update model/statistical state.
6. U1 Observation / Action / Reward semantics are unchanged.
7. U2 Base Training requires a frozen U1 normalizer.
8. Policy input is exactly U1 observation; legacy structured planes are not reintroduced.
9. U2 action is bounded before environment transport; no hidden clipping is required.
10. One exact PPO configuration exists.
11. Exactly eight precommitted seeds exist.
12. No best-seed selection.
13. One performance-eligible final checkpoint per seed.
14. No best-checkpoint selection.
15. Train symbols are balanced by the maintained router.
16. Training episodes come only from authorized FIT start sets.
17. G1/G2/G3 remain separate.
18. G3 is primary.
19. All comparisons are after cost.
20. Baselines share equivalent economic execution semantics.
21. Aggregate evidence reconstructs from immutable leaves.
22. Development failure cannot be converted to success by editing gates in place.
23. Admission remains inaccessible.
24. U2 acceptance does not imply Production readiness.

## 28. Failure Modes

### Critical

- normalizer uses post-`T_fit_end` data;
- RL update uses non-Train or post-cutoff state;
- Development/Admission refit;
- legacy observation/prior reintroduced into U2 policy;
- unbounded PPO action is legalized by external clipping;
- best seed/checkpoint selected from Development;
- G3/gates changed after results;
- Admission accessed early;
- identity drift/evidence overwrite.

### High

- symbol histories receive unequal routing probability;
- evaluation episodes are not time-aligned;
- one seed/symbol explains aggregate profit;
- gross edge collapses after costs;
- minimum episode return below `-5%`;
- turnover exceeds `1.0x/day`;
- economic termination occurs;
- baseline uses cheaper execution assumptions;
- bootstrap treats seed replicas as independent time samples.

### Medium

- source coverage leaves too few complete FIT episodes;
- training budget ends with a deterministic partial routing cycle;
- checkpoint/logging overhead materially changes throughput;
- GPU stochasticity prevents bitwise reproduction while statistical seed contract remains intact.

## 29. Test Oracle

Correctness is not "PPO finished".

### Data/leakage

Observe:

- exact authorized Train symbols;
- exact episode start/end timestamps;
- normalizer/RL cutoff equality;
- zero Development/Admission fit provenance;
- zero training episode crossing FIT boundary.

### Policy architecture

Observe:

- exact U1 keys only;
- exact state-layout digest;
- expected sequence shapes/dtypes;
- no concrete symbol identity;
- causal timeframe encoder source rows;
- fixed architecture digest;
- bounded pre-environment actions;
- external clipping identity.

### Training

Observe:

- exact model config digest;
- exact seed namespace/vector;
- exact per-seed budget;
- router cycle/symbol counts;
- final checkpoint identity;
- resume identity.

### Evaluation

Observe:

- complete immutable leaf set;
- independent wealth/accounting reconciliation;
- exact G1/G2/G3 membership;
- baseline parity;
- exact aggregation reproduction.

### Selection

Observe:

- only final checkpoints are performance-eligible;
- all eight valid seeds included;
- no economic retry;
- exact precommitted gates;
- deterministic `development_decision` from immutable inputs.

## 30. Required Test Layers

- Unit: temporal derivation, episode plans, seed derivation, codecs, gate arithmetic;
- Property: timestamp boundaries, seed determinism/uniqueness, routing balance;
- Policy unit: U1 extractor shapes, state transform binding, action boundedness;
- Integration: U0 -> temporal -> U1 -> U2 environment/model assembly;
- PPO integration: rollout action transport, log-prob reevaluation, final checkpoint/resume;
- Falsification: post-cutoff normalizer, Development leakage, start override, seed substitution, checkpoint substitution, missing/tampered leaf;
- Economic integration: fee/spread/impact/funding/borrow/margin;
- Compatibility: U0/U1, existing router, maintained PPO policy/distribution;
- Static: Ruff, format, MyPy, import architecture;
- Full suite;
- package build;
- exact-final-HEAD CI;
- independent/falsification review.

## 31. Acceptance Criteria

U2 software is complete only when all are evidenced on one exact final HEAD:

1. One frozen U0 generation is bound.
2. Temporal boundaries are deterministically derived from frozen source identity metadata.
3. Exact 12 DEV and 12 reserved SEALED 720h periods are frozen.
4. Train/Development coverage requirements pass before training.
5. U1 final normalizer cutoff equals `T_fit_end`.
6. U1 Quality Gate is complete for all U2-relevant findings.
7. U1 missing-value policy invariant is fixed and tested.
8. `normalizer=None` is rejected for U2 Base Training.
9. Only Train x FIT can fit/update.
10. U1 episode-planning read API exists and matches maintained sampler behavior.
11. U2 authorized episode wrapper cannot escape FIT.
12. U2 extractor consumes only exact U1 observation keys.
13. Existing causal timeframe/fusion primitives are reused rather than duplicated.
14. Exact U2 architecture/model config is identity-bound.
15. Policy actions are valid before environment transport.
16. Exact eight-seed vector is deterministic and frozen.
17. Every seed receives `524288` timesteps.
18. Maintained balanced symbol router is used.
19. No intermediate checkpoint is performance-selected.
20. Exactly one final checkpoint per valid seed is performance-eligible.
21. G1/G2/G3 episode grids are immutable and auditable.
22. Cash/BuyHold/Trend baseline replay is economically comparable.
23. Complete immutable leaf evidence exists.
24. Aggregate/Bootstrap results reproduce from leaves.
25. Structural Hard Gate is fail-closed.
26. Exact economic gates are tested, including `6/8`, `60%`, `-5%`, `1.0x/day`, and zero economic termination.
27. Poor but valid seeds cannot be retried/replaced.
28. Development cannot trigger refit or in-generation gate change.
29. Admission remains inaccessible.
30. Targeted/Property/Integration/Falsification/Compatibility tests pass.
31. Ruff, format, MyPy, import architecture, full suite, package build pass.
32. Self-review and independent/falsification review have no unresolved substantive finding.
33. Required CI is green on the exact final HEAD.
34. Final report separates software validity from economic acceptance.

## 32. Development Outcome States

### Software valid, economic reject

```text
U2 software = VALID
U2 generation = DEVELOPMENT_REJECTED
Admission = CLOSED
Production = NO-GO
```

The rejection is a successful scientific result and remains durable.

### Software valid, economic accept

```text
U2 software = VALID
U2 generation = DEVELOPMENT_ACCEPTED
Admission = CLOSED
Production = NO-GO
```

A separate later authorization/design is required to open Admission.

## 33. Claims Allowed After Development Acceptance

Allowed limited statement:

> Under one frozen U0/U1/U2 contract, all eight preregistered Base PPO runs were evaluated without best-seed or best-checkpoint selection, and the policy set passed the preregistered after-cost Development gates including unseen Development symbols in future time.

Not established:

- final zero-shot Admission performance;
- unseen later-regime robustness beyond the reserved but unopened SEALED interval;
- live profitability;
- 1-minute/tick execution fidelity;
- Production readiness;
- superiority to all other strategies or RL algorithms.

## 34. Implementation Handoff

No scientific degree of freedom remains to be chosen from U2 Development results.

Implementation planning must now specify tasks for:

1. U1 episode-planning read API and remaining U1 Quality Gate closure;
2. temporal-contract materializer;
3. U2 authorized episode wrapper;
4. `UniversalTradeSequenceFeatureExtractor` and model assembly;
5. exact seed/run identity;
6. Base PPO training/resume artifacts;
7. deterministic G1/G2/G3 evaluator and baseline replay;
8. immutable Development leaves/aggregation/gate;
9. falsification and full Quality Gate.

All implementation changes follow Red -> Green -> Refactor. Test failure is not permission to weaken this design.
