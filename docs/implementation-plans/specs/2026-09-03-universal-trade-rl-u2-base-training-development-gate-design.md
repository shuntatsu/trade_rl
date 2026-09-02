# Universal Trade RL U2 Base Training / Development Gate Design

> Status: **DESIGN / Production NO-GO**
>
> U2 does not authorize Admission access, production deployment, profitability claims, or live trading. It defines one preregistered Base RL experiment and the Development gate that may reject that experiment before Admission is opened.

## 1. Conclusion

U2 V1 is a **single-candidate, eight-seed PPO experiment** over the U1 one-symbol Universal Trade environment.

The scientific question is not "which RL configuration backtests best?". It is:

> Can one fixed symbol-independent policy, trained only on U0 Train symbols and only before one global time cutoff, produce positive after-cost generalization on both unseen time and unseen symbols without selecting the best seed or best checkpoint?

U2 freezes:

- one U0 universe generation with sufficient zero-shot role counts;
- one deterministic temporal partition derived from frozen source identities;
- one U1 contract and one U1 normalizer fitted only through the U2 fit cutoff;
- one U1-specific sequence-policy architecture;
- one exact PPO configuration and training budget;
- exactly eight deterministically derived training seeds;
- one performance-eligible final checkpoint per seed;
- fixed G1/G2/G3 Development scopes;
- fixed Cash / Buy-and-Hold / Trend baselines;
- fixed statistical aggregation and Development gates.

The primary Development question is **G3 joint generalization**: unseen Development symbols evaluated in a future interval excluded from both normalization fit and RL gradient updates.

## 2. Ordering Constraint: U2 Temporal Design Precedes Final U1 Freeze

U2 execution depends on a completed U1 Quality Gate, but the U2 temporal boundary must be frozen before the final U1 normalizer is fitted.

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

The U1 normalizer is learned statistical state. Fitting it with Train observations after `T_fit_end` leaks future distribution information into time-OOS evaluation even when no future return label is used.

Critical equality:

```text
U1 normalizer knowledge_cutoff
  == U0 RL_TRAINING provenance knowledge_cutoff
  == U2 T_fit_end
```

Any mismatch rejects the generation before training.

## 3. Objective

Build an auditable Base RL experiment that can answer:

1. whether a pure U1 policy learns economically useful behavior beyond Cash;
2. whether that behavior survives future time on seen Train symbols;
3. whether it transfers to unseen Development symbols in pre-cutoff time;
4. whether it survives unseen Development symbols in future time;
5. whether the conclusion is robust across stochastic training seeds;
6. whether aggregate profit is not explained by one seed, symbol, or episode;
7. whether gross edge survives maintained execution costs.

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

`EpisodeRoutedSingleInstrumentEnv` remains the symbol-routing authority and is instantiated with:

```text
instrument_context_provider = None
v4_context_provider = None
```

## 6. Discovered U1-to-PPO Boundary

The existing `hierarchical_sequence_v2` `SequenceAssetFeatureExtractor` is **not** the U2 policy encoder. It requires legacy structured keys:

```text
current_snapshot
asset_state
global_state
active
current_weights
```

U1 intentionally replaces those policy planes with a versioned `policy_state` vector. Reintroducing the legacy observation surface would undermine U1.

U2 therefore introduces one narrow policy-side adapter:

```text
UniversalTradeSequenceFeatureExtractor
```

It consumes only exact U1 observation keys and reuses existing causal temporal/fusion primitives. It does not change U1 environment economics.

## 7. U2 V1 Policy Architecture

The architecture follows the previously designed **U-Medium Direct** direction rather than the larger BC-oriented Universal U6 profile.

### 7.1 Input surface

Exact keys:

```text
sequence_15m_values / available / staleness
sequence_1h_values  / available / staleness
sequence_4h_values  / available / staleness
sequence_1d_values  / available / staleness
policy_state
```

Forbidden tensor inputs include concrete symbol strings/IDs, dataset IDs, instrument descriptors, alpha outputs, Trend state, baseline/shadow state, remaining horizon, and Admission identity.

### 7.2 Timeframe encoding

For each timeframe:

```text
input_t = concat(
  normalized_values,
  availability_float,
  log1p(staleness_hours)
)
```

Reuse:

- `CausalTimeframeEncoder`;
- `CrossTimeframeFusion`;
- `sequence_encoder_widths("compact")`.

Fixed latent widths:

```text
15m = 192
1h  = 192
4h  = 160
1d  = 128
```

Fixed fusion configuration:

```text
sequence_tcn_capacity = compact
d_model = 256
timeframe_attention_heads = 4
timeframe_attention_layers = 1
timeframe_ffn_multiplier = 3
timeframe_gate_bias = -2.0
sequence_dropout = 0.0
```

No cross-asset attention is instantiated for one instrument.

### 7.3 Policy-state context

`policy_state` width/order come from the frozen U1 state-layout contract.

Two encoders consume the same exact U1 state vector:

```text
fusion context:
  state_width -> 256 -> 256

global critic/context:
  state_width -> 256 -> 128
```

using LayerNorm + SiLU. The 256-wide state context becomes the one-instrument context token supplied to `CrossTimeframeFusion`.

### 7.4 Maintained bounded PPO policy reuse

Resolve the policy class to:

```text
SharedPerAssetActorCriticPolicy
```

with:

```text
shared_actor_n_symbols = 1
shared_actor_d_model = 256
shared_actor_global_dim = 128
shared_actor_head = "shared_target_v1"
shared_actor_net_arch = (256, 128)
critic_hidden_dims = (256, 128)
```

The config-schema policy field remains `MultiInputPolicy`, but the model-assembly identity binds the resolved concrete policy class above.

The new feature extractor emits the maintained bounded-policy layout:

```text
[fused instrument token: 256]
[pooled token: 256]
[state global: 128]
[distribution-active mask: 1]
[current_weight: 1]
```

Total feature width is therefore `642`.

For U2 V1:

- `distribution-active mask` is always `1.0`;
- U1 `asset_active` and `tradable` remain explicit inside `policy_state`;
- Risk/Execution, not the policy adapter, decide whether a requested target can execute;
- `current_weight` is copied from the exact U1 state field and is also present in the versioned state vector intentionally.

This compatibility layout reuses the maintained squashed action-distribution implementation without reintroducing the legacy observation tensor.

### 7.5 No hierarchical Gate head

U2 does **not** use `hierarchical_gate_target_v1`.

The actor learns the scalar normalized target exposure directly. Turnover suppression is learned from state and enforced by maintained Risk/Execution, not a second hidden Gate semantic.

### 7.6 Strict bounded action invariant

`SharedPerAssetActorCriticPolicy` uses the maintained squashed Gaussian distribution. Rollout and deterministic actions must be finite and inside `[-1,+1]` before environment transport.

External SB3 action-space clipping must be an identity operation. Tests must prove the float32 policy action equals the value received by the U1 strict parser. U2 never relies on external clipping to legalize an invalid action.

## 8. Exact PPO Configuration

One configuration only:

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
policy = MultiInputPolicy
resolved_policy_class = SharedPerAssetActorCriticPolicy
observation_encoder = universal_trade_sequence_v1
policy_actor_head = shared_target_v1
policy_net_arch = (256, 128)
value_net_arch = (256, 128)
sequence_tcn_capacity = compact
sequence_d_model = 256
sequence_timeframe_attention_heads = 4
sequence_timeframe_attention_layers = 1
sequence_timeframe_ffn_multiplier = 3
sequence_timeframe_gate_bias = -2.0
sequence_dropout = 0.0
sequence_compile = false
sequence_compile_mode = reduce-overhead
sequence_transfer_mode = pinned_non_blocking
vector_environment_mode = subprocess
device = cuda
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

`log_std_init=-0.5` uses the maintained pure-PPO default rather than the low exploration scale used after BC warm start.

Any change is a new model-config digest and new U2 generation.

## 9. U0 Universe Contract and Minimum Role Counts

U2 consumes exactly one U0 materialized universe generation.

- `Train`: normalization + RL training only inside FIT;
- `Development`: evaluation only;
- `Admission`: sealed and inaccessible to U2 execution;
- `Excluded`: inaccessible.

U2 preserves the established Universal minimum zero-shot universe strength:

```text
non-excluded symbols >= 15
Train symbols        >= 9
Development symbols  >= 3
Admission symbols    >= 3
```

Larger frozen universes are allowed. U2 does not reshuffle roles after observing outcomes.

## 10. Deterministic Temporal Partition

### 10.1 Episode unit

```text
E = 720 hours
DEV_EPISODES = 12
SEALED_EPISODES = 12
MIN_TRAIN_FIT_EPISODES = 24
G2_FIT_EPISODES = 12
```

DEV and reserved SEALED time are each `8640h = 360d`, aligned to the U1 horizon.

### 10.2 Boundary derivation

From frozen U0 source identity metadata only:

```text
T_sealed_end = minimum last_timestamp_ns across
               Train + Development + Admission

T_dev_end = T_sealed_end - 12 * E
T_fit_end = T_dev_end    - 12 * E
```

All boundaries must lie on the maintained 15-minute clock. Admission price/feature arrays and economic outcomes remain unopened; only already-frozen source identity metadata is used.

### 10.3 Exact period semantics

- FIT episode: final accounting timestamp `<= T_fit_end`;
- DEV episode: initial decision timestamp `>= T_fit_end` and final accounting timestamp `<= T_dev_end`;
- SEALED episode: initial decision timestamp `>= T_dev_end` and final accounting timestamp `<= T_sealed_end`.

Lookback observations may reference earlier timestamps. Reward, execution, and next-state accounting cannot cross the period's final cutoff.

### 10.4 Coverage requirements

Before training:

- every Train symbol has at least 24 valid complete FIT episodes;
- every Train symbol covers all 12 fixed DEV episodes for G1;
- every Development symbol covers the latest fixed 12 FIT episodes for G2;
- every Development symbol covers all 12 fixed DEV episodes for G3;
- Admission source identity metadata starts no later than `T_dev_end` and ends no earlier than `T_sealed_end`.

Coverage failure does not shorten periods in place. The universe must change before freeze or a new U0/U2 generation must be created.

### 10.5 Fixed evaluation grids

DEV grid, for `j=0..11`:

```text
start_j = T_fit_end + j * E
end_j   = T_fit_end + (j + 1) * E
```

G1 and G3 use these same 12 intervals.

G2 uses the 12 immediately preceding non-overlapping intervals:

```text
start_j = T_fit_end - (12 - j) * E
end_j   = T_fit_end - (11 - j) * E
```

Exact start/end indices are resolved through the maintained U1 episode planner and bound into the temporal-contract digest.

## 11. Authorized Episode Planning API

U2 does not duplicate episode-validity logic.

Before U2 training, U1 exposes a read-only planning surface equivalent to:

```text
valid_episode_starts()
episode_end_index(start_index)
```

for the fixed 720h contract, backed by the maintained episode sampler.

U2 builds immutable per-symbol authorized FIT start sets. A thin U2 training wrapper:

- samples uniformly only from the authorized FIT starts using the seed-bound RNG;
- forbids non-authorized explicit start overrides;
- records chosen start/end + temporal-contract digest;
- delegates exactly once to U1.

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

G1/G2/G3 stay separate. G1/G2 cannot compensate for G3 failure.

## 13. Fit Firewall

Only:

```text
U0 Train x FIT
```

may update learned/statistical state.

Forbidden fit/update inputs:

- Train x DEV;
- Development x FIT;
- Development x DEV;
- all Admission;
- all Excluded.

This applies to normalization, PPO gradients, optimizer state, architecture/hyperparameter selection, reward coefficients, calibration/threshold fitting, performance early stopping, and checkpoint selection.

Training-only mechanics diagnostics may detect implementation failure, NaN, zero gradients, or resource failure. They may not compare multiple economic candidates because U2 V1 has one candidate.

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

### 15.1 Non-circular seed namespace

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

For `i=0..7`:

```text
seed_digest_i = SHA256(canonical_json({
  schema_version: "universal_trade_rl_u2_seed_v1",
  seed_namespace_digest,
  index: i
}))

seed_i = unsigned big-endian uint32(first 4 bytes of seed_digest_i)
```

A collision fails materialization. The resolved ordered vector is then bound into the final U2 contract digest.

### 15.2 Seed domains

Each seed binds maintained SB3/PPO, Python/NumPy/Torch inputs where exposed, Universal router `run_seed`, child episode sampling, and execution RNG derivation.

No bit-for-bit CUDA determinism claim is made unless separately verified.

### 15.3 No best seed

Every technically valid completed seed is evidence. Poor valid seeds are not discarded, replaced, or retried. Technical failure may rerun only the same seed/run identity.

## 16. Symbol Routing Contract

Use `DeterministicBalancedInstrumentRouter` unchanged.

> Every complete routing cycle per environment contains each Train symbol exactly once.

Persist routing cycles/positions, completed episodes, per-symbol counts, and any deterministic partial final cycle. History length cannot change symbol routing probability.

## 17. Training Budget / Checkpoint Contract

Each seed receives exactly `524288` SB3 timesteps.

Intermediate checkpoints exist only for crash recovery, diagnostics, learning curves, and exact resume. They are never Development candidates. Only the canonical final checkpoint is performance-eligible.

Changing the budget creates a new generation.

## 18. Baselines

Every G1/G2/G3 leaf is replayed with:

1. `CASH_FLAT` — target always zero;
2. `BUY_AND_HOLD_LONG` — long target under U1 Risk/Execution;
3. `TREND_BASELINE` — maintained TrendStrategy used only externally.

The policy and baselines use the same episode interval, source, execution costs, liquidity/partial-fill semantics, funding/borrow, Risk/margin, terminal accounting, and matched execution RNG identity where execution is stochastic.

Trend never enters U1 policy input, reset, reward, or action composition.

## 19. Immutable Economic Leaf

Atomic record:

```text
(seed, scope, symbol, episode_index)
```

Persist at minimum:

- start/end timestamps;
- initial/final wealth;
- gross return;
- after-cost net return / net log growth;
- Cash/BuyHold/Trend metrics;
- drawdown;
- turnover and turnover/day;
- execution cost;
- funding PnL;
- borrow cost;
- requested/executed/filled target evidence;
- trade/rebalance/fill counts;
- termination reason;
- hard Risk violations;
- execution rejection reasons;
- policy/environment/source/contract identities.

All aggregates reconstruct exactly from leaves.

## 20. Statistical Aggregation

### 20.1 Equal-symbol and seed-robust summaries

Do not pool raw rows or treat repeated seeds on one market episode as independent time observations.

For every scope retain all seed/symbol/episode leaves, seed median/worst seed, per-symbol medians, positive fractions, and equal-symbol summaries.

### 20.2 G3 primary time series

For each fixed DEV episode `j`:

```text
leaf_excess(seed, symbol, j)
  = policy_net_log_growth - CASH_FLAT_net_log_growth

symbol_episode_excess(symbol, j)
  = median over 8 seeds of leaf_excess

primary_excess(j)
  = median over Development symbols of symbol_episode_excess
```

This yields exactly 12 ordered time observations after seed repetitions are collapsed.

### 20.3 Fixed bootstrap

```text
bootstrap_namespace_digest = SHA256(canonical_json({
  schema_version: "universal_trade_rl_u2_development_bootstrap_v1",
  seed_namespace_digest,
  scope: "G3",
  statistic: "median_seed_then_median_symbol_cash_excess_v1",
  n_bootstrap: 2000,
  block_size_episodes: 3
}))

bootstrap_seed = unsigned big-endian uint32(
  first 4 bytes of bootstrap_namespace_digest
)
```

Use the existing moving-block mean test with:

```text
n_bootstrap = 2000
block_size = 3 episodes
existing 2.5% / 97.5% interval implementation
```

Development cannot change these values.

## 21. Development Gate

Development is a rejection gate for one frozen configuration.

### 21.1 Structural Hard Gate

Invalid evidence/run before economic interpretation if any occur:

- NaN/Inf policy output, reward, wealth, or required metric;
- U0/U1/U2 identity mismatch;
- source/provenance drift;
- unauthorized fit/update;
- U1 contract violation;
- policy action outside `[-1,+1]` before environment transport;
- external clipping changes the policy action;
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

All required:

```text
median across 8 seed-level G3 excess net log growth vs CASH_FLAT > 0
mean(primary_excess[12]) > 0
moving-block bootstrap lower_ci(primary_excess) > 0
positive seed count >= 6 / 8
median Development-symbol absolute net log growth > 0
positive Development-symbol excess-vs-Cash fraction >= 0.60
minimum required G3 leaf net return >= -0.05
for every seed: mean G3 turnover_per_day <= 1.0
G3 economic termination count = 0
```

The `-5%` episode floor and `1.0x/day` turnover ceiling reuse established repository research guardrails rather than being fitted to U2 Development.

Economic termination includes drawdown-stop, minimum-equity, margin-call, execution-cost exhaustion, and insolvency. It is an economic rejection even when software behavior is correct.

### 21.5 Trend secondary gate

Development acceptance also requires:

```text
median across 8 seed-level G3 excess net log growth vs TREND_BASELINE > 0
```

No significance claim versus Trend is made at U2.

### 21.6 Buy-and-Hold

Always reported, diagnostic only. Long beta is not the Universal policy objective.

## 22. Seed-level and Symbol-level Statistics

For each seed/scope:

1. sum non-overlapping episode log growth per symbol;
2. compute equal-symbol median excess vs Cash;
3. seed positive iff statistic `>0`.

For each symbol/scope:

1. sum episode log growth per seed;
2. take median across eight seeds;
3. report absolute growth and excess vs all baselines.

The Development positive-symbol fraction uses excess vs Cash.

Always publish worst seed, symbol, episode, and minimum G3 leaf return.

## 23. Selection Semantics

One candidate means Development result is only:

```text
DEVELOPMENT_ACCEPTED
or
DEVELOPMENT_REJECTED
```

No ranking.

Diagnostics may classify no edge, time failure, symbol-transfer failure, joint-OOS failure, seed instability, symbol concentration, cost collapse, downside breach, turnover breach, or economic termination.

Changing scientific semantics creates a new generation.

## 24. Admission Firewall

U2 never opens Admission.

Even after acceptance:

```text
Admission = CLOSED
Production = NO-GO
```

A later authorization binds U0, frozen U1, frozen U2 model/seed identities, complete Development evidence, accepted eight-policy-set identity, and proof of no post-Development fit or gate change.

No normalization, gradient update, calibration, reward tuning, threshold tuning, seed selection, or checkpoint selection occurs between Development acceptance and Admission.

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
- role-count evidence;
- episode semantic `720h`;
- temporal derivation schema;
- `T_fit_end`, `T_dev_end`, `T_sealed_end`;
- G1/G2/G3 grids;
- per-symbol authorized FIT start-set digests;
- coverage evidence.

`u2_contract.json` binds:

- U0 universe/materialization identities;
- U1 artifact/contract/normalizer identities;
- temporal-contract digest;
- exact U-Medium Direct/U1-adapter architecture digest;
- exact PPO model-config digest;
- exact training budget;
- seed namespace + ordered seed vector;
- router semantics;
- baseline identities;
- bootstrap/statistical identity;
- exact Development thresholds;
- evaluator/gate code identity;
- `production_status = NO-GO`.

U0 `BASE_TRAINING` run identity binds U0 universe, U2 model config, U1 FEATURE_NORMALIZATION provenance, and U0 RL_TRAINING provenance.

## 26. Resume / Retry

Resume requires complete immutable identity match.

A valid final checkpoint is never retrained because performance is poor. A valid Development leaf is never recomputed/replaced after aggregates are observed. Crash-before-publication may recompute only the exact same leaf under exact immutable inputs. Partial/corrupt/drifted final evidence fails closed.

## 27. Invariants

1. U0 roles remain disjoint and minimum role counts are met.
2. Only Train x FIT updates statistical/model state.
3. U1 normalizer cutoff = RL training cutoff = `T_fit_end`.
4. No training reward/execution crosses `T_fit_end`.
5. Development/Admission never update model/statistical state.
6. U1 Observation / Action / Reward semantics are unchanged.
7. U2 requires a frozen U1 normalizer.
8. Policy input is exact U1 observation; legacy planes are not reintroduced.
9. U2 action is bounded before environment transport.
10. U2 architecture is compact d256 U-Medium Direct with direct shared target head.
11. One exact PPO configuration exists.
12. Exactly eight precommitted seeds exist.
13. No best-seed selection.
14. One performance-eligible final checkpoint per seed.
15. No best-checkpoint selection.
16. Train symbols are balanced by the maintained router.
17. Training episodes come only from authorized FIT starts.
18. G1/G2/G3 remain separate; G3 is primary.
19. All comparisons are after cost.
20. Baselines use equivalent economics.
21. Aggregates reconstruct from immutable leaves.
22. Development failure cannot be converted to success by editing gates.
23. Admission remains inaccessible.
24. U2 acceptance does not imply Production readiness.

## 28. Failure Modes

### Critical

- normalizer uses post-`T_fit_end` data;
- RL update uses non-Train or post-cutoff state;
- Development/Admission refit;
- legacy observation/prior reintroduced;
- unbounded PPO action legalized by clipping;
- best seed/checkpoint selected from Development;
- G3/gates changed after results;
- Admission accessed early;
- identity drift/evidence overwrite.

### High

- insufficient role count for credible zero-shot claim;
- unequal symbol routing probability;
- misaligned evaluation episodes;
- one seed/symbol explains aggregate profit;
- gross edge collapses after cost;
- G3 leaf below `-5%`;
- turnover above `1.0x/day`;
- economic termination;
- cheaper baseline execution;
- bootstrap treats seed replicas as independent time samples.

### Medium

- insufficient complete FIT episodes;
- deterministic partial routing cycle at budget end;
- logging/checkpoint overhead materially changes throughput;
- GPU stochasticity prevents bitwise reproduction while seed evidence remains valid.

## 29. Test Oracle

Correctness is not "PPO finished".

### Data/leakage

Observe exact roles/counts, authorized timestamps/start sets, normalizer/RL cutoff equality, zero Development/Admission fit provenance, and zero training episode crossing FIT.

### Policy architecture

Observe exact U1 keys/state-layout digest, sequence shapes/dtypes, no concrete symbol identity, causal source rows, exact architecture digest/feature width, bounded pre-environment actions, and clipping identity.

### Training

Observe model-config digest, seed namespace/vector, per-seed budget, routing counts/cycles, final checkpoint identity, and resume identity.

### Evaluation

Observe complete immutable leaves, independent wealth reconciliation, exact G1/G2/G3 membership, baseline parity, and deterministic aggregate reproduction.

### Selection

Observe only final checkpoints eligible, all eight valid seeds included, no economic retry, exact preregistered gates, and deterministic Development decision from frozen inputs.

## 30. Required Test Layers

- Unit: temporal derivation, episode plans, seed derivation, codecs, gate arithmetic;
- Property: timestamp boundaries, seed determinism/uniqueness, routing balance;
- Policy unit: U1 extractor shapes, state binding, feature width 642, action boundedness;
- Integration: U0 -> temporal -> U1 -> U2 environment/model assembly;
- PPO integration: rollout action transport, log-prob reevaluation, checkpoint/resume;
- Falsification: post-cutoff normalizer, Development leakage, unauthorized start, seed/checkpoint substitution, missing/tampered leaf;
- Economic integration: fee/spread/impact/funding/borrow/margin;
- Compatibility: U0/U1, router, maintained squashed PPO policy/distribution;
- Static: Ruff, format, MyPy, import architecture;
- Full suite;
- package build;
- exact-final-HEAD CI;
- independent/falsification review.

## 31. Acceptance Criteria

U2 software is complete only when one exact final HEAD evidences:

1. Frozen U0 generation with `>=15` non-excluded, `>=9` Train, `>=3` Development, `>=3` Admission symbols.
2. Deterministically derived temporal boundaries.
3. Exact 12 DEV + 12 reserved SEALED 720h periods.
4. All Train/Development coverage requirements pass before training.
5. U1 final normalizer cutoff equals `T_fit_end`.
6. U1 Quality Gate is complete for U2-relevant findings.
7. U1 missing-value policy invariant is fixed/tested.
8. `normalizer=None` is rejected for U2 Base Training.
9. Only Train x FIT can fit/update.
10. U1 episode-planning read API matches maintained sampler behavior.
11. U2 authorized wrapper cannot escape FIT.
12. U2 extractor consumes only exact U1 keys.
13. Existing causal timeframe/fusion primitives are reused.
14. U-Medium Direct compact d256 architecture and feature width `642` are identity-bound.
15. Direct `shared_target_v1` head is used; hierarchical Gate head is absent.
16. Policy actions are bounded before environment transport.
17. Exact PPO configuration is identity-bound.
18. Exact eight-seed vector is deterministic/frozen.
19. Every seed receives `524288` timesteps.
20. Maintained balanced router is used.
21. No intermediate checkpoint is performance-selected.
22. Exactly one final checkpoint per valid seed is performance-eligible.
23. G1/G2/G3 grids are immutable/auditable.
24. Cash/BuyHold/Trend replay is economically comparable.
25. Complete immutable leaf evidence exists.
26. Aggregate/bootstrap results reproduce from leaves.
27. Structural Hard Gate is fail-closed.
28. Economic gates test `6/8`, `60%`, `-5%`, `1.0x/day`, zero economic termination, and positive Trend median excess.
29. Poor valid seeds cannot be retried/replaced.
30. Development cannot trigger refit or in-generation gate change.
31. Admission remains inaccessible.
32. Targeted/Property/Integration/Falsification/Compatibility tests pass.
33. Ruff, format, MyPy, import architecture, full suite, package build pass.
34. Self-review + independent/falsification review have no unresolved substantive finding.
35. Required CI is green on the exact final HEAD.
36. Final report separates software validity from economic acceptance.

## 32. Development Outcome States

### Software valid, economic reject

```text
U2 software = VALID
U2 generation = DEVELOPMENT_REJECTED
Admission = CLOSED
Production = NO-GO
```

The rejection is a valid scientific result and remains durable.

### Software valid, economic accept

```text
U2 software = VALID
U2 generation = DEVELOPMENT_ACCEPTED
Admission = CLOSED
Production = NO-GO
```

A separate later authorization/design opens Admission.

## 33. Claims Allowed After Development Acceptance

Allowed limited statement:

> Under one frozen U0/U1/U2 contract, all eight preregistered Base PPO runs were evaluated without best-seed or best-checkpoint selection, and the policy set passed the preregistered after-cost Development gates including unseen Development symbols in future time.

Not established:

- final zero-shot Admission performance;
- later-regime robustness beyond unopened SEALED time;
- live profitability;
- 1-minute/tick execution fidelity;
- Production readiness;
- superiority to all strategies or RL algorithms.

## 34. Implementation Handoff

No scientific degree of freedom remains to be selected from U2 Development results.

Implementation planning must specify tasks for:

1. U1 episode-planning read API and remaining U1 Quality Gate closure;
2. temporal-contract materializer;
3. U2 authorized episode wrapper;
4. `UniversalTradeSequenceFeatureExtractor` + model assembly;
5. exact seed/run identity;
6. Base PPO training/resume artifacts;
7. deterministic G1/G2/G3 evaluator + baseline replay;
8. immutable Development leaves/aggregation/gate;
9. falsification + full Quality Gate.

All implementation follows Red -> Green -> Refactor. Test failure is not permission to weaken this design.
