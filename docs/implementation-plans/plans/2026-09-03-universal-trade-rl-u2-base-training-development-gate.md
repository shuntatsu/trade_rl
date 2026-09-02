# Universal Trade RL U2 Base Training / Development Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Every production change follows Red → Green → Refactor. Do not weaken, skip, delete, or rewrite an oracle merely to obtain Green.

**Goal:** Implement one preregistered eight-seed PPO Base RL experiment over the frozen U1 one-symbol environment, with a deterministic symbol×time split, Train×FIT-only updates, bounded identity-free policy input/output, immutable G1/G2/G3 Development evidence, and a fail-closed Development decision that never opens Admission.

**Architecture:** U2 does not create a second market simulator, symbol router, Risk engine, execution engine, reward function, or normalizer. It consumes one frozen U0 universe and one frozen U1 generation, adds a deterministic temporal contract and authorized-episode wrapper, adds a U1-specific sequence feature extractor to the maintained bounded PPO path, derives eight seeds from immutable identities, trains one final checkpoint per seed, replays G1/G2/G3 and fixed baselines through the same U1 economics, then computes one immutable Development ACCEPT/REJECT result.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Gymnasium, Stable-Baselines3 PPO, existing `ResidualMarketEnv`, `UniversalTradeEnvironment`, `EpisodeRoutedSingleInstrumentEnv`, `UniversalRoutedEnvironmentFactory`, `StableBaselines3Backend`, canonical JSON/SHA-256 artifacts, pytest, Hypothesis where useful, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-training-development-gate-design.md`

**Planning base:** U1 draft head `a26f21b2ff113b6d7e4b554ff9cc30d3355ed996`, with the U2 design branch currently stacked on that head. **Do not begin Task 1 production code until the blocking U1 gate below is satisfied and the U2 branch is synchronized to that exact final U1 head.**

---

## 0. Blocking prerequisite — not part of U2 implementation commits

U1 closure remains a separate change. Do not hide U1 fixes inside U2 commits.

Before Task 1:

1. Finish the existing U1 implementation plan Tasks 8–11:
   - `trade_rl/workflows/universal_trade_rl_u1_contract.py`;
   - `trade_rl/workflows/universal_trade_rl_u1_runner.py`;
   - adversarial accounting/state falsification;
   - docs/full verification/independent review/exact-head CI.
2. Fix the confirmed U1 unavailable-value invariant:
   - when `available == false`, the policy-facing value must be exactly zero even when no normalizer is supplied;
   - add the mutation/property test that changes the hidden raw placeholder across `0`, `±1`, `±1e9` and proves identical policy observation.
3. Freeze/document `UniversalTradeSequenceNormalizer.clip_value = 10.0` as part of U1 semantics and identity.
4. Add a read-only U1 episode-planning API backed by the existing `EpisodeContractSampler`, equivalent to:

```python
def valid_episode_starts(self) -> np.ndarray: ...
def episode_end_index(self, start_index: int) -> int: ...
```

   It must not modify reset/step economics.
5. Complete U1 Quality Gate on one exact U1 head.
6. Re-run the stronger U0/U1 stack verification on the exact final heads; do not rely on the stale U0 strong-verification SHA.
7. Synchronize the U2 branch to the exact final U1 head without force-push/history rewrite unless the user explicitly authorizes it.
8. Re-run the design-diff review. If the final U1 public surface differs materially from this plan, update this plan/spec before coding.

**Stop condition:** if any prerequisite above is not evidenced, U2 remains **NO-GO** and Task 1 does not start.

---

# Implementation Quality Contract

## Objective

- Train exactly one preregistered PPO configuration across U0 Train symbols using only FIT-time episodes.
- Preserve one-symbol deployment semantics and generic `INSTRUMENT` policy identity.
- Evaluate time-OOS, symbol-OOS, and joint-OOS separately.
- Reject a weak/unstable generation without tuning the same generation from Development results.
- Keep Admission inaccessible.

## Non-goals

- No algorithm tournament.
- No Development-driven hyperparameter search.
- No BC, teacher, Causal Alpha, DAgger, anchored residual, or Trend prior in the policy.
- No best-seed or best-checkpoint selection.
- No U1 Observation/Action/Reward change.
- No multi-asset allocation.
- No 1-minute execution-fidelity project.
- No Admission opening, live trading, profitability claim, or Production GO.

## Critical Invariants

1. `U1 normalizer cutoff == U0 RL_TRAINING cutoff == U2 T_fit_end`.
2. Only `Train × FIT` can update statistical/model/optimizer state.
3. U2 policy input is exactly the U1 Dict observation; no legacy planes or concrete symbol identity re-enter.
4. Policy output is finite and already inside `[-1, 1]` before environment transport.
5. External action clipping is an identity operation.
6. Every complete router cycle contains each Train symbol exactly once per environment.
7. Exactly eight precommitted seeds exist; a poor valid seed is never replaced.
8. Only the final `524288`-timestep checkpoint per seed is performance-eligible.
9. G1/G2/G3 use immutable episode grids.
10. Development evidence is immutable and reconstructible from leaf records.
11. Admission remains closed even if Development passes.

## Primary Failure Modes

- time leakage through normalizer or episode boundaries;
- Development/Admission fit leakage;
- U1 state layout drift;
- legacy observation or context reinjection;
- hidden action clipping;
- wrong seed substitution;
- uneven symbol routing;
- intermediate checkpoint selection;
- baseline/economic path mismatch;
- seed replicas treated as independent time samples;
- missing/tampered leaf evidence;
- aggregate winner masking a catastrophic seed/symbol/episode;
- post-result threshold edits.

## Risk

- **Critical:** leakage, Admission access, identity drift, hidden action alteration, best-seed/checkpoint selection.
- **High:** wrong episode alignment, economic execution mismatch, unstable seed result, severe downside/turnover, incomplete evidence.
- **Medium:** performance/resource overhead, deterministic incomplete router cycle, CUDA non-bitwise reproducibility.

## Test Oracle

Correctness is observed through:

- exact source roles/digests/cutoffs;
- exact authorized episode start/end timestamps;
- exact U1 observation keys/shapes/state-layout digest;
- actual pre-environment actions;
- actual PPO model architecture/config/checkpoint identity;
- exact router cycle counts;
- independent wealth/accounting reconciliation;
- immutable per-leaf Development records;
- aggregate recomputation from leaves;
- deterministic gate outcome from frozen thresholds;
- final diff/status/HEAD/CI evidence.

## Required Test Layers

Unit + Property/Falsification + Integration + PPO Integration + Economic Integration + Compatibility + Static Analysis + Full Suite + Build + exact-final-HEAD CI + Independent/Falsification Review.

## Quality Gate

Task completion requires the evidence in Task 13 on one exact final HEAD. Targeted tests alone are insufficient.

---

# File Map

## Create

- `trade_rl/workflows/universal_trade_rl_u2_temporal.py`
- `trade_rl/rl/universal_trade_u2_environment.py`
- `trade_rl/rl/universal_trade_policy.py`
- `trade_rl/workflows/universal_trade_rl_u2_contract.py`
- `trade_rl/workflows/universal_trade_rl_u2_training.py`
- `trade_rl/workflows/universal_trade_rl_u2_evaluation.py`
- `trade_rl/workflows/universal_trade_rl_u2_development.py`
- `trade_rl/workflows/universal_trade_rl_u2_runner.py`
- `scripts/run_universal_trade_rl_u2.py`
- `tests/workflows/universal_trade_rl_u2_test_support.py`
- task-specific tests listed below.

## Modify narrowly

- `trade_rl/rl/training_modes.py`
- `trade_rl/rl/training.py`
- `trade_rl/integrations/sb3_model_assembly.py`
- `trade_rl/rl/policy_identity.py`
- `trade_rl/artifacts/policy_identity_contract.py`
- `docs/UNIVERSAL_TRADE_RL.md`
- `docs/CONFIGURATION.md`
- `tests/test_architecture_contract.py` only if required by the architecture checker.

## Reuse — do not duplicate

- `trade_rl/rl/environment_episode.py::EpisodeContractSampler`
- U1 final read-only episode planning API
- `trade_rl/rl/universal_episode_router.py::DeterministicBalancedInstrumentRouter`
- `trade_rl/rl/universal_single_instrument_env.py::EpisodeRoutedSingleInstrumentEnv`
- `trade_rl/workflows/universal_training_runner.py::UniversalRoutedEnvironmentFactory`
- `trade_rl/rl/sequence_policy.py::CausalTimeframeEncoder`
- `trade_rl/rl/timeframe_fusion.py::CrossTimeframeFusion`
- `trade_rl/rl/policies.py::SharedPerAssetActorCriticPolicy`
- `trade_rl/integrations/sb3_training.py::StableBaselines3Backend`
- `trade_rl/evaluation/bootstrap.py::moving_block_mean_test`
- Risk / Execution / BookState / U1 reward.

---

# Task 1: Pure deterministic U2 temporal contract

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_temporal.py`.
- Create `tests/workflows/universal_trade_rl_u2_test_support.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_temporal.py`.

**Produces**

```python
U2_EPISODE_HOURS = 720
U2_DEV_EPISODES = 12
U2_SEALED_EPISODES = 12
U2_MIN_TRAIN_FIT_EPISODES = 24
U2_G2_FIT_EPISODES = 12

@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2EpisodeInterval:
    start_timestamp_ns: int
    end_timestamp_ns: int

@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TemporalContract:
    universe_manifest_digest: str
    t_fit_end_ns: int
    t_dev_end_ns: int
    t_sealed_end_ns: int
    g1_dev_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    g2_fit_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    g3_dev_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    authorized_fit_start_digests: tuple[tuple[str, str], ...]
    coverage_digest: str
    digest: str = ""
```

### Step 1 — RED: timestamp/role-count tests

Write tests that prove:

- Train count `< 9` rejects;
- Development count `< 3` rejects;
- Admission count `< 3` rejects;
- `T_sealed_end = min(last_timestamp)` over Train+Development+Admission manifest entries;
- `T_dev_end = T_sealed_end - 12*720h`;
- `T_fit_end = T_dev_end - 12*720h`;
- all three cutoffs are 15-minute aligned;
- G1 and G3 contain exactly the same 12 DEV intervals;
- G2 contains exactly the latest 12 FIT intervals;
- mutating one source identity changes the temporal digest.

Example RED oracle:

```python
def test_temporal_contract_uses_common_absolute_cutoffs() -> None:
    manifest = make_u2_manifest(train=9, development=3, admission=3)
    contract = derive_u2_temporal_boundaries(manifest)
    assert len(contract.g1_dev_intervals) == 12
    assert contract.g1_dev_intervals == contract.g3_dev_intervals
    assert len(contract.g2_fit_intervals) == 12
    assert contract.t_fit_end_ns < contract.t_dev_end_ns < contract.t_sealed_end_ns
```

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_temporal.py -q
```

Expected: import/missing-symbol failures because U2 temporal code does not exist.

### Step 3 — GREEN: implement pure derivation

- Consume only `UniversalTradeRLUniverseManifest` metadata.
- Do not open datasets.
- Use integer nanoseconds; no local-time datetime arithmetic.
- Canonicalize role entries by symbol.
- Fail closed on non-15-minute alignment.
- Do not silently shorten DEV/SEALED.

### Step 4 — GREEN verification

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_temporal.py -q
```

### Step 5 — Refactor + commit

```bash
git add trade_rl/workflows/universal_trade_rl_u2_temporal.py tests/workflows/universal_trade_rl_u2_test_support.py tests/workflows/test_universal_trade_rl_u2_temporal.py
git commit -m "feat: define Universal Trade RL U2 temporal contract"
```

---

# Task 2: Materialize authorized FIT starts and fixed Development episode plans

**Files**

- Modify `trade_rl/workflows/universal_trade_rl_u2_temporal.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_episode_planning.py`.

### Step 1 — RED: coverage and boundary tests

For each Train/Development test dataset, use the frozen U1 episode-planning API. Test:

- every authorized Train start has `episode_end <= T_fit_end`;
- no authorized Train start is after FIT;
- each Train symbol has at least 24 non-overlapping complete FIT episodes;
- each Train symbol resolves all 12 G1 starts;
- each Development symbol resolves all 12 G2 and 12 G3 starts;
- Admission arrays are never requested; only manifest metadata coverage is checked;
- missing one required Development interval fails instead of shortening the grid;
- boundary fragment exclusion count is deterministic.

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_episode_planning.py -q
```

### Step 3 — GREEN: add planning contracts

Add immutable structures such as:

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SymbolEpisodePlan:
    symbol: str
    authorized_fit_starts: tuple[int, ...]
    g1_starts: tuple[int, ...]
    g2_starts: tuple[int, ...]
    g3_starts: tuple[int, ...]
    digest: str = ""
```

Rules:

- Train: `authorized_fit_starts + g1_starts` only.
- Development: `g2_starts + g3_starts` only.
- Admission: no dataset/planner object is accepted.
- Bind dataset digest and temporal digest to every symbol plan.
- Store a digest of sorted authorized starts, not an unbounded opaque mutable container.

### Step 4 — Verify

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_temporal.py tests/workflows/test_universal_trade_rl_u2_episode_planning.py -q
```

### Step 5 — Commit

```bash
git add trade_rl/workflows/universal_trade_rl_u2_temporal.py tests/workflows/test_universal_trade_rl_u2_episode_planning.py
git commit -m "feat: freeze U2 authorized episode plans"
```

---

# Task 3: Authorized U1 concrete-environment wrapper

**Files**

- Create `trade_rl/rl/universal_trade_u2_environment.py`.
- Create `tests/rl/test_universal_trade_u2_environment.py`.

**Produces**

```python
class AuthorizedUniversalTradeEnvironment(gym.Wrapper):
    ...
```

The wrapper owns only episode authorization and U2 sequence-layout metadata. It does not own economics.

### Step 1 — RED: reset escape tests

Test:

1. `normalizer=None` child rejects at construction.
2. wrong U1 observation-contract digest rejects.
3. wrong U1 state-layout digest rejects.
4. no `start_idx`: wrapper chooses only from authorized FIT starts using the reset seed.
5. explicit authorized `start_idx`: accepted.
6. explicit non-authorized `start_idx`: rejected before child reset.
7. child `start_index/end_index` must exactly match the planned contract.
8. wrapper does not alter `step()` action/reward/terminated/truncated/info.
9. same seed + same authorized set => same selected start.
10. changing order of the input authorized set does not change the canonical set/digest.

### Step 2 — Verify RED

```bash
uv run pytest tests/rl/test_universal_trade_u2_environment.py -q
```

### Step 3 — GREEN: minimal wrapper

Constructor inputs should include:

```python
AuthorizedUniversalTradeEnvironment(
    child: UniversalTradeEnvironment,
    *,
    authorized_start_indices: tuple[int, ...],
    temporal_contract_digest: str,
    expected_u1_observation_contract_digest: str,
    u1_state_layout_digest: str,
    policy_state_fields: tuple[str, ...],
)
```

Expose `sequence_layout_metadata` for the later SB3 assembly:

```python
{
    "n_symbols": 1,
    "feature_counts": {"15m": ..., "1h": ..., "4h": ..., "1d": ...},
    "window_lengths": {"15m": 96, "1h": 168, "4h": 120, "1d": 60},
    "policy_state_width": ...,
    "policy_state_fields": (...,),
    "state_layout_digest": "...",
    "current_weight_state_index": policy_state_fields.index("current_weight"),
    "u1_observation_contract_digest": "...",
}
```

Important:

- derive feature counts from the actual child `observation_space`;
- reject extra/missing U1 observation keys;
- do not read private U1 fields;
- use `np.random.default_rng(seed)` only to select from the frozen start tuple; router already derives an episode-specific seed.

### Step 4 — Verify

```bash
uv run pytest tests/rl/test_universal_trade_u2_environment.py tests/rl/test_universal_trade_environment.py -q
```

### Step 5 — Commit

```bash
git add trade_rl/rl/universal_trade_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: restrict U2 episodes to frozen FIT starts"
```

---

# Task 4: Add `universal_trade_sequence_v1` as a closed training mode

**Files**

- Modify `trade_rl/rl/training_modes.py`.
- Modify `trade_rl/rl/training.py`.
- Modify `tests/rl/test_training_modes.py`.
- Modify/add the nearest `ResidualTrainingConfig` validation tests.

### Step 1 — RED: enum and inactive-field tests

Add:

```python
UNIVERSAL_TRADE_SEQUENCE_V1 = "universal_trade_sequence_v1"
```

Tests must prove:

- exact string resolves through `ObservationEncoder`;
- typo/unknown value rejects;
- mode is PPO-family compatible but U2 later freezes PPO only;
- sequence settings are active for this mode;
- asset-set-only embedding settings remain inactive/default;
- BC is not silently enabled by the encoder change.

### Step 2 — Verify RED

```bash
uv run pytest tests/rl/test_training_modes.py tests/rl/test_training_config.py -q
```

Use the actual nearest config-test filename if the repository differs; do not invent a new duplicate config suite.

### Step 3 — GREEN

- Extend the closed enum and validation messages.
- Factor sequence-mode checks into a helper instead of copying the entire `hierarchical_sequence_v2` branch.
- Keep existing three encoder semantics unchanged.

### Step 4 — Compatibility verification

```bash
uv run pytest tests/rl/test_training_modes.py tests/integrations/test_universal_sb3_model_assembly.py tests/workflows/test_universal_full_research_training.py -q
```

### Step 5 — Commit

```bash
git add trade_rl/rl/training_modes.py trade_rl/rl/training.py tests/rl/test_training_modes.py tests/rl/test_training_config.py
git commit -m "feat: register U2 Universal Trade sequence mode"
```

If the nearest config test has a different path, commit the actual path only.

---

# Task 5: U1-specific causal sequence feature extractor

**Files**

- Create `trade_rl/rl/universal_trade_policy.py`.
- Create `tests/rl/test_universal_trade_policy.py`.

**Produces**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeSequenceArchitecture:
    ...

class UniversalTradeSequenceFeatureExtractor(BaseFeaturesExtractor):
    ...
```

### Step 1 — RED: exact observation surface

Test the extractor accepts exactly:

```text
sequence_15m_values / available / staleness
sequence_1h_values  / available / staleness
sequence_4h_values  / available / staleness
sequence_1d_values  / available / staleness
policy_state
```

and rejects legacy/context keys such as:

```text
current_snapshot
asset_state
global_state
instrument_context
local_cross_market_context
```

Test architecture constants:

```text
capacity=compact
d_model=256
heads=4
layers=1
ffn=3
gate_bias=-2.0
dropout=0.0
actor/value widths=(256,128)
```

### Step 2 — RED: numerical/shape tests

Test:

- output width is `1*256 + 256 + 128 + 2 = 642`;
- per-timeframe input is `values + availability + log1p(staleness)`;
- unavailable values are defensively zeroed before encoding;
- `current_weight` comes from the state-layout index, not a hard-coded position;
- distribution-active mask output is exactly `1.0`;
- changing a field not present in U1 is impossible because the Dict space is closed;
- architecture digest changes if any structural width/head/layer/window/input-channel value changes.

### Step 3 — Verify RED

```bash
uv run pytest tests/rl/test_universal_trade_policy.py -q
```

### Step 4 — GREEN implementation

Reuse:

```python
from trade_rl.rl.sequence_policy import CausalTimeframeEncoder, sequence_encoder_widths
from trade_rl.rl.timeframe_fusion import CrossTimeframeFusion
```

Do **not** instantiate `MultiTimeframeAssetEncoder` because it reintroduces snapshot/asset-state context.

Forward outline:

```python
for timeframe in TIMEFRAMES:
    values = obs[f"sequence_{timeframe}_values"].float()
    available = obs[f"sequence_{timeframe}_available"] > 0.5
    staleness = obs[f"sequence_{timeframe}_staleness"].float()
    safe_values = torch.where(available, values, torch.zeros_like(values))
    channels = torch.cat(
        (safe_values, available.float(), torch.log1p(staleness.clamp_min(0.0))),
        dim=-1,
    )
    # flatten [batch, 1, time, channels] -> [batch, time, channels]
    ...

state_context = context_encoder(policy_state).unsqueeze(1)
fused = timeframe_fusion(..., context=state_context)
pooled = fused[:, 0]
global_state = global_state_encoder(policy_state)
active = torch.ones(...)
current_weight = policy_state[:, current_weight_state_index]
return torch.cat((fused[:, 0], pooled, global_state, active, current_weight), dim=-1)
```

### Step 5 — Verify + refactor

```bash
uv run pytest tests/rl/test_universal_trade_policy.py tests/rl/test_sequence_policy_core.py -q
```

### Step 6 — Commit

```bash
git add trade_rl/rl/universal_trade_policy.py tests/rl/test_universal_trade_policy.py
git commit -m "feat: add U1-only Universal Trade policy encoder"
```

---

# Task 6: SB3 assembly, bounded action transport, and policy identity

**Files**

- Modify `trade_rl/integrations/sb3_model_assembly.py`.
- Modify `trade_rl/rl/policy_identity.py`.
- Modify `trade_rl/artifacts/policy_identity_contract.py`.
- Create `tests/integrations/test_universal_trade_u2_sb3_model_assembly.py`.
- Extend nearest policy-identity tests.

### Step 1 — RED: assembly tests

Build a routed U2 probe and assert:

```python
assembly.policy_identifier is SharedPerAssetActorCriticPolicy
assembly.observation_encoder == "universal_trade_sequence_v1"
assembly.policy_actor_head == "shared_target_v1"
assembly.sequence_symbols == ("INSTRUMENT",)
assembly.sequence_action_names == ("target_weight:INSTRUMENT",)
```

Assert feature extractor class is `UniversalTradeSequenceFeatureExtractor` and receives the frozen U2 metadata.

Reject:

- non-generic policy symbol;
- action size != 1;
- missing U1 state-layout digest;
- `hierarchical_gate_target_v1`;
- nonzero instrument/V4 context.

### Step 2 — RED: action transport test

Create the actual SB3 policy/model and prove both stochastic rollout and deterministic actions are finite and bounded before `UniversalTradeEnvironment.step()`.

Instrument the child environment to record the incoming action and assert:

```python
np.testing.assert_array_equal(received_action, produced_action.astype(np.float32))
```

or the narrowest exact float32 transport oracle supported by the existing vector-env path.

Also inject an out-of-range fake policy action and prove the U1 strict parser rejects it; do not let wrapper clipping turn it legal.

### Step 3 — Verify RED

```bash
uv run pytest tests/integrations/test_universal_trade_u2_sb3_model_assembly.py -q
```

### Step 4 — GREEN assembly

Add a dedicated branch in `resolve_sb3_policy_assembly()` for `universal_trade_sequence_v1`.

- reuse `SharedPerAssetActorCriticPolicy`;
- use `shared_target_v1` only;
- use standard Dict rollout storage unless measured memory exceeds the already frozen cap;
- do not use the old single-dataset `SequenceRolloutReconstructor` across routed symbols;
- preserve old `hierarchical_sequence_v2` branch byte-for-byte where practical.

### Step 5 — GREEN identity

Add canonical identity vocabulary for the new encoder without weakening old payload validation.

The U2 identity must bind the **actual extractor architecture**, not merely config text:

```text
u1 observation/state-layout identity
clock order/windows/input channels
compact TCN widths/dilations
d_model=256
TF attention 4x1, FFN=3, gate bias=-2, dropout=0
state context/global encoder widths
direct actor head
squashed action distribution
INSTRUMENT / target_weight:INSTRUMENT binding
```

Existing v4 identities must continue to validate exactly as before. If implementation proves an additive v4 variant cannot be represented unambiguously, stop and version the schema explicitly; do not silently reinterpret old fields.

### Step 6 — Verify compatibility

```bash
uv run pytest \
  tests/integrations/test_universal_trade_u2_sb3_model_assembly.py \
  tests/integrations/test_universal_sb3_model_assembly.py \
  tests/integrations/test_sb3_policy_identity_v3.py \
  tests/rl/test_asset_agnostic_policy_identity.py -q
```

### Step 7 — Commit

```bash
git add trade_rl/integrations/sb3_model_assembly.py trade_rl/rl/policy_identity.py trade_rl/artifacts/policy_identity_contract.py tests/integrations/test_universal_trade_u2_sb3_model_assembly.py
git commit -m "feat: assemble bounded U2 PPO policy"
```

Include the actual modified identity-test paths in the same commit.

---

# Task 7: Frozen U2 model config, seed namespace, and run identity

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_contract.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_contract.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_seed_identity.py`.

### Step 1 — RED: exact model config

Create an oracle that rejects one-field drift from this exact profile:

```python
EXPECTED = {
    "algorithm": "ppo",
    "timesteps": 524_288,
    "gamma": 1.0,
    "learning_rate": 0.00012,
    "learning_rate_schedule": "linear",
    "learning_rate_final_ratio": 0.1,
    "n_envs": 8,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "log_std_init": -0.5,
    "target_kl": 0.02,
    "use_sde": False,
    "observation_encoder": "universal_trade_sequence_v1",
    "policy_actor_head": "shared_target_v1",
    "policy_net_arch": (256, 128),
    "value_net_arch": (256, 128),
    "sequence_tcn_capacity": "compact",
    "sequence_d_model": 256,
    "sequence_timeframe_attention_heads": 4,
    "sequence_timeframe_attention_layers": 1,
    "sequence_timeframe_ffn_multiplier": 3,
    "sequence_timeframe_gate_bias": -2.0,
    "sequence_dropout": 0.0,
    "behavior_cloning_epochs": 0,
}
```

The full digest payload must also bind remaining resource/checkpoint/tensorboard/vector-env fields from the spec.

### Step 2 — RED: cutoff/U1 dependency tests

Reject:

- missing U1 normalizer;
- U1 normalizer cutoff != `T_fit_end`;
- `RL_TRAINING` provenance cutoff != `T_fit_end`;
- U1/U0 manifest mismatch;
- U1 observation/state/action/reward digest mismatch;
- U1 `production_status` anything except `NO-GO`.

### Step 3 — RED: seed derivation

Test exact two-stage derivation:

```text
seed_namespace_digest -> 8 ordered seed digests -> uint32 seeds -> final U2 contract
```

Assert:

- deterministic across repeated construction;
- exactly 8 unique non-negative uint32 values;
- changing universe/U1/temporal/model digest changes namespace;
- forced collision raises; it does not probe a ninth seed;
- final U2 digest changes if ordered seed vector changes.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_seed_identity.py -q
```

### Step 5 — GREEN implementation

Define versioned dataclasses/codecs for:

- U2 model-config identity;
- seed namespace;
- ordered seeds artifact;
- full U2 contract.

Build the existing U0 `UniversalTradeRLFitPurpose.RL_TRAINING` provenance and `UniversalTradeRLRunStage.BASE_TRAINING` identity rather than inventing a second stage enum.

### Step 6 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_seed_identity.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
git add trade_rl/workflows/universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_seed_identity.py
git commit -m "feat: freeze U2 model and seed identities"
```

---

# Task 8: Eight-seed Base PPO training and final-checkpoint closure

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_training.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_training.py`.
- Add/extend a focused SB3 integration test if needed.

### Step 1 — RED: factory wiring tests

Build a U2 concrete factory that:

1. loads one immutable single-symbol dataset artifact;
2. verifies dataset digest against U0 manifest;
3. constructs the frozen U1 environment + frozen normalizer;
4. wraps it with `AuthorizedUniversalTradeEnvironment` using that symbol's FIT plan.

Then pass it to existing `UniversalRoutedEnvironmentFactory` with:

```python
instrument_context_provider=None
v4_context_provider=None
training_contract_digest=u2_contract.digest
run_seed=seed
```

Test `for_environment_index(0..7)` yields distinct router environment indices and balanced cycle semantics.

### Step 2 — RED: seed/checkpoint eligibility tests

Test runner behavior:

- loops exactly the 8 frozen seeds;
- each seed receives exactly the fixed `524288` target timesteps;
- a technically valid but negative-return seed is retained, not retried;
- a technical crash may resume only the same seed + identity;
- wrong checkpoint/config/environment digest rejects resume;
- intermediate checkpoint never becomes `performance_eligible=true`;
- one final checkpoint manifest per seed is required.

### Step 3 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_training.py tests/integrations/test_universal_trade_u2_sb3_model_assembly.py -q
```

### Step 4 — GREEN implementation

Reuse `StableBaselines3Backend` per seed with a seed-specific `UniversalRoutedEnvironmentFactory`. Do not fork PPO.

Persist under:

```text
training/<seed>/
  final-checkpoint/...
  training_result.json
  routing_evidence.json
```

`training_result.json` binds at minimum:

- U2 contract/model/seed/run-identity digests;
- exact final checkpoint identity;
- observed timesteps;
- worker/router evidence;
- technical completion state;
- `performance_eligible=true` only for the canonical final checkpoint.

### Step 5 — Small integration training

Use tiny test-only timesteps/config in an integration fixture to prove wiring, but do **not** weaken the production U2 contract. Production profile validation remains exact; the tiny integration path must be clearly test-only.

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_training.py -q
```

### Step 6 — Commit

```bash
git add trade_rl/workflows/universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_training.py
git commit -m "feat: train frozen U2 PPO seed set"
```

---

# Task 9: Deterministic G1/G2/G3 and baseline replay

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_evaluation.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_evaluation.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_baselines.py`.

### Step 1 — RED: evaluation-scope tests

Assert exact scope closure:

- G1 = Train × 12 DEV episodes;
- G2 = Development × latest 12 FIT episodes;
- G3 = Development × same 12 DEV episodes as G1;
- no Admission symbol accepted;
- no missing/extra symbol or episode accepted;
- final checkpoints only.

### Step 2 — RED: baseline parity tests

On a deterministic synthetic market:

- `CASH_FLAT` always requests 0;
- `BUY_AND_HOLD_LONG` requests +1 through the same signal-delay/Risk/Execution path;
- `TREND_BASELINE` uses the maintained child TrendStrategy target but still enters through U1 target-action/Risk/Execution economics;
- all four paths begin from the same cash reset and exact same start/end;
- fee/spread/impact/funding/borrow changes affect policy and baselines consistently;
- no baseline gets free terminal liquidation.

### Step 3 — RED: leaf accounting tests

Leaf schema:

```python
(seed, scope, symbol, episode_index)
```

Require:

- start/end timestamps;
- initial/final wealth;
- gross return;
- net return + net log growth;
- Cash/BuyHold/Trend metrics;
- max drawdown;
- turnover + turnover/day;
- execution cost;
- funding/borrow;
- requested/executed/fill counts;
- trade/rebalance/fill counts;
- termination reason;
- hard-risk and execution-rejection evidence;
- all source/policy/environment/U2 identities.

Independently recompute:

```python
expected_log_growth = math.log(final_wealth / initial_wealth)
```

and reject drift.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py -q
```

### Step 5 — GREEN implementation

- Load checkpoint through maintained checkpoint loader.
- Use deterministic policy prediction for Development.
- Never update optimizer/normalizer/calibration.
- Reset every leaf with the exact planned `start_idx`.
- Preserve immutable scope identity.

### Step 6 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py -q
git add trade_rl/workflows/universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py
git commit -m "feat: replay U2 Development scopes and baselines"
```

---

# Task 10: Immutable Development leaf publication and resume

**Files**

- Modify `trade_rl/workflows/universal_trade_rl_u2_evaluation.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_evidence_io.py`.

### Step 1 — RED

Test:

- leaf path is exactly `development/records/<scope>/<seed>/<symbol>/<episode>.json`;
- canonical bytes/digest round-trip;
- identical existing leaf = idempotent success;
- modified/tampered leaf = fail closed;
- extra unknown leaf = fail closed at complete-scope validation;
- partial write never appears as valid final leaf;
- crash after compute but before durable publish can recompute same immutable leaf;
- existing valid leaf is not recomputed/replaced after aggregate results exist.

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evidence_io.py -q
```

### Step 3 — GREEN

Reuse canonical JSON + existing atomic-write helpers. Do not create a parallel serialization convention.

### Step 4 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evidence_io.py tests/workflows/test_universal_trade_rl_u2_evaluation.py -q
git add trade_rl/workflows/universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_evidence_io.py
git commit -m "feat: persist immutable U2 Development leaves"
```

---

# Task 11: Statistical aggregation and fail-closed Development gate

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_development.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_aggregation.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_development_gate.py`.

### Step 1 — RED: aggregation oracle

Build a hand-calculated fixture with 8 seeds × >=3 Development symbols × 12 episodes.

For G3 compute exactly:

```python
leaf_excess = policy_net_log_growth - cash_net_log_growth
symbol_episode_excess = median(8 seed leaf_excess values)
primary_excess_j = median(symbol_episode_excess across Development symbols)
```

Assert exactly 12 ordered `primary_excess` values; seeds are collapsed before bootstrap.

### Step 2 — RED: bootstrap identity

Use existing `moving_block_mean_test` with:

```text
n_bootstrap=2000
block_size=3
CI=2.5%/97.5%
bootstrap_seed=derived from seed namespace + "u2-development-bootstrap-v1"
```

Test independently calculated deterministic fixture output/digest.

### Step 3 — RED: every gate reason

Test each condition independently:

**Structural rejection**

- NaN/Inf;
- identity/source drift;
- unauthorized fit/update evidence;
- action transport alteration;
- missing leaf;
- unexplained execution rejection;
- hard Risk invariant violation;
- evidence tamper;
- poor-seed substitution.

**G1**

```text
median 8-seed excess vs Cash > 0
```

**G2** same.

**G3 — all required**

```text
median 8-seed excess vs Cash > 0
mean(primary_excess[12]) > 0
bootstrap lower_ci > 0
positive seeds >= 6/8
median Development-symbol absolute net log growth > 0
positive Development-symbol excess-vs-Cash fraction >= 0.60
minimum G3 leaf net return >= -0.05
for every seed: mean G3 turnover/day <= 1.0
economic termination count == 0
median 8-seed G3 excess vs Trend > 0
```

Buy-and-Hold remains diagnostic.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_aggregation.py tests/workflows/test_universal_trade_rl_u2_development_gate.py -q
```

### Step 5 — GREEN

Output only:

```text
DEVELOPMENT_ACCEPTED
DEVELOPMENT_REJECTED
```

No ranking score/candidate grid.

Persist:

```text
development/summary.json
development/decision.json
```

The decision artifact binds all required leaf digests, threshold identity, evaluator code identity, bootstrap identity, and U2 contract digest.

### Step 6 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_aggregation.py tests/workflows/test_universal_trade_rl_u2_development_gate.py -q
git add trade_rl/workflows/universal_trade_rl_u2_development.py tests/workflows/test_universal_trade_rl_u2_aggregation.py tests/workflows/test_universal_trade_rl_u2_development_gate.py
git commit -m "feat: gate U2 Development evidence"
```

---

# Task 12: End-to-end resumable U2 runner and minimal CLI

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_runner.py`.
- Create `scripts/run_universal_trade_rl_u2.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_runner.py`.
- Create `tests/scripts/test_run_universal_trade_rl_u2.py`.

### Step 1 — RED: orchestration order

The runner must enforce:

```text
load/verify frozen U0 + frozen U1
-> materialize/reuse temporal contract
-> materialize/reuse U2 contract + seeds + BASE_TRAINING identity
-> train/reuse exact 8 final checkpoints
-> replay/reuse complete G1/G2/G3 leaves
-> aggregate
-> write Development decision
-> STOP
```

Test:

- missing prior-stage artifact prevents later stage;
- existing valid artifacts resume idempotently;
- corrupt/extra/wrong-identity artifact fails closed;
- Development rejected is a valid terminal research outcome, not an exception that deletes evidence;
- no code path loads/authorizes Admission datasets.

### Step 2 — RED: CLI

Keep the CLI narrow. It should accept explicit frozen artifact/source roots and output root, print terminal state, and provide `--help`. Do not expose thresholds, seed count, candidate grid, algorithm, or hyperparameters as command-line knobs.

Example intended surface:

```bash
uv run python scripts/run_universal_trade_rl_u2.py \
  --u0-root <path> \
  --u1-root <path> \
  --dataset-root <path> \
  --output-root <path>
```

### Step 3 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_runner.py tests/scripts/test_run_universal_trade_rl_u2.py -q
```

### Step 4 — GREEN

Do not delete durable evidence on Development reject.

Recommended terminal semantics:

- exit `0`: software valid + `DEVELOPMENT_ACCEPTED`;
- exit `3`: software valid + `DEVELOPMENT_REJECTED`;
- other nonzero: execution/contract failure.

If the repository already has a stronger standardized research outcome convention by implementation time, use that exact convention and update the plan/spec before changing semantics.

### Step 5 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_runner.py tests/scripts/test_run_universal_trade_rl_u2.py -q
uv run python scripts/run_universal_trade_rl_u2.py --help
git add trade_rl/workflows/universal_trade_rl_u2_runner.py scripts/run_universal_trade_rl_u2.py tests/workflows/test_universal_trade_rl_u2_runner.py tests/scripts/test_run_universal_trade_rl_u2.py
git commit -m "feat: orchestrate Universal Trade RL U2"
```

---

# Task 13: Adversarial / falsification suite

**Files**

- Create `tests/workflows/test_universal_trade_rl_u2_falsification.py`.
- Create `tests/integrations/test_universal_trade_rl_u2_economics.py`.

### Step 1 — Write falsification tests before final docs

Attempt to break the implementation with at least:

1. U1 normalizer cutoff one 15m bar after `T_fit_end`.
2. RL_TRAINING provenance containing Development symbol.
3. Train reset with a DEV `start_idx`.
4. Development reset with wrong planned episode.
5. U1 raw missing placeholder changed under `available=false`.
6. concrete symbol/ticker injected into policy observation.
7. instrument context or V4 provider enabled.
8. fake policy produces `1.0001` and relies on external clip.
9. seed vector element replaced after six successful runs.
10. ninth replacement seed added for a losing seed.
11. intermediate checkpoint substituted for final.
12. valid final checkpoint from wrong model config.
13. missing one G3 leaf.
14. duplicated G3 leaf.
15. modified execution cost on Trend baseline only.
16. funding/borrow applied twice.
17. same 8 seed replicas incorrectly fed as 96 bootstrap time samples.
18. one -5.01% G3 leaf hidden by positive mean.
19. one seed turnover `>1.0x/day` hidden by median.
20. economic termination hidden by positive final aggregate.
21. threshold artifact modified after leaves exist.
22. Admission dataset loader touched before authorization.

### Step 2 — Economic integration

Use real maintained execution/accounting objects, not mocks only, to verify:

- fee;
- spread;
- impact;
- funding;
- borrow;
- partial fill/liquidity;
- margin/economic termination;
- signal delay and no terminal liquidation.

### Step 3 — Run

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_falsification.py tests/integrations/test_universal_trade_rl_u2_economics.py -q
```

Expected: all adversarial mutations are detected by the intended independent oracle.

### Step 4 — Fix any discovered issue and rerun nearest tests first

Do not merely document a fixable substantive defect.

### Step 5 — Commit

```bash
git add tests/workflows/test_universal_trade_rl_u2_falsification.py tests/integrations/test_universal_trade_rl_u2_economics.py
git commit -m "test: falsify Universal Trade RL U2 boundaries"
```

---

# Task 14: Documentation, architecture review, full verification, independent review, exact-head CI

**Files**

- Modify `docs/UNIVERSAL_TRADE_RL.md`.
- Modify `docs/CONFIGURATION.md`.
- Modify `tests/test_architecture_contract.py` only if necessary.
- Optionally add a strict illustrative U2 input example only if the CLI needs one; label it non-production evidence.

### Step 1 — Update docs without overstating research state

Document:

- U2 symbol×time split;
- cutoff ordering and U1 normalizer dependency;
- U-Medium Direct `universal_trade_sequence_v1` architecture;
- exact eight-seed PPO profile;
- no best-seed/checkpoint selection;
- G1/G2/G3;
- Development gates;
- artifact tree/resume behavior;
- `Admission=CLOSED`, `Production=NO-GO` even after Development pass.

Resolve any stale U0/U1 handoff wording so the documented order matches:

```text
U0 freeze
-> U2 temporal contract
-> final U1 normalizer/artifact freeze
-> U1 Quality Gate
-> U2 execution
```

### Step 2 — Targeted test wave

```bash
uv run pytest \
  tests/rl/test_universal_trade_u2_environment.py \
  tests/rl/test_universal_trade_policy.py \
  tests/integrations/test_universal_trade_u2_sb3_model_assembly.py \
  tests/workflows/test_universal_trade_rl_u2_temporal.py \
  tests/workflows/test_universal_trade_rl_u2_episode_planning.py \
  tests/workflows/test_universal_trade_rl_u2_contract.py \
  tests/workflows/test_universal_trade_rl_u2_seed_identity.py \
  tests/workflows/test_universal_trade_rl_u2_training.py \
  tests/workflows/test_universal_trade_rl_u2_evaluation.py \
  tests/workflows/test_universal_trade_rl_u2_baselines.py \
  tests/workflows/test_universal_trade_rl_u2_evidence_io.py \
  tests/workflows/test_universal_trade_rl_u2_aggregation.py \
  tests/workflows/test_universal_trade_rl_u2_development_gate.py \
  tests/workflows/test_universal_trade_rl_u2_runner.py \
  tests/workflows/test_universal_trade_rl_u2_falsification.py \
  tests/integrations/test_universal_trade_rl_u2_economics.py \
  -q
```

### Step 3 — Related compatibility wave

```bash
uv run pytest \
  tests/rl/test_universal_trade_environment.py \
  tests/rl/test_universal_trade_observation.py \
  tests/rl/test_universal_trade_u1_normalization.py \
  tests/integrations/test_universal_sb3_model_assembly.py \
  tests/workflows/test_universal_full_research_training.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  -q
```

Use actual current filenames if U1 final refactors them; do not skip the layer because a path changed.

### Step 4 — Static / architecture checks

```bash
uv run ruff check trade_rl tests scripts
uv run ruff format --check trade_rl tests scripts
uv run mypy trade_rl scripts
uv run lint-imports
```

### Step 5 — Full suite + build

```bash
uv run pytest -q
uv build
```

If the repository uses a canonical wrapper command in the final U1 head, run that canonical command in addition to the explicit layers above.

### Step 6 — Inspect coverage of changed behavior

Confirm changed production lines are actually executed by tests. Pay special attention to:

- cutoff rejection;
- explicit/non-explicit `start_idx` paths;
- unavailable-value defense;
- new encoder assembly/identity;
- action transport;
- seed collision/substitution;
- resume/tamper;
- every Development rejection reason.

Coverage percentage alone is not the oracle.

### Step 7 — Self-review full diff

Review:

- Requirement compliance;
- dependency direction (`U2 -> frozen U1`, never `U1 -> U2`);
- no duplicate economics;
- no Development access from training modules;
- no Admission loader in U2;
- identity closure;
- error handling/atomic publication;
- deterministic ordering;
- dead/debug/temporary code;
- accidental generated artifacts/secrets.

Fix substantive findings and rerun impacted tests.

### Step 8 — Independent / falsification review

Give a reviewer/verifier only:

1. original U2 spec;
2. this plan;
3. final diff;
4. final tests/assertions;
5. actual verification outputs.

Ask the reviewer to find:

- a path from Development/Admission into fit state;
- a path that alters the policy action before U1 parser;
- a way to replace a poor seed/checkpoint;
- a way to change gates after results;
- a way to pass with missing/downside-breaching evidence;
- an assumption hidden by mocks.

Do not ask merely whether the implementation “looks good.”

### Step 9 — Git/HEAD hygiene

Before reporting completion:

```bash
git diff --check
git status --short
git log -1 --oneline
```

Confirm:

- no untracked debug/temp files;
- no secrets;
- no temporary workflows;
- no unrelated refactor;
- branch is based on the intended final U1 head.

### Step 10 — exact-final-HEAD CI

Push only after local gates pass. Verify required CI belongs to the exact final U2 HEAD, not an earlier commit. Do not mark Ready or merge unless the user explicitly authorizes the relevant action and the Quality Gate is complete.

### Step 11 — Final evidence report

Report separately:

1. what changed;
2. why this architecture was used;
3. Acceptance Criteria mapping;
4. Failure Modes tested;
5. exact test/static/build results;
6. independent/falsification findings;
7. exact HEAD and CI status;
8. unverified items;
9. residual risks;
10. what the verification guarantees and does **not** guarantee.

If no real production-candidate eight-seed run has been executed, explicitly state:

```text
Software implementation may be valid.
Economic Development acceptance is not established.
Admission remains closed.
Production remains NO-GO.
```

### Step 12 — Commit docs/verification-only changes

```bash
git add docs/UNIVERSAL_TRADE_RL.md docs/CONFIGURATION.md tests/test_architecture_contract.py
git commit -m "docs: document Universal Trade RL U2 gates"
```

Only add `tests/test_architecture_contract.py` if actually modified.

---

# Acceptance Criteria → Task Mapping

| Acceptance area | Primary task(s) |
| --- | --- |
| deterministic temporal cutoffs/grids | 1–2 |
| Train×FIT-only authorized episodes | 2–3 |
| frozen U1 normalizer/cutoff dependency | 0, 7, 13 |
| exact U1 observation only | 3, 5 |
| bounded direct scalar action | 5–6 |
| U-Medium Direct PPO identity | 4–7 |
| exact 8 seeds/no replacement | 7–8 |
| balanced train-symbol routing | 8 |
| final-checkpoint-only eligibility | 8 |
| G1/G2/G3 fixed evaluation | 9 |
| baseline parity | 9, 13 |
| immutable leaf evidence | 9–10 |
| block-bootstrap aggregation | 11 |
| fixed Development thresholds | 11 |
| Admission firewall | 7, 9, 12, 13 |
| falsification | 13 |
| static/full/build/CI/review | 14 |

---

# Final execution rule

Do not start a real U2 training run merely because the software tests pass.

Real execution order is:

```text
U1 final Quality Gate
-> production-candidate U0/U1 identities frozen
-> U2 temporal materialization
-> inspect temporal/coverage artifact
-> U1 final normalizer cutoff equality verified
-> U2 contract/seeds frozen
-> 8-seed Base PPO training
-> final checkpoint closure
-> G1/G2/G3 Development replay
-> immutable Development decision
-> STOP; Admission stays closed
```

If Development rejects, preserve the evidence. Any scientific change becomes a new U2 generation rather than an in-place retry.
