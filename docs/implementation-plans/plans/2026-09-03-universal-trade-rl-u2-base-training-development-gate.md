# Universal Trade RL U2 Base Training / Development Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Every production change follows Red → Green → Refactor. Do not weaken, skip, delete, or rewrite an oracle merely to obtain Green.

**Goal:** Implement one preregistered eight-seed PPO Base RL experiment over the frozen U1 one-symbol environment, with a deterministic symbol×time split, Train×FIT-only updates, bounded identity-free policy input/output, immutable G1/G2/G3 Development evidence, and a fail-closed Development decision that never opens Admission.

**Architecture:** U2 does not create a second market simulator, symbol router, Risk engine, execution engine, reward function, or normalizer. It consumes one frozen U0 universe and one frozen U1 generation. U2 adds: (1) a source-metadata-only temporal-boundary artifact, (2) dataset/U1-planner-backed immutable episode plans, (3) a thin authorized-episode wrapper, (4) a U1-specific causal sequence extractor plugged into the maintained bounded PPO policy, (5) an immutable eight-seed/model/run identity, and (6) deterministic Development replay/aggregation/gating. Development code depends on frozen training outputs; training code never imports Development-selection logic.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Gymnasium, Stable-Baselines3 PPO, existing `ResidualMarketEnv`, `UniversalTradeEnvironment`, `EpisodeRoutedSingleInstrumentEnv`, `UniversalRoutedEnvironmentFactory`, `StableBaselines3Backend`, canonical JSON/SHA-256 artifacts, pytest, Hypothesis where useful, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-training-development-gate-design.md`

**Planning base:** U1 draft head `a26f21b2ff113b6d7e4b554ff9cc30d3355ed996`. **Do not begin Task 1 production code until the blocking U1 gate below is satisfied and the final U1 head is an ancestor of the U2 implementation branch.**

---

## 0. Blocking prerequisite — separate U1 work, not U2 implementation commits

U1 closure remains a separate change. Do not hide U1 fixes inside U2 commits.

Before Task 1:

1. Finish the existing U1 implementation-plan Tasks 8–11:
   - `trade_rl/workflows/universal_trade_rl_u1_contract.py`;
   - `trade_rl/workflows/universal_trade_rl_u1_runner.py`;
   - adversarial accounting/state falsification;
   - docs/full verification/independent review/exact-head CI.
2. Fix the confirmed unavailable-value invariant in U1:
   - whenever `available == false`, the policy-facing value is exactly zero even when the low-level observation builder is invoked without a normalizer;
   - mutate the hidden raw placeholder across `0`, `±1`, `±1e9` and prove identical policy observations with and without normalization.
3. Freeze/document `UniversalTradeSequenceNormalizer.clip_value = 10.0` as U1 semantics and identity.
4. Add a **read-only public U1 episode-planning surface**, backed by the existing `EpisodeContractSampler`, equivalent to:

```python
def valid_episode_starts(self) -> np.ndarray: ...
def episode_end_index(self, start_index: int) -> int: ...
```

   This must delegate to the maintained 720h episode contract and must not modify reset/step economics.
5. Ensure the frozen U1 artifact exposes/binds enough public state-layout identity for U2 to obtain:
   - `state_layout_digest`;
   - ordered `policy_state_fields` or an equivalent canonical field-order artifact;
   - observation/action/reward/normalizer/runtime/economic digests.
   U2 must not reach into `_POLICY_STATE_LAYOUT` or other U1 private fields.
6. Complete the U1 Quality Gate on one exact U1 head.
7. Re-run stronger U0/U1 stack verification on the exact final U0/U1 heads; do not inherit the stale strong-verification claim from the older U0 SHA.
8. Bring the final U1 head into the U2 branch by a normal merge/fast-forward-compatible ancestry-preserving operation. Verify with `git merge-base --is-ancestor <final-u1-head> HEAD`. **Do not rebase, force-push, or rewrite history without explicit user authorization.**
9. Re-run the U2 design-diff review. If the final U1 public surface differs materially from this plan, update spec/plan before coding.

**Stop condition:** if any prerequisite is not evidenced, U2 remains **NO-GO** and Task 1 does not start.

---

# Implementation Quality Contract

## Objective

- Train exactly one preregistered PPO configuration across U0 Train symbols using only FIT-time episodes.
- Preserve one-symbol deployment semantics and generic `INSTRUMENT` policy identity.
- Evaluate time-OOS, symbol-OOS, and joint-OOS separately.
- Reject a weak/unstable generation without tuning the same generation from Development results.
- Keep Admission inaccessible.

## Non-goals

- No PPO/SAC/TD3/TQC tournament.
- No Development-driven hyperparameter search or checkpoint search.
- No BC, teacher, Causal Alpha, DAgger, anchored residual, or Trend prior inside the policy.
- No best-seed selection.
- No U1 Observation/Action/Reward semantic change.
- No multi-asset portfolio allocation.
- No 1-minute execution-fidelity project.
- No Admission opening, live trading, profitability claim, or Production GO.

## Critical Invariants

1. `U1 normalizer cutoff == U0 RL_TRAINING cutoff == U2 T_fit_end`.
2. Only `Train × FIT` updates statistical/model/optimizer state.
3. U2 policy input is exactly the U1 Dict observation; no legacy planes, instrument descriptors, V4 context, ticker, dataset ID, role, or horizon enter the tensor path.
4. Policy output is finite and already inside `[-1, 1]` before environment transport.
5. External action clipping is an identity operation.
6. Every complete router cycle contains each Train symbol exactly once per environment.
7. Exactly eight precommitted seeds exist; a poor valid seed is never replaced.
8. Only the canonical final `524288`-timestep checkpoint per seed is performance-eligible.
9. G1/G2/G3 use immutable, globally aligned 720h episode grids.
10. Development evidence is immutable and reconstructible from leaf records.
11. Development logic cannot be imported by the training path.
12. Admission remains closed even if Development passes.

## Primary Failure Modes

- time leakage through normalizer or episode boundaries;
- Development/Admission fit leakage;
- U1 state-layout drift;
- legacy observation/context reinjection;
- hidden action clipping;
- wrong seed substitution or retrying an economically poor seed;
- uneven symbol routing;
- intermediate checkpoint selection;
- baseline/economic path mismatch;
- seed replicas treated as independent time samples;
- missing/duplicated/tampered leaf evidence;
- aggregate winners hiding catastrophic seed/symbol/episode behavior;
- post-result threshold edits;
- execution/evidence resume accepting stale code/config identity.

## Risk

- **Critical:** leakage, Admission access, identity drift, hidden action alteration, best-seed/checkpoint selection, mutable post-result gates.
- **High:** wrong episode alignment, baseline economic mismatch, unstable seed result, severe downside/turnover, incomplete evidence.
- **Medium:** runtime/memory overhead, deterministic incomplete router cycle, CUDA non-bitwise reproducibility.

## Test Oracle

Correctness is observed through:

- exact source roles/digests/cutoffs;
- exact authorized episode start/end timestamps;
- exact U1 observation keys/shapes/state-layout digest;
- actual pre-environment actions;
- actual PPO extractor/actor/distribution/config/checkpoint identity;
- exact router cycle and per-symbol counts;
- independent wealth/accounting reconciliation;
- immutable per-leaf Development records;
- aggregate recomputation from leaves;
- deterministic gate outcome from frozen thresholds;
- final diff/status/HEAD/CI evidence.

## Required Test Layers

Unit + Property/Falsification + Integration + PPO Integration + Economic Integration + Compatibility + Static Analysis + Full Suite + Build + exact-final-HEAD CI + Independent/Falsification Review.

## Quality Gate

Task completion requires the evidence in **Task 14** on one exact final HEAD. Targeted tests alone are insufficient.

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
- `tests/workflows/test_universal_trade_rl_u2_temporal.py`
- `tests/workflows/test_universal_trade_rl_u2_episode_planning.py`
- `tests/rl/test_universal_trade_u2_environment.py`
- `tests/rl/test_universal_trade_u2_training_config.py`
- `tests/rl/test_universal_trade_policy.py`
- `tests/integrations/test_universal_trade_u2_sb3_model_assembly.py`
- `tests/workflows/test_universal_trade_rl_u2_contract.py`
- `tests/workflows/test_universal_trade_rl_u2_seed_identity.py`
- `tests/workflows/test_universal_trade_rl_u2_training.py`
- `tests/workflows/test_universal_trade_rl_u2_evaluation.py`
- `tests/workflows/test_universal_trade_rl_u2_baselines.py`
- `tests/workflows/test_universal_trade_rl_u2_evidence_io.py`
- `tests/workflows/test_universal_trade_rl_u2_aggregation.py`
- `tests/workflows/test_universal_trade_rl_u2_development_gate.py`
- `tests/workflows/test_universal_trade_rl_u2_runner.py`
- `tests/scripts/test_run_universal_trade_rl_u2.py`
- `tests/workflows/test_universal_trade_rl_u2_falsification.py`
- `tests/integrations/test_universal_trade_rl_u2_economics.py`

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
- frozen U1 read-only episode-planning API
- `trade_rl/rl/universal_episode_router.py::DeterministicBalancedInstrumentRouter`
- `trade_rl/rl/universal_single_instrument_env.py::EpisodeRoutedSingleInstrumentEnv`
- `trade_rl/workflows/universal_training_runner.py::UniversalRoutedEnvironmentFactory`
- `trade_rl/rl/sequence_policy.py::CausalTimeframeEncoder`
- `trade_rl/rl/timeframe_fusion.py::CrossTimeframeFusion`
- `trade_rl/rl/policies.py::SharedPerAssetActorCriticPolicy`
- `trade_rl/integrations/sb3_training.py::StableBaselines3Backend`
- `trade_rl/rl/training_environment_contract.py`
- `trade_rl/evaluation/bootstrap.py::moving_block_mean_test`
- existing Risk / Execution / BookState / U1 reward implementations.

---

# Task 1: Pure source-metadata temporal boundaries

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_temporal.py`.
- Create `tests/workflows/universal_trade_rl_u2_test_support.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_temporal.py`.

**Produces only source-metadata-derived boundaries:**

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
class UniversalTradeRLU2TemporalBoundaries:
    universe_manifest_digest: str
    t_fit_end_ns: int
    t_dev_end_ns: int
    t_sealed_end_ns: int
    g1_dev_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    g2_fit_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    g3_dev_intervals: tuple[UniversalTradeRLU2EpisodeInterval, ...]
    digest: str = ""
```

Do **not** put dataset-index start sets or coverage evidence in this Task 1 class; those require Task 2 and belong to the final temporal contract.

### Step 1 — RED: role/count/cutoff tests

Write tests proving:

- Train count `< 9` rejects;
- Development count `< 3` rejects;
- Admission count `< 3` rejects;
- `T_sealed_end = min(last_timestamp)` over Train+Development+Admission manifest entries;
- `T_dev_end = T_sealed_end - 12*720h`;
- `T_fit_end = T_dev_end - 12*720h`;
- all cutoffs are 15-minute aligned;
- G1 and G3 are the same 12 DEV intervals;
- G2 is the latest 12 FIT intervals ending at `T_fit_end`;
- intervals are non-overlapping and exactly 720h;
- mutating any bound source identity changes the digest;
- no dataset loader is called.

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_temporal.py -q
```

Expected RED: U2 temporal module is absent.

### Step 3 — GREEN

- Consume `UniversalTradeRLUniverseManifest` only.
- Use integer nanoseconds, not local-time arithmetic.
- Canonicalize role entries by symbol.
- Fail closed on non-15-minute alignment.
- Never shorten DEV/SEALED in place.

### Step 4 — Verify + Refactor

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_temporal.py -q
```

### Step 5 — Commit

```bash
git add trade_rl/workflows/universal_trade_rl_u2_temporal.py tests/workflows/universal_trade_rl_u2_test_support.py tests/workflows/test_universal_trade_rl_u2_temporal.py
git commit -m "feat: define Universal Trade RL U2 temporal boundaries"
```

---

# Task 2: Freeze dataset-backed episode plans and final temporal contract

**Files**

- Modify `trade_rl/workflows/universal_trade_rl_u2_temporal.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_episode_planning.py`.

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SymbolEpisodePlan:
    symbol: str
    role: str
    dataset_digest: str
    authorized_fit_starts: tuple[int, ...]
    fit_coverage_starts: tuple[int, ...]
    g1_starts: tuple[int, ...]
    g2_starts: tuple[int, ...]
    g3_starts: tuple[int, ...]
    digest: str = ""

@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TemporalContract:
    boundaries_digest: str
    universe_manifest_digest: str
    symbol_plan_digests: tuple[tuple[str, str], ...]
    coverage_digest: str
    digest: str = ""
```

### Step 1 — RED: canonical coverage and boundary tests

Use the frozen U1 public episode planner for Train/Development datasets.

For Train symbols:

- `authorized_fit_starts` contains every valid 720h U1 start whose **final accounting timestamp** is `<= T_fit_end`;
- no authorized start crosses FIT;
- `fit_coverage_starts` is the canonical **latest 24 non-overlapping 720h FIT grid ending at `T_fit_end`**;
- all 24 required coverage episodes exist;
- all fixed 12 G1 DEV episodes resolve exactly.

For Development symbols:

- all fixed 12 G2 FIT episodes resolve exactly;
- all fixed 12 G3 DEV episodes resolve exactly.

For Admission:

- no dataset/planner/price/feature array is accepted or loaded;
- only U0 manifest metadata coverage of the reserved SEALED interval is checked.

Also test:

- one missing required interval fails closed;
- boundary fragments do not become partial episodes;
- plan input ordering cannot alter digest;
- dataset digest drift fails;
- per-role fields are closed: Train plan cannot contain G2/G3 and Development plan cannot contain authorized Train starts/G1.

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_episode_planning.py -q
```

### Step 3 — GREEN

- Use public U1 planner only; no private `EpisodeContractSampler` access from U2.
- Convert timestamp intervals to exact U1 `start_idx` by matching dataset timestamps; no nearest-neighbor/coercion.
- Bind each symbol plan to dataset digest + boundary digest.
- Bind final temporal contract to all symbol-plan digests and coverage evidence.

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

**Produces:**

```python
class AuthorizedUniversalTradeEnvironment(gym.Wrapper):
    ...
```

The wrapper owns only episode authorization and public policy-layout metadata. It does not own economics.

### Step 1 — RED: constructor/reset escape tests

Test:

1. `child.sequence_normalizer is None` rejects. Do **not** inspect U1 `.normalizer`, which intentionally represents the legacy flat-normalizer slot.
2. wrong U1 observation-contract digest rejects.
3. wrong U1 state-layout digest/field order rejects.
4. child observation keys must equal the exact U1 closed set.
5. no `start_idx`: wrapper chooses only from canonical authorized FIT starts using the supplied reset seed.
6. explicit authorized `start_idx`: accepted.
7. explicit non-authorized `start_idx`: rejected before child reset.
8. child returned `start_index/end_index` must exactly equal the planned U1 episode contract.
9. wrapper delegates `step()` exactly once and preserves action/reward/terminated/truncated/info.
10. same seed + same canonical authorized set => same selected start.
11. input authorized-set ordering cannot alter canonical digest.

### Step 2 — Verify RED

```bash
uv run pytest tests/rl/test_universal_trade_u2_environment.py -q
```

### Step 3 — GREEN

Constructor inputs:

```python
AuthorizedUniversalTradeEnvironment(
    child: UniversalTradeEnvironment,
    *,
    authorized_start_indices: tuple[int, ...],
    temporal_contract_digest: str,
    expected_u1_observation_contract_digest: str,
    state_layout_digest: str,
    policy_state_fields: tuple[str, ...],
)
```

Expose public `sequence_layout_metadata` for SB3 assembly:

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

Rules:

- derive feature counts from actual child observation space;
- reject missing/extra U1 keys;
- never import U1 private layout symbols;
- use `np.random.default_rng(seed)` only to choose among the frozen start tuple;
- rely on the routed environment to derive its episode-specific reset seed.

### Step 4 — Verify compatibility

```bash
uv run pytest tests/rl/test_universal_trade_u2_environment.py tests/rl/test_universal_trade_environment.py -q
```

### Step 5 — Commit

```bash
git add trade_rl/rl/universal_trade_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: restrict U2 episodes to frozen FIT starts"
```

---

# Task 4: Register `universal_trade_sequence_v1` and exact U2 training config validation

**Files**

- Modify `trade_rl/rl/training_modes.py`.
- Modify `trade_rl/rl/training.py`.
- Create `tests/rl/test_universal_trade_u2_training_config.py`.
- Modify `tests/rl/test_training_modes.py`.

### Step 1 — RED: closed encoder vocabulary

Add tests expecting:

```python
ObservationEncoder.UNIVERSAL_TRADE_SEQUENCE_V1.value == "universal_trade_sequence_v1"
```

and prove typo/unknown values reject.

### Step 2 — RED: exact U2 config helper

Define the production U2 training profile in `trade_rl/workflows/universal_trade_rl_u2_contract.py` later in Task 7; at this task add config-level capability tests proving `ResidualTrainingConfig` can represent the new sequence mode without weakening existing modes.

Required mode behavior:

- sequence parameters are active;
- asset-set-only embedding fields remain default/inactive;
- `policy_actor_head="shared_target_v1"` is required for U2 assembly later;
- BC remains disabled in the final U2 contract;
- existing `flat_mlp`, `asset_set`, `hierarchical_sequence_v2` validation remains unchanged.

### Step 3 — Verify RED

```bash
uv run pytest tests/rl/test_training_modes.py tests/rl/test_universal_trade_u2_training_config.py -q
```

### Step 4 — GREEN

- Extend `ObservationEncoder`.
- Refactor sequence-mode validation into shared helpers only where behavior is truly shared.
- Do not relax inactive-field checks for unrelated encoders.
- Keep existing error contracts deterministic.

### Step 5 — Compatibility

```bash
uv run pytest \
  tests/rl/test_training_modes.py \
  tests/rl/test_universal_trade_u2_training_config.py \
  tests/rl/test_action_head_ablation.py \
  tests/integrations/test_universal_sb3_model_assembly.py \
  tests/workflows/test_universal_full_research_training.py \
  -q
```

### Step 6 — Commit

```bash
git add trade_rl/rl/training_modes.py trade_rl/rl/training.py tests/rl/test_training_modes.py tests/rl/test_universal_trade_u2_training_config.py
git commit -m "feat: register U2 Universal Trade sequence mode"
```

---

# Task 5: U1-specific causal sequence feature extractor

**Files**

- Create `trade_rl/rl/universal_trade_policy.py`.
- Create `tests/rl/test_universal_trade_policy.py`.

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeSequenceArchitecture:
    ...

class UniversalTradeSequenceFeatureExtractor(BaseFeaturesExtractor):
    ...
```

### Step 1 — RED: exact observation surface

The extractor accepts exactly:

```text
sequence_15m_values / available / staleness
sequence_1h_values  / available / staleness
sequence_4h_values  / available / staleness
sequence_1d_values  / available / staleness
policy_state
```

and rejects legacy/context keys:

```text
current_snapshot
asset_state
global_state
instrument_context
local_cross_market_context
```

### Step 2 — RED: fixed architecture

Test exact U-Medium Direct structure:

```text
sequence_tcn_capacity = compact
d_model = 256
timeframe_attention_heads = 4
timeframe_attention_layers = 1
timeframe_ffn_multiplier = 3
timeframe_gate_bias = -2.0
sequence_dropout = 0.0
actor MLP = (256, 128)
critic MLP = (256, 128)
```

Test:

- output width is `1*256 + 256 + 128 + 2 = 642`;
- per-timeframe input is `safe_values + availability + log1p(staleness)`;
- unavailable values are defensively zeroed before encoding even though U1 should already guarantee it;
- `current_weight` comes from `current_weight_state_index`, not a hard-coded ordinal;
- distribution-active mask output is exactly `1.0`;
- architecture digest changes if structural widths/heads/layers/windows/channels/state-layout digest change;
- no cross-asset attention module exists.

### Step 3 — Verify RED

```bash
uv run pytest tests/rl/test_universal_trade_policy.py -q
```

### Step 4 — GREEN implementation

Reuse only:

```python
from trade_rl.rl.sequence_policy import CausalTimeframeEncoder, sequence_encoder_widths
from trade_rl.rl.timeframe_fusion import CrossTimeframeFusion
```

Do **not** instantiate `MultiTimeframeAssetEncoder`; it requires the legacy snapshot/asset-state context that U1 removed.

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
    ...

state_context = context_encoder(policy_state).unsqueeze(1)
fused = timeframe_fusion(..., context=state_context)
asset_token = fused[:, 0]
pooled = asset_token
global_state = global_state_encoder(policy_state)
active = torch.ones_like(current_weight)
return torch.cat((asset_token, pooled, global_state, active, current_weight), dim=-1)
```

### Step 5 — Verify + Refactor

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
- Extend the existing policy-identity tests that cover current v4 payloads.

### Step 1 — RED: model assembly

Build a routed U2 probe and assert:

```python
assembly.policy_identifier is SharedPerAssetActorCriticPolicy
assembly.observation_encoder == "universal_trade_sequence_v1"
assembly.policy_actor_head == "shared_target_v1"
assembly.sequence_symbols == ("INSTRUMENT",)
assembly.sequence_action_names == ("target_weight:INSTRUMENT",)
```

Assert `features_extractor_class is UniversalTradeSequenceFeatureExtractor` and exact U2 architecture metadata is passed.

Reject:

- non-generic policy symbol;
- action size != 1;
- missing/wrong state-layout digest;
- `hierarchical_gate_target_v1`;
- instrument/V4 context enabled;
- incorrect feature/window metadata.

### Step 2 — RED: action transport

Create an actual PPO model/policy and instrument the U1 child to record the received action.

Prove stochastic rollout and deterministic actions are finite and already in `[-1,+1]` before U1 receives them. The transport oracle should be exact at float32 where the maintained vector-env path allows:

```python
np.testing.assert_array_equal(received_action, produced_action.astype(np.float32))
```

If the framework inserts an unavoidable dtype conversion, use a separately justified byte/dtype oracle; do not accept value clipping as transport.

Inject a fake out-of-range action (`1.0001`) and prove U1 strict parsing rejects it. External clipping must not be the safety mechanism.

### Step 3 — Verify RED

```bash
uv run pytest tests/integrations/test_universal_trade_u2_sb3_model_assembly.py -q
```

### Step 4 — GREEN assembly

Add a dedicated `universal_trade_sequence_v1` branch in `resolve_sb3_policy_assembly()`:

- reuse `SharedPerAssetActorCriticPolicy`;
- force `shared_target_v1` for the frozen U2 contract;
- use standard Dict rollout storage for routed U2 environments;
- do not use the old single-dataset `SequenceRolloutReconstructor` across symbol-routed datasets;
- preserve `hierarchical_sequence_v2` behavior.

### Step 5 — GREEN identity

Extend `sb3_policy_identity_v4` as a **discriminated additive encoder variant** only if existing v4 payloads remain byte/semantic compatible. Add explicit U2 schemas such as:

```text
universal_trade_sequence_architecture_v1
universal_trade_direct_policy_v1
```

Bind the **actual instantiated extractor**, not config text only:

- U1 observation + state-layout identity;
- clock order/window/input channels;
- compact TCN widths/dilations;
- d_model 256;
- timeframe attention 4×1, FFN 3, gate bias -2, dropout 0;
- state context/global encoder widths;
- direct shared actor head;
- squashed Gaussian exploration identity;
- generic `INSTRUMENT` / `target_weight:INSTRUMENT` binding.

Compatibility oracle: existing v4 `flat_mlp`, `asset_set`, and `hierarchical_sequence_v2` payloads still validate identically. If implementation shows the new union cannot be represented without changing existing v4 meaning, **stop**, version to a new schema, add explicit legacy read behavior, update spec/plan, then continue. Do not silently reinterpret v4.

### Step 6 — Verify compatibility

```bash
uv run pytest \
  tests/integrations/test_universal_trade_u2_sb3_model_assembly.py \
  tests/integrations/test_universal_sb3_model_assembly.py \
  tests/integrations/test_sb3_policy_identity_v3.py \
  tests/rl/test_asset_agnostic_policy_identity.py \
  -q
```

### Step 7 — Commit

```bash
git add trade_rl/integrations/sb3_model_assembly.py trade_rl/rl/policy_identity.py trade_rl/artifacts/policy_identity_contract.py tests/integrations/test_universal_trade_u2_sb3_model_assembly.py
git commit -m "feat: assemble bounded U2 PPO policy"
```

Include any actually modified existing identity-test files in the same commit.

---

# Task 7: Frozen runtime/model config, seed namespace, and U0 run identity

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_contract.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_contract.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_seed_identity.py`.
- Extend `tests/rl/test_universal_trade_u2_training_config.py`.

### Step 1 — RED: strict authored runtime config closure

U2 accepts one strict `TrainingRunConfig` as construction input. It is not a free tuning surface.

The config must satisfy two independent closures before dataset/environment creation:

**U1 economic/runtime closure**

Its environment/action/reward/risk/portfolio-risk/execution semantics must reproduce the digests frozen in `u1_contract.json`.

**U2 learner closure**

Its `training` section must equal the preregistered U-Medium Direct PPO profile.

Exact production learner values:

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
observation_encoder = universal_trade_sequence_v1
policy_actor_head = shared_target_v1
policy_net_arch = (256,128)
value_net_arch = (256,128)
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

All remaining `ResidualTrainingConfig` fields are also included in canonical model-config identity; inactive fields must be at their validated defaults.

One-field mutation tests must fail for every scientific/resource field that can change learning or retained-checkpoint behavior.

### Step 2 — RED: U1/U0/cutoff dependencies

Reject before training:

- `sequence_normalizer` absent;
- U1 normalizer cutoff != `T_fit_end`;
- U0 `RL_TRAINING` provenance cutoff != `T_fit_end`;
- U1/U0 universe manifest mismatch;
- U1 observation/state/action/reward/runtime/economic digest mismatch;
- authored runtime config does not reproduce frozen U1 digests;
- U1 `production_status` not `NO-GO`.

### Step 3 — RED: no circular seed identity

Implement/test:

```text
seed_namespace_digest
  = digest(universe + U1 + temporal + model config + seed_count=8)

seed_digest_i
  = digest(seed_namespace_digest + index i + schema)

seed_i
  = unsigned big-endian uint32(first 4 bytes)
```

Assert:

- exactly eight ordered unique uint32 seeds;
- repeated construction is deterministic;
- changing universe/U1/temporal/model identity changes namespace;
- forced collision fails closed; no ninth probing seed;
- final U2 contract digest binds the ordered resolved seed vector.

### Step 4 — RED: U0 maintained run identity

Build existing:

- `UniversalTradeRLFitPurpose.RL_TRAINING` provenance;
- `UniversalTradeRLRunStage.BASE_TRAINING` identity.

Do not add a second U2 stage enum.

### Step 5 — Verify RED

```bash
uv run pytest \
  tests/rl/test_universal_trade_u2_training_config.py \
  tests/workflows/test_universal_trade_rl_u2_contract.py \
  tests/workflows/test_universal_trade_rl_u2_seed_identity.py \
  -q
```

### Step 6 — GREEN + commit

```bash
uv run pytest \
  tests/workflows/test_universal_trade_rl_u2_contract.py \
  tests/workflows/test_universal_trade_rl_u2_seed_identity.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  -q

git add trade_rl/workflows/universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_contract.py tests/workflows/test_universal_trade_rl_u2_seed_identity.py tests/rl/test_universal_trade_u2_training_config.py
git commit -m "feat: freeze U2 model and seed identities"
```

---

# Task 8: Eight-seed Base PPO training and final-checkpoint closure

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u2_training.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_training.py`.

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeU2ConcreteEnvironmentFactory:
    ...

@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TrainingResult:
    ...
```

### Step 1 — RED: concrete factory closes U1 economics

The factory receives only frozen, identity-checked inputs:

- dataset artifact paths for U0 Train symbols;
- U0 manifest;
- frozen U1 policy contract;
- frozen U1 sequence normalizer;
- strict authored `TrainingRunConfig` already validated against U1 + U2 identities;
- per-symbol U2 FIT plans.

For each binding:

1. load immutable single-symbol dataset artifact;
2. verify dataset digest against U0 manifest/binding;
3. create `UniversalTradeMarketEnv` using the frozen U1 runtime/economic config;
4. wrap with `UniversalTradeEnvironment(..., normalizer=frozen_u1_sequence_normalizer)`;
5. wrap with `AuthorizedUniversalTradeEnvironment` using that Train symbol's authorized FIT starts.

No Development/Admission dataset path is accepted by this factory type.

### Step 2 — RED: routed vector factory

For each frozen member seed build existing `UniversalRoutedEnvironmentFactory` with:

```python
instrument_context_provider = None
v4_context_provider = None
training_contract_digest = u2_contract.digest
run_seed = member_seed
```

Test `for_environment_index(0..7)` produces distinct worker identities and every complete router cycle is balanced.

### Step 3 — RED: exactly eight valid runs

Test orchestration:

- iterates exactly the frozen 8 seeds;
- each member targets exactly `524288` timesteps;
- uses `StableBaselines3Backend` rather than a new PPO implementation;
- technically valid but economically losing members are retained;
- no ninth replacement member;
- technical retry/resume requires same seed/U2/model/environment/checkpoint identity;
- intermediate checkpoint is never performance-eligible;
- exactly one canonical final checkpoint manifest per valid seed is performance-eligible.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_training.py tests/integrations/test_universal_trade_u2_sb3_model_assembly.py -q
```

### Step 5 — GREEN

Persist under:

```text
training/<seed>/
  final-checkpoint/...
  training_result.json
  routing_evidence.json
```

`training_result.json` binds U2 contract/model/seed/BASE_TRAINING identity, final checkpoint identity, observed timesteps, worker/router evidence, and technical completion state.

### Step 6 — Test-only tiny integration

A test-only fixture may replace timesteps with a tiny number **only inside tests** to prove plumbing. It must not instantiate or serialize a production U2 contract under the reduced values. Production contract validator still rejects the reduced profile.

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_training.py -q
```

### Step 7 — Commit

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

### Step 1 — RED: exact evaluation scope

Assert:

- G1 = Train × fixed 12 DEV episodes;
- G2 = Development × latest fixed 12 FIT episodes;
- G3 = Development × exactly the same 12 DEV intervals as G1;
- Development symbol count is at least 3;
- no Admission/Excluded symbol is accepted;
- no missing/extra episode is accepted;
- only canonical final checkpoints are accepted.

### Step 2 — RED: baseline economic parity

On deterministic synthetic markets prove:

- `CASH_FLAT` requests 0;
- `BUY_AND_HOLD_LONG` requests +1;
- `TREND_BASELINE` uses the maintained TrendStrategy target only as an external action source;
- all paths start cash-only at identical start/end timestamps;
- all actions pass through the same U1 signal-delay/Risk/Execution/accounting contract;
- fees/spread/impact/funding/borrow/partial fills/margin affect baseline and policy consistently;
- no baseline receives free terminal liquidation.

### Step 3 — RED: immutable economic leaf contract

Atomic key:

```text
(seed, scope, symbol, episode_index)
```

Require:

- exact start/end timestamps;
- initial/final wealth;
- gross return when independently defined;
- after-cost net return and net log growth;
- Cash/BuyHold/Trend comparable results;
- max drawdown;
- turnover and turnover/day;
- execution cost;
- funding PnL;
- borrow cost;
- requested/executed/fill evidence;
- trade/rebalance/fill counts;
- termination reason;
- hard-Risk invariant evidence;
- execution rejection reasons;
- policy/checkpoint/environment/source/U0/U1/U2 identities.

Independent accounting oracle:

```python
expected_net_log_growth = math.log(final_wealth / initial_wealth)
```

Reject disagreement.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py -q
```

### Step 5 — GREEN

- Load final checkpoint via maintained checkpoint loader.
- Development prediction is deterministic.
- No optimizer/normalizer/calibration update path is imported or called.
- Each leaf resets with exact planned `start_idx`.
- Preserve all scope/identity evidence.

### Step 6 — Verify + commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py -q
git add trade_rl/workflows/universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_evaluation.py tests/workflows/test_universal_trade_rl_u2_baselines.py
git commit -m "feat: replay U2 Development scopes and baselines"
```

---

# Task 10: Atomic immutable Development evidence and resume

**Files**

- Modify `trade_rl/workflows/universal_trade_rl_u2_evaluation.py`.
- Create `tests/workflows/test_universal_trade_rl_u2_evidence_io.py`.

### Step 1 — RED

Test:

- path exactly `development/records/<scope>/<seed>/<symbol>/<episode>.json`;
- canonical byte/digest round-trip;
- identical existing leaf => idempotent reuse;
- modified/corrupt/wrong-path/wrong-identity leaf => fail closed;
- extra unknown leaf => complete-scope failure;
- partial write never appears as final valid evidence;
- crash after compute/before durable publish may recompute the exact same leaf;
- once a valid leaf exists, aggregate outcomes cannot cause it to be recomputed/replaced.

### Step 2 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_evidence_io.py -q
```

### Step 3 — GREEN

Reuse canonical JSON and existing atomic-write conventions; do not invent a second serializer.

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

### Step 1 — RED: independent aggregation fixture

Use hand-calculated evidence with exactly 8 seeds × at least 3 Development symbols × 12 episodes.

G3 primary series:

```python
leaf_excess(seed, symbol, j) = policy_net_log_growth - cash_net_log_growth
symbol_episode_excess(symbol, j) = median(leaf_excess over 8 seeds)
primary_excess(j) = median(symbol_episode_excess over Development symbols)
```

Assert exactly 12 ordered time observations. Seed replicas are collapsed before significance testing.

### Step 2 — RED: fixed block bootstrap

Reuse `moving_block_mean_test` with frozen parameters:

```text
n_bootstrap = 2000
block_size = 3 episodes
CI = existing 2.5% / 97.5%
bootstrap_seed = deterministic digest-derived uint32
```

Bind bootstrap schema/seed/parameters into summary identity.

### Step 3 — RED: every rejection reason

**Structural:**

- NaN/Inf;
- source/identity drift;
- unauthorized fit/update evidence;
- action transport alteration;
- missing/duplicate/extra leaf;
- unexplained execution rejection;
- hard Risk invariant violation;
- evidence tamper;
- poor-seed substitution/retry.

**G1:**

```text
median across 8 seed-level excess net log growth vs Cash > 0
```

**G2:** same.

**G3 — all required:**

```text
median across 8 seed-level G3 excess vs Cash > 0
mean(primary_excess[12]) > 0
moving-block bootstrap lower_ci(primary_excess) > 0
positive seed count >= 6 / 8
median Development-symbol absolute net log growth > 0
positive Development-symbol excess-vs-Cash fraction >= 0.60
minimum G3 leaf net return >= -0.05
for every seed: mean G3 turnover/day <= 1.0
economic termination count == 0
median across 8 seed-level G3 excess vs Trend > 0
```

Buy-and-Hold remains diagnostic.

### Step 4 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_aggregation.py tests/workflows/test_universal_trade_rl_u2_development_gate.py -q
```

### Step 5 — GREEN

Only terminal decisions:

```text
DEVELOPMENT_ACCEPTED
DEVELOPMENT_REJECTED
```

No ranking/candidate grid.

Persist:

```text
development/summary.json
development/decision.json
```

Decision binds complete leaf digest closure, threshold identity, evaluator/gate code identity, bootstrap identity, and U2 contract digest.

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

### Step 1 — RED: one-way orchestration dependency

Enforce:

```text
load/verify frozen U0 + U1 + strict runtime config
-> derive/materialize source-only temporal boundaries
-> build/materialize dataset-backed episode plans/final temporal contract
-> verify U1 normalizer cutoff == T_fit_end
-> materialize U2 contract + seeds + BASE_TRAINING identity
-> train/reuse exact 8 final checkpoints
-> replay/reuse complete G1/G2/G3 leaves
-> aggregate
-> write Development decision
-> STOP
```

Tests:

- missing prior-stage artifact prevents later stage;
- valid artifacts resume idempotently;
- corrupt/extra/wrong-identity artifacts fail closed;
- Development reject is a valid terminal research result and evidence remains;
- training module does not import Development module;
- no U2 path loads Admission datasets.

### Step 2 — RED: narrow CLI

CLI accepts frozen roots/config only, not scientific knobs:

```bash
uv run python scripts/run_universal_trade_rl_u2.py \
  --u0-root <path> \
  --u1-root <path> \
  --runtime-config <path> \
  --dataset-root <path> \
  --output-root <path>
```

Do **not** expose algorithm, seed count, thresholds, architecture, learning rate, or candidate grid.

### Step 3 — Verify RED

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_runner.py tests/scripts/test_run_universal_trade_rl_u2.py -q
```

### Step 4 — GREEN

Runner terminal status is a versioned operational contract. Before coding, check the final repository's maintained research-run exit convention. If no stronger convention exists, use:

- `0`: software valid + `DEVELOPMENT_ACCEPTED`;
- `3`: software valid + `DEVELOPMENT_REJECTED`;
- other nonzero: contract/execution failure.

If the maintained convention differs, update spec/plan first; do not silently change semantics.

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

### Step 1 — Attempt to break the implementation

At minimum test:

1. U1 normalizer cutoff one 15m bar after `T_fit_end`.
2. `RL_TRAINING` provenance containing a Development symbol.
3. Train reset with a DEV `start_idx`.
4. Development reset with a wrong planned interval.
5. U1 hidden missing placeholder changed under `available=false`.
6. concrete ticker/dataset ID injected into policy observation.
7. instrument context or V4 context enabled.
8. fake policy emits `1.0001` and relies on external clipping.
9. seed vector member replaced after valid runs exist.
10. ninth seed substituted for an economically poor seed.
11. intermediate checkpoint substituted for final.
12. final checkpoint from wrong model config/U2 identity.
13. one G3 leaf missing.
14. one G3 leaf duplicated/extra.
15. one leaf content modified while preserving path.
16. execution cost changed for Trend baseline only.
17. funding or borrow double-counted.
18. 8 seed replicas incorrectly fed as independent bootstrap time observations.
19. one `-5.01%` G3 leaf hidden by a positive mean.
20. one seed mean turnover above `1.0x/day` hidden by median.
21. economic termination hidden by positive aggregate.
22. gate/threshold artifact changed after leaves exist.
23. Admission dataset loader invoked before authorization.
24. resume artifact generated by different evaluator/gate code identity.

### Step 2 — Real-boundary economic integration

Use maintained production objects rather than mocks only to exercise:

- fee;
- spread;
- impact;
- funding;
- borrow;
- liquidity/partial fill;
- signal delay;
- margin/economic termination;
- no terminal liquidation.

### Step 3 — Run

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u2_falsification.py tests/integrations/test_universal_trade_rl_u2_economics.py -q
```

### Step 4 — Fix/retest

A substantive fixable issue discovered here must be fixed, followed by nearest targeted tests and then this suite again. Do not only document it.

### Step 5 — Commit

```bash
git add tests/workflows/test_universal_trade_rl_u2_falsification.py tests/integrations/test_universal_trade_rl_u2_economics.py
git commit -m "test: falsify Universal Trade RL U2 boundaries"
```

---

# Task 14: Documentation, full verification, architecture review, independent review, exact-HEAD CI

**Files**

- Modify `docs/UNIVERSAL_TRADE_RL.md`.
- Modify `docs/CONFIGURATION.md`.
- Modify `tests/test_architecture_contract.py` only if the architecture checker requires it.

### Step 1 — Documentation

Document without overstating research state:

- source-only temporal-boundary derivation vs dataset-backed episode-plan closure;
- U1 normalizer cutoff dependency;
- U-Medium Direct `universal_trade_sequence_v1` architecture;
- exact eight-seed PPO profile;
- no best-seed/checkpoint selection;
- G1/G2/G3;
- baseline/economic parity;
- Development gates;
- artifact/resume semantics;
- `Admission=CLOSED`, `Production=NO-GO` even after Development acceptance.

Resolve stale U0/U1 handoff wording to the actual order:

```text
U0 freeze
-> U2 temporal boundaries
-> U1 final normalizer/artifact freeze at T_fit_end
-> U1 Quality Gate
-> U2 episode-plan closure/training/evaluation
```

### Step 2 — Targeted U2 wave

```bash
uv run pytest \
  tests/rl/test_universal_trade_u2_environment.py \
  tests/rl/test_universal_trade_u2_training_config.py \
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
  tests/scripts/test_run_universal_trade_rl_u2.py \
  -q
```

### Step 3 — U1/Universal compatibility wave

Run current equivalents of:

```bash
uv run pytest \
  tests/rl/test_universal_trade_environment.py \
  tests/rl/test_universal_trade_observation.py \
  tests/rl/test_universal_trade_u1_normalization.py \
  tests/integrations/test_universal_sb3_model_assembly.py \
  tests/integrations/test_sb3_training.py \
  tests/workflows/test_universal_full_research_training.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  -q
```

If U1 finalization renames a file, replace it with the current equivalent; do not skip the coverage layer.

### Step 4 — Static / architecture

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

Also run any canonical repository verification wrapper introduced by final U1.

### Step 6 — Changed-line/assertion audit

Confirm changed production branches are actually executed and strongly asserted, especially:

- cutoff rejection;
- authorized/unauthorized `start_idx`;
- missing-value defense;
- exact U1-only extractor surface;
- bounded action transport;
- seed collision/substitution;
- resume/tamper;
- every Development rejection reason.

Coverage percentage is only a signal, not the oracle.

### Step 7 — Architecture/self-review loop

Review final diff for:

- original requirement compliance;
- dependency direction: `U2 -> frozen U1`, never `U1 -> U2`;
- training module cannot import Development module;
- no duplicated Risk/Execution/Reward/normalizer logic;
- no Development/Admission dataset access from training;
- no concrete symbol/context in policy tensors;
- action distribution/transport correctness;
- state transition/accounting correctness;
- atomicity/resume/idempotency;
- deterministic ordering;
- dead/debug/temporary code;
- secrets/generated artifacts/unrelated refactors.

For each substantive finding: fix → nearest targeted tests → falsification → broader verification again.

### Step 8 — Independent / falsification review

Give the verifier only:

1. original U2 spec;
2. this plan;
3. final diff;
4. tests and assertions;
5. actual verification outputs.

Ask the verifier to find, not justify:

- any path from Development/Admission into fit state;
- any path that changes the action before U1 strict parser;
- any way to replace a poor seed/final checkpoint;
- any way to change gates after results;
- any way to pass with missing/downside-breaching evidence;
- any integration assumption hidden by mocks.

### Step 9 — Git hygiene

```bash
git diff --check
git status --short
git log -1 --oneline
git merge-base --is-ancestor <final-u1-head> HEAD
```

Confirm no untracked debug/temp files, secrets, temporary workflows, unrelated changes, or unintended branch ancestry.

### Step 10 — exact-final-HEAD CI

Push only after local gates pass. Verify required CI belongs to the exact final U2 HEAD, not an earlier commit. Do not mark Ready or merge unless explicitly authorized and the full Quality Gate is met.

### Step 11 — Final evidence report

Report separately:

1. what changed;
2. design rationale;
3. Acceptance Criteria mapping;
4. Failure Modes exercised;
5. exact tests/static/build outputs;
6. independent/falsification findings;
7. exact HEAD and CI state;
8. unverified items;
9. residual risks;
10. what this verification guarantees and does not guarantee.

If a real production-candidate eight-seed run has **not** been executed, state explicitly:

```text
Software implementation may be valid.
Economic Development acceptance is not established.
Admission remains closed.
Production remains NO-GO.
```

### Step 12 — Docs commit

```bash
git add docs/UNIVERSAL_TRADE_RL.md docs/CONFIGURATION.md
git add tests/test_architecture_contract.py  # only if actually modified
git commit -m "docs: document Universal Trade RL U2 gates"
```

---

# Acceptance Criteria → Task Mapping

| Acceptance area | Primary tasks |
| --- | --- |
| deterministic source-only cutoffs/grids | 1 |
| dataset-backed coverage/authorized starts | 2 |
| Train×FIT-only reset enforcement | 3, 8 |
| frozen U1 normalizer/cutoff dependency | 0, 7, 13 |
| exact U1 observation only | 3, 5 |
| bounded direct scalar action | 5, 6 |
| U-Medium Direct PPO identity | 4–7 |
| exact 8 seeds/no replacement | 7–8 |
| balanced Train-symbol routing | 8 |
| final-checkpoint-only eligibility | 8 |
| G1/G2/G3 fixed evaluation | 9 |
| baseline parity | 9, 13 |
| immutable leaf evidence | 9–10 |
| block-bootstrap aggregation | 11 |
| fixed Development thresholds | 11 |
| Admission firewall | 7, 9, 12, 13 |
| falsification | 13 |
| static/full/build/CI/independent review | 14 |

---

# Final execution rule

Do not start a real U2 training run merely because software tests pass.

Real execution order:

```text
final U0/U1 Quality Gate
-> U2 source-only temporal-boundary materialization
-> verify/freeze T_fit_end
-> final U1 normalizer/artifact identity at T_fit_end
-> U2 dataset-backed episode-plan/coverage closure
-> U2 runtime/model/seed contract freeze
-> 8-seed Base PPO training
-> final-checkpoint closure
-> G1/G2/G3 Development replay
-> immutable Development decision
-> STOP; Admission stays closed
```

If Development rejects, preserve the evidence. Any scientific change becomes a new U2 generation rather than an in-place retry.