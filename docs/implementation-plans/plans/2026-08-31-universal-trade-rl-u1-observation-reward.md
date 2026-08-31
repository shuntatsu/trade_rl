# Universal Trade RL U1 Observation / Action / Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the audited U1 contract: one-symbol causal observations without manual strategy priors, scalar normalized target exposure with one-decision signal delay, Train-only equal-symbol market normalization, and reward exactly reconciled to realized after-cost wealth.

**Architecture:** Keep `ResidualMarketEnv` as the single risk/execution/accounting authority and add a focused U1 wrapper. Add one named read-only runtime snapshot to expose the already-maintained distinctions between current submitted policy target, signal-delay pending target, post-risk execution target, realized position, execution diagnostics, and pending-order lifecycle. U1 market normalization uses equal-symbol moments over unique native source rows; endogenous policy state uses versioned deterministic transforms and is never fitted from policy-generated state distributions. U1 artifacts (`u1_contract.json`, `normalizer.json`) are materialized atomically and bind the frozen U0 generation.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, dataclasses, existing `MarketDataset`, `SequenceObservationBuilder`, `ResidualMarketEnv`, PreTradeRisk, execution/accounting, canonical SHA-256 digests, pytest, Hypothesis, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`

**Pinned design base:** `b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c` (U0 PR #426 head used to create this branch). Before implementation begins, verify PR #426 has not advanced. If its intended U0 head has changed, synchronize this design branch first and amend this pinned comparator before writing production code; do not verify against a stale base.

## Global Constraints

- Production status remains `NO-GO`.
- U1 does not train an RL algorithm, open Admission, claim profitability, or change Production status.
- `dataset.n_symbols == 1` is mandatory for the U1 wrapper.
- Existing U3-U6 and Causal Alpha V9/V10/V11 behavior/economics remain unchanged.
- Existing `ResidualMarketEnv` remains the only risk/execution/accounting engine; U1 must not duplicate fee/spread/impact/funding/borrow/margin/order lifecycle logic.
- U1 V1 requires `ActionMode.TARGET_WEIGHT`, target count 1, `signal_delay_decisions == 1`, structured sequence observation, external truncation, `finite_horizon_observation == False`, `liquidate_on_end == False`, and sampled `initial_state_modes == ("cash",)`.
- Policy input must not expose symbol/dataset ID, raw absolute OHLC, raw nominal volume/cash/quantity, `TrendTargets`, alpha-provider output, factor priors, shadow/baseline state, manually latched ownership, or remaining episode fraction.
- Market windows are exactly `15m×96`, `1h×168`, `4h×120`, `1d×60`.
- Market feature kinds are restricted to the frozen U1 allowlist in the spec; cross-asset/BTC-reference kinds are rejected.
- All market source rows must be causal: `source_index <= decision_index`.
- Missing value, availability, and staleness remain separate.
- `policy_requested_weight`, signal-delay `pending_target_weight`+`pending_target_active`, `risk_projected_weight`, and `current_weight` have distinct meanings.
- Signal-delay pending target is not pending-order lifecycle. Pending orders use existing `PendingOrderObservationState`.
- Existing `ObservationExecutionState.requested_weights` is post-risk target because `ResidualMarketEnv.step()` populates it from `hybrid_risk.weights`; do not rename its existing semantic globally.
- Existing `ObservationExecutionState.execution_cost` is `cost_by_symbol / initial_capital`; U1 exposes it as `execution_cost_rate` without re-scaling.
- U0 `require_normalization_scope()` must run before supplied normalization datasets are read.
- Only Train data contribute to fitted market statistics.
- Normalizer uses equal-symbol moment aggregation and unique native source rows; row-count weighting and repeated base-clock copies of high-timeframe values are prohibited.
- Endogenous policy state uses deterministic transforms; it is not fitted from rollout/action distributions.
- Reward is exactly `100 * log(W_after / W_before)` from realized hybrid `BookState.portfolio_value`. No extra reward shaping or cost penalty.
- Non-positive/non-finite wealth fails closed in the U1 reward layer.
- Sample horizon is Gymnasium truncation; no forced end liquidation/bonus.
- No merge to `main` without explicit user permission.

---

## File Map

**Create:**

- `trade_rl/rl/universal_trade_contract.py` — schema constants, allowlist, policy contract and digest.
- `trade_rl/rl/universal_trade_action.py` — strict scalar action parsing.
- `trade_rl/rl/universal_trade_runtime.py` — immutable named runtime snapshot.
- `trade_rl/rl/universal_trade_observation.py` — causal U1 observation builder/space.
- `trade_rl/rl/universal_trade_normalization.py` — equal-symbol Train-only normalizer, payload codec.
- `trade_rl/rl/universal_trade_reward.py` — pure wealth reward/reconciliation.
- `trade_rl/rl/universal_trade_environment.py` — U1 Gym wrapper.
- `trade_rl/workflows/universal_trade_rl_u1_contract.py` — frozen U1 contract artifact/identity.
- `trade_rl/workflows/universal_trade_rl_u1_runner.py` — atomic materialization of `u1_contract.json` + `normalizer.json`.

**Modify:**

- `trade_rl/rl/environment.py` — one read-only `universal_trade_runtime_snapshot()` accessor only; no economic behavior change.
- `docs/UNIVERSAL_TRADE_RL.md` — U1 maintained documentation and U2 gate.

**Create tests:**

- `tests/rl/test_universal_trade_contract.py`
- `tests/rl/test_universal_trade_action.py`
- `tests/rl/test_universal_trade_runtime.py`
- `tests/rl/test_universal_trade_observation.py`
- `tests/rl/test_universal_trade_normalization.py`
- `tests/rl/test_universal_trade_reward.py`
- `tests/rl/test_universal_trade_environment.py`
- `tests/rl/test_universal_trade_falsification.py`
- `tests/workflows/test_universal_trade_rl_u1_contract.py`
- `tests/workflows/test_universal_trade_rl_u1_runner.py`

---

### Task 1: Freeze U1 policy/schema contract

**Files:**
- Create: `trade_rl/rl/universal_trade_contract.py`
- Test: `tests/rl/test_universal_trade_contract.py`

**Produces:**

```python
UNIVERSAL_TRADE_OBSERVATION_SCHEMA = "universal_trade_observation_v1"
UNIVERSAL_TRADE_ACTION_SCHEMA = "normalized_target_exposure_v1"
UNIVERSAL_TRADE_REWARD_SCHEMA = "universal_net_log_growth_reward_v1"
UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA = "universal_trade_policy_state_v1"
UNIVERSAL_TRADE_SEQUENCE_WINDOWS = (
    ("15m", 96),
    ("1h", 168),
    ("4h", 120),
    ("1d", 60),
)
```

and `UniversalTradePolicyContract` with ordered `FeatureSpec`s, `policy_weight_scale=1.0`, `reward_scale=100.0`, fixed signal delay 1, cash-only reset, external truncation, and content digest.

- [ ] **Step 1: Write failing tests**

```python
def test_contract_rejects_cross_asset_feature() -> None:
    spec = FeatureSpec(name="relative_btc", kind=FeatureKind.RELATIVE_RETURN_TO_BTC)
    with pytest.raises(ValueError, match="U1 feature"):
        UniversalTradePolicyContract(feature_specs=(spec,))


def test_contract_binds_ordered_feature_specs() -> None:
    ret = FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    vol = FeatureSpec(name="vol", kind=FeatureKind.REALIZED_VOLATILITY, lookback=16)
    assert UniversalTradePolicyContract(feature_specs=(ret, vol)).digest != UniversalTradePolicyContract(feature_specs=(vol, ret)).digest
```

Also test empty/duplicate names, unsupported timeframe, invalid scale, wrong windows, signal delay other than 1, non-cash sampled reset, finite-horizon observation, and finite-horizon termination.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
```

- [ ] **Step 3: Implement exact allowlist**

```python
UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS = frozenset(
    {
        FeatureKind.LOG_RETURN,
        FeatureKind.BODY_RETURN,
        FeatureKind.HIGH_LOW_RANGE,
        FeatureKind.GAP_RETURN,
        FeatureKind.REALIZED_VOLATILITY,
        FeatureKind.DOWNSIDE_VOLATILITY,
        FeatureKind.UPSIDE_VOLATILITY,
        FeatureKind.VOLATILITY_OF_VOLATILITY,
        FeatureKind.ATR_PCT,
        FeatureKind.ATR_CHANGE,
        FeatureKind.EMA_DISTANCE,
        FeatureKind.EMA_SLOPE,
        FeatureKind.LINEAR_REGRESSION_SLOPE,
        FeatureKind.TREND_R2,
        FeatureKind.VOLUME_ZSCORE,
        FeatureKind.VOLUME_LOG_CHANGE,
        FeatureKind.RELATIVE_VOLUME,
        FeatureKind.FUNDING_BPS,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.FUNDING_ZSCORE,
    }
)
```

Digest payload must include every semantic knob above and `FeatureSpec.canonical_payload()` in exact order.

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
uv run ruff check trade_rl/rl/universal_trade_contract.py tests/rl/test_universal_trade_contract.py
uv run mypy trade_rl/rl/universal_trade_contract.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/universal_trade_contract.py tests/rl/test_universal_trade_contract.py
git commit -m "feat: define Universal Trade RL U1 contract"
```

---

### Task 2: Implement strict scalar action semantics

**Files:**
- Create: `trade_rl/rl/universal_trade_action.py`
- Test: `tests/rl/test_universal_trade_action.py`

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class NormalizedTargetExposureAction:
    normalized: float
    policy_requested_weight: float


def parse_normalized_target_exposure(
    value: np.ndarray,
    *,
    policy_weight_scale: float,
) -> NormalizedTargetExposureAction: ...
```

- [ ] **Step 1: Write failing mapping/range tests**

```python
@pytest.mark.parametrize("raw", (-1.0, -0.5, 0.0, 0.5, 1.0))
def test_action_is_linear(raw: float) -> None:
    parsed = parse_normalized_target_exposure(
        np.asarray([raw], dtype=np.float32),
        policy_weight_scale=1.0,
    )
    assert parsed.policy_requested_weight == pytest.approx(raw)


def test_action_rejects_hidden_clipping() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        parse_normalized_target_exposure(np.asarray([1.01]), policy_weight_scale=1.0)
```

Add wrong shape, NaN/inf, invalid scale tests.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
```

- [ ] **Step 3: Implement minimal strict parser**

```python
vector = np.asarray(value, dtype=np.float64).reshape(-1)
if vector.shape != (1,) or not np.isfinite(vector).all():
    raise ValueError("Universal Trade RL action must be one finite scalar")
normalized = float(vector[0])
if not -1.0 <= normalized <= 1.0:
    raise ValueError("Universal Trade RL action must be within [-1, 1]")
```

Do not call dynamic risk code here.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
uv run ruff check trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
uv run mypy trade_rl/rl/universal_trade_action.py
git add trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
git commit -m "feat: add normalized target exposure action"
```

---

### Task 3: Add named runtime snapshot without changing economics

**Files:**
- Create: `trade_rl/rl/universal_trade_runtime.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_universal_trade_runtime.py`

**Consumes existing audited semantics:**

- current submitted action after a step: `self._previous_action[0]` in target-weight mode → `policy_requested_weight`.
- signal-delay queue after a step: `self._pending_hybrid_target` → `pending_target_weight` + active mask.
- post-risk target actually handed to execution: `self._execution_state.requested_weights[0]` → `risk_projected_weight`.
- realized target: `self.hybrid.weights[0]` → `current_weight`.
- execution diagnostics: existing `ObservationExecutionState`.
- order lifecycle: existing `_pending_order_observation_state()` / `PendingOrderObservationState`.
- `execution_cost` already equals cost/initial capital.

**Produces:** `UniversalTradeRuntimeSnapshot` and `ResidualMarketEnv.universal_trade_runtime_snapshot()`.

- [ ] **Step 1: Write failing signal-delay/risk/partial-fill test**

Use a single-symbol target-weight environment with signal delay 1. Drive two decisions so decision 2 simultaneously has:

```text
current submitted policy target = +0.80
previous pending target executed this step = +0.60
post-risk target = +0.35
realized weight after partial fill != +0.35
current submitted target remains pending = +0.80
```

Assert:

```python
snapshot = env.universal_trade_runtime_snapshot()
assert snapshot.policy_requested_weight == pytest.approx(0.80)
assert snapshot.pending_target_active is True
assert snapshot.pending_target_weight == pytest.approx(0.80)
assert snapshot.risk_projected_weight == pytest.approx(0.35)
assert snapshot.current_weight != pytest.approx(snapshot.risk_projected_weight)
```

Also assert pending-order fields are separate from `pending_target_*`.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
```

- [ ] **Step 3: Implement immutable snapshot**

The accessor must:

1. require target-weight mode and one symbol;
2. read, never mutate, existing state;
3. map `None` pending target to `pending_target_weight=0.0`, `pending_target_active=False`;
4. preserve true pending flat target as `0.0`, `pending_target_active=True`;
5. convert `position_age` from bars to hours using `dataset.bar_hours`;
6. expose `execution_cost_rate` directly from existing normalized execution cost;
7. derive drawdown/gross/net/cash/risk/margin from the hybrid book/current risk state;
8. build pending order observation through the existing maintained helper.

No `_last_risk_projected_target` field is added: existing `_execution_state.requested_weights` is already the correct post-risk oracle.

- [ ] **Step 4: Regression verify**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/test_universal_trade_runtime.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/test_universal_trade_runtime.py
git commit -m "feat: expose Universal Trade RL runtime state"
```

---

### Task 4: Build strategy-prior-free causal U1 observation

**Files:**
- Create: `trade_rl/rl/universal_trade_observation.py`
- Test: `tests/rl/test_universal_trade_observation.py`

**Produces:**

```python
UNIVERSAL_TRADE_POLICY_STATE_NAMES: tuple[str, ...]

class UniversalTradeObservationBuilder:
    def build(
        self,
        dataset: MarketDataset,
        *,
        index: int,
        runtime: UniversalTradeRuntimeSnapshot,
        normalizer: UniversalTradeSequenceNormalizer | None = None,
    ) -> dict[str, np.ndarray]: ...
```

- [ ] **Step 1: Write failing exact-key/layout test**

Expected keys are only:

```text
sequence_15m_values / available / staleness
sequence_1h_values / available / staleness
sequence_4h_values / available / staleness
sequence_1d_values / available / staleness
policy_state
```

Policy-state names must contain the exact fields in the spec, including `pending_target_active`, and no `trend`, `alpha`, `shadow`, `baseline`, `remaining`, `symbol`, or `dataset` field.

- [ ] **Step 2: Write future-mutation RED test**

Create two datasets identical through `t`, mutate all later market arrays in one, and assert every U1 tensor at `t` is equal.

- [ ] **Step 3: Write symbol rename and price/unit scale tests**

Symbol text-only change must not change policy tensors. Economically equivalent price/unit rescaling must leave dimensionless U1 tensors equal within defined tolerance.

- [ ] **Step 4: Confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
```

- [ ] **Step 5: Implement directly from named sources**

Market plane must call existing `SequenceObservationBuilder`; do **not** slice `baseline_residual_observation_v5`, because that vector contains `TrendTargets`, alpha, shadow, and baseline-relative state.

Policy-state deterministic transforms are versioned, e.g.:

```python
position_age_days = np.log1p(runtime.position_age_hours / 24.0)
pending_age_days = np.log1p(runtime.pending_order_age_hours / 24.0)
mark_index_basis_scaled = np.tanh(100.0 * runtime.mark_index_basis)
borrow_rate_scaled = np.tanh(runtime.borrow_rate)
```

Weights, masks, ratios, fill ratio, participation, execution cost rate, drawdown/gross/net/cash/risk/margin use their dimensionless maintained meanings. No fitted state normalizer.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
uv run ruff check trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
uv run mypy trade_rl/rl/universal_trade_observation.py
git add trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
git commit -m "feat: add Universal Trade RL observation"
```

---

### Task 5: Implement Train-only equal-symbol market normalization

**Files:**
- Create: `trade_rl/rl/universal_trade_normalization.py`
- Test: `tests/rl/test_universal_trade_normalization.py`
- Regression: `tests/workflows/test_universal_trade_rl_data_provenance.py`

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeNormalizationSource:
    symbol: str
    dataset: MarketDataset
    source_dataset_digest: str

@dataclass(frozen=True, slots=True)
class UniversalTradeChannelStatistics:
    timeframe: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    per_symbol_sample_counts: tuple[tuple[str, tuple[int, ...]], ...]

@dataclass(frozen=True, slots=True)
class UniversalTradeSequenceNormalizer:
    ...
    statistics_digest: str
    digest: str
    def to_payload(self) -> dict[str, object]: ...
    @classmethod
    def from_payload(cls, payload: object) -> "UniversalTradeSequenceNormalizer": ...
```

and `fit_universal_trade_sequence_normalizer(...)`.

- [ ] **Step 1: Write firewall-before-source-access test**

Pass a Development/Admission source whose dataset property raises on access. Expected: `PermissionError` from `require_normalization_scope()` before the exploding dataset is touched.

- [ ] **Step 2: Write exact equal-symbol moment oracle**

For each feature channel, compute per-symbol:

```python
mu_s = np.mean(samples_s)
q_s = np.mean(np.square(samples_s))
```

then expected:

```python
mean = np.mean([mu_s for each train symbol])
second = np.mean([q_s for each train symbol])
variance = max(second - mean * mean, 0.0)
scale = 1.0 if math.sqrt(variance) <= epsilon else math.sqrt(variance)
```

Use unequal row counts in the fixture so a row-concatenation implementation produces a different number and fails.

- [ ] **Step 3: Write unique-native-row test**

For 4h features on a 15m base clock, construct repeated aligned values and assert each unique native source row contributes once, not 16 times or once per overlapping window. Bind sampling semantic `native_unique_source_rows_v1`.

- [ ] **Step 4: Write Train-statistics vs artifact-identity test**

Two synthetic U0 generations share identical Train numeric samples but differ only in Admission source identity:

```python
assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
assert normalizer_a.digest != normalizer_b.digest
```

- [ ] **Step 5: Confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py -q
```

- [ ] **Step 6: Implement fit sequence**

Exact order:

1. canonicalize supplied symbols;
2. call `access.require_normalization_scope(symbols)`;
3. build `FEATURE_NORMALIZATION` provenance;
4. validate each `source_dataset_digest` against manifest entry before using numeric arrays;
5. validate single-symbol dataset and required feature layout;
6. derive unique native source rows per timeframe up to `knowledge_cutoff`;
7. ignore unavailable samples;
8. compute per-symbol first/second moments;
9. equal-weight symbol moments;
10. create content-addressed statistics and artifact digests.

`transform()` changes only `sequence_<tf>_values`; `available`, `staleness`, and `policy_state` pass through unchanged.

- [ ] **Step 7: Verify**

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
uv run mypy trade_rl/rl/universal_trade_normalization.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
git commit -m "feat: add Train-only universal market normalizer"
```

---

### Task 6: Implement pure wealth reward and reconciliation

**Files:**
- Create: `trade_rl/rl/universal_trade_reward.py`
- Test: `tests/rl/test_universal_trade_reward.py`

**Produces:**

```python
def universal_net_log_growth_reward(
    *, before_value: float, after_value: float, scale: float = 100.0
) -> float: ...

def reconcile_universal_trade_reward(
    *, rewards: Sequence[float], initial_value: float, final_value: float,
    scale: float = 100.0, atol: float = 1e-12
) -> None: ...
```

- [ ] **Step 1: Write telescoping and invalid-wealth tests**

```python
values = [100.0, 101.0, 99.5, 103.25]
rewards = [
    universal_net_log_growth_reward(before_value=a, after_value=b)
    for a, b in zip(values, values[1:], strict=True)
]
assert sum(rewards) / 100.0 == pytest.approx(math.log(values[-1] / values[0]))
```

Reject zero/negative/NaN/inf before/after wealth and invalid scale.

- [ ] **Step 2: Confirm RED, implement minimal formula, verify**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
```

Core implementation:

```python
return scale * math.log(after_value / before_value)
```

after explicit finite/positive validation.

- [ ] **Step 3: Checks and commit**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
uv run ruff check trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
uv run mypy trade_rl/rl/universal_trade_reward.py
git add trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
git commit -m "feat: add Universal Trade RL wealth reward"
```

---

### Task 7: Add the U1 Gym wrapper around maintained execution

**Files:**
- Create: `trade_rl/rl/universal_trade_environment.py`
- Test: `tests/rl/test_universal_trade_environment.py`

**Produces:** `UniversalTradeMarketEnv(gym.Wrapper)`.

- [ ] **Step 1: Write constructor fail-closed tests**

Reject each independently:

```text
n_symbols != 1
not TARGET_WEIGHT
target_weight_count != 1
structured_sequence_observation == false
sequence windows mismatch
signal_delay_decisions != 1
reward not pure net log growth
episode_boundary_mode != EXTERNAL_TRUNCATION
finite_horizon_observation == true
liquidate_on_end == true
initial_state_modes != ("cash",)
```

- [ ] **Step 2: Write two-decision signal-delay integration test**

Reset cash. Submit action A on decision 1; verify it becomes pending. Submit action B on decision 2; verify A is the executed/risk-projected target path while B becomes the next pending target. Then verify returned observation fields match runtime snapshot.

- [ ] **Step 3: Write reward drift guard test**

For every step:

```python
before = base_env.hybrid.portfolio_value
base_obs, base_reward, terminated, truncated, info = base_env.step(
    np.asarray([parsed.policy_requested_weight], dtype=np.float32)
)
after = base_env.hybrid.portfolio_value
u1_reward = universal_net_log_growth_reward(before_value=before, after_value=after)
if not math.isclose(base_reward, u1_reward, rel_tol=0.0, abs_tol=1e-10):
    raise RuntimeError("base environment reward drifted from U1 wealth oracle")
```

- [ ] **Step 4: Write truncation test**

At ordinary sample horizon:

```python
assert terminated is False
assert truncated is True
assert base_env.config.liquidate_on_end is False
```

and no remaining-horizon field exists in U1 policy state.

- [ ] **Step 5: Confirm RED and implement wrapper**

`reset()` only permits normal cash sampled reset; U1 V1 does not expose baseline/stress/partial-fill sampled modes through the wrapper.

`step()` must not call PreTradeRisk or execution directly; it delegates once to base env.

- [ ] **Step 6: Verify regressions**

```bash
uv run pytest tests/rl/test_universal_trade_environment.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
uv run mypy trade_rl/rl/universal_trade_environment.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
git commit -m "feat: add Universal Trade RL environment wrapper"
```

---

### Task 8: Build strict U1 contract artifact bound to U0

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u1_contract.py`
- Test: `tests/workflows/test_universal_trade_rl_u1_contract.py`

**Produces:**

```python
U1_CONTRACT_SCHEMA = "universal_trade_rl_u1_contract_v1"

@dataclass(frozen=True, slots=True)
class UniversalTradeRLU1Contract:
    universe_manifest_digest: str
    u0_identity_digest: str
    policy_contract_digest: str
    observation_schema_digest: str
    normalizer_digest: str
    normalizer_provenance_digest: str
    state_layout_digest: str
    runtime_config_digest: str
    execution_policy_digest: str
    pretrade_risk_digest: str
    portfolio_risk_digest: str
    production_status: str = "NO-GO"
    digest: str = ""
```

- [ ] **Step 1: Write U0 binding tests**

Require the supplied U0 identity to be `UNIVERSE_MATERIALIZATION` and its `universe_manifest_digest` to match the manifest. Reject missing/mismatched U0 identity and any normalizer provenance that is not Train-only `FEATURE_NORMALIZATION` for this manifest.

- [ ] **Step 2: Write runtime identity drift tests**

Changing signal delay, episode boundary, initial-state contract, execution cost, pretrade risk, portfolio risk, policy contract, state layout, or normalizer must change U1 contract digest.

`runtime_config_digest` must explicitly bind at least:

```text
decision_hours
episode_hours
signal_delay_decisions=1
episode_boundary_mode=external_truncation
finite_horizon_observation=false
liquidate_on_end=false
initial_state_modes=[cash]
```

without binding symbol text.

- [ ] **Step 3: Confirm RED and implement strict payload codec**

Add `to_payload()` / `from_payload()` with exact keys and content digest validation. Bind `production_status="NO-GO"`; reject other status.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_u1_contract.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_contract.py
git add trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_u1_contract.py
git commit -m "feat: bind Universal Trade RL U1 contract identity"
```

---

### Task 9: Materialize `u1_contract.json` and `normalizer.json` atomically

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u1_runner.py`
- Test: `tests/workflows/test_universal_trade_rl_u1_runner.py`

**Produces:**

```python
def materialize_universal_trade_rl_u1(
    *,
    contract: UniversalTradeRLU1Contract,
    normalizer: UniversalTradeSequenceNormalizer,
    output_root: str | Path,
) -> tuple[UniversalTradeRLU1Contract, UniversalTradeSequenceNormalizer]: ...
```

- [ ] **Step 1: Write atomic/idempotent RED tests**

Expected output exactly:

```text
output-root/
  normalizer.json
  u1_contract.json
```

Test canonical bytes, sorted JSON keys, one trailing newline, byte-identical rerun success, drifted existing artifact rejection, extra file rejection, partial/staging cleanup after injected failure, and no one-file published final state.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py -q
```

- [ ] **Step 3: Implement using the same publish discipline as U0 runner**

Use staging directory, write+flush+`fsync`, directory `fsync` where supported, then one `os.replace(staging, output_root)`. Existing output is success only when both canonical files are byte-identical. Never auto-repair drift.

Before publish, require `contract.normalizer_digest == normalizer.digest` and `contract.normalizer_provenance_digest == normalizer.provenance_digest`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_runner.py
git add trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
git commit -m "feat: materialize Universal Trade RL U1 artifacts"
```

---

### Task 10: Falsification, documentation, and final Quality Gate

**Files:**
- Create: `tests/rl/test_universal_trade_falsification.py`
- Modify: `docs/UNIVERSAL_TRADE_RL.md`
- Review: all U1 files above.

- [ ] **Step 1: Add future mutation falsification**

Mutate all market arrays after `t`; raw and normalized U1 Observation(t) must remain unchanged.

- [ ] **Step 2: Add symbol rename / price-unit falsification**

Symbol text-only change must not alter policy tensors. Economically equivalent price/unit rescaling must preserve dimensionless policy tensors within explicit tolerance.

- [ ] **Step 3: Add Admission poisoning/equal-symbol falsification**

Prove Development/Admission cannot be read before U0 normalization authorization; prove identical Train numeric samples across generations produce identical `statistics_digest` while U0-bound artifact digest changes.

- [ ] **Step 4: Add signal-delay/risk/partial-fill falsification**

Force distinct current submitted target, executed previous pending target, risk-projected target, and realized current weight. Assert pending-order lifecycle remains separately observable.

- [ ] **Step 5: Add flat-market cost reconciliation**

Execute a flat-price round trip through maintained accounting. Assert final wealth loss is exactly the existing execution/accounting effect and:

```python
assert sum(rewards) / 100.0 == pytest.approx(math.log(final_value / initial_value))
```

No turnover/cost reward penalty is added.

- [ ] **Step 6: Add reset-prior/state-leak falsification**

U1 wrapper must reject baseline/stress/partial-fill sampled reset and cash reset must clear prior action, signal-delay target, pending orders, position age, and realized position.

- [ ] **Step 7: Update maintained docs**

Document U1 semantics, exact four exposure stages, separate pending-order state, equal-symbol/native-row normalization, cash-only reset, pure reward, artifact materialization, Quality Gate, and U2 handoff. Keep `Production status: NO-GO` explicit.

- [ ] **Step 8: Run targeted U1/U0 suite**

```bash
uv run pytest \
  tests/rl/test_universal_trade_contract.py \
  tests/rl/test_universal_trade_action.py \
  tests/rl/test_universal_trade_runtime.py \
  tests/rl/test_universal_trade_observation.py \
  tests/rl/test_universal_trade_normalization.py \
  tests/rl/test_universal_trade_reward.py \
  tests/rl/test_universal_trade_environment.py \
  tests/rl/test_universal_trade_falsification.py \
  tests/workflows/test_universal_trade_rl_u1_contract.py \
  tests/workflows/test_universal_trade_rl_u1_runner.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  tests/workflows/test_universal_trade_rl_universe_access.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  -q
```

- [ ] **Step 9: Run static/architecture checks**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

If a repository-maintained CI command differs, run the exact CI command as an additional check; do not replace the commands above with a weaker partial check.

- [ ] **Step 10: Run full tests and build**

```bash
uv run pytest -q
uv build
```

Record exact pass/skip/fail counts and build outputs.

- [ ] **Step 11: Self-review final diff and status**

```bash
git diff b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c...HEAD
git status --short
git log --oneline --decorate -n 30
```

If the pinned U0 base has changed before implementation, this step is invalid until the branch and plan are synchronized to the new U0 base.

Inspect for manual-prior leakage, ID/raw nominal leakage, duplicated accounting, reward double counting, row-weighted normalizer, Development/Admission fit leakage, signal-delay/pending-order semantic confusion, truncation leakage, reset prior leakage, debug code, secrets, generated files, and unrelated refactors.

- [ ] **Step 12: Independent/falsification review**

Rebuild the acceptance oracle from the spec and try to construct a wrong implementation that still passes. Specifically challenge future leakage, row-count weighting, repeated native rows, zero-vs-missing, pending flat target vs no pending target, post-risk vs realized exposure, cost double-counting, and old-path compatibility. Fix substantive findings and rerun from targeted tests through full verification.

- [ ] **Step 13: Commit docs/falsification tests**

```bash
git add tests/rl/test_universal_trade_falsification.py docs/UNIVERSAL_TRADE_RL.md
git commit -m "test: close Universal Trade RL U1 quality gate"
```

- [ ] **Step 14: Draft PR + exact-final-HEAD CI evidence**

Push the branch and use a Draft PR. Record final HEAD SHA and confirm the repository's required CI, PostgreSQL Catalog, Nautilus Capability, and any applicable training/static jobs all run on that exact SHA. Earlier-SHA green runs are not final evidence.

---

## Completion Gate

Do not call U1 complete unless all are evidenced on final HEAD:

1. one-symbol U1 wrapper and fixed V1 runtime config are enforced;
2. policy tensors exclude IDs/raw nominal/manual priors/horizon fraction;
3. causal future-mutation test passes;
4. missing/availability/staleness remain distinct;
5. current policy request, signal-delay pending+active, post-risk target, realized weight, and pending-order lifecycle are separately observable;
6. scalar action meaning is static, not dynamic-risk-scaled;
7. cash-only sampled reset prevents TrendStrategy prior injection;
8. U0 normalization firewall runs before source access;
9. equal-symbol moment oracle passes with unequal row counts;
10. unique native source-row oracle prevents high-timeframe repetition weighting;
11. Admission-only generation drift keeps statistics digest equal and artifact digest different when Train numeric samples are identical;
12. pure reward telescopes to realized after-cost wealth on normal, cost, funding/borrow, partial-fill paths that are part of maintained test fixtures;
13. non-positive wealth fails closed;
14. external truncation produces no forced liquidation/terminal bonus;
15. `u1_contract.json` + `normalizer.json` materialize atomically/canonically/idempotently;
16. existing U3-U6 and Causal Alpha regression tests show no intended economic behavior change;
17. targeted/property/integration/falsification tests, Ruff, format, MyPy, import architecture, full suite, and build pass;
18. diff self-review and independent/falsification review find no unresolved substantive issue;
19. required CI is green on exact final HEAD;
20. report explicitly states U1 does not prove RL learnability, zero-shot economics, profitability, Admission, real-market fidelity, or Production readiness.

## U2 Handoff

U2 starts only after a real production-candidate U0 generation and U1 artifacts are frozen. U2 must create `UniversalTradeRLFitPurpose.RL_TRAINING` provenance from U0 Train symbols and bind the frozen U1 contract digest into the existing `UniversalTradeRLRunStage.BASE_TRAINING` model/checkpoint identity. Development remains evaluation-only and Admission remains unopened.
