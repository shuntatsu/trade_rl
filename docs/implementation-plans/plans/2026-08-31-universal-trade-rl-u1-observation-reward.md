# Universal Trade RL U1 Observation / Action / Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a zero-shot-first, one-symbol Universal Trade RL U1 contract that exposes symbol-independent causal observations, a scalar normalized target-exposure action, Train-only pooled market normalization, and reward exactly reconciled to after-cost portfolio wealth without changing existing U3-U6 or Causal Alpha economics.

**Architecture:** Reuse `ResidualMarketEnv` as the sole execution/accounting engine and add a U1-specific Gym wrapper rather than a second simulator. The wrapper requires the base environment to be single-symbol, `target_weight`, structured-sequence, continuing/truncation semantics, and pure net-log-growth; it projects only approved market/runtime state into `universal_trade_observation_v1`, passes a fixed-semantic scalar target exposure into the existing risk/execution path, and recomputes/validates reward from realized `BookState.portfolio_value`. Normalization fits only approved market sequence channels from U0 Train symbols; endogenous state uses dimensionless values or deterministic transforms and is not fitted from Development/Admission.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, dataclasses, canonical SHA-256 content digests, existing `MarketDataset`/`ResidualMarketEnv`/PreTradeRisk/execution/accounting, pytest, Hypothesis, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`

## Global Constraints

- Production status remains **NO-GO** throughout U1.
- U1 must not train PPO/SAC/TD3, run Admission, claim profitability, or promote Production.
- One environment instance controls exactly one concrete symbol and one capital budget; `dataset.n_symbols == 1` is mandatory.
- Policy action is one scalar in `[-1, +1]`; its semantic scale is static and never the current dynamic risk cap.
- Existing `ResidualMarketEnv` remains the only execution/accounting engine. Do not duplicate fee, spread, impact, funding, borrow, margin, partial-fill, latency, or order-state logic.
- Existing Universal U3-U6 and Causal Alpha V9/V10/V11 observation/economic behavior must remain unchanged unless a shared read-only accessor is added with regression coverage.
- U1 policy input must not expose symbol ID, dataset ID, raw absolute OHLC, raw nominal volume, raw quantity, `TrendTargets`, alpha-provider output, factor priors, shadow-book state, baseline-relative state, manually latched ownership state, or remaining-episode fraction.
- U1 allowed market FeatureKinds are limited to the groups frozen in the spec. Cross-asset/BTC-reference FeatureKinds are rejected.
- Market sequence source rows must satisfy `source_index <= decision_index`; future mutation must not change the current U1 observation.
- Missing value, availability, and staleness remain distinct channels.
- U0 `UniversalTradeRLUniverseAccess.require_normalization_scope()` and `UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION` are the authority for normalization fit scope.
- Development and Admission data must never contribute values to fitted market means/scales. If the full U0 universe identity changes while Train sample content is unchanged, statistics may remain equal but the bound artifact identity must change.
- Reward is exactly `100 * log(W_after / W_before)` over the realized hybrid accounting book. No extra turnover, cost, drawdown, baseline, projection, terminal, or margin shaping is added.
- `W_before <= 0`, `W_after <= 0`, NaN, or infinity is fail-closed; do not silently epsilon-clip a broken accounting transition inside U1 reward.
- Episode horizon is a training sample boundary. Time-limit completion is truncation, not a market terminal state, and remaining-horizon fraction is not visible to the policy.
- U1 implementation must preserve the U0 branch ancestry and must not merge to `main` without explicit user permission.

---

## File Structure

### New focused modules

- `trade_rl/rl/universal_trade_contract.py` — U1 schema constants, frozen FeatureKind allowlist, sequence-window/model contract, content digest.
- `trade_rl/rl/universal_trade_action.py` — strict scalar action parsing and fixed semantic mapping to policy-requested target weight.
- `trade_rl/rl/universal_trade_runtime.py` — read-only U1 runtime snapshot types; no simulator logic.
- `trade_rl/rl/universal_trade_observation.py` — build `universal_trade_observation_v1` from causal sequence observations plus the runtime snapshot.
- `trade_rl/rl/universal_trade_normalization.py` — U0 Train-only pooled market-sequence normalization artifact and statistics digest.
- `trade_rl/rl/universal_trade_reward.py` — pure net-log-growth calculation and telescoping reconciliation helper.
- `trade_rl/rl/universal_trade_environment.py` — `UniversalTradeMarketEnv` Gym wrapper around an already-configured `ResidualMarketEnv`.
- `trade_rl/workflows/universal_trade_rl_u1_identity.py` — immutable U1 environment identity binding universe, policy contract, normalizer, execution/risk, and reward semantics.

### Existing modules modified only where needed

- `trade_rl/rl/environment.py` — add one read-only `universal_trade_runtime_snapshot()` accessor; no step/economic behavior change.
- `docs/UNIVERSAL_TRADE_RL.md` — document U1 contract and explicit U2 gate.
- `tests/test_architecture_contract.py` — add new module/script allowlist only if architecture checks require it.

### New tests

- `tests/rl/test_universal_trade_contract.py`
- `tests/rl/test_universal_trade_action.py`
- `tests/rl/test_universal_trade_runtime.py`
- `tests/rl/test_universal_trade_observation.py`
- `tests/rl/test_universal_trade_normalization.py`
- `tests/rl/test_universal_trade_reward.py`
- `tests/rl/test_universal_trade_environment.py`
- `tests/rl/test_universal_trade_falsification.py`
- `tests/workflows/test_universal_trade_rl_u1_identity.py`

---

### Task 1: Freeze the U1 policy contract and feature surface

**Files:**
- Create: `trade_rl/rl/universal_trade_contract.py`
- Test: `tests/rl/test_universal_trade_contract.py`

**Interfaces:**
- Consumes: `trade_rl.data.contracts.FeatureKind`, `FeatureSpec`; `trade_rl.artifacts.hashing.content_digest`.
- Produces:
  - `UNIVERSAL_TRADE_OBSERVATION_SCHEMA = "universal_trade_observation_v1"`
  - `UNIVERSAL_TRADE_ACTION_SCHEMA = "normalized_target_exposure_v1"`
  - `UNIVERSAL_TRADE_REWARD_SCHEMA = "universal_net_log_growth_reward_v1"`
  - `UNIVERSAL_TRADE_SEQUENCE_WINDOWS = (("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60))`
  - `UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS: frozenset[FeatureKind]`
  - `UniversalTradePolicyContract(feature_specs: tuple[FeatureSpec, ...], policy_weight_scale: float = 1.0, reward_scale: float = 100.0, ...)`
  - `UniversalTradePolicyContract.digest: str`

- [ ] **Step 1: Write failing tests for exact schemas, allowlist, and forbidden cross-asset features**

```python
from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS,
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)


def test_u1_contract_uses_frozen_sequence_windows() -> None:
    assert UNIVERSAL_TRADE_SEQUENCE_WINDOWS == (
        ("15m", 96),
        ("1h", 168),
        ("4h", 120),
        ("1d", 60),
    )


def test_u1_contract_rejects_cross_asset_feature() -> None:
    spec = FeatureSpec(name="relative_btc", kind=FeatureKind.RELATIVE_RETURN_TO_BTC)
    with pytest.raises(ValueError, match="Universal Trade RL U1 feature"):
        UniversalTradePolicyContract(feature_specs=(spec,))


def test_u1_contract_digest_changes_when_feature_order_changes() -> None:
    a = FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    b = FeatureSpec(name="vol", kind=FeatureKind.REALIZED_VOLATILITY, lookback=16)
    assert UniversalTradePolicyContract(feature_specs=(a, b)).digest != UniversalTradePolicyContract(feature_specs=(b, a)).digest
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
```

Expected: collection/import failure because `trade_rl.rl.universal_trade_contract` does not exist.

- [ ] **Step 3: Implement the exact allowlist and immutable digest-bound contract**

Use this allowlist and do not broaden it in U1:

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

The contract must reject empty feature specs, duplicate names, unsupported timeframes, non-finite/non-positive scales, and any FeatureKind outside the allowlist. Digest payload must include ordered `FeatureSpec.canonical_payload()`, windows, schema names, `policy_weight_scale`, `reward_scale`, and a policy-state schema version.

- [ ] **Step 4: Run targeted tests and static checks**

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
uv run ruff check trade_rl/rl/universal_trade_contract.py tests/rl/test_universal_trade_contract.py
uv run mypy trade_rl/rl/universal_trade_contract.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/universal_trade_contract.py tests/rl/test_universal_trade_contract.py
git commit -m "feat: define Universal Trade RL U1 policy contract"
```

---

### Task 2: Implement fixed-semantic scalar target exposure

**Files:**
- Create: `trade_rl/rl/universal_trade_action.py`
- Test: `tests/rl/test_universal_trade_action.py`

**Interfaces:**
- Consumes: `UniversalTradePolicyContract.policy_weight_scale`.
- Produces:
  - `NormalizedTargetExposureAction(normalized: float, policy_requested_weight: float)`
  - `parse_normalized_target_exposure(value: np.ndarray, *, policy_weight_scale: float) -> NormalizedTargetExposureAction`

- [ ] **Step 1: Write failing unit/property tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(-1.0, -1.0), (-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
)
def test_action_maps_linearly_without_dynamic_risk_semantics(raw: float, expected: float) -> None:
    action = parse_normalized_target_exposure(
        np.asarray([raw], dtype=np.float32),
        policy_weight_scale=1.0,
    )
    assert action.policy_requested_weight == pytest.approx(expected)


def test_action_rejects_out_of_range_instead_of_hidden_clip() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        parse_normalized_target_exposure(np.asarray([1.01]), policy_weight_scale=1.0)
```

Add tests for wrong shape, NaN/inf, zero/negative/greater-than-one scale, and deterministic `float32`/`float64` equivalence.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
```

- [ ] **Step 3: Implement strict parsing**

```python
@dataclass(frozen=True, slots=True)
class NormalizedTargetExposureAction:
    normalized: float
    policy_requested_weight: float


def parse_normalized_target_exposure(
    value: np.ndarray,
    *,
    policy_weight_scale: float,
) -> NormalizedTargetExposureAction:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (1,) or not np.isfinite(vector).all():
        raise ValueError("Universal Trade RL action must be one finite scalar")
    if not np.isfinite(policy_weight_scale) or not 0.0 < policy_weight_scale <= 1.0:
        raise ValueError("policy_weight_scale must be within (0, 1]")
    normalized = float(vector[0])
    if not -1.0 <= normalized <= 1.0:
        raise ValueError("Universal Trade RL action must be within [-1, 1]")
    return NormalizedTargetExposureAction(
        normalized=normalized,
        policy_requested_weight=normalized * policy_weight_scale,
    )
```

- [ ] **Step 4: Run tests/static checks**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
uv run ruff check trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
uv run mypy trade_rl/rl/universal_trade_action.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
git commit -m "feat: add normalized target exposure action"
```

---

### Task 3: Expose one read-only runtime snapshot from `ResidualMarketEnv`

**Files:**
- Create: `trade_rl/rl/universal_trade_runtime.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_universal_trade_runtime.py`
- Regression test: existing `tests/rl/test_environment*.py`

**Interfaces:**
- Produces `UniversalTradeRuntimeSnapshot` with one-symbol fields:
  - `policy_requested_weight`
  - `risk_projected_weight`
  - `pending_target_weight`
  - `current_weight`
  - `previous_action`
  - `fill_ratio`
  - `unfilled_turnover_ratio`
  - `participation_ratio`
  - `execution_cost_rate`
  - `position_age_hours`
  - pending-order fields already represented by `PendingOrderObservationState`
  - `asset_active`, `tradable`, `borrow_available`, `borrow_rate`, `mark_index_basis`
  - `current_drawdown`, `current_gross_exposure`, `current_net_exposure`, `cash_weight`, `risk_scale`, `margin_utilization`
- Adds `ResidualMarketEnv.universal_trade_runtime_snapshot() -> UniversalTradeRuntimeSnapshot` as a read-only accessor only.

- [ ] **Step 1: Write a failing runtime-state test that distinguishes all four exposure stages**

Construct a single-symbol `ResidualMarketEnv` in `TARGET_WEIGHT` mode with a risk cap and partial fill such that:

```text
policy requested = +0.80
risk projected   = +0.35
realized current = +0.22
```

Assert after the step:

```python
snapshot = env.universal_trade_runtime_snapshot()
assert snapshot.policy_requested_weight == pytest.approx(0.80)
assert snapshot.risk_projected_weight == pytest.approx(0.35)
assert snapshot.current_weight == pytest.approx(0.22)
assert snapshot.pending_target_weight != snapshot.current_weight
```

The test oracle is semantic, not just field existence: the fixture must force risk projection and incomplete fill so the values cannot accidentally be aliases.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
```

- [ ] **Step 3: Implement the immutable snapshot type and accessor without changing step logic**

`UniversalTradeRuntimeSnapshot` validates all scalars are finite, weights are within the environment contract, masks are in `[0,1]`, and ratios/cost/ages are non-negative where applicable.

In `ResidualMarketEnv.universal_trade_runtime_snapshot()`:

- require `dataset.n_symbols == 1`;
- derive `policy_requested_weight` from the last one-dimensional target-weight action accepted by the environment;
- define `risk_projected_weight` as the target handed from the risk projector into the execution coordinator, not as realized `BookState.weights`;
- derive `pending_target_weight` from the current hybrid pending target;
- derive `current_weight` from `hybrid.weights[0]`;
- reuse the maintained `ObservationExecutionState` and `PendingOrderObservationState` sources rather than recomputing order lifecycle differently;
- return copies/read-only values so callers cannot mutate environment state.

If the current environment does not persist the post-risk/pre-execution target as a distinct field, add exactly one `_last_risk_projected_target` state field at the existing decision→execution handoff and update it there. Do not infer it later from realized weights.

- [ ] **Step 4: Run targeted and base-environment regressions**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment.py tests/rl/test_environment_reduce_only_integration.py tests/rl/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/test_universal_trade_runtime.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py
```

Expected: U1 snapshot test passes and pre-existing environment behavior stays green.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/test_universal_trade_runtime.py
git commit -m "feat: expose Universal Trade RL runtime snapshot"
```

---

### Task 4: Build the strategy-prior-free U1 observation

**Files:**
- Create: `trade_rl/rl/universal_trade_observation.py`
- Test: `tests/rl/test_universal_trade_observation.py`

**Interfaces:**
- Consumes: `SequenceObservationBuilder`, `UniversalTradePolicyContract`, `UniversalTradeRuntimeSnapshot`.
- Produces:
  - `UNIVERSAL_TRADE_POLICY_STATE_NAMES: tuple[str, ...]`
  - `UniversalTradeObservationBuilder(contract: UniversalTradePolicyContract)`
  - `build(dataset: MarketDataset, *, index: int, runtime: UniversalTradeRuntimeSnapshot, sequence_normalizer: UniversalTradeSequenceNormalizer | None = None) -> dict[str, np.ndarray]`
  - `observation_space(dataset: MarketDataset) -> gym.spaces.Dict`
  - `schema_digest(dataset: MarketDataset) -> str` that excludes symbol text from tensor semantics but binds ordered feature names/layout.

- [ ] **Step 1: Write failing tests for the exact observation keys and forbidden priors**

Expected U1 observation keys:

```python
{
    "sequence_15m_values",
    "sequence_15m_available",
    "sequence_15m_staleness",
    "sequence_1h_values",
    "sequence_1h_available",
    "sequence_1h_staleness",
    "sequence_4h_values",
    "sequence_4h_available",
    "sequence_4h_staleness",
    "sequence_1d_values",
    "sequence_1d_available",
    "sequence_1d_staleness",
    "policy_state",
}
```

Assert `policy_state` names are exactly the ordered U1 endogenous fields and contain no `trend`, `alpha`, `shadow`, `baseline`, `remaining`, `symbol`, or `dataset` field.

- [ ] **Step 2: Add the causal future-mutation test before implementation**

Create `dataset_a` and `dataset_b` identical through decision index `t` and mutate every market value after `t` in `dataset_b`. Build U1 observations at `t` and assert every tensor is exactly equal:

```python
for key in obs_a:
    np.testing.assert_array_equal(obs_a[key], obs_b[key])
```

- [ ] **Step 3: Add symbol-rename and price-unit falsification tests**

For symbol rename, create equivalent datasets whose only identity difference is symbol text; U1 tensor values must be equal.

For price-unit scaling, scale all raw OHLC/mark/index prices and compatible execution-rule units by a constant while preserving dimensionless feature paths. The U1 observation must remain equal within floating tolerance. This test must fail if raw absolute price or raw quantity is accidentally added.

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
```

- [ ] **Step 5: Implement the builder directly from maintained causal sequence outputs plus the explicit runtime snapshot**

Do not parse or slice `baseline_residual_observation_v5`; that would couple U1 to hidden field offsets and could leak strategy priors. The builder must call `SequenceObservationBuilder` for market history and construct `policy_state` only from named `UniversalTradeRuntimeSnapshot` fields.

Use deterministic fixed transforms for endogenous state where necessary, for example:

```python
position_age_days = np.log1p(runtime.position_age_hours / 24.0)
pending_age_days = np.log1p(runtime.pending_order_age_hours / 24.0)
mark_index_basis = np.tanh(100.0 * runtime.mark_index_basis)
borrow_rate = np.tanh(runtime.borrow_rate)
```

Bind the exact transform names/version in the observation schema digest. Do not fit these endogenous transforms on Development/Admission data.

- [ ] **Step 6: Run targeted tests/static checks**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
uv run ruff check trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
uv run mypy trade_rl/rl/universal_trade_observation.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
git commit -m "feat: add Universal Trade RL observation contract"
```

---

### Task 5: Fit one Train-only pooled market-sequence normalizer

**Files:**
- Create: `trade_rl/rl/universal_trade_normalization.py`
- Test: `tests/rl/test_universal_trade_normalization.py`
- Reuse tests: `tests/workflows/test_universal_trade_rl_data_provenance.py`

**Interfaces:**
- Consumes:
  - `UniversalTradeRLUniverseManifest`
  - `UniversalTradeRLUniverseAccess`
  - `build_universal_trade_rl_fit_provenance(... purpose=FEATURE_NORMALIZATION ...)`
  - ordered `Mapping[str, MarketDataset]` for Train symbols
  - `UniversalTradePolicyContract`
- Produces:
  - `UniversalTradeSequenceStatistics(timeframe, feature_names, mean, scale, sample_count)`
  - `UniversalTradeSequenceNormalizer(...)`
  - `UniversalTradeSequenceNormalizer.statistics_digest` — only fitted numeric statistics/schema
  - `UniversalTradeSequenceNormalizer.digest` — statistics plus U0 universe/provenance/contract identity
  - `fit_universal_trade_sequence_normalizer(...) -> UniversalTradeSequenceNormalizer`

- [ ] **Step 1: Write the Train-only scope failure first**

```python
def test_normalizer_rejects_development_before_reading_dataset() -> None:
    with pytest.raises(PermissionError, match="normalization scope is Train-only"):
        fit_universal_trade_sequence_normalizer(
            manifest=manifest,
            access=development_access,
            datasets={"BTCUSDT": train_ds, "LINKUSDT": ExplodingDataset()},
            contract=contract,
            knowledge_cutoff=cutoff,
        )
```

`ExplodingDataset` raises if any array/property is touched. The test proves the U0 firewall is checked before source data lookup.

- [ ] **Step 2: Add pooling/invariance tests**

Test that two Train symbols contribute to each `(timeframe, feature)` mean/scale, missing/unavailable samples are excluded, zero-variance channels get scale `1.0`, and transform clips to `[-10, 10]`.

Add the identity distinction test:

```python
assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
assert normalizer_a.digest != normalizer_b.digest
```

where only the Admission source identity changes while Train source arrays, policy contract, and knowledge cutoff stay identical.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py -q
```

- [ ] **Step 4: Implement pooled fitting**

Algorithm per timeframe/feature:

1. Call `access.require_normalization_scope(tuple(sorted(datasets)))` before touching `datasets[symbol]`.
2. Build U0 provenance with `UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION`.
3. Require each dataset is single-symbol and manifest dataset identity matches the supplied dataset artifact identity used by the repository.
4. Use only samples with causal source timestamp `<= knowledge_cutoff` and feature availability true.
5. Concatenate across Train symbols; compute population mean/std (`ddof=0`); replace `std <= 1e-8` by `1.0`.
6. Store ordered feature names and sample counts; reject missing required features or zero available samples.
7. `transform()` normalizes only `sequence_<tf>_values`; availability/staleness and `policy_state` are passed through unchanged after the builder's deterministic transforms.

- [ ] **Step 5: Run targeted tests plus U0 provenance regression**

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
uv run mypy trade_rl/rl/universal_trade_normalization.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
git commit -m "feat: add Train-only Universal Trade RL normalization"
```

---

### Task 6: Implement the pure after-cost reward oracle

**Files:**
- Create: `trade_rl/rl/universal_trade_reward.py`
- Test: `tests/rl/test_universal_trade_reward.py`

**Interfaces:**
- Produces:
  - `universal_net_log_growth_reward(*, before_value: float, after_value: float, scale: float = 100.0) -> float`
  - `reconcile_universal_trade_reward(*, rewards: Sequence[float], initial_value: float, final_value: float, scale: float = 100.0, atol: float = 1e-12) -> None`

- [ ] **Step 1: Write the telescoping identity test**

```python
def test_reward_telescopes_to_final_wealth() -> None:
    values = [100.0, 101.0, 99.5, 103.25]
    rewards = [
        universal_net_log_growth_reward(before_value=a, after_value=b)
        for a, b in zip(values, values[1:], strict=True)
    ]
    assert sum(rewards) / 100.0 == pytest.approx(math.log(values[-1] / values[0]))
    reconcile_universal_trade_reward(
        rewards=rewards,
        initial_value=values[0],
        final_value=values[-1],
    )
```

- [ ] **Step 2: Add fail-closed tests**

Reject `0`, negative, NaN, infinity for before/after wealth and non-positive/non-finite scale. No epsilon substitution is allowed.

- [ ] **Step 3: Verify RED and implement**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
```

Implementation core:

```python
def universal_net_log_growth_reward(*, before_value: float, after_value: float, scale: float = 100.0) -> float:
    for name, value in (("before_value", before_value), ("after_value", after_value)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reward scale must be finite and positive")
    return scale * math.log(after_value / before_value)
```

- [ ] **Step 4: Run checks and commit**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
uv run ruff check trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
uv run mypy trade_rl/rl/universal_trade_reward.py
git add trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
git commit -m "feat: add Universal Trade RL wealth reward oracle"
```

---

### Task 7: Wrap the maintained environment instead of duplicating the simulator

**Files:**
- Create: `trade_rl/rl/universal_trade_environment.py`
- Test: `tests/rl/test_universal_trade_environment.py`
- Regression: existing `tests/rl/test_environment*.py`, `tests/learning/test_rollout_execution_lifecycle.py`

**Interfaces:**
- Consumes:
  - configured `ResidualMarketEnv`
  - `UniversalTradePolicyContract`
  - optional frozen `UniversalTradeSequenceNormalizer`
  - `UniversalTradeObservationBuilder`
  - `parse_normalized_target_exposure`
  - `universal_net_log_growth_reward`
- Produces:
  - `UniversalTradeMarketEnv(gym.Wrapper)`
  - scalar `action_space = Box(-1, 1, shape=(1,), dtype=float32)`
  - U1 `observation_space`

- [ ] **Step 1: Write constructor rejection tests**

Reject a base environment when any of these are false:

```text
dataset.n_symbols == 1
action_spec.mode == TARGET_WEIGHT
action_spec.target_weight_count == 1
config.structured_sequence_observation is True
reward_tracker.config.is_pure_net_log_growth() is True
finite_horizon_observation is False
time-limit boundary is truncation/continuing semantics
```

Also reject policy-contract feature names/layout that do not match the dataset.

- [ ] **Step 2: Write reset/step behavior tests before code**

```python
obs, info = wrapper.reset(seed=7)
assert wrapper.observation_space.contains(obs)

before = wrapper.base_env.hybrid.portfolio_value
obs, reward, terminated, truncated, info = wrapper.step(np.asarray([0.5], dtype=np.float32))
after = wrapper.base_env.hybrid.portfolio_value
assert reward == pytest.approx(100.0 * math.log(after / before))
assert wrapper.observation_space.contains(obs)
```

Assert the wrapper passes a one-element target-weight vector into the base env and never creates a portfolio action over multiple symbols.

- [ ] **Step 3: Write the no-hidden-shaping drift test**

The base env must be configured pure net-log-growth. After every step compare the base reward and U1 recomputed reward; fail closed if they diverge beyond `1e-10`:

```python
if not math.isclose(base_reward, reward, rel_tol=0.0, abs_tol=1e-10):
    raise RuntimeError("base environment reward drifted from U1 wealth oracle")
```

This turns future accidental shaping changes into a contract failure.

- [ ] **Step 4: Write continuing-boundary test**

Force the configured episode sample boundary without insolvency/emergency termination. Expected:

```python
assert terminated is False
assert truncated is True
assert "remaining_episode_fraction" not in wrapper.policy_state_names
```

- [ ] **Step 5: Verify RED and implement the wrapper**

`step()` sequence:

```text
parse scalar normalized action
 -> fixed policy_requested_weight
 -> record W_before from base_env.hybrid.portfolio_value
 -> base_env.step([policy_requested_weight])
 -> record W_after
 -> compute U1 reward
 -> require base reward reconciliation
 -> read base_env.universal_trade_runtime_snapshot()
 -> build U1 observation
 -> preserve base terminated/truncated/info
```

Do not intercept or reimplement risk/execution/accounting.

- [ ] **Step 6: Run integration/regression checks**

```bash
uv run pytest tests/rl/test_universal_trade_environment.py -q
uv run pytest tests/rl/test_environment.py tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
uv run mypy trade_rl/rl/universal_trade_environment.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
git commit -m "feat: add Universal Trade RL environment wrapper"
```

---

### Task 8: Bind U1 environment identity to U0 without opening Admission

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u1_identity.py`
- Test: `tests/workflows/test_universal_trade_rl_u1_identity.py`
- Reuse: `trade_rl/workflows/universal_trade_rl_run_identity.py`

**Interfaces:**
- Produces `UniversalTradeRLU1EnvironmentIdentity` with:
  - `universe_manifest_digest`
  - `policy_contract_digest`
  - `normalizer_digest`
  - `normalizer_provenance_digest`
  - `observation_schema_digest`
  - `action_schema`
  - `reward_schema`
  - `execution_policy_digest`
  - `pretrade_risk_digest` or canonical risk config digest
  - `portfolio_risk_digest` or canonical risk config digest
  - `schema_version = "universal_trade_rl_u1_environment_identity_v1"`
  - `digest`

- [ ] **Step 1: Write strict identity tests**

Test that changing any of contract, normalizer, execution policy, risk config, or universe generation changes the identity digest. Tampered supplied digest must be rejected.

- [ ] **Step 2: Prove Development does not authorize Admission**

Construct identity using `UniversalTradeRLUniverseAccess.for_phase(... DEVELOPMENT)`. It must accept only the already-frozen Train normalizer provenance and must not accept or contain an Admission authorization digest.

- [ ] **Step 3: Verify RED and implement canonical identity**

The identity module validates:

```python
require_universal_trade_rl_train_only_provenance(normalizer.provenance, manifest=manifest)
if normalizer.provenance.purpose is not UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION:
    raise ValueError("U1 normalizer provenance purpose mismatch")
```

Do not add a new U0 `UniversalTradeRLRunStage` for U1. U1 environment identity is a model/environment input artifact; U2 later binds its digest into the existing `BASE_TRAINING` run identity as `model_config_digest` or as part of the frozen model-config digest.

- [ ] **Step 4: Run tests/checks and commit**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_identity.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_identity.py tests/workflows/test_universal_trade_rl_u1_identity.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_identity.py
git add trade_rl/workflows/universal_trade_rl_u1_identity.py tests/workflows/test_universal_trade_rl_u1_identity.py
git commit -m "feat: bind Universal Trade RL U1 environment identity"
```

---

### Task 9: Add falsification tests that try to break the U1 contract

**Files:**
- Create: `tests/rl/test_universal_trade_falsification.py`
- May extend: `tests/rl/test_universal_trade_environment.py`

**Interfaces:**
- No new production API unless a test exposes a missing observable contract.

- [ ] **Step 1: Future-leakage falsification**

Mutate all bars after `t`; U1 observation at `t` must remain identical.

- [ ] **Step 2: Symbol-ID falsification**

Rename the symbol while keeping economically equivalent data; all policy tensors must remain equal.

- [ ] **Step 3: Price-scale falsification**

Multiply price units and compatible contract/execution units by a positive constant; U1 policy tensors and normalized action semantics must remain economically equivalent.

- [ ] **Step 4: Exposure-stage falsification**

Force:

```text
policy_requested_weight != risk_projected_weight != pending_target_weight != current_weight
```

and assert all four fields retain their distinct oracle values after one step and after a retry/partial fill.

- [ ] **Step 5: Cost double-counting falsification**

Use a flat-price fixture with `0 -> long -> flat`. Let the existing accounting engine calculate fee/spread/impact loss. Assert:

```python
assert final_value < initial_value
assert sum(rewards) / 100.0 == pytest.approx(math.log(final_value / initial_value))
```

Do not separately subtract turnover/cost in reward.

- [ ] **Step 6: Admission poisoning falsification**

Change only Admission dataset content/identity while keeping Train arrays identical. Refit under the new universe generation:

```python
assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
assert normalizer_a.digest != normalizer_b.digest
```

Attempt to include Admission in normalization input and assert `PermissionError` before dataset access.

- [ ] **Step 7: Episode restart/state-leak falsification**

Reset an environment after a trajectory with pending order/position state. Assert no previous episode pending order, previous action, risk projection, or position age leaks into the new episode unless the selected existing initial-state mode explicitly defines a carried baseline state. Cash-mode reset must be clean.

- [ ] **Step 8: Run falsification suite**

```bash
uv run pytest tests/rl/test_universal_trade_falsification.py -q
```

Expected: PASS without weakening assertions or skipping cases.

- [ ] **Step 9: Commit**

```bash
git add tests/rl/test_universal_trade_falsification.py tests/rl/test_universal_trade_environment.py
git commit -m "test: falsify Universal Trade RL U1 boundaries"
```

---

### Task 10: Document U1 and run the full quality gate

**Files:**
- Modify: `docs/UNIVERSAL_TRADE_RL.md`
- Modify only if required: `tests/test_architecture_contract.py`
- Review all U1 production/test files from Tasks 1-9.

**Interfaces:**
- Documentation must state U1 is NO-GO and U2 is blocked until a real production-candidate U0 universe is materialized/frozen.

- [ ] **Step 1: Update maintained documentation**

Add sections covering:

```text
U0 prerequisite gate
U1 observation schema
forbidden strategy priors
scalar action semantics
policy-requested vs risk-projected vs pending vs realized weight
Train-only pooled normalizer
pure after-cost reward equation
continuing/truncation episode semantics
U1 Quality Gate
U2 handoff
Production status: NO-GO
```

- [ ] **Step 2: Run focused U1 + U0 regression tests**

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
  tests/workflows/test_universal_trade_rl_u1_identity.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  tests/workflows/test_universal_trade_rl_universe_access.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  -q
```

- [ ] **Step 3: Run static/architecture checks**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

If repository conventions use a narrower maintained MyPy command, execute that exact documented CI command in addition to the affected-file command; do not claim full MyPy success from a partial invocation.

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest -q
```

Record exact `passed / skipped / failed` counts and compare with the U0 exact base. A green full suite proves regression coverage only; it does not prove profitability or zero-shot transfer quality.

- [ ] **Step 5: Build package**

```bash
uv build
```

Expected: successful sdist/wheel build with no untracked generated artifacts intended for commit.

- [ ] **Step 6: Self-review the complete diff**

Explicitly inspect:

```bash
git diff <U0_BASE_SHA>...HEAD
git status --short
git log --oneline --decorate -n 20
```

Review for:

- accidental U3-U6/Causal Alpha behavior changes;
- strategy priors leaking into U1 tensors;
- symbol/dataset identifiers in policy observations;
- private-state mutation through the runtime snapshot;
- duplicated accounting/risk logic in the wrapper;
- reward cost double-counting;
- Development/Admission normalization leakage;
- episode-end information leakage;
- stale debug code, generated files, temporary workflows, or secrets.

- [ ] **Step 7: Independent/falsification review**

Reconstruct the Acceptance Criteria from the spec without relying on implementation conclusions. Review the actual diff and assertions and answer:

```text
Can a wrong implementation still pass these tests?
Can future data affect obs(t)?
Can symbol identity be inferred from an explicit ID/raw price scale?
Can Admission values enter fitted statistics?
Can policy-requested and realized exposure collapse into one field?
Can reward differ from realized after-cost wealth?
Can time-limit termination teach end-of-episode behavior?
Did any existing economic path change?
```

Fix any substantive issue found and rerun from the smallest affected test through the required full checks.

- [ ] **Step 8: Commit documentation/final test adjustments**

```bash
git add docs/UNIVERSAL_TRADE_RL.md tests/test_architecture_contract.py
git commit -m "docs: document Universal Trade RL U1 contract"
```

Only stage `tests/test_architecture_contract.py` if it actually changed.

- [ ] **Step 9: CI gate on the exact final HEAD**

Push the branch and use a Draft PR. Required checks must execute against the same final HEAD that is reported. Confirm at minimum the repository's normal CI, PostgreSQL Catalog, Nautilus Capability, and any U1-specific workflow added by repository convention. Do not treat CI from an earlier SHA as final evidence.

Record:

```text
final HEAD SHA
base SHA
workflow/run IDs
conclusion per required check
full-suite counts
coverage evidence if CI publishes it
unverified external/economic claims
remaining risks
```

Do not mark the PR Ready solely because tests are green; the complete U1 Quality Gate below must be satisfied.

---

## U1 Quality Gate

U1 may be described as implementation-complete only when all of the following are evidenced on the final HEAD:

1. `dataset.n_symbols == 1` is enforced for the U1 wrapper.
2. U1 tensors contain no symbol/dataset ID, raw absolute OHLC, raw nominal volume/quantity, manual trend/alpha/factor/shadow/baseline prior, or remaining-horizon fraction.
3. Every sequence source row is causal and future mutation cannot alter the current observation.
4. Missing value, availability, and staleness are distinct.
5. `policy_requested_weight`, `risk_projected_weight`, `pending_target_weight`, and `current_weight` are independently observable and proven distinct under a forced projection/partial-fill test.
6. Scalar action meaning is static and independent of current risk cap.
7. Normalizer scope is checked by U0 before any supplied dataset is read.
8. Only Train samples contribute to fitted market statistics.
9. Admission-only generation drift changes the identity-bound normalizer digest but not `statistics_digest` when Train samples are identical.
10. Reward telescopes exactly to realized after-cost wealth and does not double-count costs.
11. Broken/non-positive wealth fails closed rather than being hidden by epsilon clipping in the U1 reward layer.
12. Time-limit boundary is truncation/continuing semantics and horizon fraction is absent from policy state.
13. Existing U3-U6 and Causal Alpha regression suites pass with no intended economic behavior change.
14. Ruff, format check, required MyPy/static analysis, import architecture, full test suite, and package build pass.
15. Falsification review finds no known implementation path that violates the spec while satisfying current assertions.
16. CI/required checks are green on the exact final HEAD.
17. Remaining limitations explicitly state that U1 does **not** prove RL learnability, zero-shot economics, profitability, Admission performance, real-market execution fidelity, or Production readiness.

## U2 Handoff

U2 Base RL training is blocked until both conditions are true:

1. U1 Quality Gate above is satisfied and its U1 environment identity is frozen.
2. A real production-candidate U0 source catalog/role config—not illustrative example data—has been materialized and its `universe.json` / `identity.json` digests are frozen.

U2 must then create `UniversalTradeRLFitPurpose.RL_TRAINING` provenance from U0 Train symbols and bind the frozen U1 environment/model config digest into `UniversalTradeRLRunStage.BASE_TRAINING`. Development remains evaluation-only and Admission remains unopened.
