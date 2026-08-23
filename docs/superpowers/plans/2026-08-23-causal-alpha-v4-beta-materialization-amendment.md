# Causal Alpha V4 Beta Materialization Plan Amendment

> **For agentic workers:** Read this file immediately after `docs/superpowers/plans/2026-08-23-causal-alpha-v4-hierarchical-teacher.md`. This amendment overrides the beta ownership/order portions of Tasks 1, 2, 4, 5, and 6 only. All other tasks and quality gates in the base plan remain authoritative.

**Spec amendment:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-beta-materialization-amendment.md`

## Reason

Plan self-review found a dependency error: Task 5 exposed `causal_beta` to the Student before Task 6 created beta for Teacher samples. Beta must instead be immutable decision-time context shared by Teacher, Student, and serving.

No empirical V4 result has been observed; this correction does not tune the scientific hypothesis.

## Corrected Task 1 additions

`trade_rl/data/v4_context.py` also owns:

```python
@dataclass(frozen=True, slots=True)
class CausalBetaConfig:
    return_horizon_hours: float = 4.0
    lookback_hours: float = 720.0
    minimum_complete_samples: int = 90
    minimum_market_variance: float = 1e-12
    minimum_beta: float = -3.0
    maximum_beta: float = 3.0


@dataclass(frozen=True, slots=True)
class CausalBetaSeries:
    decision_indices: np.ndarray
    beta: np.ndarray
    available: np.ndarray
    source_start_indices: np.ndarray
    source_end_indices: np.ndarray
    config: CausalBetaConfig
    source_digest: str
    digest: str = ""


def build_causal_beta_series(
    *,
    symbol: str,
    decision_indices: object,
    target_close: object,
    btc_close: object,
    bars_per_4h: int,
    config: CausalBetaConfig,
    target_source_digest: str,
    btc_source_digest: str,
) -> CausalBetaSeries:
    """Build one immutable causal beta series from trailing completed 4h returns."""
```

The implementation validates the explicit config rather than using a mutable/implicit default. The primitive constructs trailing, fully observed 4h log-return pairs. For BTCUSDT, every row with sufficient valid source prices is exactly beta `1.0`. For other targets, each decision uses only 4h returns whose closes are at or before that decision.

Add deterministic tests with a shortened *test-only* support contract while retaining the production defaults above:

```python
def _prices_from_log_returns(returns: np.ndarray) -> np.ndarray:
    return np.exp(np.concatenate(([0.0], np.cumsum(returns, dtype=np.float64))))


def test_causal_beta_recovers_known_two_beta():
    btc_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005, 0.02, -0.012],
        dtype=np.float64,
    )
    btc_close = _prices_from_log_returns(btc_returns)
    target_close = _prices_from_log_returns(2.0 * btc_returns)
    decision_indices = np.arange(len(btc_close), dtype=np.int64)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=24.0,
        minimum_complete_samples=3,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    result = build_causal_beta_series(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        target_close=target_close,
        btc_close=btc_close,
        bars_per_4h=1,
        config=config,
        target_source_digest="1" * 64,
        btc_source_digest="2" * 64,
    )
    assert np.count_nonzero(result.available) > 0
    np.testing.assert_allclose(result.beta[result.available], 2.0, atol=1e-12, rtol=0.0)


def test_causal_beta_future_mutation_does_not_change_prefix():
    btc_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005, 0.02, -0.012],
        dtype=np.float64,
    )
    btc_close = _prices_from_log_returns(btc_returns)
    target_close = _prices_from_log_returns(1.5 * btc_returns)
    decision_indices = np.arange(len(btc_close), dtype=np.int64)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=24.0,
        minimum_complete_samples=3,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    common = dict(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        btc_close=btc_close,
        bars_per_4h=1,
        config=config,
        target_source_digest="1" * 64,
        btc_source_digest="2" * 64,
    )
    before = build_causal_beta_series(target_close=target_close, **common)
    prefix_stop = 6
    mutated_target = target_close.copy()
    mutated_target[prefix_stop + 1 :] *= 5.0
    after = build_causal_beta_series(target_close=mutated_target, **common)
    np.testing.assert_array_equal(before.beta[:prefix_stop], after.beta[:prefix_stop])
    np.testing.assert_array_equal(
        before.available[:prefix_stop], after.available[:prefix_stop]
    )
```

## Corrected `V4TargetContext`

Replace the base-plan Task 1 definition with:

```python
@dataclass(frozen=True, slots=True)
class V4TargetContext:
    symbol: str
    local: V4ContextBlock
    global_market: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    beta_source_digest: str
    profile_name: str
    digest: str = ""

    def policy_row_digest(self, row: int) -> str:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        if not 0 <= row < len(self.beta):
            raise IndexError("row is outside V4 target context")
        return content_and_arrays_digest(
            {
                "context_digest": self.digest,
                "decision_index": int(self.local.decision_indices[row]),
                "profile_name": self.profile_name,
                "schema_version": "causal_alpha_v4_policy_context_row_v1",
                "symbol": self.symbol,
            },
            (
                ("local_values", self.local.values[row : row + 1]),
                ("local_available", self.local.available[row : row + 1]),
                (
                    "local_staleness_hours",
                    self.local.staleness_hours[row : row + 1],
                ),
                ("global_values", self.global_market.values[row : row + 1]),
                (
                    "global_available",
                    self.global_market.available[row : row + 1],
                ),
                (
                    "global_staleness_hours",
                    self.global_market.staleness_hours[row : row + 1],
                ),
                ("beta", self.beta[row : row + 1]),
                ("beta_available", self.beta_available[row : row + 1]),
            ),
        )
```

`beta` and `beta_available` are row-aligned with local/global decision indices and included in the target-context digest.

## Corrected Task 2 artifact identity

`causal_alpha_v4_target_context_artifact_v1` persists these additional NPZ members:

```text
beta
beta_available
```

and manifest field:

```text
beta_source_digest
```

Round-trip/corruption tests mutate beta bytes independently and prove strict rejection.

## Corrected Task 4 materialization

Before constructing `V4TargetContext`, the materializer computes `CausalBetaSeries` from the target USD-M perpetual history and BTCUSDT USD-M perpetual history already required by the V4 source bundle.

Required construction order per symbol:

```text
load/validate target Spot + target perp + BTC/ETH context sources
-> build local/global value context
-> build causal beta series
-> require identical decision indices
-> create V4TargetContext including beta
-> write immutable artifact
```

`CausalAlphaV4ContextManifest.context_digests` therefore commits to beta through each context digest.

## Corrected Task 5 provider signature

Replace caller-supplied beta with artifact-backed beta.

```python
class V4ContextProvider:
    local_width: int
    global_width: int
    schema_digest: str

    def resolve(
        self,
        *,
        symbol: str,
        decision_index: int,
    ) -> V4PolicyContext:
        context = self._contexts[symbol]
        row = _exact_context_row(context, decision_index)
        return V4PolicyContext(
            local_values=context.local.values[row : row + 1],
            local_available=context.local.available[row : row + 1],
            local_staleness_hours=context.local.staleness_hours[row : row + 1],
            global_values=context.global_market.values[row : row + 1],
            global_available=context.global_market.available[row : row + 1],
            global_staleness_hours=context.global_market.staleness_hours[row : row + 1],
            beta=context.beta[row : row + 1, None],
            beta_available=context.beta_available[row : row + 1, None],
            digest=context.policy_row_digest(row),
        )
```

The routed environment must never accept beta as an external observation argument.

## Corrected Task 6 title and responsibility

Rename Task 6 conceptually to:

```text
Task 6: 4h labels, persisted-beta audit, and residual reconstruction
```

Remove teacher-private beta computation from this task.

`CausalAlphaV4SymbolSamples` receives beta from `V4TargetContext`:

```python
beta = np.asarray(context.beta, dtype=np.float64)
beta_available = np.asarray(context.beta_available, dtype=np.bool_)
```

Before sample construction, add focused assertions/tests equivalent to:

```python
np.testing.assert_array_equal(
    sample_decision_indices,
    context.local.decision_indices,
)
np.testing.assert_array_equal(
    context.local.decision_indices,
    context.global_market.decision_indices,
)
assert context.beta.shape == sample_decision_indices.shape
assert context.beta_available.shape == sample_decision_indices.shape
assert np.all(context.beta[context.beta_available] >= -3.0)
assert np.all(context.beta[context.beta_available] <= 3.0)
if context.symbol == "BTCUSDT":
    np.testing.assert_array_equal(
        context.beta[context.beta_available],
        np.ones(np.count_nonzero(context.beta_available), dtype=np.float64),
    )
```

For focused correctness tests, reconstruct beta from the same synthetic raw histories and compare it with the persisted context beta. Production runtime itself must not silently replace persisted beta by recomputation.

Residual labels remain:

```python
residual = symbol_label - context.beta * btc_market_proxy_label
```

for beta-available rows.

## Corrected verification additions

Add these focused tests to Task 14:

```text
tests/data/test_v4_context.py::test_causal_beta_recovers_known_two_beta
tests/data/test_v4_context.py::test_causal_beta_future_mutation_does_not_change_prefix
tests/data/test_v4_context_artifact.py::test_context_artifact_rejects_tampered_beta
tests/rl/test_universal_v4_context.py::test_student_beta_equals_artifact_beta
tests/workflows/test_universal_causal_alpha_v4_runtime.py::test_teacher_samples_use_artifact_beta_exactly
```

The falsification review additionally asks:

```text
Can Teacher and Student obtain different beta for the same symbol/decision?
Can serving calculate beta from a different source window?
Can beta bytes change without changing artifact identity?
```

Any `yes` blocks V4 execution.
