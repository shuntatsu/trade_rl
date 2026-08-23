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
    config: CausalBetaConfig = CausalBetaConfig(),
    target_source_digest: str,
    btc_source_digest: str,
) -> CausalBetaSeries: ...
```

The primitive constructs trailing, fully observed 4h log-return pairs. For BTCUSDT, every row with sufficient valid source prices is exactly beta `1.0`. For other targets, each decision uses only 4h returns whose closes are at or before that decision.

Add tests:

```python
def test_causal_beta_recovers_known_two_beta():
    # Build BTC 4h returns and target returns exactly 2x BTC.
    result = build_causal_beta_series(...)
    assert np.allclose(result.beta[result.available], 2.0)


def test_causal_beta_future_mutation_does_not_change_prefix():
    before = build_causal_beta_series(...)
    mutated_target = target_close.copy()
    mutated_target[future_start:] *= 5.0
    after = build_causal_beta_series(target_close=mutated_target, ...)
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

Round-trip/corruption tests must mutate beta bytes independently and prove strict rejection.

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
            digest=_policy_context_digest(...),
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
beta=np.asarray(context.beta, dtype=np.float64)
beta_available=np.asarray(context.beta_available, dtype=np.bool_)
```

Before sample construction, independently verify alignment and identity:

```python
np.testing-equivalent requirements:
sample decision index == context decision index
context beta length == context row count
available beta inside [-3, 3]
BTC available beta == 1.0
```

For focused correctness tests, reconstruct beta from the same synthetic raw histories and compare it with the persisted context beta. Production runtime itself must not silently replace a persisted beta by recomputation.

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
