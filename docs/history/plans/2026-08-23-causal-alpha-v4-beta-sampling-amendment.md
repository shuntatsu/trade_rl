# Causal Alpha V4 Beta Sampling Plan Amendment

> **For agentic workers:** This follows the base V4 plan and both prior V4 amendments. It overrides only beta return-sample construction.

**Spec amendment:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-beta-sampling-amendment.md`

## Corrected beta primitive

`build_causal_beta_series` continues to receive aligned decision indices, target/BTC closes, and `bars_per_4h`, but it must construct non-overlapping 4h samples.

Use this exact rule:

```python
first_decision = int(decision_indices[0])
relative = decision_indices - first_decision
sample_row_mask = (relative >= bars_per_4h) & (relative % bars_per_4h == 0)
sample_rows = np.flatnonzero(sample_row_mask)
```

For each `sample_row`, the return start row is `sample_row - bars_per_4h`. The return is valid only when every source row from start through end is available for both target and BTC.

The beta input contract therefore also adds:

```python
target_row_available: object
btc_row_available: object
```

to `build_causal_beta_series`.

For each decision row `t`:

```python
eligible_sample_rows = sample_rows[sample_rows <= t]
eligible_sample_rows = eligible_sample_rows[-180:]
```

Then retain only rows whose target and BTC 4h intervals are both complete. Require at least `config.minimum_complete_samples` pairs. Compute population covariance and variance consistently with NumPy `mean` semantics:

```python
x = btc_returns
 y = target_returns
x_centered = x - x.mean(dtype=np.float64)
y_centered = y - y.mean(dtype=np.float64)
market_variance = np.mean(np.square(x_centered), dtype=np.float64)
covariance = np.mean(x_centered * y_centered, dtype=np.float64)
beta = covariance / market_variance
```

The implementation line must be `y = target_returns`; the leading space above is prose formatting only.

Use at most 180 paired return samples because `720h / 4h = 180`.

## Test additions

Add this focused test before implementing beta:

```python
def test_causal_beta_ignores_intermediate_fifteen_minute_path_when_four_hour_closes_match():
    bars_per_4h = 16
    four_hour_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005] * 31,
        dtype=np.float64,
    )
    endpoints = np.exp(
        np.concatenate(([0.0], np.cumsum(four_hour_returns, dtype=np.float64)))
    )
    row_count = (len(endpoints) - 1) * bars_per_4h + 1
    btc_left = np.empty(row_count, dtype=np.float64)
    btc_right = np.empty(row_count, dtype=np.float64)
    for block in range(len(endpoints) - 1):
        start = block * bars_per_4h
        stop = start + bars_per_4h
        btc_left[start : stop + 1] = np.linspace(
            endpoints[block], endpoints[block + 1], bars_per_4h + 1
        )
        phase = np.linspace(0.0, np.pi, bars_per_4h + 1)
        curved = np.exp(0.01 * np.sin(phase))
        curved[0] = 1.0
        curved[-1] = 1.0
        btc_right[start : stop + 1] = (
            np.linspace(endpoints[block], endpoints[block + 1], bars_per_4h + 1)
            * curved
        )
    target_left = np.square(btc_left)
    target_right = np.square(btc_right)
    decision_indices = np.arange(row_count, dtype=np.int64)
    available = np.ones(row_count, dtype=np.bool_)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=720.0,
        minimum_complete_samples=90,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    common = dict(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        target_row_available=available,
        btc_row_available=available,
        bars_per_4h=bars_per_4h,
        config=config,
        target_source_digest="1" * 64,
        btc_source_digest="2" * 64,
    )
    first = build_causal_beta_series(
        target_close=target_left,
        btc_close=btc_left,
        **common,
    )
    second = build_causal_beta_series(
        target_close=target_right,
        btc_close=btc_right,
        **common,
    )
    np.testing.assert_allclose(
        first.beta[first.available],
        second.beta[second.available],
        atol=1e-12,
        rtol=0.0,
    )
```

Also add a test that makes one intermediate source row unavailable inside a 4h interval and proves that whole paired return sample is excluded.

## Identity addition

`beta_source_digest` binds:

```text
target source digest
BTC source digest
decision indices digest
target/BTC availability digests
bars_per_4h = 16
lookback_hours = 720
maximum sample count = 180
minimum complete sample count = 90
clip [-3, 3]
```

This prevents a beta produced from overlapping returns from sharing identity with the V4 contract.
