# Causal Alpha V4 Derivatives Staleness Plan Amendment

> **For agentic workers:** Read this immediately after `2026-08-23-causal-alpha-v4-source-availability-amendment.md`. It adds one pre-outcome Task 3 constraint; all other V4 plan sections remain authoritative.

**Spec:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-source-availability-amendment.md`

## Reason

The source-availability contract already required exact futures-metrics staleness, but did not state when an old as-of value stops being actionable. Without a cutoff, one historical metrics row could remain semantically available forever.

No V4 model or trading outcome has been observed. The first generation therefore freezes the cutoff from the maintained decision clock rather than from performance:

```text
maximum_derivatives_staleness_hours = 0.25
```

## Task 3 implementation contract

Define in `trade_rl/integrations/binance_v4_context.py`:

```python
BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS: Final = 0.25
```

`align_futures_metrics_to_decisions(...)` performs a backward as-of join using only `create_time <= decision_timestamp`.

For the latest eligible source row:

```python
age_hours = (
    decision_timestamp_ns - metrics_create_time_ns
) / 3_600_000_000_000.0
available = 0.0 <= age_hours <= BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS
```

When `available` is false because the row is older than 15 minutes:

- keep exact source values only as inert storage;
- set `derivatives_available=False`;
- keep exact `derivatives_staleness_hours=age_hours` for diagnostics;
- do not substitute zero as market state;
- do not choose a future metrics row.

## Required RED tests

Add before production implementation:

```python
def test_metrics_older_than_one_decision_are_unavailable():
    decisions = np.asarray(
        [np.datetime64("2026-01-01T00:30")],
        dtype="datetime64[ns]",
    )
    metrics = make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:10")], dtype="datetime64[ns]")
    )
    aligned = align_futures_metrics_to_decisions(decisions, metrics)
    assert not aligned.available[0]
    assert aligned.staleness_hours[0] == pytest.approx(20.0 / 60.0)


def test_metrics_exactly_fifteen_minutes_old_remain_available():
    decisions = np.asarray(
        [np.datetime64("2026-01-01T00:30")],
        dtype="datetime64[ns]",
    )
    metrics = make_metrics(
        np.asarray([np.datetime64("2026-01-01T00:15")], dtype="datetime64[ns]")
    )
    aligned = align_futures_metrics_to_decisions(decisions, metrics)
    assert aligned.available[0]
    assert aligned.staleness_hours[0] == pytest.approx(0.25)
```

## Quality gate addition

V4 source integration is not complete unless tests prove all three cases independently:

```text
future metric -> unavailable
latest metric age <= 15m -> available
latest metric age > 15m -> unavailable with preserved positive staleness
```
