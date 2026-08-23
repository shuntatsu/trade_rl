# Causal Alpha V4 Source Availability Plan Amendment

> **For agentic workers:** Read this after the base V4 plan and the beta-materialization plan amendment. This file overrides the raw `V4CrossMarketInputs` definition and source-timing portions of Tasks 1 and 3. Other plan sections remain authoritative.

**Spec amendment:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-source-availability-amendment.md`

## Corrected Task 1 source contract

Use this exact input dataclass in `trade_rl/data/v4_context.py`:

```python
@dataclass(frozen=True, slots=True)
class V4CrossMarketInputs:
    decision_indices: np.ndarray
    decision_timestamps: np.ndarray
    spot_close: np.ndarray
    spot_quote_volume: np.ndarray
    spot_taker_buy_quote_volume: np.ndarray
    spot_row_available: np.ndarray
    perp_close: np.ndarray
    perp_mark_price: np.ndarray
    perp_quote_volume: np.ndarray
    perp_taker_buy_quote_volume: np.ndarray
    perp_row_available: np.ndarray
    funding_event_rate: np.ndarray
    funding_event_available: np.ndarray
    open_interest_value: np.ndarray | None
    global_long_short_ratio: np.ndarray | None
    top_position_long_short_ratio: np.ndarray | None
    derivatives_available: np.ndarray | None
    derivatives_staleness_hours: np.ndarray | None
    source_digest: str
```

Validation requirements:

```text
decision_indices: rank 1 int64, strictly increasing, non-negative
decision_timestamps: rank 1 datetime64[ns], same length, strictly increasing
price arrays: finite positive float64 on available rows
quote/taker volumes: finite non-negative float64 on available rows
taker buy quote <= total quote on available rows
row/event/derivative availability: rank 1 bool, same length
staleness: finite non-negative float64 on derivative-available rows
optional derivative arrays: all present together or all absent
source_digest: exactly 64 lower-case hexadecimal characters
```

`V4GlobalMarketInputs` is:

```python
@dataclass(frozen=True, slots=True)
class V4GlobalMarketInputs:
    btc: V4CrossMarketInputs
    eth: V4CrossMarketInputs
    source_digest: str
```

BTC and ETH decision indices/timestamps must match exactly.

## Corrected Task 1 funding helpers

Implement a pure helper:

```python
@dataclass(frozen=True, slots=True)
class FundingContextSeries:
    rate: np.ndarray
    change: np.ndarray
    robust_z_7d: np.ndarray
    available: np.ndarray
    staleness_hours: np.ndarray


def build_funding_context_series(
    *,
    decision_timestamps: object,
    funding_event_rate: object,
    funding_event_available: object,
    maximum_staleness_hours: float = 24.0,
    z_window_hours: float = 168.0,
    minimum_z_events: int = 8,
) -> FundingContextSeries:
    """Carry actual funding events causally without duplicating event weight."""
```

Algorithm for row `t`:

```text
observed_event_indices = event rows <= t
latest = last observed event
previous = event immediately before latest
window = observed events with timestamp >= timestamp[t] - 168h
rate[t] = event_rate[latest]
change[t] = event_rate[latest] - event_rate[previous] when previous exists
z[t] = robust z of event_rate[latest] within window when event count >= 8
staleness[t] = hours(timestamp[t] - timestamp[latest])
available[t] = latest exists and staleness[t] <= 24h
```

The `rate` channel can be available with one event. `change` requires two events. `robust_z_7d` requires eight events. The builder returns separate availability internally for these three channels; `FundingContextSeries.available` is therefore a `(row, 3)` Boolean array and `staleness_hours` is `(row, 3)`.

Write these exact tests before production code:

```python
def test_funding_context_carries_event_age_then_expires():
    timestamps = np.datetime64("2026-01-01T00:00", "ns") + np.arange(101) * np.timedelta64(15, "m")
    rates = np.zeros(101, dtype=np.float64)
    events = np.zeros(101, dtype=np.bool_)
    rates[0] = 0.001
    events[0] = True
    result = build_funding_context_series(
        decision_timestamps=timestamps,
        funding_event_rate=rates,
        funding_event_available=events,
    )
    assert result.available[0, 0]
    assert result.staleness_hours[4, 0] == 1.0
    assert result.available[96, 0]
    assert not result.available[100, 0]


def test_funding_z_counts_events_not_carried_rows():
    timestamps = np.datetime64("2026-01-01T00:00", "ns") + np.arange(8 * 32 + 1) * np.timedelta64(15, "m")
    rates = np.zeros(len(timestamps), dtype=np.float64)
    events = np.zeros(len(timestamps), dtype=np.bool_)
    event_rows = np.arange(0, 8 * 32, 32)
    rates[event_rows] = np.asarray([0.001, 0.002, -0.001, 0.003, 0.0, 0.0025, -0.0005, 0.004])
    events[event_rows] = True
    result = build_funding_context_series(
        decision_timestamps=timestamps,
        funding_event_rate=rates,
        funding_event_available=events,
    )
    first_z_row = int(event_rows[-1])
    assert not np.any(result.available[:first_z_row, 2])
    assert result.available[first_z_row, 2]
```

## Corrected trailing-window availability semantics

Define exact bar constants in `v4_context.py`:

```python
BARS_1H = 4
BARS_4H = 16
BARS_24H = 96
BARS_7D = 672
```

A trailing window feature is available only if every required raw row in the window is available.

Write a failing test before implementing the window helper:

```python
def test_missing_spot_bar_invalidates_four_hour_spot_return():
    inputs = make_v4_cross_market_inputs(row_count=40)
    available = inputs.spot_row_available.copy()
    available[25] = False
    inputs = replace(inputs, spot_row_available=available)
    block = build_cross_market_context(inputs, include_derivatives=False)
    index = block.feature_names.index("spot_log_return_4h")
    assert not block.available[32, index]
```

Use contiguous row support, not forward-filled close, for return/volume/flow windows.

## Corrected feature staleness rules

Task 1 builders assign feature staleness as follows:

```text
Spot/perp completed-bar-only feature                -> 0.0h
funding-derived feature                            -> funding source age
OI/L-S derivative-only feature                     -> derivatives_staleness_hours
interaction/difference using multiple source types -> max(component ages)
```

Unavailable features store inert numeric `0.0`, `available=False`, and staleness equal to the maximum authored source staleness limit rather than pretending to be fresh.

## Corrected Task 3 Binance kline parsing

`trade_rl/integrations/binance_v4_context.py` parses V4 kline fields directly from immutable Binance Vision ZIP/CSV rows. The generic `RawMarketSeries` contract remains unchanged.

Freeze this position mapping for headerless Binance kline rows:

```python
KLINE_OPEN_TIME = 0
KLINE_OPEN = 1
KLINE_HIGH = 2
KLINE_LOW = 3
KLINE_CLOSE = 4
KLINE_BASE_VOLUME = 5
KLINE_CLOSE_TIME = 6
KLINE_QUOTE_VOLUME = 7
KLINE_TRADE_COUNT = 8
KLINE_TAKER_BUY_BASE_VOLUME = 9
KLINE_TAKER_BUY_QUOTE_VOLUME = 10
KLINE_IGNORE = 11
KLINE_FIELD_COUNT = 12
```

Parser tests require exactly 12 fields, positive OHLC, non-negative volumes, `taker_buy_quote_volume <= quote_volume + 1e-12`, and close time not earlier than open time.

The adapter produces fully closed 15-minute decision-clock source rows and marks source availability only when source `available_at <= decision close`. Missing archive/bar data produces `row_available=False`; it does not forward-fill a usable source row for V4 features.

## Corrected Task 3 derivatives as-of semantics

For each decision timestamp, choose the latest metrics event satisfying:

```text
metrics.create_time <= decision_timestamp
```

Emit:

```python
derivatives_staleness_hours = (
    decision_timestamp_ns - metrics_create_time_ns
) / 3_600_000_000_000.0
```

If no earlier metric exists, all derivative fields are unavailable. Future metrics events are never consulted.

Add exact tests:

```python
def test_future_metrics_event_is_not_visible_to_earlier_decision():
    decisions = np.asarray(
        [np.datetime64("2026-01-01T00:00"), np.datetime64("2026-01-01T00:15")],
        dtype="datetime64[ns]",
    )
    metrics_time = np.asarray([np.datetime64("2026-01-01T00:10")], dtype="datetime64[ns]")
    aligned = align_futures_metrics_to_decisions(decisions, make_metrics(metrics_time))
    assert not aligned.available[0]
    assert aligned.available[1]
    assert aligned.staleness_hours[1] == pytest.approx(5.0 / 60.0)
```

## Verification additions

Task 14 focused suite must also run:

```text
tests/data/test_v4_context.py::test_funding_context_carries_event_age_then_expires
tests/data/test_v4_context.py::test_funding_z_counts_events_not_carried_rows
tests/data/test_v4_context.py::test_missing_spot_bar_invalidates_four_hour_spot_return
tests/integrations/test_binance_v4_context.py::test_future_metrics_event_is_not_visible_to_earlier_decision
```

Falsification review additionally asks:

```text
Can a missing Spot/perp bar become a valid flat period?
Can one funding event gain artificial sample weight through 15m carry-forward?
Can a future metrics row enter an earlier decision by nearest-neighbor rather than backward as-of join?
Can a stale event appear with zero staleness in the Student observation?
```

Any `yes` blocks V4 execution.
