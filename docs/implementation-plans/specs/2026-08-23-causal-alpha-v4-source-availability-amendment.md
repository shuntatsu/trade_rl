# Causal Alpha V4 Source Availability Amendment

## Status and precedence

This amendment corrects a second pre-implementation contract gap found while preparing Task 1 RED tests. It applies after the base V4 design and beta-materialization amendment.

No V4 model/trading outcome has been observed. The correction fixes information timing and missingness semantics; it does not change the economic hypothesis, reward, model strengths, feature names, or target parameters.

## Problem found

The base plan described `V4CrossMarketInputs` primarily as value arrays. That is insufficient for features whose raw source may be missing or whose newest usable observation is older than the 15-minute decision close.

In particular:

- Binance funding is an event series and must be carried causally with age;
- futures metrics can be as-of aligned from a finer/different clock and therefore need source age;
- Spot/perpetual bars can be absent and must not become informative forward-filled values;
- Student observations explicitly expose V4 feature staleness, so the same staleness must exist before feature construction.

## Corrected source input contract

`V4CrossMarketInputs` is an aligned source-event contract, not merely a matrix of filled values.

It contains:

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

All arrays are row-aligned to the maintained 15-minute decision clock before entering the pure feature builder.

`decision_timestamps` are `datetime64[ns]`, strictly increasing, and represent decision-bar close times.

For Spot/perpetual rows, `*_row_available` means the fully closed source bar was observable by the decision close. A numeric storage value on an unavailable row has no semantic value.

For futures metrics, the adapter performs an as-of join using source `create_time <= decision_timestamp` and emits both `derivatives_available` and exact non-negative `derivatives_staleness_hours`.

The first V4 generation freezes:

```text
maximum_derivatives_staleness_hours = 0.25
```

This value is derived from the maintained 15-minute decision clock, not from model or trading outcomes. The latest metrics row is available at a decision only when its age is in `[0.0, 0.25]` hours. If the latest source row is older than 15 minutes, the adapter keeps its stored numeric values inert, sets `derivatives_available=False`, and preserves the exact age in `derivatives_staleness_hours` for diagnostics. A later generation may change this threshold only through a new pre-outcome authored contract.

## Funding event semantics

`funding_event_rate` contains the actual funding value only on source funding-event rows and uses inert numeric zero elsewhere. `funding_event_available` identifies those actual events.

Feature construction processes funding as an event stream:

1. `funding_rate` is the latest observed funding event carried forward for at most 24 hours.
2. `funding_rate_change` is the difference between the latest and previous observed funding events, then carried with the latest event.
3. `funding_rate_robust_z_7d` is calculated over funding events whose timestamps lie within the trailing 168 hours, requires at least 8 observed events, and is then carried with the latest event.
4. The staleness of all three funding-derived channels is the age in hours since the latest funding event used.
5. With no qualifying event, or when the latest event is older than 24 hours, all funding-derived channels are unavailable.

This prevents repeated 15-minute copies of one funding event from being counted as independent samples in the z-score.

## Trailing bar-derived feature semantics

For a trailing feature ending at decision row `t`, every source bar required by its window must be available. Otherwise the derived feature is unavailable.

The first V4 profile uses exact decision-bar counts:

```text
1h   = 4 x 15m bars
4h   = 16 x 15m bars
24h  = 96 x 15m bars
7d   = 672 x 15m bars
```

Return windows use `log(close_t / close_{t-window})` and therefore require both endpoints and all intervening source rows to be available, preventing a missing interval from masquerading as a valid flat period.

Quote-volume sums and taker-buy quote-volume sums require every row in the window.

A bar-derived feature has staleness `0.0` at a usable decision because its final source bar closes at that decision. A feature that uses an as-of event/metric carries the maximum staleness among the source components that actually determine the feature.

## Interaction feature staleness

For products/differences of derived channels:

```text
available = all(component_available)
staleness = max(component_staleness_hours)
```

This applies to basis/flow divergence, basis/OI, funding/OI, and global cross-anchor features.

## Binance V4 adapter responsibility

The V4 Binance adapter may not rely on `RawMarketSeries.volume` for quote-volume/taker-flow features because the maintained generic raw-series contract does not expose those fields.

`trade_rl/integrations/binance_v4_context.py` therefore parses the required public kline fields directly from the same immutable Binance Vision archives:

```text
open time
open
high
low
close
base volume
close time
quote asset volume
number of trades
taker buy base asset volume
taker buy quote asset volume
ignore
```

It binds the raw archive digest and header/layout parser version into `source_digest`.

The generic maintained market dataset remains unchanged.

## Added Test Oracle

The source availability contract is correct when:

1. Removing one Spot bar from a 4h window makes the affected Spot 4h features unavailable.
2. A missing bar is not treated as a carried flat close.
3. A funding event remains available with increasing staleness until 24h, then expires.
4. Duplicating the same carried funding value on 15-minute rows does not change the event-based funding z-score.
5. A futures metrics event with `create_time` after decision `t` cannot affect context at `t`.
6. An as-of metrics value has staleness equal to `decision_time - create_time`.
7. A futures metrics value older than 15 minutes is unavailable even though its exact staleness remains observable for diagnostics.
8. Interaction feature staleness is the maximum of its inputs.
9. Mutating any future source row leaves all earlier feature values, availability, and staleness unchanged.

Any violation is a causality/information-contract failure and blocks V4 Teacher admission.
