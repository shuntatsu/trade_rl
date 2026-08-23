# Causal Alpha V4 Beta Sampling Amendment

## Status

This pre-implementation amendment refines only the beta sampling semantics. It follows the V4 beta-materialization amendment. No V4 outcome has been read.

## Problem

Using a trailing 4h return recalculated every 15 minutes would create heavily overlapping beta observations. With a 720h lookback that would produce thousands of correlated rows and make the `minimum_complete_samples=90` rule mean something very different from 90 independent-ish 4h observations.

## Corrected beta sampling contract

Causal beta uses **non-overlapping 4h returns**.

For an aligned V4 target context with ordered decision indices and `bars_per_4h=16`, define 4h return sample ends relative to the first decision index:

```text
sample_end(k) = first_decision_index + k * 16
```

A return sample ending at `sample_end(k)` is:

```text
r_4h(k) = log(close[sample_end(k)] / close[sample_end(k) - 16])
```

and is usable only when both endpoints and every intervening source row are valid/observable.

At decision `t`, beta may use only samples whose sample end is `<= t`. The 720h window contains at most:

```text
720h / 4h = 180
```

non-overlapping return pairs. At least 90 complete target/BTC paired samples are required.

For each decision, take the most recent at most 180 complete paired samples ending at or before the decision, then calculate covariance/variance and apply the existing `[-3, 3]` clip.

Between two completed 4h sample ends, beta remains unchanged except for availability expiry caused by source invalidity; it is not recalculated from overlapping partial shifts.

The first sample end is the first decision index at least 16 bars after the context start that satisfies the fixed step above. The context start, decision-index vector, and `bars_per_4h` are bound into `beta_source_digest`, so changing alignment changes identity.

## Test Oracle addition

A production-like 15-minute synthetic series with 181 non-overlapping 4h returns must provide at most 180 observations to any one beta estimate. Duplicating the same 4h path at intermediate 15-minute offsets must not change beta.

The following property is required:

```text
beta(original 4h closes)
== beta(same 4h closes with arbitrary causal intermediate 15m path changes)
```

provided the 4h sample endpoints and source-validity masks are unchanged.
