# Causal Alpha V8 Exposure State Machine Design

## Status and evidence

V7 passed Signal and failed Selection on the fixed nine-symbol, eight-episode
contract. The best candidate, symmetric contrarian, produced symbol-balanced
gross wealth 0.968042 and net wealth 0.961676. Its net/gross retention was
0.993424, so execution costs were not the primary failure.

Two episodes dominated the loss:

- cutoff 40218: all candidates held long for all 25,920 decisions and produced
  net wealth 0.703162;
- cutoff 51742: all candidates shared mostly-long exposure and produced net
  wealth 0.861566.

In both episodes `cadence_hold` and nominally supportive slow state cells held
most of the negative gross return. Candidate-specific forecasts collapsed to
the same target path because the shared V6 target compiler treats an
unprofitable/no-entry proposal as `hold previous`. That behavior is safe while
flat but unsafe when the previous position is inherited or no longer supported.

## Objective and unchanged invariants

Maximize independent per-symbol after-cost wealth using the pure reward
`100 * net_log_return`. Preserve all existing symbols, chronological splits,
costs, liquidity/risk caps, initial-state contracts, and fixed universal gates.

The following remain forbidden:

- threshold relaxation after observing Selection;
- holdout access before Admission;
- symbol exclusion or symbol-identity lookup;
- BC/RL before Signal, Selection, and Admission pass;
- direct position flips that combine exit and entry into one change.

One-minute data remains out of scope until a positive-gross strategy is shown to
fail because of execution or slippage. V7 is negative gross, so finer execution
resolution cannot repair the observed failure.

## Two-stage decomposition

### Stage 1: causal signal

Reuse the frozen V7 feature and calibration contract. It remains pooled,
symbol-free, chronological, purged on the 4h label end, and fitted separately at
each cutoff.

The three predeclared V8 candidates are:

1. `v7_control`: the exact V7 raw forecast and V6 exposure compiler;
2. `robust_contrarian`: symmetrically negate raw 4h return and direction, then
   use the V8 exposure state machine;
3. `robust_calibrated`: use the pooled causal calibration heads, then use the
   V8 exposure state machine.

The unchanged V7 control makes the exposure-state-machine effect identifiable.

### Stage 2: exposure state machine

At every fixed fast rebalance decision, score the current and reachable target
levels by robust position utility:

```text
U(w) = w * expected_return
       - abs(w) * (uncertainty_multiplier * uncertainty + edge_margin)

transition_score(previous, target)
    = U(target) - U(previous)
      - abs(target - previous)
        * execution_cost_multiplier * one_way_cost_rate
```

This is not a new fitted threshold. It reuses the authored V6 uncertainty,
margin, cost multiplier, target levels, delta cap, cadence, and liquidity/risk
caps.

The state transitions are:

- `flat -> position` (entry): require return/direction consensus, the existing
  confirmation count, and strictly positive transition score;
- `position -> larger same-sign position` (add): apply the same entry rules;
- `position -> smaller same-sign position` (reduce): always eligible when the
  robust transition score is positive;
- `position -> flat` (exit): always eligible when the robust transition score
  is positive;
- `long <-> short`: prohibited directly; exit to flat first and require a later
  confirmed entry;
- non-rebalance rows: hold, except immediate liquidity/risk deleveraging.

The critical asymmetry is mathematically explicit. For an entry, uncertainty
reduces the value of taking exposure. For a reduction, removing uncertain
exposure increases robust utility. Therefore uncertainty cannot simultaneously
block a new trade and force an unsupported inherited position to remain open.

Long waves of hours or days remain possible: a position is held without changes
while its robust continuation utility stays positive. Low-confidence/no-volume
periods remain flat, and exits cannot immediately turn into opposite entries.

## Evidence and gates

V8 must emit its own content-addressed schemas and candidate names. Every replay
must bind:

- source forecast, calibration fit, target path, contract, and config digests;
- simulator-authoritative gross/net log return, reward, costs, turnover, risk,
  and holding duration;
- attribution by confidence quartile, exposure, centered relative-volume
  quartile, slow state, transition, and volatility quartile;
- entry, add, reduce, exit, direct-flip, and inherited-exit counts.

Signal, Selection, and Admission retain the V7 gate values. Selection must still
require balanced gross and net wealth above one, every symbol net wealth at
least one, median symbol wealth at least one, at least half of scopes positive,
bounded turnover, meaningful execution, and no hard-risk/unexplained rejection.

## Durable resume

Selection runtime grows with the cutoff. Persist each completed
episode/symbol/candidate replay as an immutable leaf before advancing. A resume
must validate run, context, config, generator, contract, forecast, target, and
replay digests, reject partial or conflicting leaves, and skip only exact
validated identities. Final Selection must be reconstructed from the complete
set of validated replay leaves.

The checkpoint JSONL remains a progress/diagnostic stream; it is not itself
sufficient to reconstruct final Selection.

## Admission and learning

Admission remains sealed until one candidate passes Selection. Only an admitted
candidate may be packaged for BC/RL. If no candidate passes, preserve all
evidence, issue a terminal rejection report, and redesign from the identified
gross-loss state rather than relaxing gates.

