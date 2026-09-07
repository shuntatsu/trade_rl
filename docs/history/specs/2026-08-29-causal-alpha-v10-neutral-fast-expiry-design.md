# Causal Alpha V10 neutral-fast-expiry candidate

## Scope

This is a preregistered comparison candidate after the rejected r8
`flatten_then_reset` run. It changes only how an already-owned non-flat
position handles repeated neutral fast (4-hour) qualification. Signal fitting,
72-hour slow regime, entry/exit confirmation, execution eligibility,
risk/liquidity caps, cost model, data, and all numerical Selection gates remain
identical to V10.

## Fixed behavior

- Candidate mode: `neutral_fast_expiry`.
- On a cadence decision while a position is non-flat, a fast qualified
  direction of zero increments a consecutive neutral-fast counter.
- After `slow_neutral_expiry_count` consecutive neutral-fast cadence decisions
  (the frozen default is 6), request flat with hierarchy reason
  `neutral_fast_expiry` and reset ownership/confirmation state.
- Any non-neutral fast direction resets that counter. A risk projection takes
  precedence and resets the counter, so this candidate does not alter risk
  handling.
- The flat decision is not allowed to re-enter on the same row; the existing
  two-observation coherent fast/slow entry confirmation remains required.
- Initial inherited positions, direct-sign-flip protection, realized-weight
  tracing, and all artifact/gate contracts remain unchanged.

The candidate is intentionally separate from boundary ownership and
flat-on-risk-breach. A Selection reject does not open Admission, BC/RL, or
holdout.

