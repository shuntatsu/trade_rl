# Causal Alpha V10 flat-on-risk-breach candidate

## Scope

This is a preregistered comparison candidate after the rejected r10
`neutral_fast_expiry` run. It changes only the ownership response to a
realized position above the active hard risk cap. Signal fitting, horizons,
confirmation, liquidity/cost model, data, and all numerical Selection gates
remain identical.

## Fixed behavior

- Candidate mode: `flatten_on_risk_breach`.
- When realized exposure exceeds the active risk cap, request flat and record
  hierarchy reason `risk_cap_flatten` instead of requesting a partial cap
  reduction.
- Keep requesting flat until a realized flat observation is received; then
  reset hierarchy state and require the existing entry confirmation before any
  re-entry.
- Existing default partial risk projection and non-executable partial
  reduction behavior remain unchanged for other modes.
- Direct realized sign-flip protection, trace/economics identity, and all
  stage gates remain unchanged.

This candidate is intentionally separate from boundary ownership and neutral
fast expiry. A Selection reject does not open Admission, BC/RL, or holdout.
