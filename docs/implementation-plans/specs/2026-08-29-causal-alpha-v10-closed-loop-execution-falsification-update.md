# Causal Alpha V10 Closed-Loop Execution Falsification Update

This document is a normative amendment to `2026-08-29-causal-alpha-v10-closed-loop-execution-design.md`, discovered during the implementation falsification pass. Where this amendment conflicts with the earlier hard-risk latch wording, this amendment takes precedence.

## Discovered failure mode

The original design said that after a non-executable hard-cap partial reduction causes V10 to request flat, the policy should keep requesting flat until realized exposure is exactly zero.

That is too strong under the maintained `PreTradeRisk` no-trade band. Example:

```text
realized current = 0.10
risk cap         = 0.04
entry threshold  = 0.10
exit threshold   = 0.03
no-trade band    = 0.05
```

V10 correctly requests `0.0` because the same-direction `0.04` target would be held by entry hysteresis. If downstream execution or an external constraint leaves realized exposure at `0.04`, repeatedly requesting `0.0` can itself be suppressed because the remaining delta is below the `0.05` no-trade band. The old latch could therefore create a permanent ineffective flatten loop even though the actual hard-risk invariant is already satisfied.

## Updated hard-risk invariant

The safety condition is:

```text
abs(realized_exposure) <= active_risk_cap
```

not mandatory exact flatness.

After `risk_cap_flatten` is requested:

1. While realized exposure remains above the active risk cap, keep the flatten intent active and keep requesting flat.
2. Once realized exposure is at or below the active risk cap (within the policy observation tolerance), release the flatten latch and resume ordinary closed-loop reasoning from that realized exposure.
3. Do not restore the pre-reduction requested target merely because the latch was released.
4. If a later active risk cap falls below realized exposure again, re-enter hard-risk projection/flatten logic.
5. `PreTradeRisk` remains the final safety authority.

## Updated acceptance criteria

In addition to the original criteria:

- Starting from `0.10` with active risk cap `0.04`, a non-executable partial reduction requests flat and records `risk_cap_flatten`.
- If the next simulator observation reports realized exposure `0.04`, V10 releases the flatten latch because the cap invariant is now satisfied and the next ordinary hold is based on `0.04`, not another unconditional flat request and not the old `0.10` target.
- If the next simulator observation remains above `0.04`, the flatten intent remains active.

## Updated test oracle

A regression must drive the policy sequentially with `current_weights=0.10` followed by `current_weights=0.04` under a `0.04` risk cap. The first action must be flat; the second action must be based on the realized `0.04` position and must not be forced flat solely by the old latch.

This change does not weaken the risk cap. It removes an impossible stronger condition that could cause a no-trade-band loop after the actual cap invariant has already been met.
