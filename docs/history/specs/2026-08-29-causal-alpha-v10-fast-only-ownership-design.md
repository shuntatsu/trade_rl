# Causal Alpha V10 fast-only ownership candidate

## Scope

This is a preregistered comparison candidate after the rejected r11
`flatten_on_risk_breach` run. It tests whether the 72h slow head is causing
ownership loss after entry. Only the hierarchical candidate's decision
ownership changes; fitting, 4h/72h horizons, entry/exit confirmation counts,
risk/liquidity caps, execution, costs, data, and numerical Selection gates
remain identical.

## Fixed behavior

- Candidate mode: `fast_only_ownership`.
- Flat entry uses the qualified fast direction and execution eligibility only;
  the slow direction is retained in the trace but is not required for entry.
- An inherited non-flat position earns ownership confirmation from the fast
  direction only. A slow-opposite observation does not force an inherited exit
  when fast direction and execution eligibility remain coherent.
- Owned-position exits use fast-opposite confirmation and the existing six-
  cadence fast-neutral expiry. Slow-opposite and slow-neutral counters are
  diagnostic only for this candidate and cannot trigger an exit.
- Risk projection, hard caps, liquidity handling, execution/cost semantics,
  direct sign-flip protection, action trace fields, and step economics remain
  unchanged.
- The hierarchy reason `fast_support_hold` is recorded while the underlying V6
  reason remains `hold_position`, preserving the V6 target contract.

## Falsification and gates

- Regression tests must prove fast-only entry with slow neutral and inherited
  retention with slow opposite.
- The stage identity digest must differ from every existing V10 boundary mode.
- The run uses a new output root and exact-HEAD Docker image; no prior leaf or
  result is reused.
- Signal must pass all 72 paired scopes and Selection must complete all 216
  leaves before any promotion decision.
- A Selection reject does not open Admission, BC/RL, or holdout.

The candidate is intended to falsify the hypothesis that slow-head ownership,
rather than the fast signal or execution cost, is the dominant source of the
observed inherited/neutral loss. A positive aggregate alone is insufficient;
minimum and median per-symbol net wealth and positive-scope fraction must pass
the existing gates.
