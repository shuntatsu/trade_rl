# Cloud trend-strength exit gate and ATR entry sizing (software-only)

Status: Active — synthetic software prototype. No new economic run authorized.

## G0 / mechanism and falsifier

Cloud divergence exits can cut successful trend exposure prematurely. Test a role-separated
mechanism: daily visible Ichimoku provides direction; confirmed opposing 4h
divergence indicates potential loss of momentum; normalized 4h ADX measures
strength (not direction); 4h ATR/close measures one-bar price range relative
to price. High ADX can also occur during a reversal, so suppressing a reversal
warning is a counter-hypothesis, not inherently safer. ATR sizing can suppress
useful risk exposure. An isolated synthetic gate, unit, clock and native fill
oracle is the cheapest falsifier. Stop if any oracle fails. No performance-based
parameter selection or modification of an earlier closed study.

## G1 / fixed synthetic candidate definitions

Use canonical ADX(14) (already normalized by 100) and ATR_PCT(14) from completed UTC 4h
bars, aligned as-of the 1h decision clock. No new oscillator numerics.
Mask a row if its latest completed 4h bar is unavailable, any preceding
dependency is late, 1h row unavailable, or canonical indicator unready.
Require 28 completed contiguous 4h bars; mask the earlier canonical seed row
whose readiness otherwise depends on one more completed bar. Never backfill
a prior decision. ATR must be strictly positive, ADX in [0,1].

All three modes delegate six-hour cloud entry and exact quantity hold to
existing C2. Cloud failure or context failure flattens held positions.
A requested exit latches until the account confirms flat; no re-entry on
that decision. A divergence exit additionally blocks re-entry for 24 hours
from the first confirmed-flat decision. Insolvent/terminated accounts preserve
the base policy's hold and hard risk/execution/accounting remain authoritative.

- T0 (all_exits): all confirmed opposite 4h divergence events can exit.
- T1 (weak_trend_exits): opposing divergence can exit only when ADX < 0.25.
  Strong ADX never prevents an exit caused by loss of cloud support.
- T2 (weak_trend_atr_entry): T1 plus entry absolute weight capped at
  min(base entry weight, 0.0025 / ATR_PCT). The intended target weight times the last completed 4h ATR/close
  fraction is at most 0.25%. This is not the realized fill exposure, a stop,
  or a maximum loss; price drift, gaps, execution and subsequent risk remain. Exact quantity hold thereafter.

T0 is the matched implementation control, not a claim of bitwise C3 parity:
it shares the new availability and latched-exit contract. C2/C3 remain
unchanged. The earlier standalone Donchian/ATR stop line is closed to additional
economic runs; its existing Donchian/ATR policy is reused solely as a
synthetic common-engine comparator (20/10 bars on a generated hourly path).

## G2 / tests and G3 / evidence

Synthetic oracles: monotonic literal true range/Wilder ADX and ATR units,
4h clock and publication delay, prefix equivalence, tradable-gap restart,
threshold equality, mirrored long/short, partial exits, terminated account,
strict context validation, size cap, and actual native common replay.

No price archive is read or new real-market performance computed in this
change. A separate result-blind author packet, independent fresh review and
owner G4 authorization on exact heads are required before economic evaluation.
No sealed/unused data, no economic winner, no paper/live or main changes.
