# Proposed trend spot cash-account safety revision (`trend_long_spot_oos_v2`)

Status: Active (documentation lifecycle only), 2026-11-21.

**Economic authorization: HOLD / software-only proposal.** This is **not** an
owner-approved one-shot economic protocol, an authorization to revisit data,
or a replacement for any immutable `trend_long_spot_oos_v1` run.

## Historical boundary and objective

The historical [v1 frozen protocol](trend-long-spot-oos.md), original source commits,
owner G4 authorizations, source-bound results and reported BACKCAST `OOS_SUPPORTED`
remain preserved **as originally recorded**, without rewriting or relabeling them.
[Issue #1306](https://github.com/shuntatsu/trade_rl/issues/1306) reproduces a validity
defect in the v1 no-borrowing account; therefore its positive OOS label **does not
constitute verified cash-safe or profitable evidence**. The 2018–2020 BACKCAST
window has been consumed, and must never be presented as an unseen OOS test for v2.

This v2 source is a synthetic-only safety successor. The unchanged lowturn strategy,
original scenario costs, controls, risk scaling, end-of-replay settlement, and
causality probes are used for mechanical regression tests, **not** economic approval.
The v1 BACKCAST/FORWARD windows are retained as software layout fixtures only; they
are **not** an eligible new research holdout. The public `tools.trend_oos` CLI
**refuses real `run` and `coverage` for either stage before opening data or
querying the clock/GitHub**, even with an otherwise well-formed v2 authorization
comment. In-process synthetic fixture injection is used for software tests only;
it must not be used as an operator economic execution entry point. Any future economic stage must receive
a result-blind G0–G3 design review, an explicitly authorized new protocol/window,
its own G4 relay quoting the new exact digest and commit, and fresh independent
software review. Old v1 G4 comments cannot authorize v2. Neither v1 nor v2 grants
FS1, unused/final, paper/live order, or profitability authorization.

## Cash budget and falsifiers

All spot legs must stay long-only, all perpetual legs unavailable, and all arms use
the shared `run_shared_portfolio_replay` ledger. At decision time, form the *effective*
weights with exact-quantity HOLD legs kept at current weights and other legs scaled
**once** by the existing drawdown risk factor. For a real retargeting decision, define
`g0 = sum(abs(current_spot_weights))`, `g1 = sum(abs(target_spot_weights))`, and
`r` as the maximum deterministic *configured* transaction cost fraction over the
three frozen scenarios: `r = 0.0016` (HIGH_COST: fee 0.0008 + spread 0.0004 +
maximum participation impact 0.0004; source-specific extra rates excluded).

The v2 nominal gross bound is

```text
g1 <= max(0, (1 - r*g0)/(1+r))
```

Because no same-mark turnover can exceed `g0 + g1`, this leaves enough equity
at the decision mark to pay the configured total fill cost. **Equity is not free
cash:** native fills process symbols in ascending index, so a new lower-index buy
may execute before any higher-index sale. Independently of the gross bound, for
a uniform multiplier `f` on the full effective target `t`, require:

```text
B(f) = sum_spot(max(0, f*t_i - current_i))
B(f) * (1 + r) <= max(0, min(cash_before / equity, 1 - g0))
```

The `B(f)` constraint reserves actual current cash for all newly bought positions
and their configured buy fees **without counting future sale proceeds**. Solve for
the greatest admissible `f` by monotonic bisection; an all-sell exit can proceed
without prior free cash because its own proceeds pay its fees. The first gross
bound still reserves *total* worst-configured buy-and-sell costs. Whenever either
constraint binds, clear all quantity-HOLDs and uniformly scale the full effective
target, then invert the risk factor once. Pure HOLDs and genuine exact no-trade
pre-images do not incur new orders. Relative `np.allclose` tolerance is forbidden
(`rtol=0`).

**Execution-time safety is a separate hard gate.** In each observer ledger interval:

- Reject non-finite cash/cost totals, non-monotone cumulative costs, or discontinuous
  cost boundaries. Track only *unabsorbed* fee overdraft: add the current interval
  cost, then cap the allowance by the current negative cash; any cash recovery
  extinguishes the corresponding allowance. Older paid fees are never recycled.
- A long-only spot interval cannot incur short-inventory borrowing charges.
  Require a **finite, exactly zero** native `interval_borrow_cost`; any nonzero
  debit or malformed value refuses the software run before an OOS cell can be
  claimed valid. This does not infer financing timing or rewrite historical v1
  receipts.
- Inspect native ordered `OrderEvent` fills (not merely period-end cash).
  The current event schema does not carry each fill's execution fee. To ensure a
  **conservative lower bound**, reserve all negative non-trade cashflows from
  `cash_before` before processing orders, but deduct the **full verified
  interval execution cost immediately after the first filled order's signed
  notional**. Subsequent fills receive no additional cost deduction, because
  all nonnegative fees are already charged in the lower bound. If the interval
  has no fill, reserve any reported cost at its start. Omit positive non-trade
  credits until after the audit. This respects the native book's execution
  order: sell proceeds arrive before that fill's cost, but later sell proceeds
  cannot fund an earlier purchase. It prevents the known false INVALID on a
  cash-safe passive exit that begins with a sale, while remaining conservative
  on buy-first execution. It **can still falsely invalidate some cash-safe
  paths**, because fees actually paid on later fills are charged at the first
  fill in the lower bound. Such INVALID results are not economic findings.
  Per-fill cost/cash evidence remains necessary to eliminate these residual
  false positives. Unexpected perp fills or contradictory fee totals fail closed.
- `min(cash_after, conservative ordered fill cash)` below
  `-0.000001 × initial_capital` is `negative_cash_borrowing` and INVALID **even
  if the deficit comes solely from fees**. The separately reported maximum
  borrowing *net of unabsorbed fees* remains diagnostic, not financing credit.
- Forward price movement, unfavorable fill sequencing, unfilled sells, unmodeled
  fee additions and pending orders may exceed the nominal decision reserve. Such
  cells must fail the raw-cash gate; there is no post-hoc retry or rescue.

Synthetic RED/GREEN oracles include recovered cash followed by a new buy, partial
fee recovery, rising fees, nonfinite/discontinuous ledger costs, tiny real buys that
would evade default NumPy relative tolerance, and drawdown/partial-fill replay.
They do not prove real-data source provenance, achieved profitability or all
possible next-bar fills.

## Dependency and approval boundary

The narrow microscopic-trade regression in [PR #1313](https://github.com/shuntatsu/trade_rl/pull/1313)
is a source dependency, not an independent cash validity approval. This v2 software
revision is stacked on it until it is independently reviewed and merged. Any v2
protocol edit changes its canonical SHA-256 digest; all exact-head tests/reviews
are renewed. AGY's own already-executed `agy_anchor_realistic_v1` remains separate;
its import of the shared `_cash_extremes` diagnostic must be evaluated independently
for a future authorized protocol revision, with no retroactive change to its INVALID
BACKCAST_2020 verdict.
