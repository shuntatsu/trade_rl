# Explicit reduce-only exits

Status: Active implementation; Stage A integrated, Stage B in progress

## Objective and evidence

Implement correctly bounded closing orders so directional strategies can realize
their exits under an explicitly supported execution profile. A profitable marked
balance with residual positions is not a qualified strategy. This work is one
execution capability within the broader profit objective; it does not establish
that PPO, a passive control, or any strategy has qualified.

A separately frozen diagnostic replay of the original constant-long control
completed all 17544 reused development intervals with the accepted-lot accounting
correction, unchanged strategy, costs and risk. Its protocol is
`666c864868f09dce66e3c1ae87dea934756ed00380177135aaeda08404ccee22`.
BTC and BNB ended at exact zero, while ETH 0.004, XRP 0.1 and ADA 1 remained.
The raw order trace records minimum-notional rejection for these final exits.
The diagnostic is not an execution-profile experiment or a winner selection.

Binance's [USD-M error contract](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/error-code)
documents a minimum-notional exception for an explicitly reduce-only order.
Its [order contract](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade)
restricts the flag to one-way mode and distinguishes it from close-all orders.
These documents justify a separate supported capability, not a universal waiver
for any order whose sign opposes a position. No signed request is needed for this
implementation and synthetic verification.

## Decisions

- Use an explicit, identity-bound order attribute, initially for MARKET orders.
  A negative target or risk-controller mask alone does not establish order semantics.
- Preserve ordinary-order identities and behavior by default. Historical study
  evidence and the frozen prospective carry worktree remain unchanged.
- Build inventory safety before enabling any minimum-notional exception. This
  separates the reduction invariant from venue assumptions.
- Bind any later exception to an explicit profile, dataset and symbol roster,
  one-way account mode, supported order type and quantity-rule evidence.
  `MarketDataset` currently lacks these venue/account assertions.
- Keep the compounding drawdown projection investigation separate. This change
  does not modify risk scale, thresholds, desired quantities, reward or training.

Rejected alternatives: setting minimum notional to zero for an entire symbol;
silently converting ordinary closing orders into exempt orders; promoting true
sub-lot quantities; writing off dust; or reporting a successful terminal close
without costed fills.

## Stage A: explicit order safety

Add a validated boolean `reduce_only` attribute with default false. Initially
only MARKET orders support true. True must change order identity and survive
persistence and event evidence; old mappings without the attribute restore as
ordinary orders. A changed flag with an unchanged identity must be rejected on
read-back. Ordinary order IDs and execution-policy identities stay unchanged.

Admission requires an existing opposite signed position and a reducing quantity;
flat, increasing and oversized reduce-only requests are rejected conservatively.
Existing tradability, side permission, quantity, funding, cost and margin rules
still apply. Stage A does not grant a minimum-notional exception.

The stateful caller supplies actual exact inventory separately from the pending
order projection used for economic admission. Unfilled openings cannot supply
inventory for a reduce-only request.

Admission's projected book is insufficient as the final inventory authority.
Symbol allocation must enforce the restriction in actual deterministic fill
order against exact inventory, including previous accepted fills in that batch.
A later reduce-only fill is bounded by the opposite position that remains;
it cannot reopen a flat position, increase it, or cross zero. Competing ordinary
orders must be included in this inventory sequence. An exhausted closing order
is explicitly expired with a reason; rejected or unfilled quantity is never
charged as a fill. A genuine sub-lot remainder remains visible.

Any safety clipping must preserve capacity conservation and accepted integer-lot
evidence. Unused closing capacity remains unused or is allocated deterministically
to later requests under the existing priority contract. Partial fills retain
the same attribute after serialization and restart.

A synchronous inventory check before each closing fill also covers margin
handling that flattens the account after an earlier fill. A newly exhausted or
insufficient position expires the remainder without execution. Such released
capacity stays unused; later orders keep their original reservations. Capacity
events and final totals are reconstructed from actual fills.

Stage A does not change automatic target reconciliation. Later activation must
cancel and replace an ordinary residual when the required order semantics change,
even if its remaining quantity happens to match the new target.

The implemented Stage A readers accept missing `reduce_only` as false. Explicit
intent/event canonical payloads omit false; generic dataclass serialization adds
the field. This preserves ordinary order IDs and canonical event evidence, while
remaining backward-readable rather than promising identical generic JSON bytes.

## Stage B: explicit execution profile

Before any exemption reaches an economic replay, implement and review a profile
that binds the selected dataset and symbol identities, one-way mode, MARKET-only
support and the retained source-rule evidence. It must distinguish:

- minimum order quantity and maximum order quantity;
- `LOT_SIZE` and `MARKET_LOT_SIZE` step constraints;
- minimum notional for ordinary orders;
- the documented exception for an actual reduce-only order.

The current generic metadata drops market quantity bounds, so existing dataset
lot/minimum arrays alone do not prove this profile. Use an additive explicit
adapter/profile rather than silently changing existing data contracts or studies.
Current rule evidence applied to old bars remains a disclosed historical
assumption; it is not a point-in-time reconstruction of missing past filters.

Enforce the per-order exception both at admission and at fill allocation.
Opening and crossing orders retain ordinary minimum notional. An automatic
reversal must either retain ordinary semantics or explicitly finish a closing
order before submitting a separately qualified opening order; its opening
quantity must never inherit an exemption. Freeze the selected behavior before
the subsequent comparison.

### Stage B implementation decisions

An immutable `data/market_order_rules.py` profile will carry the full ordered
dataset symbol roster and Dataset ID, selected symbol indices/names, the declared
Binance USD-M perpetual venue and one-way account model, source URI/retrieval time
and raw SHA-256, derived MARKET quantity bounds and combined lot increments, and
an explicit `reduce_only_exits` boolean. The boolean controls automatic same-side
reductions and the minimum-notional exception together. False provides a matched
ordinary-order profile for the later comparison. The retained raw exchange-info
bytes remain the authority for the separate LOT_SIZE and MARKET_LOT_SIZE fields.

The Binance adapter will reuse the strict current-rule parser, validate raw
response bytes/digest/URI and the selected TRADING USDT perpetual rows, and publish
a write-once profile plus its raw source. Loading with an externally supplied
profile digest must rederive the rules from those bytes and compare the entire
profile. The supported in-memory construction paths are the Binance builder and
loader only; the immutable data value rejects public construction and dataclass
replacement without the private factory capability. This is an API contract,
not protection from arbitrary Python reflection. Selected dataset contract
multipliers must equal one: source quantity filters are base-asset quantities.
Dataset venue/account identity is a declared research assumption, since
MarketDataset has no such fields; an account configuration has not been queried.
The profile explicitly records current-snapshot historical application. Unselected
symbols remain under existing rules and cannot inherit the exemption.

MarketExecutor accepts this optional profile, checks its dataset/symbol binding
and MARKET configuration, and adds the complete profile and rule-stress identities
to its execution-policy digest. Omission preserves the default policy digest.
Selected symbols use the intersection of dataset/runtime lot constraints and the
source MARKET lot constraints. Ordinary minimum notional remains at least the
existing dataset/runtime and source floor; an eligible reduce-only order waives
the declared venue minimum only, retaining any explicit runtime minimum floor.
The profile asserts that the selected dataset minimum-notional arrays represent
venue constraints. This is part of its disclosed execution assumption, not a
silent reinterpretation of old artifacts.

Intersect dataset, runtime and source grids separately using rational decimal
LCM; do not first collapse dataset/runtime grids with max. Stress adds a further
grid equal to that common quantum times the lot stress factor, intersected with
the original common grid. For example .002/.003/.003 gives .006; a 1.5 stress
adds .009 and yields .018. Reject a final quantum that cannot round-trip through
the supported decimal float representation. Both ordinary and retained runtime
minimum-notional floors receive the configured notional stress factor.
Profile lot-burden diagnostics report the actual stressed/nominal grid ratio,
which can exceed the configured multiplicative factor after intersection.

Admission and per-request allocation both enforce the applicable minimum notional
and quantity bounds. Requests above the source maximum are rejected rather than
silently split; capacity/position clipping cannot produce a fill below the source
minimum quantity. Unsupported selected-symbol order types fail closed. Tick,
tradability, side permissions, accounting, funding and margin retain their current
owners; this profile does not claim to reproduce every live exchange filter.
The compatibility `liquidate_at_close` shortcut rejects profile mode; closing must
use explicit stateful orders so this shortcut cannot bypass profile constraints.

Automatic reconciliation enables reduce-only only for a same-side decrease or a
target of zero, using exact current inventory. Reversals retain ordinary semantics
and their minimum-notional requirement. In explicit profile mode, an equal residual
quantity is reusable only when its reduce-only flag and execution-policy identity
also match; otherwise cancel and replace it. No deferred reversal state or implicit
opening exemption is introduced. Default reconciliation remains unchanged.
The order request is still a float: an exact closing delta is projected
conservatively toward zero when necessary. This may leave an executable lot for
unusual non-representable inventories; it never promotes a sub-lot remainder or
claims arbitrary one-order flattening. The exact inventory remains authoritative.

Stage B verification includes false/true profile pairs, omitted-profile golden
compatibility, raw/rule/profile tampering, mixed selected/unselected symbols,
long/short zero and partial reductions, ordinary reversals, stale semantic order
replacement, quantity-limit admission and clipped-fill limits, and fill/fee/capacity
evidence. No financial replay runs until this implementation and its fresh protocol
are frozen separately from the preserved diagnostic.

## Stage C: fresh economic evidence

Freeze the source, runtime, data, full candidate/control roster and saved model
hashes or fitting budgets before replay. Keep both years, full clock, costs,
drawdown, actual flatness and stress criteria visible. A single profitable
control cannot be promoted by the original candidate-family selector, and a
best PPO seed cannot replace a family gate. Preserve the original failed study,
the unchanged diagnostic and the new profile comparison separately.

Record actual fills, inventories, cash, funding and final residual reasons so an
independent implementation can reconcile the result. If no candidate qualifies,
retain that outcome. The fixed 90-day carry observation continues independently.

## Verification

Stage A requires synthetic long/short closes, flat/increasing/oversized rejection,
competing and differently prioritized orders, stale partial orders, restart,
strict sub-lot preservation, lot/capacity/cost conservation, identity tampering,
unsupported types and unchanged-default characterization. Stage B adds mixed
eligible/ineligible instruments, quantity bounds, exemption scope, rule/profile
tampering, and reconciliation semantics. Both require independent review and
the full repository quality gate on the exact head containing current main.
