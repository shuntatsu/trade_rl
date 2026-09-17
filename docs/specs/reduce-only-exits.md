# Explicit reduce-only exits

Status: Active design

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

Stage A does not change automatic target reconciliation. Later activation must
cancel and replace an ordinary residual when the required order semantics change,
even if its remaining quantity happens to match the new target.

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
