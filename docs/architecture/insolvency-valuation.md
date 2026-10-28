# Native insolvency valuation

`MarketExecutor` has two explicit insolvency valuation modes. The default
`floor_zero` mode preserves the historical execution-policy digest and behavior:
an economic termination flattens quantities and sets cash to the non-negative
pre-flatten portfolio value. Existing callers therefore keep their prior
identity and accounting semantics.

The opt-in `retain_debt` mode binds its choice into
`MarketExecutor.execution_policy_digest`, which also binds every `OrderIntent`
identity. Intents prepared under the other mode are rejected by native order
admission before a fill. On the first insolvent state, the native executor
cancels active orders, zeros quantities and margin, preserves the signed
pre-flatten portfolio value as cash, and stops advancing the execution
interval. It does not process later bars or accrue later cash interest, funding,
or borrow. Cash interest or borrow already applied before an insolvency is
detected remains in the account; the mode does not reorder the existing
within-interval accounting phases.

The returned `BookState` is the source of the retained signed terminal NAV.
Existing return fields retain their numerical lower bound near -100% so their
log-return representation remains finite; they do not encode debt beyond that
bound. A result consumer that needs the amount of retained debt must inspect
the native terminal book rather than infer it from the bounded return fields.

The retained-debt book is terminal output, not a resumable starting state:
execution still rejects non-positive starting equity, before advancing a bar or
mutating the returned account/order-book snapshots.

This mode changes execution/accounting semantics only when explicitly selected.
It does not qualify the underlying data, historical venue rules, insolvency
model, or research result, and it does not make an insolvent endpoint valid for
selection or deployment.
