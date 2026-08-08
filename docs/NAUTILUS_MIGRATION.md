# NautilusTrader migration status

Trade RL is migrating the maintained single-instrument execution authority toward an external NautilusTrader runtime. Production remains **NO-GO** throughout this migration.

## Pinned runtime

- Package: `nautilus_trader`
- Version: `1.230.0`
- Python: 3.12
- Formal CI platform for the current probe: Ubuntu 22.04 x86_64
- Maintained instrument: `BTCUSDT-PERP.BINANCE`
- OMS: NETTING
- Account type: MARGIN

The Nautilus dependency is optional. Importing the Trade RL core, read-only serving, or non-Nautilus tests does not require the wheel. Nautilus-only paths validate the exact runtime version before importing upstream APIs.

## Implemented evidence

The dedicated `Nautilus Capability` workflow exercises direct `BacktestEngine` tests in isolated Python processes because the pinned runtime has process-global kernel state which does not permit multiple independent engines in one process. RL dual-shadow execution uses a spawned episode worker which owns exactly one `BacktestEngine`; the parent process does not construct Nautilus engine state.

Current migration slices cover:

- exact runtime identity guard;
- frozen BTCUSDT perpetual instrument construction and registration;
- causal OHLC event projection where a queued target activates only after the next open quote;
- policy-independent high/low ordering;
- deterministic L1 quote projection and replay;
- Market IOC open and reduce-only close lifecycle;
- framework-neutral `TargetExposureController` with working-order commitment, cancel-before-replace, no-trade band, sign-flip reduce-to-flat, emergency flatten, and HALT behavior;
- exact-wheel partial-fill stale-working cancellation evidence;
- exact canonical fill/economic closure and integer-valued execution trace types;
- canonical funding settlement records with exact tick/lot identity, integer funding minor units, preserved signed position lots, and post-settlement equity evidence;
- Binance USDⓈ-M funding reference accounting based on signed position notional at the funding-boundary mark price and contract multiplier;
- Stage A replay-v4 historical interval evidence binding each factual action and before/after equity observation to its exact consumed `(start, end]` source bars and non-overlapping funding boundaries, without synthesizing fill-level equity;
- causal historical interval timelines where the queued Stage A target activates exactly once after the first consumed open quote, while adjacent bar close/open timestamps may share one physical boundary;
- actual historical `BacktestEngine` target replay over the Stage A source-bar contract, including same-side changes and safe sign reversal through an explicit flat state;
- candidate position snapshots taken from actual Nautilus fills at requested factual boundaries;
- Stage A funding settlement that requires actual candidate position quantity to match the factual funding-boundary quantity before canonical funding evidence is emitted;
- fresh-process full-prefix historical execution via a JSON request/result subprocess boundary, retained as the deterministic reference implementation for differential checks;
- exact pinned-runtime streaming lifecycle evidence using one `BacktestEngine` across successive `run(streaming=True)` batches with `clear_data()` and final `end()`;
- a persistent spawned historical worker which keeps one child PID and one `BacktestEngine` for an episode-like stream while receiving only the new target interval on each step;
- exact cumulative streaming-versus-full-prefix parity for round trip, safe sign reversal, and same-side target changes, including terminal-flat and zero-open-order closure;
- an opt-in RL dual-shadow execution observer that leaves legacy reward/execution authority unchanged and streams authoritative hybrid targets through one Nautilus child per episode;
- a dedicated dual-shadow residual-environment wrapper whose environment identity includes the candidate runtime identity, whose reset synchronizes the candidate from the actual initial book, and whose `close()` releases the episode worker;
- exact-wheel three-step SB3 PPO and Lagrangian PPO training smokes on the single-BTCUSDT streaming dual-shadow runtime;
- legacy-versus-Nautilus dual-shadow conformance for Flat → Long → Flat, safe Flat → Long → Flat → Short → Flat sign reversal, and same-side target increases/reductions to Flat;
- fresh-process deterministic execution digests;
- fail-closed RL dual-shadow symbol validation for the maintained `BTCUSDT` dataset symbol;
- persisted Stage A historical structural differential evidence that binds the authoritative replay/request/dataset identities and compares exact terminal position lots, zero candidate open orders, and canonical funding records without claiming economic fill equivalence;
- exact historical economic normalization that adds each runtime's own non-funding execution-cost burden back to final equity in integer settlement minor units, compares the resulting cost-neutral equity exactly, and never allows that normalization to override structural or funding mismatch;
- an observational CI throughput artifact comparing the accelerated legacy environment with the streaming Nautilus dual-shadow path on the same deterministic synthetic BTCUSDT eight-step CPU PPO fixture. The verified run recorded about `11.47 step/s` for legacy and `1.275 step/s` for streaming dual-shadow, an elapsed-time slowdown ratio of about `8.99x`. This evidence explicitly records `performance_approved=false` and does not define a production promotion threshold.

The passive GTC limit used by the partial-fill capability test is a fixture only. It exists to create an authentic Nautilus working remainder and does not add Limit/GTC as a maintained Trade RL child-order type. Maintained target replacement and flattening continue to use the existing Market IOC adapter.

The legacy high-level target reconciler is not the migration authority for sign reversals because it can represent a positive-to-negative target change as one cross-through-flat delta order. The maintained migration contract instead uses `TargetExposureController`: first reduce the realized position to flat with a reduce-only child order, wait for terminal execution evidence, and only then open the opposite side. Dual-shadow sign-reversal conformance therefore compares the resulting safe child-order lifecycle rather than preserving the legacy cross-through behavior.

## Funding limitation in v1.230.0 Python BacktestEngine

NautilusTrader v1.230.0 contains funding-settlement logic in the Rust simulated exchange, including pending funding rates and `FundingSettlement` processing. The Python low-level `BacktestEngine` used by the current Trade RL integration, however, does not dispatch `FundingRateUpdate`, `MarkPriceUpdate`, or `IndexPriceUpdate` into that simulated exchange. Exact-wheel testing confirms that those data objects can be added to the backtest data stream but no native funding adjustment is produced on the position.

Trade RL therefore does **not** claim native Nautilus funding support for this pinned Python runtime. Funding is settled at the Trade RL Nautilus integration boundary by `CanonicalFundingLedger` using an explicit settlement boundary, signed position quantity, mark/settlement price, contract multiplier, and funding rate. The adapter:

- settles each boundary exactly once;
- requires strictly increasing boundaries;
- debits positive-rate longs and credits positive-rate shorts;
- never changes position quantity;
- performs conservative settlement-currency rounding;
- emits integer minor-unit evidence;
- projects an already-settled boundary into a canonical `funding` execution record with exact price ticks, signed position lots, zero fill quantity, zero fee, and post-settlement equity.

The exact-wheel test deliberately locks the absence of native Python-engine funding settlement. If a future Nautilus release closes the dispatch gap, that test must fail and trigger a reviewed migration from the adapter back to native settlement rather than silently double-settling funding.

## RL dual-shadow runtime

The maintained RL observer no longer replays the complete target prefix on every step. A reset creates one spawned child for the episode; that child owns one pinned `BacktestEngine` and receives only the newly executed interval. The parent remains free of Nautilus engine state. The full-prefix fresh-process runner remains available as a reference implementation, and exact-wheel tests require the streaming worker's cumulative execution to equal that reference across round-trip, safe sign-flip, and same-side target-change scenarios.

This removes the previous prefix-replay work-growth pattern from the RL observer, but it does not by itself constitute performance approval. The current eight-step CPU PPO microbenchmark is intentionally observational and shows substantial overhead relative to legacy execution. Memory behavior, representative workloads, and an explicit reviewed performance threshold remain required before authority promotion.

## Authority modes

Trade RL recognizes three execution authority modes:

1. `legacy_authoritative`
2. `dual_shadow`
3. `nautilus_authoritative`

`legacy_authoritative` remains the fail-closed default. `dual_shadow` requires successful capability, causal bridge, funding, and terminal-flat evidence. `nautilus_authoritative` additionally requires representative historical parity, deterministic replay, and explicit performance approval.

Passing capability, historical synthetic evidence, persisted structural/economic comparison contracts, streaming parity, or training smoke does not automatically change the runtime mode. Selected-final and sealed-test authority must not switch until all promotion evidence is persisted and the workflow integration enforces the promotion decision.

## Remaining work before authority promotion

- run differential dual-shadow replay on persisted representative **real** maintained BTCUSDT historical windows using factual market/replay evidence rather than synthetic fixtures;
- evaluate the implemented structural, funding, and cost-neutral economic comparison contracts on those representative windows and persist the resulting evidence;
- benchmark memory and broader representative training throughput, define an explicit reviewed performance threshold, and keep `performance_approved=false` until that review is complete;
- connect persisted promotion evidence to walk-forward, selected-final, sealed-test, export, and Studio runtime reporting without silently changing the fail-closed authority default;
- retain `NO-GO` until production execution, reconciliation, secrets, kill switch, and operational controls are separately implemented and authorized.

## Upstream relationship

NautilusTrader is an external project developed by Nautech Systems. Trade RL is independent and is not affiliated with, endorsed by, sponsored by, or an official work of Nautech Systems. See `THIRD_PARTY_NOTICES.md` and `docs/LICENSING.md`.
