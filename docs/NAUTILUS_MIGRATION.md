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

The dedicated `Nautilus Capability` workflow exercises each direct `BacktestEngine` test in an isolated Python process because the pinned runtime has process-global kernel state which does not permit multiple independent engines in one process.

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
- fresh-process historical execution via a JSON request/result subprocess boundary, allowing repeated replays from one parent process without violating the pinned kernel constraint;
- an opt-in RL dual-shadow execution observer that leaves legacy reward/execution authority unchanged, replays each authoritative hybrid target prefix in a fresh Nautilus child, and records structural terminal-position parity;
- a dedicated dual-shadow residual-environment wrapper whose environment identity includes the candidate runtime identity and whose reset synchronizes the candidate from the actual initial book;
- exact-wheel three-step SB3 PPO and Lagrangian PPO training smokes on the single-BTCUSDT dual-shadow runtime;
- legacy-versus-Nautilus dual-shadow conformance for Flat → Long → Flat, safe Flat → Long → Flat → Short → Flat sign reversal, and same-side target increases/reductions to Flat;
- fresh-process deterministic execution digests;
- fail-closed RL dual-shadow symbol validation for the maintained `BTCUSDT` dataset symbol;
- persisted Stage A historical structural differential evidence that binds the authoritative replay/request/dataset identities and compares exact terminal position lots, zero candidate open orders, and canonical funding records without claiming economic fill equivalence.

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

## RL dual-shadow runtime limitation

The current RL observer replays the complete target prefix in a fresh child process on every authoritative hybrid step. This is deliberately conservative and solves the pinned Nautilus process-global kernel constraint, but its work grows quadratically with the number of decision steps in an episode. The three-step PPO/Lagrangian smoke proves functional integration only; it is not a throughput or memory approval.

The maintained execution authority therefore remains legacy. A persistent or streaming episode worker may replace prefix replay only after that exact pinned-runtime capability is separately verified and deterministic parity is preserved.

## Authority modes

Trade RL recognizes three execution authority modes:

1. `legacy_authoritative`
2. `dual_shadow`
3. `nautilus_authoritative`

`legacy_authoritative` remains the fail-closed default. `dual_shadow` requires successful capability, causal bridge, funding, and terminal-flat evidence. `nautilus_authoritative` additionally requires representative historical parity, deterministic replay, and explicit performance approval.

Passing capability, historical synthetic evidence, persisted structural comparison contracts, or three-step training smoke does not automatically change the runtime mode. Selected-final and sealed-test authority must not switch until all promotion evidence is persisted and the workflow integration enforces the promotion decision.

## Remaining work before authority promotion

- run differential dual-shadow replay on representative maintained BTCUSDT historical windows using the persisted structural differential contract and factual market/replay evidence rather than synthetic fixtures;
- define and verify the historical economic comparison contract where Nautilus L1 one-tick spread execution and the accelerated legacy cost model differ, without weakening structural or funding parity checks;
- replace or optimize the current per-step prefix subprocess replay only after an exact `nautilus_trader==1.230.0` streaming/persistent-worker capability test proves the lifecycle safe;
- benchmark memory and throughput against the accelerated legacy training backend before any performance approval;
- persist full historical differential evidence and wire promotion evidence into walk-forward, selected-final, sealed-test, export, and Studio runtime reporting;
- retain `NO-GO` until production execution, reconciliation, secrets, kill switch, and operational controls are separately implemented and authorized.

## Upstream relationship

NautilusTrader is an external project developed by Nautech Systems. Trade RL is independent and is not affiliated with, endorsed by, sponsored by, or an official work of Nautech Systems. See `THIRD_PARTY_NOTICES.md` and `docs/LICENSING.md`.
