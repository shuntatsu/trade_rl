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
- checked-in factual Binance USDⓈ-M BTCUSDT 15-minute representative windows selected at time quantiles `0.1`, `0.5`, and `0.9` from the canonical `2021-01-01T00:00:00Z` to `2026-07-01T00:00:00Z` / 192,672-bar range. The fixtures bind OHLC, mark/index close values, quote-notional volume, and factual funding data and are replayed through fresh Nautilus children;
- factual representative-window Stage A evidence persisted through the content-addressed promotion store, including exact structural closure, funding closure from actual candidate position snapshots, cost-neutral economic comparison, source identity, and an aggregate evidence artifact over all three time-quantile windows;
- canonical isolated-process runtime-performance evidence over deterministic synthetic BTCUSDT CPU PPO workloads of `8`, `32`, and `128` timesteps. The artifact binds the benchmark source digest, legacy and streaming-Nautilus elapsed throughput, self/child/process-tree peak RSS, and process count. A verified CI run recorded a worst elapsed slowdown ratio of about `3.62x` and a worst process-tree RSS ratio of about `1.55x`. The artifact remains explicitly `performance_approved=false` because no reviewed production threshold is bound to it;
- an explicit representative persisted-artifact benchmark path. The parent accepts only an already-published canonical market dataset artifact, validates its file closure, exact `BTCUSDT` symbol, and minimum bar count, binds the published artifact digest into the benchmark source identity, and passes only the resolved artifact root across the isolated worker boundary. Each legacy and streaming worker revalidates and reloads that same canonical artifact before training. The no-argument synthetic CI benchmark remains unchanged, and no reviewed representative persisted run is claimed by this implementation alone;
- immutable runtime-performance evidence and approval-policy sidecars with content digests, deterministic threshold assessment, and explicit review references. Approved evidence can only be materialized from observational evidence after a reviewed policy passes; already-approved evidence is rejected rather than rebound to a different policy;
- immutable runtime-promotion reports whose digest is separately bound into the signed `SelectionProposal` alongside the walk-forward and gate-evidence identities. The proposal is the cryptographic join point; the runtime report is not injected into the generic walk-forward or sealed-test APIs;
- selected-final training that revalidates the signed proposal and retained runtime-promotion report, preserves both inside the training artifact, and binds exported model artifacts into the same `TrainingRunManifest` file closure;
- training and full-research promotion paths that retain and revalidate the approved runtime-performance evidence and its reviewed policy sidecar, including policy/evidence digest binding and deterministic threshold reassessment before authoritative promotion or finalization;
- finalization and release packaging that re-read and revalidate the proposal/runtime-report binding before approval or serving-bundle materialization. New artifact generations fail closed on missing, mismatched, or unauthorized sidecars while sidecar-less historical selected-final artifacts retain explicit backward compatibility;
- serving bundles whose existing file-closure digest carries the retained proposal, runtime-promotion report, export artifacts, and selected-final training evidence without introducing a second execution-authority field in the serving schema;
- Studio Evidence Explorer read-only reporting for runtime-promotion evidence at the research-run level, including required/missing/unbound evidence states;
- Studio Serving Monitor read-only revalidation of the proposal digest, dataset identity, walk-forward identity, gate-evidence identity, and proposal-to-runtime-report digest against the serving bundle before showing the promotion evidence as verified. Legacy sidecar-less selected-final bundles are surfaced as a warning rather than being falsely promoted.

The sealed outer-test remains controlled by the existing one-shot walk-forward ledger and is closed into the walk-forward run identity. The signed `SelectionProposal` joins that walk-forward identity to the gate evidence and runtime-promotion report digest; selected-final training then rechecks its execution replay against the proposal's walk-forward digest. This preserves one evidence chain without making the generic walk-forward implementation depend on NautilusTrader.

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

This removes the previous prefix-replay work-growth pattern from the RL observer, but it does not by itself constitute performance approval. CI now records isolated legacy-versus-streaming measurements across `8`, `32`, and `128` deterministic synthetic BTCUSDT CPU PPO workloads, including process-tree memory rather than parent-only RSS. A canonical persisted-artifact benchmark path is now implemented as a separate explicit opt-in. Authority promotion still requires a retained run on a reviewed representative persisted/catalog artifact plus an explicit reviewed slowdown/process-tree-memory threshold policy; implementing the path itself does not satisfy either requirement.

## Representative persisted performance benchmark

The parent benchmark can now measure the same isolated legacy-versus-streaming training path against an already-published canonical market dataset artifact:

```bash
uv run python scripts/nautilus_training_throughput_benchmark.py \
  --dataset-artifact /path/to/published-market-dataset \
  --timesteps 8 \
  --timesteps 32 \
  --timesteps 128 \
  --output var/nautilus/representative-performance.json
```

`--dataset-artifact` accepts the directory produced by the canonical market-dataset artifact publisher, not a raw NPZ/JSON file or an arbitrary digest. The artifact must validate successfully, contain exactly `BTCUSDT`, and contain at least `max(80, max(requested_timesteps) + 32)` bars. The parent binds the published artifact digest into `source_digest`; each isolated worker receives only the resolved directory path and revalidates/reloads the artifact itself before the training timer starts. The emitted runtime-performance evidence remains `performance_approved=false`; a reviewed policy must independently pass before approved evidence may be materialized through the existing approval path.

The repository does not retain a reviewed representative persisted/catalog benchmark artifact or result yet. The command above documents the implemented execution path only and must not be read as evidence that the representative authority-promotion run has occurred.

## Authority modes

Trade RL recognizes three execution authority modes:

1. `legacy_authoritative`
2. `dual_shadow`
3. `nautilus_authoritative`

`legacy_authoritative` remains the fail-closed default. `dual_shadow` requires successful capability, causal bridge, funding, and terminal-flat evidence. `nautilus_authoritative` additionally requires representative historical parity, deterministic replay, and explicit performance approval.

Passing capability, historical synthetic evidence, factual representative-window differential evidence, persisted structural/economic comparison contracts, streaming parity, training smoke, or a signed promotion-report chain does not automatically change the runtime mode. The evidence chain authorizes only what its persisted decision permits; production authority remains separately gated.

## Remaining work before authority promotion

- execute the now-implemented `--dataset-artifact` runtime-performance benchmark on a reviewed representative persisted/catalog artifact, retain the resulting observational evidence, define and externally review an explicit slowdown/process-tree-memory threshold policy, retain that policy beside the evidence, and keep `performance_approved=false` until the reviewed policy passes and approved evidence is materialized through the fail-closed approval path;
- execute and retain the now-wired signed promotion chain on the same reviewed representative persisted/catalog dataset, including reviewed authorization/confirmation artifacts. The code path already spans the sealed walk-forward identity, signed proposal, selected-final training and export closure, release packaging, Studio Evidence Explorer, and Studio Serving Monitor;
- retain `NO-GO` until production execution, reconciliation, secrets, kill switch, and operational controls are separately implemented and authorized.

The repository still intentionally excludes local runtime data and generated artifacts (`/data/`, `/var/`, and `*.npz`). The checked-in factual three-window Binance fixtures are bounded deterministic CI evidence for the representative differential/evidence pipeline; they do not replace a reviewed representative local/catalog run or signed authorization/confirmation artifacts for authority promotion.

## Upstream relationship

NautilusTrader is an external project developed by Nautech Systems. Trade RL is independent and is not affiliated with, endorsed by, sponsored by, or an official work of Nautech Systems. See `LICENSES/THIRD_PARTY_NOTICES.md` and `docs/LICENSING.md`.
