# Stateful Execution Decomposition Verification

Date: 2026-07-23

## Scope

This record verifies PR #107, the `AUD-SIM-001` remediation which replaces the 614-line `execute_stateful_orders()` policy monolith with a bounded orchestration function and four focused runtime boundaries:

- `StatefulExecutionRuntime` for invocation-local mutable state, ordered evidence, accumulators, and final result construction;
- `StatefulBarLifecycle` for split and inactive-asset handling, open revaluation, carry, mark-to-market, margin, and insolvency phases;
- `StatefulOrderTransitionProcessor` for expiry, latency, admission, eligibility, projected reservation, and post-attempt remainder expiry;
- `StatefulSymbolFillProcessor` for OHLC path selection, trigger evaluation, shared symbol capacity, fill application, costs, and fill evidence.

The public `StatefulExecutionResult` remains in `trade_rl.simulation.stateful_execution`, and `MarketExecutor.execute_orders()` keeps its existing public contract.

## Architecture boundary

The maintained orchestration now:

1. validates bars, indices, and book shape;
2. creates one invocation-local runtime;
3. submits caller-ordered intents and initializes metrics;
4. applies the existing per-bar order of lifecycle begin, transition preparation, symbol fills, remainder expiry, and lifecycle finish;
5. constructs the unchanged public result type from the runtime payload.

The architecture contract requires all four service modules, limits `execute_stateful_orders()` to at most 180 source lines, forbids low-level admission, path, capacity, dividend, and cash-interest calls in the orchestration function, and keeps the public result class in the orchestration module.

The resulting function spans 111 source lines, down from 614.

## TDD RED evidence

The formatted architecture contract was executed against the pre-service implementation on temporary PR #109.

RED head:

- `3ee45c164261b7d663c919ca356212221c7c638a`

GitHub Actions CI run `29962113726`: expected failure.

The run passed Studio, fixed viewport, workflow security, Ruff, format, Mypy, Import Linter, dead-code analysis, Serving smoke, Ubuntu compatibility, Windows compatibility, and the complete training-image probe. The full suite failed only in the new architecture contract:

- `stateful_runtime.py` did not exist;
- `execute_stateful_orders()` spanned lines 228–841, or 614 source lines, exceeding the 180-line orchestration bound.

Result:

- `2 failed, 1207 passed, 2 skipped, 11 warnings`;
- total coverage remained above the repository minimum at `83.45%`.

RED artifact:

- Pytest diagnostics ID `8546443797`;
- digest `sha256:03dac7b42c0cb5945f9d936d174cf335f35cdacfd57696cfaeef866f1e23e12f`.

The temporary RED PR was closed without merge.

## Pre-refactor characterization evidence

The complete result payload was captured from unchanged `main` execution code before the decomposition. The temporary capture branch added only a diagnostic script and workflow; it did not modify any simulation module.

Capture run:

- run `29961814877`: success;
- artifact ID `8546269274`;
- artifact digest `sha256:9e541690648694bf6b138a959d3b60350493b6ba963836c9b32fdb509d3912c8`.

The canonical payload contains the final `BookState`, `OrderBookState`, all 13 ordered `OrderEvent` payloads, all three `SymbolCapacityEvidence` payloads, scalar interval metrics, counters, and per-symbol arrays. Its permanent regression digest is:

- `sha256:3856e696c998e727c78690222d418e070c71eeb56f7f747f0932a17eb8ff2cc2`.

The scenario includes market, delayed limit, and stop-market orders, constrained processing-bar volume, non-zero fees, spread and impact, latency, stop triggering, capacity contention, partial-fill carry, and a later below-lot-size no-fill.

The fixed event sequence is:

```text
submitted, submitted, submitted,
eligible, eligible, latency_wait,
triggered, filled, no_fill,
eligible, filled, partial_fill, no_fill
```

Focused service regressions additionally cover cloned caller-book ownership, ordered submission evidence, non-positive-equity fail-closed behavior, partial-fill carry-disabled expiry, and zero-capacity no-fill preservation.

The temporary capture PR was closed without merge.

## Implementation exact-head verification

Strengthened implementation-and-test head:

- `b4411caf2025238c71dd52f6661c20d4ab1cfa05`.

GitHub Actions CI run `29962031316`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed viewport verification: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter and architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1223 passed, 2 skipped, 11 warnings`;
- total coverage: `83.56%`;
- total branch coverage: `4872 / 6916 = 70.45%`;
- stateful execution service branches: `67 / 72 = 93.06%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29962031311`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

## Implementation artifacts

- Pytest diagnostics: ID `8546418509`, digest `sha256:7ab119a4231d308bf370cfde347cafae8ceeba53f4f7e8fed66f7a0469eb0bac`;
- architecture diagnostics: ID `8546383047`, digest `sha256:576698ecb0985d4f43c090bc0d0527193f16bde8b59643c5e4b0157ce7aeaadc`;
- static diagnostics: ID `8546382435`, digest `sha256:337f9dadb63374796d25e4184ce7f5fef3a6e5f61423ddd67c5f1c0e1b55a104`;
- training-image evidence: ID `8546376468`, digest `sha256:6742c83360f26f93ec5ed885caaffcaba3dfcc6f8d1844c0ea90fc3ac8507700`;
- Studio layout diagnostics: ID `8546373192`, digest `sha256:2e286cd5ae1835097e2e9020d5710d334e364d3a5e8a2f5b0df32774fa1daeea`;
- Windows compatibility: ID `8546368470`, digest `sha256:0edf98daaa535e1d365f99f160421d84d3cf83a1589ba77bc59fe2ae93ef7a75`;
- Ubuntu compatibility: ID `8546364752`, digest `sha256:41d028025c67028605e288a05dd6b3551ce0547526f9d12533ae8bcd6d6647b3`.

## Coverage ratchet

The initial implementation run measured `64 / 72 = 88.89%` aggregate branch coverage for the four services. Additional service and complete-payload characterization tests increased the measured result to `67 / 72 = 93.06%`.

The configured `stateful_execution_services` threshold is therefore `93.0%`, the observed percentage rounded down to one decimal place. No existing threshold was reduced.

## Review result

The effective implementation diff preserves:

- validation exception messages;
- caller intent order and active-order stable sorting;
- every event sequence and event field;
- OHLC path and trigger semantics;
- shared processing-bar capacity and allocation order;
- tick, lot, minimum-notional, fee, spread, impact, and slippage behavior;
- split, inactive asset, dividend, cash interest, funding, borrow, mark-to-market, margin, and insolvency order;
- final books, interval metrics, counters, and per-symbol arrays;
- public result and evidence schema identities.

Mutable execution state is centralized in the invocation-local runtime. The phase services introduce no global, persisted, or cross-invocation state. No temporary workflow, capture script, generated fixture file, or duplicate legacy implementation is present in PR #107.

No critical or important review issue remains at the implementation head.

This verification note and the tightened `93.0%` threshold create a later exact head. Normal CI and PostgreSQL Catalog must pass again on that documentation-complete head before merge.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No profitability or exchange-equivalent fill claim is introduced.
- No primary execution evidence schema changes.
