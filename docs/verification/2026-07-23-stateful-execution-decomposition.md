# Stateful Execution Decomposition Verification

Date: 2026-07-23

## Scope

This record verifies Draft PR #107, the `AUD-SIM-001` remediation which replaces the approximately 614-line `execute_stateful_orders()` policy monolith with bounded orchestration and four focused runtime boundaries:

- `StatefulExecutionRuntime` for invocation-local mutable state, ordered evidence, accumulators, and final result construction;
- `StatefulBarLifecycle` for split/inactive handling, open revaluation, carry, mark-to-market, margin, and insolvency phases;
- `StatefulOrderTransitionProcessor` for expiry, latency, admission, eligibility, projected reservation, and post-attempt expiry;
- `StatefulSymbolFillProcessor` for OHLC path selection, trigger evaluation, shared symbol capacity, fill application, costs, and fill evidence.

The public `StatefulExecutionResult` remains in `trade_rl.simulation.stateful_execution`, and `MarketExecutor.execute_orders()` keeps its existing public contract.

## Architecture boundary

The maintained orchestration now:

1. validates bars, indices, and book shape;
2. creates one invocation-local runtime;
3. submits caller-ordered intents and initializes metrics;
4. applies the existing per-bar order of lifecycle begin, transition preparation, symbol fills, remainder expiry, and lifecycle finish;
5. constructs the unchanged public result type from the runtime payload.

The architecture contract requires all four service modules, limits `execute_stateful_orders()` to at most 180 source lines, forbids low-level admission/path/capacity/accounting calls in the orchestration function, and keeps the public result class in the orchestration module.

## RED evidence

The architecture contract was committed before the four production service modules existed. At that state it required missing module files and a bounded orchestration function while the maintained implementation still contained the original monolith.

The synchronized RED head was:

- `68d17ad3b6d12176b07eb8d6ce062ca8b441145a`

GitHub Actions run `29960213066` was started for that head but was cancelled automatically when implementation commits advanced the branch. Therefore it is not represented as a completed failing workflow. The source state itself retains the fail-first contract, and the later exact-head runs demonstrate the same contract turning GREEN without weakening its assertions.

## Characterization evidence

A mixed three-bar scenario fixes the complete result payload behind SHA-256:

- `3856e696c998e727c78690222d418e070c71eeb56f7f747f0932a17eb8ff2cc2`

The fixture includes market, delayed limit, and stop-market orders, constrained processing-bar volume, non-zero fees/spread/impact, partial-fill carry, ordered events, capacity evidence, accounting values, per-symbol arrays, and final books.

The expected event sequence is:

```text
submitted, submitted, submitted,
eligible, eligible, latency_wait,
triggered, filled, no_fill,
eligible, filled, partial_fill, no_fill
```

Focused service regressions additionally cover cloned caller-book ownership, ordered submission evidence, non-positive-equity fail-closed behavior, partial-fill carry disabled expiry, and zero-capacity no-fill preservation.

## Implementation exact-head verification

Implementation and measured-coverage head:

- `b4411caf2025238c71dd52f6661c20d4ab1cfa05`

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
- measured service-group threshold: `93.0%`;
- all other critical branch-coverage ratchets: passed;
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

## Exact-head artifacts

- Pytest diagnostics: ID `8546418509`, digest `sha256:7ab119a4231d308bf370cfde347cafae8ceeba53f4f7e8fed66f7a0469eb0bac`;
- architecture diagnostics: ID `8546383047`, digest `sha256:576698ecb0985d4f43c090bc0d0527193f16bde8b59643c5e4b0157ce7aeaadc`;
- static diagnostics: ID `8546382435`, digest `sha256:337f9dadb63374796d25e4184ce7f5fef3a6e5f61423ddd67c5f1c0e1b55a104`;
- training-image evidence: ID `8546376468`, digest `sha256:6742c83360f26f93ec5ed885caaffcaba3dfcc6f8d1844c0ea90fc3ac8507700`;
- Studio layout diagnostics: ID `8546373192`, digest `sha256:2e286cd5ae1835097e2e9020d5710d334e364d3a5e8a2f5b0df32774fa1daeea`;
- Windows compatibility: ID `8546368470`, digest `sha256:0edf98daaa535e1d365f99f160421d84d3cf83a1589ba77bc59fe2ae93ef7a75`;
- Ubuntu compatibility: ID `8546364752`, digest `sha256:41d028025c67028605e288a05dd6b3551ce0547526f9d12533ae8bcd6d6647b3`.

## Review result

The effective implementation diff preserves exception messages, event and capacity ordering, OHLC path semantics, processing-bar capacity, rounding, costs, corporate actions, carry, margin, insolvency, and all public result fields. Mutable execution state is centralized in the invocation-local runtime; phase services do not introduce global or cross-invocation state.

No critical or important review issue remains at the implementation head.

This verification note and the tightened `93.0%` threshold are documentation/configuration-only follow-up changes. The final PR head must pass exact-head CI and PostgreSQL verification again before merge.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No profitability or exchange-equivalent fill claim is introduced.
- No primary execution evidence schema changes.
