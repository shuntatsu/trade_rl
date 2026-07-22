# Architecture Audit Remediation — 2026-07-22

## Scope

This record closes the behavioral and dependency-boundary findings identified by `2026-07-22-documentation-and-architecture-audit.md`. It does not change the project capability boundary: direct exchange routing remains unavailable, production remains `NO-GO`, and repository verification is not profitability evidence.

## Remediated findings

### Stateful execution parity

The compatibility `MarketExecutor.execute_interval()` path now delegates to the same target reconciliation, persistent order lifecycle, processing-bar liquidity allocation, and `BookState` accounting used by the RL environment. Compatibility residual orders persist only when the caller chains the exact returned `BookState` through the same executor; unrelated books start with an empty compatibility order state.

Baseline reward pre-roll creates an isolated executor and chains its returned books, so latency, partial-fill carry, costs, funding, borrow, corporate actions, and margin use the same economic implementation as ordinary baseline transitions without leaking pending state into the episode.

Regression tests cover processing-bar volume, explicit stateful-path parity, and one residual order carried across chained compatibility calls.

### Telemetry dependency boundary

`trade_rl.telemetry` is now an explicit Import Linter layer below `artifacts` and above `domain`. A dedicated forbidden-import contract prevents telemetry from importing NumPy, Gymnasium, model frameworks, psycopg, numerical/research layers, or upper application layers.

### Incremental and strict telemetry

Telemetry JSON booleans are required to be actual JSON booleans. Missing, string, integer, or null values fail closed.

Status and cursor reads maintain an atomic sparse sidecar index bound to the JSONL file's device, inode, and indexed byte size. Appended complete lines are scanned from the previous EOF, cursor reads seek from a sparse sequence checkpoint, and replacement, truncation, malformed sidecar data, or identity mismatch rebuilds the index from byte zero. JSONL remains authoritative.

Studio rejects multiple distinct telemetry files that claim the same seed rather than selecting one by discovery order. Existing root-escape, symlink, unknown-job, and seed-identity checks remain in force.

### Canonical JSON identity

Canonical JSON conversion now has one standard-library implementation in `trade_rl.domain.canonical_json`. Artifact serialization and catalog cache-key hashing use the same bytes. Cross-module vectors cover nested mappings, tuple/list sequences, Unicode, finite floats, and non-finite rejection.

### PostgreSQL responsibility split

A dedicated `PostgresSealedTestReservationStore` owns evaluation-specific one-time reservation SQL. `PostgresArtifactCatalog.reserve_sealed_test_access()` remains a temporary compatibility delegate so existing workflow construction does not break. The database uniqueness boundary and migrations are unchanged.

## Partial structural remediation

`ResidualMarketEnv` remains a large Gymnasium facade, but target-to-order reconciliation and compatibility execution have been extracted into focused simulation collaborators. Further decomposition of episode initialization, terminal liquidation, and observation export remains maintainability work; it is no longer required to correct the economic divergence identified by the audit.

## Verification contract

Before merge, one unchanged final head must pass:

- Studio frontend tests, TypeScript, production build, and fixed-viewport validation;
- workflow security, Ruff, format, Mypy, Import Linter, and dead-code reporting;
- recovery and structured Serving smoke;
- full pytest with branch coverage and critical coverage ratchets;
- CLI smoke and Ubuntu/Windows compatibility;
- PostgreSQL Compose, migrations, and unit/integration tests;
- complete training-image build and non-root runtime probe.

Passing these checks establishes integration and contract integrity only. It does not establish trading profitability, exchange-equivalent execution, or authorization for capital deployment.
