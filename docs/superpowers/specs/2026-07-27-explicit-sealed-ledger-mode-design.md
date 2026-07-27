# Explicit Sealed-Test Ledger Mode Design

## Problem

The public market walk-forward workflow currently chooses its outer-test ledger from the ambient `TRADE_RL_DATABASE_URL` environment variable. The same configuration therefore has two different trust semantics:

- without the variable, a process-local in-memory ledger is used and can be reset by restarting the process;
- with the variable, a PostgreSQL-backed durable ledger is used.

The workflow also runs database migration during research execution. This hides an operational prerequisite inside the evaluation path and makes evidence identity independent of the actual ledger semantics.

## Decision

Add an explicit configuration field:

```json
"sealed_test_ledger_mode": "local_exploratory"
```

Supported values:

- `local_exploratory`: process-local one-shot protection for tests, smoke runs, and exploratory research;
- `durable_postgres`: durable cross-process authorization required for maintained formal walk-forward evidence.

The selected mode is part of `MarketWalkForwardConfig.digest_payload()` and therefore part of the experiment-plan identity.

## Runtime behavior

`local_exploratory` always constructs an in-memory `SealedTestLedger`, even if a database URL is present. Ambient environment can no longer upgrade or change the trust semantics.

`durable_postgres` requires a non-empty `TRADE_RL_DATABASE_URL` and fails closed otherwise. It creates `PostgresArtifactCatalog` and `PostgresSealedTestLedger`, but does not call migrations. Database migration remains an explicit operational step through the existing catalog migration command and PostgreSQL workflow.

## Evidence labeling

`walk-forward.json` records:

- `sealed_test_ledger_mode`;
- `sealed_test_ledger_durable`;
- `evidence_tier`, equal to `exploratory_process_local` or `durable_sealed`.

This prevents local process evidence from being described as durable sealed evidence.

## Compatibility

The Python dataclass and parser default to `local_exploratory` for legacy tests and ad-hoc configurations. Maintained full walk-forward examples are updated to `durable_postgres`; the small smoke example remains explicitly `local_exploratory`.

No selection rule, candidate training, return calculation, reward, execution model, or Serving behavior changes.

## Testing

Tests must prove:

1. local mode ignores a configured database URL and never constructs PostgreSQL resources;
2. durable mode fails closed without a database URL;
3. durable mode does not run migrations inside the workflow;
4. changing only the ledger mode changes configuration and experiment-plan identity;
5. published evidence records the selected mode and tier;
6. maintained full configs select durable mode while smoke config selects local mode.
