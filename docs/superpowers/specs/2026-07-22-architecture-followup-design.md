# Architecture Follow-up Remediation Design

## Goal

Remove the remaining runtime indirection and duplicated critical paths identified by the post-merge architecture audit while preserving public behavior and research evidence identities.

## Scope

This change addresses the five highest-priority findings:

1. Remove package-initializer monkey patching from simulation, telemetry, Studio, and catalog.
2. Route normal environment target execution and compatibility execution through one shared stateful target helper.
3. Make regime-balanced and stress-tail episode sampling fail closed when the selected global feature is unavailable for every candidate.
4. Remove duplicated catalog canonical-JSON and sealed-test SQL implementations while preserving compatibility APIs through explicit delegates.
5. Strengthen PostgreSQL workflow evidence with exact-head checkout, main-push execution, and complete path filters.

## Architecture

### Explicit public facades

Package initializers may re-export canonical classes and functions but must not mutate already imported modules with `setattr`. Callers that require the maintained compatibility executor import it from `trade_rl.simulation`; the original implementation in `trade_rl.simulation.execution` remains an internal base until a later removal migration.

Telemetry and Studio strict behavior becomes the implementation imported by consumers directly. Compatibility modules may re-export symbols, but import order must not alter runtime behavior.

### Single stateful target path

`EnvironmentExecutionCoordinator.execute_target()` constructs the environment-specific target identity and delegates target reconciliation and order execution to `execute_target_statefully()`. The shared helper accepts the already supported TIF and expiry values and remains the only implementation of target-to-order reconciliation.

### Fail-closed episode sampling

For `regime_balanced` and `stress_tail`, at least one candidate must have the configured global feature available. When none are available, sampling raises `ValueError` rather than interpreting placeholder values. Uniform sampling remains unchanged.

### Catalog boundaries

`trade_rl.catalog.contracts` imports canonical JSON from `trade_rl.domain.canonical_json`; its local duplicate encoder is removed. `PostgresArtifactCatalog.reserve_sealed_test_access()` remains as an explicit compatibility delegate that instantiates `PostgresSealedTestReservationStore`; the SQL exists only in the dedicated adapter.

### CI evidence

The PostgreSQL workflow runs for pull requests and pushes to `main`, checks out `${{ github.event.pull_request.head.sha || github.sha }}`, and watches catalog, sealed-test evaluation, workflow construction, CLI, compose, and relevant tests.

## Error handling

All new boundaries fail closed. Missing regime data raises an explanatory `ValueError`; malformed targets continue to fail before execution; catalog reservation conflicts retain the existing user-facing error; CI checkout remains read-only except for ordinary test execution.

## Testing

Tests must first fail on the current architecture and then prove:

- package initializers contain no runtime `setattr` replacement;
- maintained imports resolve explicitly;
- environment execution delegates to the shared helper;
- unavailable regime/stress features cannot influence sampling;
- canonical JSON and sealed-test SQL have single owners;
- PostgreSQL workflow runs on exact heads and `main` pushes.

The full repository CI, PostgreSQL integration workflow, critical branch-coverage ratchet, Windows compatibility, Studio build, and training-image probe must pass on the final exact head.

## Non-goals

This change does not add direct exchange routing, claim exchange-equivalent fills, change reward semantics, alter policy architecture, or authorize production deployment. Production remains `NO-GO`.