# Architecture Follow-up Remediation Verification

## Scope

This verification records the follow-up remediation for the post-merge architecture audit. The implementation:

- removes import-time runtime symbol replacement from simulation, telemetry, Studio, and catalog package initializers;
- makes maintained facades explicit at import sites;
- routes normal environment target execution through the same `execute_target_statefully()` helper as the compatibility API;
- fails closed when regime-balanced or stress-tail episode sampling has no available causal feature values;
- removes the duplicate catalog canonical-JSON encoder and duplicate sealed-test reservation SQL body;
- retains `PostgresArtifactCatalog.reserve_sealed_test_access()` as an explicit compatibility delegate;
- runs PostgreSQL verification on exact pull-request heads and pushes to `main`;
- adds a non-regressing branch-coverage group for the four environment runtime services.

Production remains `NO-GO`. This change does not add direct exchange routing or make a profitability claim.

## TDD RED evidence

The architecture contracts were committed before implementation.

- RED head: `5405370e298d43e8803b03bbf86e79393b4e366a`
- CI run: `29895464240`
- result: failed as intended in `tests/architecture/test_architecture_followup.py`
- failures covered package initializer mutation, duplicated environment target execution, unavailable regime/stress sampling fallback, duplicated catalog ownership, and incomplete PostgreSQL workflow evidence
- pytest diagnostics artifact: `8519849241`
- artifact digest: `sha256:e72f938ab7ebaa699ccd3a32f0db721876ee148f0446caf7a36350d3099cec53`

## GREEN implementation evidence

Implementation and coverage-ratchet head:

`ad32a3c2dfa464c55b974156855488143926cf95`

### GitHub Actions CI

- run: `29897864670`
- conclusion: success
- Studio unit tests, TypeScript checking, production build, and fixed-viewport layout: passed
- workflow-security validation: passed
- Ruff and format: passed
- Mypy: passed
- Import Linter architecture contracts: passed
- dead-code report: passed
- recovery and structured Serving smoke: passed
- full pytest: `1161 passed, 2 skipped, 11 warnings`
- total coverage: `83.44%`
- critical branch-coverage ratchet: passed
- environment runtime group: `54 / 84` branches, `64.29%`, minimum `64.0%`
- CLI smoke: passed
- Ubuntu and Windows compatibility: passed
- complete training-image build and packaged non-root runtime probe: passed

Pytest diagnostics:

- artifact: `8520720949`
- digest: `sha256:cd7d37edb4e911bef8c26ddc1623e9d65d7f84a82b8e7272b533793580dd21f7`

### PostgreSQL Catalog

- run: `29897864747`
- conclusion: success
- exact-head checkout: passed
- Compose validation: passed
- PostgreSQL startup and readiness: passed
- migrations: passed
- catalog unit and integration tests: passed
- cleanup: passed

## Coverage rationale

The environment runtime group starts at the exact measured aggregate branch coverage rather than excluding untested branches or claiming a higher threshold than the suite proves. The ratchet prevents regression while allowing later PRs to raise the minimum with targeted branch tests.

## Compatibility and safety

- Public `trade_rl.simulation.MarketExecutor` continues to resolve to the maintained stateful compatibility executor.
- Telemetry public imports continue to return strict records and indexed readers/writers.
- Studio continues to use the strict duplicate-seed reader explicitly.
- Existing PostgreSQL catalog callers retain the compatibility reservation method.
- Action, observation, reward, execution-evidence, and artifact schema versions are unchanged.
