# Environment Provider Contract Extraction Verification — 2026-07-23

## 1. Scope

This verification covers the behavior-preserving extraction of deterministic
trend, alpha, and factor-provider resolution from `ResidualMarketEnv.__init__()`
into the typed `EnvironmentProviderContractBuilder`.

The extracted boundary owns:

- trend-strategy and `MarketInputResolver` reconciliation;
- compatibility wrapping of causal alpha providers;
- alpha enablement, default `AlphaContract`, and artifact identity resolution;
- static factor-basis validation, `float64` copying, and deterministic identity;
- factor-count inference and static-basis consistency;
- provider minimum-index validation and aggregation.

Portfolio-risk provider selection, action specification, reward construction,
execution construction, observation construction, runtime-service wiring, and
mutable Gymnasium state remain outside this boundary.

Production remains `NO-GO`. This is a maintainability remediation and does not
establish profitability, exchange-equivalent execution, operational authorization,
or direct venue connectivity.

## 2. TDD evidence

The architecture and characterization tests were committed before production
implementation.

The clean RED commit was:

- commit: `663c13adb5b359c5e752e9f95e3d08b0625d50a2`;
- CI run: `29972011796`.

At that commit Ruff, Ruff formatting, Mypy, Import Linter, Ubuntu compatibility,
Windows compatibility, training-image validation, and the maintained recovery and
structured-serving smoke checks passed. Complete pytest collection then failed
because `trade_rl.rl.environment_provider_contract` did not exist.

This demonstrated that the new ownership tests detected the missing boundary rather
than merely confirming a completed implementation.

## 3. Implemented boundary

`trade_rl/rl/environment_provider_contract.py` now owns:

- `AlphaProvider` and `FactorBasisProvider` protocols;
- frozen, slotted `EnvironmentProviderContract`;
- `EnvironmentProviderContractBuilder`;
- resolver/trend reconciliation and compatibility alpha wrapping;
- explicit, provider, and static artifact-digest precedence;
- SHA-256 identity validation;
- static factor-basis shape, finiteness, and copy validation;
- factor-count inference and consistency;
- provider minimum-index validation and maximum aggregation.

`trade_rl.rl.environment` imports the provider protocols from the new module, so
existing imports from `trade_rl.rl.environment` continue to resolve while the
maintained behavior owner is explicit and statically discoverable.

`ResidualMarketEnv.__init__()` performs one provider-builder call and assigns the
resolved fields before continuing with risk, configuration, action, reward,
execution, observation, runtime-service, and mutable-state construction.

The facade no longer owns `_resolve_provider_digest`, `_validated_static_basis`, or
`_resolve_factor_count`, and it no longer constructs `MarketInputResolver` or
manually loops over alpha/factor provider minimum indices.

## 4. Constructor reduction and architecture control

Before this extraction, the constructor source span was 321 lines after PR #114.
It is now 262 lines, a reduction of 59 constructor lines.

Across `trade_rl/rl/environment.py`, the change added 29 lines and removed 154
lines, for a net reduction of 125 source lines including the three deleted helper
methods.

The architecture test:

- requires the dedicated provider-contract module and local class ownership;
- requires constructor delegation;
- prohibits restoration of low-level provider construction and helper calls;
- prohibits restoration of the three provider helper methods on the facade;
- limits the constructor source span to 270 lines.

## 5. Characterization coverage

The tests cover:

- default trend and disabled-provider state;
- explicit resolver ownership of trend and alpha mode;
- compatibility wrapping of causal alpha providers;
- explicit artifact digest precedence;
- provider `artifact_digest` and `identity_digest` fallback;
- static factor-basis conversion, copying, count inference, and deterministic digest;
- factor-provider count, digest, and minimum-index inference;
- alpha/factor minimum-index order and maximum aggregation;
- resolver/trend mismatch;
- missing enabled alpha provider and missing component identities;
- invalid SHA-256 identities;
- invalid factor-basis shape and non-finite values;
- invalid explicit and provider factor counts, including boolean values;
- static factor-count mismatch;
- invalid provider minimum indices and alpha-before-factor validation order.

The extracted module measured:

- 109 / 109 statements covered;
- 44 / 44 branches covered;
- 100.0% statement and branch coverage.

A permanent 100.0% critical branch-coverage ratchet is recorded in
`pyproject.toml`.

## 6. Exact-head verification

The implementation and coverage-ratchet exact head was:

- commit: `46a30ba1d6d0a277c28e27b5d5aae1322632c324`;
- CI run: `29973384473`;
- PostgreSQL Catalog run: `29973384477`.

The CI run passed:

- Studio frontend and fixed-viewport verification;
- workflow security checks;
- Ruff and Ruff formatting;
- Mypy;
- Import Linter;
- dead-code reporting;
- recovery and structured-serving smoke;
- complete pytest and coverage;
- critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- complete training-image build and non-root runtime probe.

The complete test result was:

- 1,280 passed;
- 2 skipped;
- 11 warnings;
- 84.06% total coverage;
- 71.17% total branch coverage.

The PostgreSQL Catalog run passed Compose validation, PostgreSQL startup and
readiness, migrations, catalog/unit/integration tests, and cleanup on the same
exact head.

## 7. Architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral
defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Provider resolution and identity binding for trend, alpha, and factor inputs are now
typed, independently tested, fully branch covered, and protected against returning
to the environment constructor.

The remaining construction density consists primarily of:

- portfolio-risk provider selection and identity validation;
- environment config, emergency-risk, action-spec, and episode-schedule validation;
- reward and executor construction;
- typed runtime-service wiring;
- mutable Gymnasium book, order, observation, and episode-state initialization.

Those concerns should not be mechanically combined or split without another
behavior-preserving seam and characterization evidence.

This item does not block causal research use. Production remains `NO-GO` until the
maintained research, evidence, operational, authorization, and profitability gates
pass independently.
