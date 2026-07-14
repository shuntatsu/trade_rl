# Critical Architecture Remediation Design

## Status

Approved by the user's instruction to implement the critical findings from the architecture audit.

## Goal

Close the contract-multiplier, external release-attestation and disconnected portfolio-risk gaps while preserving causal evaluation, immutable bundle identity and backward-compatible research configurations.

## Book construction

`ResidualMarketEnv` remains the owner of episode book creation. Every cash, weighted, reward-preroll and restored book must carry `dataset.contract_multipliers`. Restore mode rejects books whose multiplier vector differs from the dataset before revaluation or execution.

## Registry-owned external attestations

The release attestation remains outside the source candidate bundle so approval cannot alter the bundle digest. During activation, `ServingRegistry` copies the verified bundle and its external attestation into the registry staging namespace, validates the staged pair, then atomically installs both. The installed attestation is stored as the sibling of the installed bundle directory, matching `default_attestation_path()` semantics. A failed copy or validation never changes `active.json`.

Legacy internal `release.json` bundles remain readable for compatibility.

## Portfolio risk integration

`PortfolioRiskConfig` becomes part of `TrainingRunConfig`, `EnvironmentExperimentManifest`, walk-forward candidate equality and the environment digest. `ResidualMarketEnv` accepts a `PortfolioRiskModel` and applies it after `PreTradeRisk` and before execution.

The maintained environment derives current market notional causally from the dataset row. Concentration, net-exposure and liquidity caps work without additional providers. Configurations requiring covariance, beta or stress inputs fail closed until causal providers are supplied; this change does not fabricate estimates.

## Compatibility

Missing `portfolio_risk` JSON resolves to `PortfolioRiskConfig()`. Existing configurations therefore retain behavior while receiving a new explicit identity component. Schema versions whose digest payload changes are incremented.

## Error handling

- Executor-facing books must match dataset multipliers.
- Restore mode rejects incompatible books immediately.
- Registry staging removes partial bundle and attestation copies after failure.
- Portfolio risk rejects unavailable required inputs and non-finite data.
- Activation remains fail closed without verified release evidence.

## Tests

Regression tests cover:

1. an environment step with a non-unit contract multiplier;
2. restored-book multiplier rejection;
3. activation and reload of an externally attested bundle;
4. preservation of the previous active bundle after attestation failure;
5. portfolio-risk projection changing the executed target and environment identity;
6. training, experiment and walk-forward digests binding portfolio risk.

## Non-goals

No live exchange routing, covariance estimation, beta estimation, stress-model generation, cryptographic approver signatures or market-ingestion redesign is included.