# Architecture Follow-up Hardening Design

## Goal

Close the architecture gaps found after PR #42 without weakening the existing causal, content-addressed, fail-closed boundaries.

## Scope

This change set addresses the confirmed defects and low-risk structural gaps that directly affect maintained execution paths:

1. Preserve dataset contract multipliers in every environment-created `BookState`.
2. Make external non-circular release attestations survive registry installation and revalidation.
3. Connect `PortfolioRiskModel` to training, walk-forward evaluation, environment identity, and CLI experiment manifests.
4. Validate covariance matrices fail-closed instead of silently treating invalid negative variance as zero risk.
5. Reject unknown training configuration keys.
6. Advance the canonical dataset identity schema to v6 consistently in code and documentation.
7. Preserve complete loaded bundle state from registry activation and make artifact directory fsync portable on Windows.

The broader raw execution-metadata ingestion and semantic-normalizer redesign remain separate work because each changes a public data contract and requires its own migration plan.

## Chosen architecture

### Book construction

`ResidualMarketEnv` remains the owner of episode book creation, but all book constructors receive `dataset.contract_multipliers`. Restore mode validates multiplier identity before accepting the supplied book. This keeps quantity semantics centralized in `BookState` and `MarketExecutor`.

### External release attestation installation

A registry version is an installation directory containing the immutable bundle directory plus its external attestation sidecar. The sidecar remains outside the candidate bundle digest, preserving non-circular approval. During activation the registry copies both objects into staging, validates the staged bundle using the staged sidecar, then atomically installs the directory and updates the active pointer.

The installed layout is:

```text
versions/<bundle-digest>/
  bundle/
    bundle.json
    ...declared bundle files...
  bundle.release.json
```

Legacy in-bundle `release.json` remains readable for compatibility.

### Portfolio risk inputs

`PortfolioRiskConfig` gains a causal `lookback_hours` parameter. `PortfolioRiskModel` resolves required inputs directly from the immutable `MarketDataset` at the current decision index:

- market notional from explicit volume-unit semantics;
- covariance from trailing log returns;
- beta against the trailing equal-weight market return;
- stress losses from the worst trailing per-asset log return.

Only indices at or before the decision index are used. Advanced limits extend the environment minimum-history requirement. The resolved model configuration and implementation digest are included in environment, training, walk-forward, and experiment identities.

Risk order is:

```text
residual composition
  -> pre-trade operational and hard limits
  -> portfolio concentration/liquidity/statistical limits
  -> execution
```

Portfolio constraints are hard constraints and may override the soft turnover limit. The combined `RiskConstrainedTarget` records prefixed portfolio reasons and the total projection distance from the original proposal.

### Covariance validation

Externally supplied covariance matrices must be finite, symmetric within tolerance, and positive semidefinite within tolerance. Small numerical negative eigenvalues are tolerated only within the configured numerical threshold; materially indefinite matrices raise `ValueError`.

### Configuration closure

`TrainingRunConfig.from_mapping` validates exact allowed keys at every maintained nested mapping before constructing dataclasses. Typos therefore fail closed instead of silently selecting defaults.

### Identity and compatibility

Canonical market dataset identity becomes `market_dataset_identity_v6`. Existing verified v5 artifacts remain loadable only through existing compatibility behavior; newly built and published datasets use v6.

## Error handling

All identity mismatches, unavailable statistical history, invalid covariance, missing attestations, unsafe registry paths, and unknown configuration keys raise before training, evaluation, publication, or activation proceeds. No new silent fallback is introduced.

## Testing

Tests are added before production changes and must demonstrate the original failures:

- non-unit contract multiplier environment step;
- external-attestation registry activation and active reload;
- environment portfolio-risk projection and digest changes;
- causal risk input history requirement;
- asymmetric and indefinite covariance rejection;
- unknown top-level and nested config key rejection;
- v6 identity emission;
- Windows-safe artifact directory sync behavior.

Targeted tests, Ruff, formatting, Mypy, import-linter, dead-code checks, the full branch-coverage suite, critical coverage ratchet, CLI smoke test, and Ubuntu/Windows compatibility tests must pass before the branch is ready.