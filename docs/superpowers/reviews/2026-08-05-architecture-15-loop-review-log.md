# Architecture 15-Loop Review Log

## Baseline

- Fifteen regression contracts were added before production changes.
- Focused RED run: 15 failed, 0 passed.
- The generic private-import scan found three real violations.

## Loop 1: Public training-environment contract

**Check:** Environment identity and validation lived as private helpers inside `trade_rl.rl.training`.

**Fix:** Added `trade_rl.rl.training_environment_contract` with framework-neutral public functions. Kept private aliases in `rl.training` for compatibility.

**Review:** Numerical and serialized behavior is unchanged; only ownership and API visibility moved. Focused result: 1 passed, 14 failed.

## Loop 2: SB3 adapter uses the public contract

**Check:** `integrations.sb3_training` imported two private names from `rl.training`.

**Fix:** Replaced both imports and call sites with `training_environment_identity()` and `validate_training_environment()` from the public contract.

**Review:** Framework adapter behavior is unchanged; the dependency is now explicit and versionable. Focused result: 2 passed, 13 failed.

## Loop 3: Repository-wide private-import boundary

**Check:** The generic AST guard found one remaining cross-package private dependency: the Oracle benchmark imported `_portfolio_states`.

**Fix:** Published `portfolio_states()`, retained `_portfolio_states` as an internal compatibility alias, and moved the Operations consumer to the public name.

**Review:** The state enumeration algorithm is unchanged; only API visibility changed. Focused result: 3 passed, 12 failed.

## Loop 4: Generic catalog excludes sealed evaluation

**Check:** `PostgresArtifactCatalog` imported a sealed-evaluation record and delegated reservation persistence.

**Fix:** Removed the evaluation import and sealed-test method from the generic Artifact Catalog. The dedicated sealed-test stores remain intact.

**Review:** Generic catalog behavior is unchanged; sealed reservation ownership is no longer duplicated. Focused result: 4 passed, 11 failed.

## Loop 5: Public PostgreSQL connection construction

**Check:** The sealed-test adapter imported `_default_connection_factory` from the generic catalog implementation.

**Fix:** Added `catalog.postgres_connection` and moved optional psycopg loading plus default connection construction there. Both adapters now consume the public utility.

**Review:** Connection behavior and optional-dependency errors are preserved. Focused result: 5 passed, 10 failed. PostgreSQL validation remains active on the branch.

## Loop 6: RL terminal info stops importing Evaluation

**Check:** `rl.environment_info` calculated terminal metrics through `trade_rl.evaluation`, coupling the runtime environment to the reporting layer.

**Fix:** Added the lower `simulation.performance` contract with unchanged return-series validation and performance formulas, then redirected RL to that contract.

**Review:** Environment behavior and metric values are unchanged; the dependency now follows the maintained layer direction. Focused result: 6 passed, 9 failed.

## Loop 7: Performance implementation has one owner

**Check:** The lower performance contract existed, but Evaluation still owned duplicate implementations of return-series and metric calculations.

**Fix:** Replaced `evaluation.series` and `evaluation.metrics` with compatibility facades that re-export the Simulation-owned contracts.

**Review:** Existing import paths remain valid and the formulas now have one implementation owner. Focused result: 7 passed, 8 failed.

## Loop 8: Neutral policy identifier contract

**Check:** Artifact and RL modules duplicated maintained policy schema, encoder, and timeframe identifiers.

**Fix:** Added standard-library-only `domain.policy_contracts` as their neutral owner.

**Review:** All serialized identifier values remain unchanged. Focused result: 8 passed, 7 failed.

## Loop 9: Structured export consumes neutral identifiers

**Check:** `artifacts.structured_policy_contract` still defined its own policy schema and timeframe constants.

**Fix:** Redirected schema, encoder, and timeframe validation to `domain.policy_contracts` and removed local duplicates.

**Review:** Structured export schema and manifest values remain unchanged. Focused result: 9 passed, 6 failed.

## Loop 10: RL consumes neutral policy identifiers

**Check:** `rl.policy_identity` and sequence-window construction still carried duplicated schema, encoder, and timeframe ordering values.

**Fix:** Redirected both modules to `domain.policy_contracts` while preserving the exact schema string and window lengths.

**Review:** Policy digests and sequence tensor ordering are intended to remain byte-identical. The reviewed change set is staged for focused verification.
