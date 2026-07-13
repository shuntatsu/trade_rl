# Causal Market Data Final Hardening Design

## Goal

Close the remaining policy-information, Serving-contract, artifact-publication, and dataset-identity gaps in PR #24 without changing execution truth or the research NO-GO status.

## Design

Execution keeps the realized `MarketDataset.tradable` array. Policy inputs use a separate derived value, `tradable & information_available`, and the market-wide tradable fraction is built from that same observable mask. Delayed rows therefore remain usable by the simulator as realized history but cannot influence the policy before their declared availability time.

A shared `MarketInputResolver` owns deterministic Trend and optional Alpha construction. Alpha providers receive a copied causal prefix rather than the full dataset and expose a stable identity digest. The RL environment and Serving runtime use this resolver. Serving recomputes market inputs instead of trusting caller-supplied Trend or Alpha values, and the resolver digest is bound into the serving bundle.

Raw vector inference remains available only with explicit dataset and observation-schema identities. Structured and raw inference both fail closed before policy execution on dataset, schema, market-input, or vector-size mismatch.

Dataset identity is computed by one shared function over an explicit identity payload and the stored canonical arrays. The payload is persisted in the artifact manifest. `MarketDataset` and the artifact loader independently recompute the identity and reject mismatches.

Dataset artifacts are immutable directories. Publication writes the complete manifest and array archive into a sibling staging directory, validates the staged artifact, and renames the directory into a previously nonexistent destination. Existing destinations are rejected, so readers can never observe a mixed old/new pair.

## Error handling

All identity, schema, and publication mismatches raise before mutable runtime state or output paths are changed. Staging directories are removed after any failed build. Existing artifact directories are never overwritten.

## Tests

Regression tests cover delayed current-row tradability, observable market fractions, raw Serving contract mismatches, caller-supplied future-derived market inputs, causal Alpha views, dataset-ID recomputation, manifest tampering, atomic publication failure, and existing-output preservation. The complete fail-fast CI remains the final gate.
