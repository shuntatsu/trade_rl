# External research admission boundaries

The maintained typed extension API remains in [package-boundaries.md](package-boundaries.md).
These additive interfaces leave existing provenance, protocol digests and receipts unchanged.

## Loaded package observation

After independently verifying both wheel/source closures via verify_external_source_provenance, use a trusted fresh noneditable Python process (-I -B). Import the explicit entry points and call trade_rl.evaluation.runs.external_imports.verify_loaded_package_imports for each package with the reviewed manifest digest and sorted required module names including the root. Each loaded module must map to one pinned source, with regular SourceFileLoader and matching origin, loader path and package search locations. The installed source closure is rechecked and the observed roster must remain stable. LoadedPackageImports contains names and relative paths. Its digest can be compared at the same import boundary. This does not prove executed code objects, dependency bytes, malicious interpreter integrity or absence of transient mutation. It does not import unknown modules or confer authority.

## External GitHub owner-comment observation

tools.external_g4_comment is a read-only repository-side adapter, not an installed engine dependency. Trusted policy independently pins repository name/ID, owner login/ID, exact research PR HEAD/tree, core HEAD/tree/wheel, protocol and source digests, attempt/stage, assurance/review-body digests and both workflow run/attempt/job rosters. The existing authenticated result-blind review verifier is reused. Distinct reviewer, exact review, successful required jobs and unique authenticated PR association are required. An empty run PR roster may use an exact authenticated check-suite association; absent evidence fails closed. The standalone owner comment must equal the canonical binding, be unedited and created strictly after the review and CI. All fetched objects are re-read. Only trusted authenticated GitHub GET responses may establish metadata; the injected callback is also used for fabricated test oracles.

ExternalG4CommentEvidence.economic_execution_authorized is always false. Approved study policy, actual runtime/core pin checks, one-shot ledger, fresh admission and separate authorization remain study obligations. No historical evidence or economic permission changes. Consumer adoption requires reviewed installed and hosted validation.
