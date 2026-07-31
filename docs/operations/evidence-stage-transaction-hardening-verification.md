# Evidence and stage transaction hardening verification

This note records the verification boundary for draft PR #311.

## Implemented contracts

- execution events are reconstructed through the maintained order-domain types;
- event streams require canonical schemas, contiguous sequences, legal transitions, consistent fill arithmetic, and valid replacement identities;
- cancel-and-replace reconciliation emits the cancellation before the replacement submission;
- invocation-local event batches are validated and explicitly resequenced at the replay aggregation boundary;
- promotion evidence binds candidate configuration, evaluation run, fold, seed, action trace, observation trace, equity trace, event trace, and terminal states;
- maintained evaluation code publishes replay and promotion evidence together through a content-addressed artifact root;
- selected-final training resolves evidence from the content-addressed promotion root rather than accepting unrelated loose files;
- symbol-triplet completion and cursor state are committed as one immutable generation and exposed through one atomic pointer;
- stage pointer reads validate one file snapshot and do not reject a concurrent atomic pointer replacement;
- stale stage writers use cursor-digest compare-and-swap;
- post-pointer durability uncertainty preserves the committed generation;
- structured export manifests are parsed from the exact verified bytes;
- replay artifacts use exclusive, fsynced, content-addressed publication.

## Verification history

Focused regression tests were added for duplicate submissions, unsupported event schemas, cancel-and-replace evidence, replacement-chain identity, invocation-local batch aggregation, and concurrent stage-pointer replacement. Each defect was first reproduced by a failing test before the production correction.

## Merge gate

The pull request remains a draft until the complete CI and PostgreSQL Catalog workflows pass on one unchanged head after synchronization with `main`. The repository workflows are the source of truth for Ruff, formatting, MyPy, import architecture, dead-code checks, full pytest and branch coverage, critical coverage, platform compatibility, container probes, and PostgreSQL integration. Exact final counts and the verified head SHA must be recorded in the pull request before it is marked ready.
