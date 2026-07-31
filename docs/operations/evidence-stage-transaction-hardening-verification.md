# Evidence and stage transaction hardening verification

This note records the verification boundary for PR #309.

## Implemented contracts

- execution events are reconstructed through the maintained order-domain types;
- event streams require contiguous sequences, legal transitions, and consistent fill arithmetic;
- promotion evidence binds candidate configuration, evaluation run, fold, seed, action trace, observation trace, equity trace, event trace, and terminal states;
- maintained evaluation code publishes replay and promotion evidence together through a content-addressed artifact root;
- selected-final training resolves evidence from the content-addressed promotion root rather than accepting unrelated loose files;
- symbol-triplet completion and cursor state are committed as one immutable generation and exposed through one atomic pointer;
- stale stage writers use cursor-digest compare-and-swap;
- post-pointer durability uncertainty preserves the committed generation;
- structured export manifests are parsed from the exact verified bytes;
- replay artifacts use exclusive, fsynced, content-addressed publication.

## Focused local verification

- focused regression suite: 71 passed;
- all Python sources and tests compile;
- `git diff --check` passed;
- no temporary transfer workflow or payload remains in the production tree.

## Merge gate

The pull request remains a draft until the complete CI and PostgreSQL Catalog workflows pass on one unchanged final head. The repository workflows are the source of truth for Ruff, formatting, MyPy, import architecture, dead-code checks, full pytest and branch coverage, critical coverage, platform compatibility, container probes, and PostgreSQL integration.
