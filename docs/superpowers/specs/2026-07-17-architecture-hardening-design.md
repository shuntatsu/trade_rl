# Architecture Hardening Design

## Status and goal

This change hardens the maintained research workflow without expanding the repository into direct exchange trading. It addresses the architecture audit findings around privileged GPU execution, historical execution-rule evidence, unenforced package boundaries, release-key separation, final-training authorization, compatibility API ambiguity, and workflow supply-chain checks.

## Trust boundaries

The privileged GPU workflow is executable only from the protected default branch through `workflow_dispatch` or a scheduled status check. The dispatch job requires the `gpu-full-training` GitHub Environment, runs only when `github.actor == github.repository_owner`, and refuses non-main refs. Pull requests never execute code on the self-hosted GPU runner.

Long-running training remains detached because the maintained full recipe may exceed a single Actions job limit. The detached container becomes a supervised operation: it receives immutable labels, writes an identity-bound status document in the shared volume, exposes start/status/stop operations, and is inspected by scheduled status jobs. Start, terminal exit, evidence collection, and cleanup are explicit states rather than an eight-second launch check.

## Historical metadata

Authenticated Binance execution-rule history uses schema `binance_instrument_rule_history_v3`. The signed payload binds market, ordered symbols, coverage start/end, issue time, source URI, policy version, and every effective rule. The maintained runner accepts only an exact USD-M, symbol-order, and coverage match. Binance tick size, lot size, and minimum notional are strictly positive for metadata and every historical rule. Histories must be ordered, non-empty, cover the first research timestamp, and contain no rule after the declared coverage end.

## Dependency boundaries

`trade_rl.learning` and `trade_rl.release` become explicit import-linter layers. Learning may depend on data, risk, simulation, artifacts, and domain; integrations may depend on learning. Release remains below serving and depends only on artifacts/domain and standard-library cryptography contracts. Tests assert the layer declarations remain present.

## Release verification

The current HMAC format remains readable for research compatibility, but released-mode activation requires verification-only key material declared with purpose `release-verification`. Signing helpers are moved behind an offline approval module and are not re-exported by runtime-facing packages. Attestations bind a key algorithm and purpose, and runtime rejects signing-purpose keys. This is an architectural separation improvement without adding an unpinned cryptography dependency; migration to public-key signatures remains a schema-versioned follow-up.

## Final training authorization

A final training run selected from walk-forward carries an immutable `SelectionAuthorization` sidecar binding the walk-forward run digest, selected configuration, selected candidate digest, fixed seed set, dataset ID, and gate-evidence digest. `execute_training_run` accepts an optional authorization; when final-training mode is requested, missing or mismatched authorization fails before model construction. Exploratory direct training remains supported and is labeled `research_exploratory`.

## Compatibility and supply chain

The duplicate deprecated dataset writer in `trade_rl.data.artifacts` is removed from the public export surface; canonical publication remains `write_market_dataset_files`, `publish_market_dataset_artifact`, and `load_market_dataset_artifact`. CI adds a repository-owned workflow policy checker that rejects privileged PR triggers, writable default permissions, unapproved self-hosted workflows, and mutable action references in the privileged GPU workflow.

## Verification

Regression tests cover workflow policy, signed-history scope and positivity, dependency-layer declarations, release key purpose, final-training authorization, and canonical dataset artifact exports. The full CI matrix remains the final merge gate.