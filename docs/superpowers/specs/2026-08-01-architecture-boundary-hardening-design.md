# Architecture Boundary Hardening Design

## Goal

Strengthen the current Stage A architecture without implementing Stage B or changing training, evaluation, reward, action, execution, or serving behavior.

## Scope

This change has three responsibilities:

1. Move the reusable `TrainingRunConfig` contract and parser below the workflow layer so Studio can validate authored configs without importing workflow orchestration.
2. Move the structured-policy export manifest contract below the RL implementation so Serving validates a neutral artifact contract rather than importing the training/export implementation.
3. Add enforced dependency contracts and architecture tests that prevent these boundaries from regressing.

The existing public import paths remain compatible during this refactor. `trade_rl.workflows.training_run` continues to expose `TrainingRunConfig`, and the structured export module continues to expose its public manifest names through imports from the neutral contract module.

## Stage B boundary

Stage B is not implemented in this change.

The future cross-market objective is explicitly asymmetric:

- Binance Spot represents the long-side book only.
- Binance USDⓈ-M futures represents the short-side book only.
- A future portfolio coordinator may stage and combine those two market-specific books.
- The design must not silently reinterpret Spot as short-capable or USDⓈ-M as an unrestricted long/short venue.

No Spot downloader, futures short adapter, borrow model, cross-market accounting, or Stage B policy is added here.

## Training configuration boundary

`TrainingRunConfig` currently mixes a reusable authored configuration contract with `execute_training_run` orchestration in one workflow module. The contract will move to `trade_rl.rl.training_run_config` because it composes RL, risk, simulation, strategy, and artifact identities while remaining framework independent.

`trade_rl.workflows.training_run` will import and re-export the contract. Studio config discovery will import `trade_rl.rl.training_run_config` directly. This preserves compatibility while removing the upper-layer dependency from Studio config validation.

Generic closed-field helpers move from `trade_rl.workflows.config_fields` to `trade_rl.domain.config_fields`. The old workflow module becomes a compatibility re-export so existing imports do not break.

## Structured export boundary

Schema constants, `StructuredInputSpec`, `StructuredExportManifest`, and manifest decoding belong to a neutral artifact contract module:

`trade_rl.artifacts.structured_policy_contract`

The Torch-based exporter remains in `trade_rl.rl.structured_export`. The Serving loaders import only the neutral artifact contract. This separates:

- contract and validation,
- model export and parity execution,
- runtime model loading and inference.

## Enforced dependency rules

Import Linter and architecture tests will enforce:

- Studio cannot import `trade_rl.workflows`.
- Serving cannot import `trade_rl.rl.structured_export`.
- The neutral structured-policy contract cannot import Torch or RL implementation modules.
- Compatibility exports remain available from the old public modules.

## Error handling and compatibility

All existing fail-closed field validation, schema version checks, digest validation, checkpoint identity checks, and path resolution behavior remain unchanged. This is a relocation and dependency-boundary refactor, not a schema migration.

## Verification

The implementation uses test-first architecture checks, focused config and structured-export tests, Import Linter, Ruff, MyPy, and the complete repository CI. The final PR must remain unmerged until its exact head passes all required checks.
