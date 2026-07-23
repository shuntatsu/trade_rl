# Serving Package Canonical Loader Design

Date: 2026-07-24

## Goal

Remove the hand-maintained reconstruction of `ExecutionCostConfig` from `trade_rl.serving.package`, make the training-environment artifact fail closed, and protect the complete selected-final packaging boundary with critical branch-coverage ratchets.

## Scope

This change is limited to the selected-final Serving packaging path. It does not change execution semantics, bundle schema, release approval, paper-reconciliation tolerances, or the production `NO-GO` status.

## Architecture

A new `trade_rl.serving.training_environment` module owns decoding of `environment.json` for Serving promotion. Its public function is:

```python
load_training_execution_cost(path: Path) -> ExecutionCostConfig
```

The loader validates `training_environment_v2`, requires the `environment.execution_cost` object to contain exactly the fields defined by `ExecutionCostConfig`, converts JSON list fields to their canonical tuple form, and delegates semantic validation to `ExecutionCostConfig.__post_init__`.

`trade_rl.serving.package` keeps a small private adapter for local call-site stability, but it no longer knows individual execution-cost fields or default values. Unknown fields, missing fields, malformed mappings, and unsupported schemas fail before execution evidence is compared.

## Coverage contract

Critical branch coverage is enforced independently for:

- `trade_rl/serving/package.py`: 90.0%
- `trade_rl/serving/training_environment.py`: 100.0%

Tests cover the valid artifact, unsupported schema, missing and unknown execution fields, malformed mappings, evidence and identity rejection branches, output collision, staging cleanup, and manifest construction failures.

## Compatibility

Current `training_environment_v2` artifacts produced by `execute_training_run()` contain the complete `ExecutionCostConfig` mapping and remain valid. Incomplete legacy-like artifacts that previously relied on local defaults are intentionally rejected because they cannot prove the execution-policy identity used during training.

## Safety

The filesystem training run remains authoritative. The canonical loader does not infer values, mutate artifacts, or authorize deployment. Packaging still requires selected-final authorization, promotable metadata, conservative execution evidence, signed fresh confirmation, and passing paper reconciliation.