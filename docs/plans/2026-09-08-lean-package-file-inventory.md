# Lean Package File Migration Inventory

Date: 2026-09-08 (JST)
Base: `fbfe78553e414ae75674a81ab6bf606d969257e9`
Design: `docs/specs/2026-09-08-lean-package-boundaries-design.md` + Amendment 1

This inventory classifies every Python production file observed under `trade_rl/` on the exact base. Every source path has exactly one migration action: KEEP, MOVE, or DELETE. A MOVE may split a god/compatibility module into multiple explicit owners when the source currently mixes responsibilities.

## Package root

```text
KEEP   trade_rl/__init__.py
DELETE trade_rl/_source_checkout.py : no maintained caller found at design-time; exact implementation-head search must reconfirm before deletion
KEEP   trade_rl/_version.py
```

## artifacts

```text
KEEP   trade_rl/artifacts/__init__.py
KEEP   trade_rl/artifacts/atomic_pointer.py
KEEP   trade_rl/artifacts/atomic_write.py
DELETE trade_rl/artifacts/codec.py : forwarding-only compatibility module; canonical implementation moves from domain to artifacts/canonical.py
KEEP   trade_rl/artifacts/hashing.py : update import to artifacts/canonical.py
KEEP   trade_rl/artifacts/store.py
KEEP   trade_rl/artifacts/verified_file.py : update validation import only
```

New owner created from domain:

```text
MOVE   trade_rl/domain/canonical_json.py -> trade_rl/artifacts/canonical.py
```

## domain

```text
DELETE trade_rl/domain/__init__.py : package has no independent responsibility after migration
MOVE   trade_rl/domain/common.py -> trade_rl/_validation.py : retain only maintained require_* helpers; delete unused domain_content_digest
MOVE   trade_rl/domain/evaluation.py -> trade_rl/evaluation/gates/models.py : GateCheck/GateDecision ownership
```

The `trade_rl/domain/` directory must not exist after Phase 1.

## data

```text
KEEP   trade_rl/data/__init__.py : update exports to new subpackages
MOVE   trade_rl/data/artifact.py -> trade_rl/data/artifacts/publication.py : remove deprecated write_market_dataset_artifact wrapper
MOVE   trade_rl/data/artifact_codec.py -> trade_rl/data/artifacts/codec.py
MOVE   trade_rl/data/artifacts.py -> trade_rl/data/view.py + trade_rl/data/artifacts/ loader ownership : keep MarketDatasetView, remove legacy module
MOVE   trade_rl/data/builder.py -> trade_rl/data/build/builder.py
MOVE   trade_rl/data/config.py -> trade_rl/data/build/config.py
KEEP   trade_rl/data/contracts.py : update validation import only
MOVE   trade_rl/data/cross_asset_features.py -> trade_rl/data/features/cross_asset.py
MOVE   trade_rl/data/economic_semantics.py -> trade_rl/data/features/economic.py
MOVE   trade_rl/data/features.py -> trade_rl/data/features/core.py
KEEP   trade_rl/data/identity.py
KEEP   trade_rl/data/market.py : update validation/artifact imports only
MOVE   trade_rl/data/multitimeframe.py -> trade_rl/data/features/multitimeframe.py
KEEP   trade_rl/data/source.py
```

## integrations

```text
KEEP   trade_rl/integrations/__init__.py : preserve package-level public exports
MOVE   trade_rl/integrations/binance.py -> trade_rl/integrations/binance/{transport.py,vision.py,metadata.py,dataset.py,cache.py}
MOVE   trade_rl/integrations/binance_cache.py -> trade_rl/integrations/binance/cache.py
MOVE   trade_rl/integrations/frozen_binance_metadata.py -> trade_rl/integrations/binance/metadata.py
```

The old flat Binance files must not survive as forwarding shims.

## risk

```text
KEEP   trade_rl/risk/__init__.py
KEEP   trade_rl/risk/emergency.py
KEEP   trade_rl/risk/inputs.py : update validation/data/artifact imports only
KEEP   trade_rl/risk/portfolio.py
KEEP   trade_rl/risk/pretrade.py
```

## simulation

```text
KEEP   trade_rl/simulation/__init__.py : preserve package-level accounting/execution exports
KEEP   trade_rl/simulation/accounting.py
KEEP   trade_rl/simulation/bar_path.py
KEEP   trade_rl/simulation/execution.py
MOVE   trade_rl/simulation/execution_stress.py -> trade_rl/simulation/diagnostics/execution_stress.py
MOVE   trade_rl/simulation/funding_evidence.py -> trade_rl/simulation/diagnostics/funding.py
KEEP   trade_rl/simulation/liquidity.py
MOVE   trade_rl/simulation/order_admission.py -> trade_rl/simulation/orders/admission.py
MOVE   trade_rl/simulation/order_reconciliation.py -> trade_rl/simulation/orders/reconciliation.py
MOVE   trade_rl/simulation/orders.py -> trade_rl/simulation/orders/model.py
MOVE   trade_rl/simulation/runtime_performance.py -> trade_rl/simulation/diagnostics/runtime_performance.py
MOVE   trade_rl/simulation/runtime_performance_io.py -> trade_rl/simulation/diagnostics/runtime_performance_io.py
MOVE   trade_rl/simulation/stateful_bar_lifecycle.py -> trade_rl/simulation/stateful/bar_lifecycle.py
MOVE   trade_rl/simulation/stateful_execution.py -> trade_rl/simulation/stateful/execution.py
MOVE   trade_rl/simulation/stateful_order_transitions.py -> trade_rl/simulation/stateful/order_transitions.py
MOVE   trade_rl/simulation/stateful_runtime.py -> trade_rl/simulation/stateful/runtime.py
MOVE   trade_rl/simulation/stateful_symbol_fills.py -> trade_rl/simulation/stateful/symbol_fills.py
MOVE   trade_rl/simulation/target_execution.py -> trade_rl/simulation/targets/execution.py
MOVE   trade_rl/simulation/target_exposure_controller.py -> trade_rl/simulation/targets/exposure_controller.py
```

## strategies

```text
KEEP   trade_rl/strategies/__init__.py : preserve package-level public exports
KEEP   trade_rl/strategies/controls.py
MOVE   trade_rl/strategies/forecast.py -> trade_rl/strategies/forecasts/controller.py
KEEP   trade_rl/strategies/interface.py
MOVE   trade_rl/strategies/lightgbm.py -> trade_rl/strategies/forecasts/lightgbm.py
MOVE   trade_rl/strategies/mean_reversion.py -> trade_rl/strategies/rules/mean_reversion.py
KEEP   trade_rl/strategies/position_intent.py
MOVE   trade_rl/strategies/ppo.py -> trade_rl/strategies/rl/ppo.py
MOVE   trade_rl/strategies/ridge.py -> trade_rl/strategies/forecasts/ridge.py
MOVE   trade_rl/strategies/supervised.py -> trade_rl/strategies/forecasts/supervised.py
MOVE   trade_rl/strategies/trend.py -> trade_rl/strategies/rules/trend.py
```

## evaluation root

```text
KEEP   trade_rl/evaluation/__init__.py : preserve package-level public exports
MOVE   trade_rl/evaluation/_perfect_information_lp.py -> trade_rl/evaluation/robustness/perfect_information/solver.py
MOVE   trade_rl/evaluation/bootstrap.py -> trade_rl/evaluation/comparison/bootstrap.py
MOVE   trade_rl/evaluation/candidate_run.py -> trade_rl/evaluation/runs/candidate.py
MOVE   trade_rl/evaluation/candidate_suite.py -> trade_rl/evaluation/runs/candidate_suite.py
MOVE   trade_rl/evaluation/capacity.py -> trade_rl/evaluation/robustness/capacity.py
MOVE   trade_rl/evaluation/closed_trades.py -> trade_rl/evaluation/robustness/closed_trades.py
MOVE   trade_rl/evaluation/comparisons.py -> trade_rl/evaluation/comparison/paired.py
KEEP   trade_rl/evaluation/evidence.py
MOVE   trade_rl/evaluation/fold_metrics.py -> trade_rl/evaluation/robustness/fold_metrics.py
MOVE   trade_rl/evaluation/gates.py -> trade_rl/evaluation/gates/resolve.py
KEEP   trade_rl/evaluation/metrics.py
MOVE   trade_rl/evaluation/perfect_information_bound.py -> trade_rl/evaluation/robustness/perfect_information/bound.py
KEEP   trade_rl/evaluation/replay.py
MOVE   trade_rl/evaluation/seed_robustness.py -> trade_rl/evaluation/comparison/seed_robustness.py
KEEP   trade_rl/evaluation/series.py : ReturnSeries/ReturnKind primitive omitted from original target-tree diagram; Amendment 1 makes this explicit
MOVE   trade_rl/evaluation/strategy_comparison.py -> trade_rl/evaluation/comparison/strategies.py
```

Gate models created from domain:

```text
MOVE   trade_rl/domain/evaluation.py -> trade_rl/evaluation/gates/models.py
```

## evaluation/walk_forward

```text
MOVE   trade_rl/evaluation/walk_forward/__init__.py -> trade_rl/evaluation/robustness/walk_forward/__init__.py
MOVE   trade_rl/evaluation/walk_forward/capabilities.py -> trade_rl/evaluation/robustness/walk_forward/capabilities.py
MOVE   trade_rl/evaluation/walk_forward/folds.py -> trade_rl/evaluation/robustness/walk_forward/folds.py
MOVE   trade_rl/evaluation/walk_forward/sealed_test.py -> trade_rl/evaluation/robustness/walk_forward/sealed_test.py
MOVE   trade_rl/evaluation/walk_forward/stitching.py -> trade_rl/evaluation/robustness/walk_forward/stitching.py
```

## Phase ownership

```text
Phase 1: architecture tests, _validation, artifacts/canonical, evaluation/gates, domain deletion, optional _source_checkout deletion
Phase 2: data + Binance
Phase 3: strategies + simulation
Phase 4: evaluation organization + docs routing
```

## Inventory invariants

1. A MOVE old path must be absent when its owning phase is complete.
2. A DELETE path must not be replaced by a forwarding shim.
3. A KEEP path may change imports/exports only as needed for the new ownership boundaries; its semantic responsibility stays unchanged.
4. Every later phase must update this inventory if independent dependency evidence requires a different destination.
5. Files newly introduced only as temporary migration helpers must be deleted before the final quality gate.
