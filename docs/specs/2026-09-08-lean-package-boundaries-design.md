# Lean Package Boundaries Cleanup Design

Date: 2026-09-08 (JST)
Status: design for review
Base: `fix/lean-license-contract` at `fbfe78553e414ae75674a81ab6bf606d969257e9`
Upstream stack: PR #436 -> PR #437

## 1. Conclusion

Continue the 2026-09-08 lean redesign by making the physical package tree match maintained runtime responsibilities.

This change is intentionally allowed to be destructive. Internal module paths, obsolete compatibility layers, transitional abstractions, stale documents, and dead code may be removed when they are not part of an intentionally supported package-level public API or a required current contract.

The cleanup does **not** rename every top-level concept. These current lean responsibilities remain valid:

- `artifacts`: deterministic generic artifact primitives;
- `data`: causal market data and dataset artifacts;
- `integrations`: external-source adapters;
- `risk`: pre-trade safety and feasibility;
- `simulation`: execution and accounting;
- `strategies`: symbol-agnostic strategies;
- `evaluation`: replay, metrics, comparison, robustness, and immutable research runs.

The cleanup removes transitional layers and decomposes large flat packages into responsibility-oriented subpackages. The next architectural sub-project, after this one is merged and verified, is `evaluation/experiments` for the Controlled Experiment Loop.

## 2. Objective

Create a repository layout in which a maintainer or agent can infer ownership and dependency direction from the filesystem itself, while preserving observable semantics of the current lean research core.

The immediate outcomes are:

1. remove the transitional `trade_rl/domain` package;
2. move generic validation and canonical serialization to their real owners;
3. delete compatibility-only APIs that the current tree itself identifies as deprecated/legacy;
4. split data, Binance, strategy, simulation, and evaluation by responsibility;
5. preserve intentional package-level public APIs while allowing internal module paths to break;
6. make package-boundary rules executable in architecture tests and CI;
7. replace stale documentation with a small current-only documentation tree;
8. use Git history, not `docs/history`, as the archive;
9. leave a stable boundary for the later Controlled Experiment Loop without implementing it here.

## 3. Non-goals

This cleanup must not:

- change strategy thresholds, strategy-family behavior, model architecture, or training budget;
- change fit/evaluation symbol-scope semantics introduced by PR #436;
- change `MarketExecutor` / `BookState` economic accounting semantics;
- change fee, spread, impact, funding, borrow, liquidation, or order-admission semantics;
- choose a candidate winner or make a profitability claim;
- run the real-data M2 development comparison;
- add database persistence, Studio/UI, GraphPatch, Marketplace, hosted services, or live order routing;
- implement the Controlled Experiment Loop itself;
- preserve private/internal import paths merely for historical compatibility;
- combine algorithmic behavior changes with path migration.

## 4. Baseline and prerequisites

The implementation must start from the exact stacked head containing:

- PR #436: explicit fit-symbol scope for Ridge, LightGBM, and PPO;
- PR #437: current licensing provenance and full `tests/` execution in CI.

At design time:

`fbfe78553e414ae75674a81ab6bf606d969257e9`

Normal CI run `34195183477` is successful on that exact head.

If an upstream head moves, the implementation branch must first be recreated/rebased on the new verified stacked head. Green evidence from another SHA is not evidence for the new HEAD.

## 5. Quality contract

### 5.1 Invariants

After every structural change:

- the same dataset and resolved candidate config produce semantically equivalent replay and metrics;
- dataset identity and candidate-run identity remain deterministic and fail closed;
- fit symbols and evaluation symbols remain separate scopes;
- every selected fit symbol still contributes eligible fit data where the current contract requires it;
- the same frozen strategy is still replayed independently for each evaluation symbol;
- `MarketExecutor` plus `BookState` remain the single economic accounting authority;
- immutable output directories are never overwritten;
- source-data availability and causality rules do not change;
- maintained package-level exports remain available unless this design explicitly removes them;
- compatibility shims are not retained for private old paths solely to make obsolete imports work;
- `LICENSES/LICENSING.md`, `LICENSES/PROVENANCE.md`, SPDX copies, and third-party notices remain intact.

### 5.2 Failure modes

The implementation must explicitly look for:

- a move leaving a second implementation in the old location;
- circular imports;
- compatibility re-export modules surviving with no independent responsibility;
- canonical bytes or content digests changing because of a move alone;
- candidate identity changing because of module path rather than semantic configuration;
- fit/evaluation scope drift during import rewrites;
- Binance retry/cache/source-selection behavior changing during decomposition;
- strategy fit/inference changing during family moves;
- fill/order/accounting ordering changing during simulation moves;
- CI omitting a newly created package or test directory;
- tests that validate only stale private paths instead of observable behavior;
- obsolete residual/U-series/Causal-Alpha wording remaining as current architecture;
- accidental deletion of license/provenance material;
- temporary migration helpers or generated files surviving in the final tree.

### 5.3 Test oracle

Correctness is judged by observable contracts, not successful imports alone. Required oracles include:

- exact maintained package-level public symbol availability;
- deterministic canonical JSON bytes and SHA-256 digests;
- dataset artifact publish/load/inspect equivalence;
- deterministic replay equivalence on fixtures;
- candidate-run `summary.json` identity/scope fields and immutable publication;
- per-symbol metrics and raw returns on deterministic datasets;
- risk projection and execution/accounting state transitions;
- Binance fixture transport/cache/parsing/metadata/dataset outputs;
- full `tests/` collection;
- architecture checks for forbidden dependency directions and retired modules.

## 6. Dependency direction

The intended dependency graph is:

```text
_validation     artifacts
      \          /
       \        /
          data <------ integrations
         /  \
        /    \
 strategies   risk
      \        \
       \        v
        \---- simulation
               ^
               |
           evaluation
```

`evaluation` is the top research orchestration/readout layer and may consume the lower lean packages. Lower packages must not depend upward on evaluation.

Normative rules:

- `artifacts` must not depend on `data`, `integrations`, `risk`, `simulation`, `strategies`, or `evaluation`;
- `_validation.py` is standard-library-only and imports no other `trade_rl` package;
- `data` may depend on `artifacts` and `_validation`, not on strategies/evaluation/simulation logic;
- `integrations` may depend on `data`, `artifacts`, and `_validation`, not on strategies/evaluation;
- `strategies` may depend on `data`, not on `evaluation`;
- `risk` may depend on data contracts and artifact identity helpers, not on strategies/evaluation;
- `simulation` may depend on data/risk contracts, not on strategies/evaluation;
- `evaluation` may depend on the lower lean core packages;
- future `evaluation/experiments` may depend on `evaluation/runs` and comparison primitives, never the reverse.

These rules become architecture tests.

## 7. File-classification rule

Before production moves begin, the implementation plan must classify **every current production file** as exactly one of:

- `KEEP`: same owner/path;
- `MOVE`: new explicit owner/path;
- `DELETE`: no maintained responsibility and deletion evidence recorded.

No file may disappear merely because it was omitted from an example tree.

A `DELETE` requires at least one of:

- repository search proves no maintained consumer and the file has no public package export;
- the code is explicitly marked deprecated/legacy and only compatibility tests preserve it;
- its complete responsibility is replaced by a named new authority and equivalence tests cover the migration.

This classification manifest belongs in the implementation plan rather than becoming a permanent repository artifact.

## 8. Target package layout

The final ownership model is:

```text
trade_rl/
├── __init__.py
├── _version.py
├── _validation.py
│
├── artifacts/
│   ├── __init__.py
│   ├── canonical.py
│   ├── hashing.py
│   ├── atomic_pointer.py
│   ├── atomic_write.py
│   ├── store.py
│   └── verified_file.py
│
├── data/
│   ├── __init__.py
│   ├── market.py
│   ├── contracts.py
│   ├── identity.py
│   ├── source.py
│   ├── view.py
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── codec.py
│   │   └── publication.py
│   ├── build/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── builder.py
│   └── features/
│       ├── __init__.py
│       ├── core.py
│       ├── cross_asset.py
│       ├── economic.py
│       └── multitimeframe.py
│
├── integrations/
│   ├── __init__.py
│   └── binance/
│       ├── __init__.py
│       ├── transport.py
│       ├── cache.py
│       ├── vision.py
│       ├── metadata.py
│       └── dataset.py
│
├── risk/
│   ├── __init__.py
│   ├── inputs.py
│   ├── portfolio.py
│   ├── pretrade.py
│   └── emergency.py
│
├── simulation/
│   ├── __init__.py
│   ├── accounting.py
│   ├── execution.py
│   ├── bar_path.py
│   ├── liquidity.py
│   ├── orders/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── admission.py
│   │   └── reconciliation.py
│   ├── stateful/
│   │   ├── __init__.py
│   │   ├── runtime.py
│   │   ├── execution.py
│   │   ├── bar_lifecycle.py
│   │   ├── order_transitions.py
│   │   └── symbol_fills.py
│   ├── targets/
│   │   ├── __init__.py
│   │   ├── execution.py
│   │   └── exposure_controller.py
│   └── diagnostics/
│       ├── __init__.py
│       ├── execution_stress.py
│       ├── funding.py
│       ├── runtime_performance.py
│       └── runtime_performance_io.py
│
├── strategies/
│   ├── __init__.py
│   ├── interface.py
│   ├── position_intent.py
│   ├── controls.py
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── trend.py
│   │   └── mean_reversion.py
│   ├── forecasts/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── supervised.py
│   │   ├── ridge.py
│   │   └── lightgbm.py
│   └── rl/
│       ├── __init__.py
│       └── ppo.py
│
└── evaluation/
    ├── __init__.py
    ├── replay.py
    ├── metrics.py
    ├── evidence.py
    ├── gates/
    │   ├── __init__.py
    │   ├── models.py
    │   └── resolve.py
    ├── comparison/
    │   ├── __init__.py
    │   ├── paired.py
    │   ├── strategies.py
    │   ├── bootstrap.py
    │   └── seed_robustness.py
    ├── robustness/
    │   ├── __init__.py
    │   ├── capacity.py
    │   ├── closed_trades.py
    │   ├── fold_metrics.py
    │   ├── perfect_information/
    │   │   ├── __init__.py
    │   │   ├── bound.py
    │   │   └── solver.py
    │   └── walk_forward/
    │       ├── __init__.py
    │       ├── capabilities.py
    │       ├── folds.py
    │       ├── sealed_test.py
    │       └── stitching.py
    └── runs/
        ├── __init__.py
        ├── candidate.py
        └── candidate_suite.py
```

`evaluation/experiments/` is deliberately absent from this sub-project.

## 9. Explicit removals and migrations

### 9.1 Remove `trade_rl/domain`

`trade_rl/domain` is transitional and must not survive.

Migration:

- `domain/canonical_json.py` -> `artifacts/canonical.py`;
- generic `require_*` helpers from `domain/common.py` -> `_validation.py`;
- `GateCheck` / `GateDecision` -> `evaluation/gates/models.py`;
- `resolve_gate` -> `evaluation/gates/resolve.py`;
- `domain_content_digest()` is deleted unless implementation-time search finds a maintained consumer. At design time, search finds only its definition.

No `trade_rl.domain.*` compatibility package remains.

### 9.2 Remove redundant/deprecated artifact compatibility

Current `artifacts/codec.py` forwards canonical JSON from `domain/canonical_json.py`. `artifacts/canonical.py` becomes the single implementation authority.

If no intentional public consumer remains after internal migration, `artifacts/codec.py` is deleted rather than retained as a forwarding shim.

Current `data/artifact.py` contains `write_market_dataset_artifact()`, explicitly marked as a deprecated compatibility wrapper. Repository search finds the production definition plus tests that preserve/check its legacy behavior, not a maintained production caller. The cleanup removes that deprecated writer and removes/rewrites tests whose only purpose is to preserve it.

The maintained dataset publication API is `publish_market_dataset_artifact()`.

### 9.3 Remove legacy `data/artifacts.py` ownership ambiguity

Current `data/artifacts.py` mixes dataset loading with `MarketDatasetView` and is referred to as a legacy module in tests.

Migration:

- `MarketDatasetView` -> `data/view.py`;
- canonical load/inspect/publication -> `data/artifacts/` authority;
- old `data/artifacts.py` -> DELETE after all maintained imports migrate.

### 9.4 Remove unused source-checkout helper if final search confirms no consumer

At design time repository search finds no maintained caller of `source_checkout_root()` in `trade_rl/_source_checkout.py`.

The implementation classification must repeat the search on the exact base. If still unused and not package-exported, `_source_checkout.py` is deleted. It must not be kept solely because it existed before the lean redesign.

### 9.5 Split Binance integration

Current `integrations/binance.py` is roughly 60 KB and mixes:

- REST/Vision transport and retry;
- cache evidence validation;
- Vision URL planning and archive parsing;
- exchange metadata and contract conversion;
- dataset assembly.

Replace it with `integrations/binance/`:

- `transport.py`: bounded public transport/retry;
- `cache.py`: cache identity/evidence/read-write behavior;
- `vision.py`: URL planning and archive parsing;
- `metadata.py`: exchange snapshots/instrument metadata/contract conversion;
- `dataset.py`: `MarketDataset` source assembly.

`trade_rl.integrations` package-level exports remain stable for intentionally public types. The old single `binance.py` file is deleted; no compatibility twin remains beside the new package.

### 9.6 Group strategy families

Physical families match the research comparison:

- `rules`: trend, mean reversion;
- `forecasts`: controller, supervised fit data, Ridge, LightGBM;
- `rl`: teacher-free PPO;
- `controls.py`: comparison baselines at package root.

`trade_rl.strategies` keeps maintained high-level exports. Old private family module paths are not shimmed.

### 9.7 Group simulation without rewriting economics

Current simulation files are assigned explicitly:

- `orders.py` -> `orders/model.py`;
- `order_admission.py` -> `orders/admission.py`;
- `order_reconciliation.py` -> `orders/reconciliation.py`;
- `stateful_runtime.py` -> `stateful/runtime.py`;
- `stateful_execution.py` -> `stateful/execution.py`;
- `stateful_bar_lifecycle.py` -> `stateful/bar_lifecycle.py`;
- `stateful_order_transitions.py` -> `stateful/order_transitions.py`;
- `stateful_symbol_fills.py` -> `stateful/symbol_fills.py`;
- `target_execution.py` -> `targets/execution.py`;
- `target_exposure_controller.py` -> `targets/exposure_controller.py`;
- `execution_stress.py` -> `diagnostics/execution_stress.py`;
- `funding_evidence.py` -> `diagnostics/funding.py`;
- `runtime_performance.py` -> `diagnostics/runtime_performance.py`;
- `runtime_performance_io.py` -> `diagnostics/runtime_performance_io.py`.

`accounting.py`, `execution.py`, `bar_path.py`, and `liquidity.py` remain at simulation root because they are core economic/execution primitives.

No move may reorder fill, funding, borrow, termination, target, or risk semantics.

### 9.8 Group data by lifecycle

Current data ownership becomes:

- central model/contracts: `market.py`, `contracts.py`, `identity.py`, `source.py`;
- artifact publication/codec: `data/artifacts/`;
- bounded views: `data/view.py`;
- build configuration/assembly: `data/build/`;
- feature computation: `data/features/`.

The implementation plan must map each current data file explicitly. It may not leave both old and new authorities.

### 9.9 Group evaluation by research responsibility

- `replay.py`, `metrics.py`, `evidence.py`: universal primitives;
- `gates/`: evidence-bound gate models/resolution;
- `comparison/`: paired/strategy comparison and sampling helpers;
- `robustness/`: capacity, closed-trade diagnostics, fold summaries, perfect-information bounds, walk-forward;
- `runs/`: concrete immutable candidate-suite execution/publication.

A `run` is one computation. A future `experiment` is a higher-level object that binds a hypothesis, baseline/candidate run identities, controlled differences, comparison evidence, and a decision. These concepts must remain structurally separate.

## 10. Public API policy

### Preserved

Where currently exported, imports of this form remain supported:

```python
from trade_rl.data import MarketDataset
from trade_rl.strategies import RidgeForecastStrategy
from trade_rl.evaluation import UniversalStrategyComparison
from trade_rl.simulation import MarketExecutor, BookState
from trade_rl.risk import PreTradeRisk
```

Package `__all__` surfaces are snapshotted before moves and tested after them.

### Not preserved by default

Private/internal paths may break, for example:

```python
from trade_rl.strategies.ridge import RidgeForecastStrategy
from trade_rl.domain.evaluation import GateCheck
```

Internal repository imports migrate in the same change.

A compatibility shim is allowed only when repository evidence proves a path is intentionally public. Historical existence alone is not evidence.

## 11. Documentation layout and retention

The current tree describes the current system; Git history stores past designs.

Target:

```text
docs/
├── README.md
├── AGENTS.md
├── architecture/
│   ├── lean-core.md
│   └── package-boundaries.md
├── research/
│   └── current-status.md
├── specs/       # active design specifications only
└── plans/       # active implementation plans only
```

Rules:

- `docs/README.md`: human entry point;
- `docs/AGENTS.md`: agent routing/update contract;
- `architecture/`: only current authoritative architecture;
- `research/`: current research state and evidence protocol;
- `specs/`: active/unimplemented or in-review designs;
- `plans/`: active implementation plans;
- do not create `docs/history/` or `docs/archive/`;
- delete completed plans after durable content is reflected in architecture docs;
- delete completed specs when they cease to be independently normative;
- obsolete docs remain recoverable from Git history;
- licensing/provenance under `LICENSES/` is permanent compliance material, not disposable history.

`docs/trade_rl_lean_redesign_20260908.md` is decomposed into current architecture/research docs and removed once all normative current content is preserved.

## 12. Tests and CI

### 12.1 Architecture tests

Add `tests/architecture/` with at least:

- `test_package_layout.py`: required locations and forbidden old locations;
- `test_dependency_boundaries.py`: AST import-direction checks;
- `test_public_api.py`: maintained package-level exports;
- `test_no_legacy_modules.py`: removed `domain`, old single-file Binance, deprecated data artifact paths, and other explicitly retired modules stay absent.

These tests inspect the source tree directly.

### 12.2 Contract regression layers

Run targeted tests for:

- artifacts canonicalization/digest/store;
- dataset artifacts/views/market dataset;
- Binance integration fixtures;
- risk;
- simulation accounting/execution/orders/stateful/targets;
- strategies rules/forecast/PPO;
- evaluation replay/comparison/robustness/candidate runs;
- licensing.

### 12.3 Full gates

On final exact HEAD:

1. install all extras required by the complete current tests;
2. `uv run ruff check trade_rl tests`;
3. `uv run ruff format --check trade_rl tests`;
4. `uv run mypy trade_rl`;
5. `uv run pytest -q tests`;
6. package identity check;
7. maintained package build if present in the final workflow;
8. normal GitHub CI on the exact PR HEAD.

CI must not maintain a hand-written package-directory list that can omit newly created packages. Prefer checking the `trade_rl` and `tests` roots directly.

## 13. TDD strategy

This structural refactor uses architectural RED tests before production moves:

1. add failing package-layout tests for target/forbidden paths;
2. add dependency-boundary tests;
3. add package-level public API tests;
4. add serialization/digest equivalence tests where a move can alter identity;
5. migrate one responsibility group at a time to Green.

Behavioral assertions are not weakened to accommodate path moves. Test imports may change only while their observable assertions remain intact.

## 14. Migration sequence

1. Build complete `KEEP / MOVE / DELETE` production-file classification in the implementation plan.
2. Add architecture RED tests and public API snapshot tests.
3. Migrate `_validation.py` and artifact canonicalization; delete `domain` and compatibility-only artifact shims.
4. Migrate data artifacts/views/build/features and delete deprecated dataset writer/legacy module.
5. Decompose Binance.
6. Group strategy families.
7. Group simulation orders/stateful/targets/diagnostics.
8. Group evaluation gates/comparison/robustness/runs.
9. Mirror source ownership in tests where useful; do not create redundant test nesting.
10. Update CI to whole-root source/test checks.
11. Rewrite docs/AGENTS/current architecture docs.
12. Delete obsolete files, forwarding modules, stale docs, migration helpers, and confirmed-unused `_source_checkout.py`.
13. Run targeted tests, full tests, static checks, package verification, exact-head CI.
14. Perform requirements-first falsification review.

Each migration step must be reviewable and must not mix economic behavior changes with path changes.

## 15. Acceptance criteria

The cleanup is specification-complete only when all are true:

1. Every pre-refactor production file has an explicit `KEEP / MOVE / DELETE` disposition in the implementation plan.
2. `trade_rl/domain` does not exist.
3. Canonical JSON has one implementation authority under `artifacts`.
4. Generic validation helpers have one standard-library-only authority.
5. Deprecated `write_market_dataset_artifact()` is removed and no maintained code depends on it.
6. Legacy `data/artifacts.py` is removed after its maintained responsibilities move.
7. `_source_checkout.py` is removed if exact-base search still proves no maintained consumer.
8. `integrations/binance.py` is replaced by a responsibility-split Binance package.
9. Strategy rule/forecast/RL families have explicit physical boundaries.
10. Simulation orders/stateful/targets/diagnostics have explicit physical boundaries while core accounting/execution remain discoverable.
11. Evaluation gates/comparison/robustness/runs have explicit physical boundaries.
12. Data artifacts/build/features have explicit boundaries without hiding `MarketDataset`.
13. No compatibility-only forwarding module remains unless backed by an explicit public API contract.
14. Maintained package-level imports remain available.
15. Architecture tests reject forbidden dependency directions and retired modules.
16. CI checks the complete maintained `trade_rl` and `tests` roots.
17. Current docs contain only current architecture/research state plus active specs/plans.
18. No `docs/history` or `docs/archive` is introduced.
19. License/provenance and third-party notices remain intact.
20. Fit-symbol scope semantics from #436 remain intact.
21. Deterministic candidate-run identity and immutable publication remain intact.
22. Targeted tests, full `tests/`, Ruff, format, Mypy, package identity, and exact-head CI pass.
23. Final diff contains no temporary migration workflow/helper or generated output.
24. Requirements-first falsification review finds no unresolved Critical/High contract violation; any lower-risk limitation is explicit.

## 16. Follow-up boundary

This cleanup completes one architectural sub-project. The Controlled Experiment Loop starts only afterward.

The next sub-project will add:

```text
trade_rl/evaluation/experiments/
├── contract.py
├── comparison.py
├── decision.py
└── runner.py
```

It will bind an immutable experiment definition to baseline/candidate run identities, verify controlled-condition equivalence, persist comparison evidence, and record `continue / reject / retest / hold` decisions.

It must consume the cleaned `evaluation/runs` and comparison APIs and must not reintroduce database, UI, GraphPatch, or historical teacher-generation abstractions.
