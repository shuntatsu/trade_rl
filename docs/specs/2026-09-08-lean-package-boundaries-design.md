# Lean Package Boundaries Cleanup Design

Date: 2026-09-08 (JST)
Status: design for review
Base: `fix/lean-license-contract` at `fbfe78553e414ae75674a81ab6bf606d969257e9`
Upstream stack: PR #436 -> PR #437

## 1. Conclusion

The repository should continue the 2026-09-08 lean redesign by making the physical package tree match the maintained runtime responsibilities.

This change is intentionally allowed to be destructive. Internal module paths, obsolete compatibility layers, transitional abstractions, stale documents, and dead code may be removed when they are not part of an intentionally supported public package API or a required current contract.

The redesign does **not** rename every top-level concept. The current top-level lean responsibilities remain valid:

- `data`: causal market data and dataset artifacts;
- `integrations`: external-source adapters;
- `risk`: pre-trade safety and feasibility;
- `simulation`: execution and accounting;
- `strategies`: symbol-agnostic strategy implementations;
- `evaluation`: replay, metrics, comparison, robustness, and immutable research runs;
- `artifacts`: generic deterministic artifact primitives.

The cleanup instead removes transitional layers and decomposes large flat packages into responsibility-oriented subpackages. After this cleanup, the next architectural sub-project is `evaluation/experiments`, which will implement the Controlled Experiment Loop without reintroducing the retired database/UI/teacher pipeline.

## 2. Objective

Create a repository layout in which a maintainer or agent can infer ownership and dependency direction from the filesystem itself, while preserving the observable semantics of the current lean research core.

The immediate outcomes are:

1. remove the transitional `trade_rl/domain` package;
2. move shared validation and canonical serialization to their real owners;
3. split large or flat data, Binance, strategy, simulation, and evaluation areas by responsibility;
4. preserve intentional package-level public APIs while allowing internal module imports to break;
5. make package-boundary rules executable in architecture tests and CI;
6. replace stale documentation with a small current-only documentation tree;
7. make Git history, not `docs/history`, the archive for obsolete implementation material;
8. leave a stable location for the later Controlled Experiment Loop without implementing that loop in this sub-project.

## 3. Non-goals

This cleanup must not:

- change strategy thresholds, strategy family behavior, model architecture, or training budget;
- change fit/evaluation symbol-scope semantics introduced by PR #436;
- change `MarketExecutor` / `BookState` economic accounting semantics;
- change fee, spread, impact, funding, borrow, liquidation, or order-admission semantics;
- change candidate profitability or choose a winner;
- run the real-data M2 development comparison as part of the refactor;
- add database persistence, Studio/UI, GraphPatch, Marketplace, hosted services, or live order routing;
- implement the Controlled Experiment Loop itself;
- preserve private/internal import paths merely for compatibility;
- perform unrelated algorithmic refactors while moving files.

## 4. Baseline and prerequisites

The implementation branch must be based on the exact integrated head that contains both current prerequisite fixes:

- PR #436: explicit fit-symbol scope for Ridge, LightGBM, and PPO;
- PR #437: current licensing provenance plus full `tests/` execution in CI.

At design time that head is:

`fbfe78553e414ae75674a81ab6bf606d969257e9`

Its normal CI run `34195183477` is successful.

If either upstream PR changes before implementation starts, the implementation branch must first be rebased or recreated on the new verified stacked head. Old Green evidence must not be attributed to a different HEAD.

## 5. Quality contract

### 5.1 Invariants

After every structural change:

- the same dataset and same resolved candidate config produce semantically equivalent replay and metric results;
- dataset identity and candidate-run identity rules remain deterministic and fail closed;
- fit symbols and evaluation symbols remain separate scopes;
- each selected fit symbol must still contribute eligible fit data where required;
- the same frozen strategy is still evaluated independently per evaluation symbol;
- `MarketExecutor` plus `BookState` remain the single economic accounting authority;
- immutable output directories are never overwritten;
- source-data availability and causality rules do not change;
- package-level public exports documented by each maintained package `__init__.py` remain available unless the design explicitly removes that public API;
- no compatibility shim is kept for a private old module path solely to keep obsolete imports working;
- license/provenance artifacts under `LICENSES/` remain intact.

### 5.2 Failure modes

The implementation must actively test or inspect for:

- moving a file but leaving a second implementation in the old location;
- circular imports introduced by the new folders;
- a compatibility re-export package surviving with no independent responsibility;
- serialization bytes or content digests changing because of a move alone;
- candidate identity changing because of module path rather than semantic configuration;
- fit/evaluation scope narrowing or widening during import rewrites;
- Binance transport/cache/source-selection behavior changing during decomposition;
- strategy family moves changing model fit or inference behavior;
- simulation moves changing fill/order/accounting ordering;
- tests passing because they only import stale private paths rather than exercising package-level behavior;
- CI omitting a newly created package or test directory;
- obsolete docs or package names continuing to describe the repository as the retired residual/U-series architecture;
- accidental deletion of licensing or third-party provenance;
- generated, temporary, or migration helper files remaining in the final tree.

### 5.3 Test oracle

Correctness is judged by observable contracts, not by successful imports alone. Required oracles include:

- exact public symbol availability from maintained package `__init__.py` modules;
- deterministic canonical JSON bytes and SHA-256 digests for representative immutable objects;
- dataset artifact publish/load/inspect equivalence;
- replay result equivalence for deterministic fixtures;
- candidate-run `summary.json` identity/scope fields and immutable publication behavior;
- per-symbol strategy metrics and raw returns for deterministic test datasets;
- risk projection and execution/accounting state transitions;
- Binance fixture transport, cache validation, parsing, metadata, and dataset assembly outputs;
- full test collection under `tests/`;
- an architecture dependency scan that fails on forbidden import directions and forbidden legacy modules.

## 6. Dependency direction

The desired dependency direction is intentionally simple.

```text
integrations ──> data
                 │
                 v
strategies ───> data
     │
     v
risk ─────────> data
     │
     v
simulation ───> data
     ^          ^
     │          │
     └──── evaluation
              │
              └──> strategies / risk / simulation / artifacts
```

More precisely:

- `artifacts` must not depend on `data`, `risk`, `simulation`, `strategies`, `evaluation`, or `integrations`;
- `_validation.py` must be standard-library-only and must not import another `trade_rl` package;
- `data` may depend on `artifacts` and `_validation`, but not on strategy/evaluation/simulation logic;
- `integrations` may depend on `data`, `artifacts`, and `_validation`, but not on `strategies` or `evaluation`;
- `strategies` may depend on `data` and its own subpackages, but not on `evaluation`;
- `risk` may depend on data contracts and artifact identity helpers, but not on `strategies` or `evaluation`;
- `simulation` may depend on data/risk contracts as required for execution, but not on `strategies` or `evaluation`;
- `evaluation` is the orchestration/readout layer and may depend on the lower lean core packages;
- the future `evaluation/experiments` package may depend on `evaluation/runs` and comparison primitives, never the reverse.

These rules become tests, not prose-only conventions.

## 7. Target package layout

The exact file split may be adjusted during implementation when a file's real dependency boundary proves different, but the ownership model below is normative.

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
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── model.py
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
│   │   ├── reconciliation.py
│   │   └── lifecycle.py
│   └── diagnostics/
│       ├── __init__.py
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

`evaluation/experiments/` is deliberately absent from this sub-project. It is the next sub-project after these boundaries are merged and verified.

## 8. Explicit removals and migrations

### 8.1 Remove `trade_rl/domain`

`trade_rl/domain` is a transitional abstraction and must not survive the cleanup.

Migration:

- `domain/canonical_json.py` -> `artifacts/canonical.py`;
- generic `require_*` validators from `domain/common.py` -> `trade_rl/_validation.py`;
- `GateCheck` and `GateDecision` from `domain/evaluation.py` -> `evaluation/gates/models.py`;
- `resolve_gate` -> `evaluation/gates/resolve.py`;
- `domain_content_digest()` is removed unless implementation-time repository search finds a real maintained consumer. At design time, code search finds no call site beyond its definition.

After migration, `trade_rl/domain` must not exist.

No `trade_rl.domain.*` compatibility package is retained.

### 8.2 Remove redundant artifact indirection

`artifacts/codec.py` currently re-exports canonical JSON owned by `domain/canonical_json.py`. The cleanup makes `artifacts/canonical.py` the implementation authority and updates internal imports directly.

If `artifacts/codec.py` has no intentional public consumer after import migration, it is deleted rather than kept as a forwarding shim.

`content_digest()` remains artifact-owned and uses the canonical artifact encoder directly.

### 8.3 Split Binance integration

The current `integrations/binance.py` is roughly 60 KB and mixes transport, cache verification, Binance Vision planning/parsing, exchange metadata, contract mapping, and dataset assembly.

It is replaced by the `integrations/binance/` subpackage.

Required responsibility boundaries:

- `transport.py`: bounded HTTP/REST/Vision transport and retry behavior;
- `cache.py`: cache paths, immutable evidence validation, and cache read/write behavior;
- `vision.py`: Vision URL planning and archive parsing;
- `metadata.py`: exchange-info snapshots, instrument metadata, and contract conversion;
- `dataset.py`: source assembly into `MarketDataset`.

`trade_rl.integrations` package-level exports remain stable for intentionally public integration types.

The old `trade_rl/integrations/binance.py` file is deleted; no same-name compatibility file remains next to the new package.

### 8.4 Group strategy families

The maintained candidate families are represented physically:

- `rules`: trend and mean reversion;
- `forecasts`: common forecast controller, supervised dataset construction, Ridge, LightGBM;
- `rl`: teacher-free PPO;
- `controls`: cash/constant direction controls remain at package root because they are comparison baselines rather than a fitted family.

`trade_rl.strategies` continues to export the maintained high-level public strategy classes/functions.

Internal imports from the old module paths are migrated; old private module paths are not shimmed.

### 8.5 Group evaluation by research responsibility

`evaluation` is split by what an operation means:

- `replay.py`, `metrics.py`, `evidence.py`: small universal primitives;
- `gates/`: evidence-bound decisions;
- `comparison/`: paired and strategy comparisons plus sampling-based comparison helpers;
- `robustness/`: capacity, closed-trade diagnostics, fold summaries, perfect-information bounds, and walk-forward logic;
- `runs/`: concrete immutable candidate-suite execution and publication.

A `run` is one computation. An `experiment` is a higher-level object that will later bind hypotheses, baseline/candidate runs, controlled differences, and a decision. This distinction is structural and must not be collapsed.

### 8.6 Group simulation without rewriting economics

Only modules with clear ownership are grouped.

- order model/admission/reconciliation/state transitions -> `simulation/orders/`;
- funding/runtime-performance evidence -> `simulation/diagnostics/`.

Core economic authorities `accounting.py` and `execution.py` remain visible at `simulation/` root. This is intentional: hiding them deeper would make the authoritative accounting path less discoverable.

No move is allowed to reorder fill, funding, borrow, termination, or risk application semantics.

### 8.7 Group data by data-lifecycle responsibility

The data package retains the central `MarketDataset` and core contracts at its root.

- dataset artifact schema/codec/publication -> `data/artifacts/`;
- build config and builder -> `data/build/`;
- feature computation -> `data/features/`.

This avoids both a flat package and an over-nested model where basic `MarketDataset` use requires navigating through implementation folders.

## 9. Public API policy

The cleanup intentionally distinguishes public package APIs from private module paths.

### Preserved

The following style of import remains supported where currently exported:

```python
from trade_rl.data import MarketDataset
from trade_rl.strategies import RidgeForecastStrategy
from trade_rl.evaluation import UniversalStrategyComparison
from trade_rl.simulation import MarketExecutor, BookState
from trade_rl.risk import PreTradeRisk
```

Package `__all__` declarations are reviewed and tested as part of the refactor.

### Not preserved by default

Private paths such as:

```python
from trade_rl.strategies.ridge import RidgeForecastStrategy
from trade_rl.domain.evaluation import GateCheck
```

may break when their owner changes. Internal repository imports are migrated in the same change.

A compatibility shim is allowed only if repository evidence proves the path is an intentionally supported external API and removing it would create a real compatibility contract violation. Mere historical existence is insufficient.

## 10. Documentation layout and retention

The current tree should describe the current system, not duplicate Git history.

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
├── specs/
│   └── <active design specifications only>
└── plans/
    └── <active implementation plans only>
```

Rules:

- `docs/README.md` is the human entry point;
- `docs/AGENTS.md` is the agent routing/update contract;
- `architecture/` contains only currently authoritative architecture;
- `research/` contains current research state and current evidence protocol;
- `specs/` contains active/unimplemented or in-review design contracts;
- `plans/` contains only active implementation plans;
- no `docs/history/` or `docs/archive/` is created;
- completed plans are deleted after their useful content is reflected in current architecture docs;
- completed specs are deleted when they cease to be independently normative;
- deleted docs remain recoverable from Git history;
- stale residual/U-series/Causal-Alpha descriptions must not remain as current architecture text;
- licensing/provenance documentation under `LICENSES/` is not treated as disposable historical docs.

The existing `docs/trade_rl_lean_redesign_20260908.md` is decomposed into the target current docs and then removed if all current normative content is preserved.

## 11. Tests and CI

### 11.1 Architecture tests

Add `tests/architecture/` with at least:

- `test_package_layout.py`: required package locations exist and forbidden legacy locations do not;
- `test_dependency_boundaries.py`: AST-based import-direction checks;
- `test_public_api.py`: maintained package-level exports are present;
- `test_no_legacy_modules.py`: `trade_rl/domain`, old single-file Binance module, and other explicitly retired module paths are absent.

The tests must inspect source structure rather than mock it.

### 11.2 Contract regression layers

Required targeted layers:

- artifacts canonicalization/digest/store tests;
- dataset artifact and market dataset tests;
- Binance integration fixture tests;
- risk tests;
- simulation accounting/execution/order tests;
- strategy rule/forecast/PPO tests;
- evaluation replay/comparison/robustness/candidate-run tests;
- license contract test.

### 11.3 Full gates

Before readiness, run on the final exact HEAD:

1. `uv sync --extra dev --extra forecast-gbm --extra train-sb3` or the dependency set required by the complete current tests;
2. `uv run ruff check trade_rl tests`;
3. `uv run ruff format --check trade_rl tests`;
4. `uv run mypy trade_rl`;
5. `uv run pytest -q tests`;
6. package identity check;
7. package build if configured/available in the maintained workflow;
8. normal GitHub CI on the exact PR HEAD.

CI must stop maintaining a hand-written list that can silently omit a newly created source/test package. Prefer checking `trade_rl` and `tests` roots directly unless an external-tool constraint requires a narrower scope.

## 12. TDD strategy for a structural refactor

This change still follows TDD, but RED is architectural rather than economic.

Before moving production files:

1. add failing package-layout tests for the target tree and forbidden old tree;
2. add dependency-boundary tests for the target import directions;
3. add package-level public API tests that capture the intended stable imports;
4. add equivalence/digest tests where a move could accidentally alter serialization identity;
5. then migrate one responsibility group at a time until each local RED becomes Green.

Do not weaken old behavioral tests to make path moves pass. Update imports in behavioral tests only after the test's observable assertion is confirmed to remain the same.

## 13. Migration sequence

The implementation plan should use this order because later moves depend on earlier shared boundaries.

1. Architecture RED tests and current public API contract.
2. `_validation.py` + artifact canonicalization migration; delete `domain`.
3. Data subpackage grouping.
4. Binance subpackage decomposition.
5. Strategy family grouping.
6. Simulation order/diagnostic grouping.
7. Evaluation grouping.
8. Test-tree grouping where needed to mirror source ownership.
9. CI root-level coverage of `trade_rl` + `tests`.
10. Documentation/AGENTS rewrite.
11. Delete obsolete files, forwarding modules, temporary migration helpers, and stale docs.
12. Targeted tests -> full tests -> static checks -> package verification -> exact-head CI.
13. Falsification review from the original invariants and failure modes.

Each step must leave the repository in a reviewable state. Do not combine an algorithmic behavior change into the same commit as a path migration.

## 14. Acceptance criteria

The cleanup is specification-complete only when all of the following are true:

1. `trade_rl/domain` does not exist.
2. Canonical JSON has one implementation authority under `artifacts`.
3. Generic validation helpers have one standard-library-only authority.
4. `integrations/binance.py` is replaced by a responsibility-split Binance subpackage.
5. Strategy rule, forecast, and RL families have explicit physical package boundaries.
6. Evaluation comparison, robustness, runs, and gates have explicit physical package boundaries.
7. Core execution/accounting authorities remain discoverable and semantically unchanged.
8. Data artifacts/build/features have explicit physical package boundaries without hiding the central dataset contract.
9. No compatibility-only forwarding package/module remains unless backed by an explicit public API contract.
10. Maintained package-level imports remain available.
11. Architecture tests reject forbidden dependency directions and retired legacy modules.
12. CI executes the complete maintained source/test tree and cannot silently omit a new package by directory-list drift.
13. Current docs explain only the current lean architecture/research state plus active specs/plans.
14. No `docs/history`/`docs/archive` is introduced; obsolete design/plan material is removed from the current tree.
15. `LICENSES/LICENSING.md`, `LICENSES/PROVENANCE.md`, SPDX license copies, and third-party notices remain intact.
16. Fit-symbol scope semantics from #436 remain intact.
17. Deterministic candidate-run identity and immutable publication remain intact.
18. Targeted behavioral tests, full `tests/`, Ruff, format, Mypy, package identity, and exact-head CI all pass.
19. The final diff contains no temporary migration workflow/helper and no accidental generated output.
20. A requirements-first falsification review finds no unresolved Critical/High contract violation; any remaining lower-risk limitation is documented explicitly.

## 15. Completion and follow-up boundary

This package cleanup is one architectural sub-project and must be completed before adding the Controlled Experiment Loop.

After it is merged and verified, the next design/implementation cycle will add:

```text
trade_rl/evaluation/experiments/
├── contract.py
├── comparison.py
├── decision.py
└── runner.py
```

That next sub-project will bind an immutable experiment definition to baseline/candidate run identities, verify controlled-condition equivalence, persist comparison evidence, and record `continue / reject / retest / hold` decisions.

It must use the cleaned `evaluation/runs` and comparison APIs rather than reintroducing database, UI, GraphPatch, or historical teacher-generation abstractions.
