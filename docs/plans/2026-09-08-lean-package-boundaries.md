# Lean Package Boundaries Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the physical `trade_rl` package tree match the current lean research responsibilities, remove transitional/compatibility-only artifacts, and install executable package-boundary contracts without changing trading, fitting, execution, accounting, or evaluation semantics.

**Architecture:** Preserve the meaningful top-level packages (`artifacts`, `data`, `integrations`, `risk`, `simulation`, `strategies`, `evaluation`) while decomposing each by responsibility. Remove `domain` entirely, keep package-level public APIs stable, allow private module paths to break, and use tests plus same-head CI as the migration oracle. The future Controlled Experiment Loop remains a separate sub-project under `evaluation/experiments` after this plan is complete.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, optional LightGBM/SB3/Torch, pytest, Hypothesis, Ruff, Mypy, GitHub Actions, immutable filesystem artifacts.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md`

## Global Constraints

- Base implementation work on the latest verified stacked head containing PR #436 and PR #437; design-time exact base is `fbfe78553e414ae75674a81ab6bf606d969257e9`.
- Preserve fit-symbol/evaluation-symbol separation and fail-closed zero-eligible-row behavior from PR #436.
- Preserve `MarketExecutor + BookState` as the sole economic accounting authority.
- Preserve deterministic dataset and candidate-run identities; module relocation alone must not alter canonical bytes, digests, metrics, returns, or scope fields.
- Preserve immutable output publication and fail on existing destinations.
- Preserve package-level public exports; do not retain forwarding shims for private old module paths unless independent repository evidence proves the path is intentionally public.
- Keep `LICENSE`, `LICENSES/*`, `pyproject.toml` license metadata, and third-party provenance semantics intact.
- Do not change strategy thresholds, model architecture, training budgets, reward, risk limits, execution/accounting semantics, or profitability status.
- Every pre-refactor production file must end with exactly one classification: KEEP, MOVE, or DELETE.
- CI must run Ruff, Format, Mypy and the complete `tests/` tree on the final layout.

---

## Task 1: Freeze the migration inventory and architecture RED contracts

**Files:**
- Create: `docs/plans/2026-09-08-lean-package-file-inventory.md`
- Create: `tests/architecture/__init__.py`
- Create: `tests/architecture/test_lean_package_layout.py`
- Create: `tests/architecture/test_lean_dependency_boundaries.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: exact pre-refactor `trade_rl/` tree on the implementation base.
- Produces: a complete KEEP/MOVE/DELETE table and executable rules used by every later task.

- [ ] **Step 1: Write the full inventory before moving code**

The inventory must enumerate every Python file under `trade_rl/` and assign exactly one of:

```text
KEEP   <current path>
MOVE   <current path> -> <new path>
DELETE <current path> : <independent reason/evidence>
```

At minimum it must explicitly classify `_source_checkout.py`, all `domain/*`, all `data/*`, every current `simulation/stateful_*`, `target_*`, `order_*`, all strategy modules, all evaluation modules/subpackages, and every Binance integration file.

- [ ] **Step 2: Write failing layout tests**

`tests/architecture/test_lean_package_layout.py` must assert the final required structure, including:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_retired_domain_package_is_absent() -> None:
    assert not (PACKAGE / "domain").exists()


def test_strategy_families_have_explicit_packages() -> None:
    assert (PACKAGE / "strategies" / "rules" / "trend.py").is_file()
    assert (PACKAGE / "strategies" / "forecasts" / "ridge.py").is_file()
    assert (PACKAGE / "strategies" / "rl" / "ppo.py").is_file()


def test_binance_is_a_package_not_a_god_module() -> None:
    assert not (PACKAGE / "integrations" / "binance.py").exists()
    for name in ("transport.py", "cache.py", "vision.py", "metadata.py", "dataset.py"):
        assert (PACKAGE / "integrations" / "binance" / name).is_file()
```

The same test module must assert absence of the explicitly deleted compatibility files from the inventory.

- [ ] **Step 3: Write failing AST dependency tests**

`tests/architecture/test_lean_dependency_boundaries.py` must parse imports under `trade_rl/` with `ast` and enforce:

```text
artifacts -> no data/risk/simulation/strategies/evaluation/integrations
_validation -> stdlib only
data -> no strategies/evaluation/simulation
integrations -> no strategies/evaluation
strategies -> no evaluation
risk -> no strategies/evaluation
simulation -> no strategies/evaluation
evaluation -> may use lower core packages
```

It must report the exact importer and forbidden target when a violation is found.

- [ ] **Step 4: Run RED architecture tests**

Run:

```bash
uv run pytest -q tests/architecture
```

Expected: FAIL because the old layout still contains `trade_rl/domain`, flat strategy modules, and `integrations/binance.py`.

- [ ] **Step 5: Make CI discover the whole package and whole test tree**

Replace hand-maintained Ruff/Format/Mypy/test directory lists with:

```yaml
- name: Ruff
  run: uv run ruff check trade_rl tests
- name: Format
  run: uv run ruff format --check trade_rl tests
- name: Mypy
  run: uv run mypy trade_rl
- name: Tests
  run: uv run pytest -q tests
```

- [ ] **Step 6: Commit**

```bash
git add docs/plans/2026-09-08-lean-package-file-inventory.md tests/architecture .github/workflows/ci.yml
git commit -m "test: define lean package boundary migration"
```

---

## Task 2: Remove `domain` and make artifact/evaluation ownership real

**Files:**
- Create: `trade_rl/_validation.py`
- Create: `trade_rl/artifacts/canonical.py`
- Create: `trade_rl/evaluation/gates/__init__.py`
- Create: `trade_rl/evaluation/gates/models.py`
- Create: `trade_rl/evaluation/gates/resolve.py`
- Delete: `trade_rl/domain/__init__.py`
- Delete: `trade_rl/domain/common.py`
- Delete: `trade_rl/domain/canonical_json.py`
- Delete: `trade_rl/domain/evaluation.py`
- Delete when no intentional public consumer remains: `trade_rl/artifacts/codec.py`
- Modify: `trade_rl/artifacts/__init__.py`
- Modify: `trade_rl/artifacts/hashing.py`
- Modify: all imports currently using `trade_rl.domain.*`
- Modify: `trade_rl/evaluation/__init__.py`
- Test: `tests/artifacts/test_canonical_json_shared.py`
- Test: `tests/artifacts/test_codec.py`
- Test: `tests/evaluation/test_gates.py`
- Test: `tests/architecture/test_lean_package_layout.py`

**Interfaces:**
- Produces `trade_rl._validation.require_non_empty`, `require_sha256`, `require_git_sha`, `require_aware_datetime`, `require_unique_non_empty`.
- Produces `trade_rl.artifacts.canonical.canonical_json_bytes` and `to_json_value`.
- Produces `trade_rl.evaluation.gates.GateCheck`, `GateDecision`, `resolve_gate`.

- [ ] **Step 1: Strengthen RED against forwarding compatibility**

Add assertions that `trade_rl/domain` is absent and no source file imports `trade_rl.domain`.

- [ ] **Step 2: Move validation helpers without semantic change**

Copy only maintained `require_*` functions into `trade_rl/_validation.py`. Do not copy unused `domain_content_digest()`.

- [ ] **Step 3: Make artifact canonical encoding authoritative**

Move canonical JSON implementation into `artifacts/canonical.py` and update `hashing.py` to import it directly. Tests must compare exact bytes for nested dataclasses/enums/datetimes/paths and must reject NaN/Inf and non-string mapping keys.

- [ ] **Step 4: Move gate records next to gate resolution**

Move `GateCheck`/`GateDecision` verbatim in behavior to `evaluation/gates/models.py`; move `resolve_gate` to `evaluation/gates/resolve.py`; export all three from `evaluation/gates/__init__.py` and preserve `from trade_rl.evaluation import resolve_gate`.

- [ ] **Step 5: Delete `domain` and old forwarding codec**

Search repository imports first. If only internal/test compatibility references remain, update them and delete `artifacts/codec.py`. Do not create `domain` or `codec.py` forwarding shims.

- [ ] **Step 6: Run targeted verification**

```bash
uv run pytest -q tests/artifacts tests/evaluation/test_gates.py tests/architecture
uv run ruff check trade_rl/_validation.py trade_rl/artifacts trade_rl/evaluation/gates tests/artifacts tests/evaluation/test_gates.py tests/architecture
uv run ruff format --check trade_rl/_validation.py trade_rl/artifacts trade_rl/evaluation/gates tests/artifacts tests/evaluation/test_gates.py tests/architecture
uv run mypy trade_rl/_validation.py trade_rl/artifacts trade_rl/evaluation/gates
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl tests
git commit -m "refactor: remove transitional domain package"
```

---

## Task 3: Consolidate dataset artifacts, builder, and feature ownership

**Files:**
- Create: `trade_rl/data/artifacts/__init__.py`
- Create: `trade_rl/data/artifacts/model.py`
- Create: `trade_rl/data/artifacts/codec.py`
- Create: `trade_rl/data/artifacts/publication.py`
- Create: `trade_rl/data/build/__init__.py`
- Create: `trade_rl/data/build/config.py`
- Create: `trade_rl/data/build/builder.py`
- Create: `trade_rl/data/features/__init__.py`
- Create: `trade_rl/data/features/core.py`
- Create: `trade_rl/data/features/cross_asset.py`
- Create: `trade_rl/data/features/economic.py`
- Create: `trade_rl/data/features/multitimeframe.py`
- Delete: `trade_rl/data/artifact.py`
- Delete: `trade_rl/data/artifact_codec.py`
- Delete: `trade_rl/data/artifacts.py`
- Delete: `trade_rl/data/builder.py`
- Delete: `trade_rl/data/config.py`
- Delete: `trade_rl/data/features.py`
- Delete: `trade_rl/data/cross_asset_features.py`
- Delete: `trade_rl/data/economic_semantics.py`
- Delete: `trade_rl/data/multitimeframe.py`
- Modify: `trade_rl/data/__init__.py`
- Modify: all internal imports and affected tests

**Interfaces:**
- Preserve `from trade_rl.data import MarketDataset, PublishedDatasetArtifact, load_market_dataset_artifact, inspect_published_market_dataset_artifact, publish_market_dataset_artifact, write_market_dataset_files`.
- Move `MarketDatasetView` to `data/artifacts/model.py`.
- Remove deprecated `write_market_dataset_artifact()` completely.

- [ ] **Step 1: Add RED asserting deprecated writer absence**

Replace warning-based compatibility expectation with:

```python
import trade_rl.data as data


def test_deprecated_dataset_writer_is_removed() -> None:
    assert not hasattr(data, "write_market_dataset_artifact")
```

Also assert the legacy modules `trade_rl.data.artifact`, `artifact_codec`, and file `data/artifacts.py` no longer exist after the migration.

- [ ] **Step 2: Move artifact model/codec/publication preserving exact serialized bytes**

Use existing dataset artifact tests as byte/digest oracle; output manifest/NPZ identity must not change from module relocation alone.

- [ ] **Step 3: Move build config/builder**

Move code without algorithmic edits. Package-level exports stay unchanged for maintained public build types.

- [ ] **Step 4: Move feature families**

Move existing implementations into `features/core.py`, `cross_asset.py`, `economic.py`, `multitimeframe.py`; `features/__init__.py` may re-export maintained feature functions but old private module files are deleted.

- [ ] **Step 5: Run data oracles**

```bash
uv run pytest -q tests/data tests/architecture
uv run ruff check trade_rl/data tests/data tests/architecture
uv run ruff format --check trade_rl/data tests/data tests/architecture
uv run mypy trade_rl/data
```

Expected: all PASS and no dataset identity/digest regression.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/data tests/data tests/architecture
git commit -m "refactor: organize causal data package"
```

---

## Task 4: Split Binance ingestion into an explicit adapter package

**Files:**
- Create: `trade_rl/integrations/binance/__init__.py`
- Create: `trade_rl/integrations/binance/transport.py`
- Create: `trade_rl/integrations/binance/cache.py`
- Create: `trade_rl/integrations/binance/vision.py`
- Create: `trade_rl/integrations/binance/metadata.py`
- Create: `trade_rl/integrations/binance/dataset.py`
- Delete: `trade_rl/integrations/binance.py`
- Move/update: `trade_rl/integrations/binance_cache.py` responsibilities into `binance/cache.py`
- Move/update: `trade_rl/integrations/frozen_binance_metadata.py` responsibilities into `binance/metadata.py`
- Modify: `trade_rl/integrations/__init__.py`
- Modify: `tests/integrations/*`, `tests/data/test_binance_contract_metadata.py`

**Interfaces:**
- Preserve package-level `BinanceMarket`, `BinancePublicTransport`, `BinanceTransportMode`, `FrozenBinanceExchangeInfoTransport`.
- `transport.py` owns HTTP/retry mechanics only.
- `cache.py` owns cache paths/evidence verification only.
- `vision.py` owns Vision URL planning and archive parsing only.
- `metadata.py` owns exchange-info snapshots and contract conversion.
- `dataset.py` owns assembly into `MarketDataset`.

- [ ] **Step 1: Add RED package-layout and public-export tests**
- [ ] **Step 2: Extract cache/transport while preserving retry/cache fixture behavior**
- [ ] **Step 3: Extract Vision URL/parsing behavior**
- [ ] **Step 4: Extract metadata and dataset assembly**
- [ ] **Step 5: Delete old Binance modules and update imports**
- [ ] **Step 6: Run adapter verification**

```bash
uv run pytest -q tests/integrations tests/data/test_binance_contract_metadata.py tests/architecture
uv run ruff check trade_rl/integrations tests/integrations tests/data/test_binance_contract_metadata.py
uv run ruff format --check trade_rl/integrations tests/integrations tests/data/test_binance_contract_metadata.py
uv run mypy trade_rl/integrations
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/integrations tests/integrations tests/data/test_binance_contract_metadata.py tests/architecture
git commit -m "refactor: split binance ingestion responsibilities"
```

---

## Task 5: Group strategy families without changing fit/inference semantics

**Files:**
- Create: `trade_rl/strategies/rules/{__init__.py,trend.py,mean_reversion.py}`
- Create: `trade_rl/strategies/forecasts/{__init__.py,controller.py,supervised.py,ridge.py,lightgbm.py}`
- Create: `trade_rl/strategies/rl/{__init__.py,ppo.py}`
- Keep: `trade_rl/strategies/{__init__.py,interface.py,position_intent.py,controls.py}`
- Delete old flat private modules: `trend.py`, `mean_reversion.py`, `forecast.py`, `supervised.py`, `ridge.py`, `lightgbm.py`, `ppo.py`
- Modify: `trade_rl/strategies/__init__.py`
- Modify: `trade_rl/evaluation/runs/*` after Task 8 or current candidate imports during this task
- Modify: `tests/strategies/*`, affected evaluation tests

**Interfaces:**
- Preserve all symbols currently exported from `trade_rl.strategies`.
- Preserve explicit fit-symbol scope arguments from #436 for Ridge/LightGBM/PPO.

- [ ] **Step 1: Add RED structure/public-export tests**
- [ ] **Step 2: Move rule strategies**
- [ ] **Step 3: Move forecast controller/training/models**
- [ ] **Step 4: Move PPO**
- [ ] **Step 5: Remove old private modules and rewrite repository imports**
- [ ] **Step 6: Run strategy and candidate-scope oracles**

```bash
uv run pytest -q tests/strategies tests/evaluation/test_candidate_suite.py tests/evaluation/test_candidate_run_artifact.py tests/architecture
uv run ruff check trade_rl/strategies tests/strategies
uv run ruff format --check trade_rl/strategies tests/strategies
uv run mypy trade_rl/strategies
```

Expected: fit scope, equal-symbol supervised weighting, PPO round-robin scope, and public strategy exports all pass unchanged.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/strategies trade_rl/evaluation tests/strategies tests/evaluation tests/architecture
git commit -m "refactor: organize strategy families"
```

---

## Task 6: Group simulation order/state/target/diagnostic responsibilities

**Files:**
- Keep at root: `accounting.py`, `execution.py`, `bar_path.py`, `liquidity.py`, `execution_stress.py`, `__init__.py`
- Create/move into `simulation/orders/`: `model.py` (from `orders.py`), `admission.py`, `reconciliation.py`
- Create/move into `simulation/stateful/`: `execution.py`, `bar_lifecycle.py`, `order_transitions.py`, `runtime.py`, `symbol_fills.py`
- Create/move into `simulation/targets/`: `execution.py`, `exposure_controller.py`
- Create/move into `simulation/diagnostics/`: `funding.py`, `runtime_performance.py`, `runtime_performance_io.py`
- Delete corresponding old flat modules after import migration
- Modify all internal consumers and tests

**Interfaces:**
- Preserve `from trade_rl.simulation import BookState, EconomicTerminationReason, ExecutionCostConfig, ExecutionResult, MarketExecutor, ExecutionEnvironmentStress`.
- No fill/funding/borrow/termination/risk ordering changes are permitted.

- [ ] **Step 1: Add RED layout tests for every current stateful/target/order module**
- [ ] **Step 2: Move order model/admission/reconciliation**
- [ ] **Step 3: Move stateful lifecycle modules**
- [ ] **Step 4: Move target controller modules**
- [ ] **Step 5: Move diagnostics**
- [ ] **Step 6: Delete all old flat copies and search for stale imports**
- [ ] **Step 7: Run economic/state-transition oracles**

```bash
uv run pytest -q tests/simulation tests/risk tests/evaluation/test_execution_evidence.py tests/architecture
uv run ruff check trade_rl/simulation tests/simulation
uv run ruff format --check trade_rl/simulation tests/simulation
uv run mypy trade_rl/simulation
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/simulation tests/simulation tests/architecture
git commit -m "refactor: organize simulation responsibilities"
```

---

## Task 7: Keep risk small and remove only stale structural debt

**Files:**
- Keep: `trade_rl/risk/{__init__.py,inputs.py,portfolio.py,pretrade.py,emergency.py}`
- Modify imports only as required by Tasks 2/3/6
- Test: `tests/risk/*`, architecture tests

**Interfaces:** unchanged public `trade_rl.risk` exports.

- [ ] **Step 1: Assert risk remains a small flat package**
- [ ] **Step 2: Rewrite imports away from deleted `domain`/old simulation/data paths**
- [ ] **Step 3: Run risk projection tests and Mypy**

```bash
uv run pytest -q tests/risk tests/architecture
uv run mypy trade_rl/risk
```

- [ ] **Step 4: Commit**

```bash
git add trade_rl/risk tests/risk tests/architecture
git commit -m "refactor: align risk imports with lean boundaries"
```

---

## Task 8: Group evaluation into gates, comparison, robustness, and runs

**Files:**
- Keep: `evaluation/{__init__.py,replay.py,metrics.py,evidence.py}`
- Gates already created in Task 2
- Move `comparisons.py`, `strategy_comparison.py`, `bootstrap.py`, `seed_robustness.py` -> `evaluation/comparison/`
- Move `capacity.py`, `closed_trades.py`, `fold_metrics.py` -> `evaluation/robustness/`
- Move `_perfect_information_lp.py` and `perfect_information_bound.py` -> `evaluation/robustness/perfect_information/{solver.py,bound.py}`
- Move existing `evaluation/walk_forward/*` -> `evaluation/robustness/walk_forward/*`
- Move `candidate_run.py`, `candidate_suite.py` -> `evaluation/runs/{candidate.py,candidate_suite.py}`
- Delete all old private flat paths after import migration
- Modify: `trade_rl/evaluation/__init__.py`, tests/evaluation/*, README/docs invocation examples

**Interfaces:**
- Preserve current package-level evaluation exports.
- Preserve executable invocation through `python -m trade_rl.evaluation.runs.candidate` and update README/docs accordingly.
- Preserve candidate artifact schema/identity/scope fields unless a deliberate schema migration is separately specified; this plan does not authorize one.

- [ ] **Step 1: Add RED layout/public-export/run-entry tests**
- [ ] **Step 2: Move comparison modules**
- [ ] **Step 3: Move robustness/perfect-information/walk-forward modules**
- [ ] **Step 4: Move candidate run/suite**
- [ ] **Step 5: Delete old paths and update all imports/docs**
- [ ] **Step 6: Run evaluation oracles**

```bash
uv run pytest -q tests/evaluation tests/architecture
uv run ruff check trade_rl/evaluation tests/evaluation
uv run ruff format --check trade_rl/evaluation tests/evaluation
uv run mypy trade_rl/evaluation
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation tests/evaluation README.md docs tests/architecture
git commit -m "refactor: organize evaluation research responsibilities"
```

---

## Task 9: Rebuild current-only docs and agent routing

**Files:**
- Create: `AGENTS.md`
- Create: `docs/README.md`
- Create: `docs/AGENTS.md`
- Create: `docs/architecture/README.md`
- Create: `docs/architecture/lean-core.md`
- Create: `docs/architecture/package-boundaries.md`
- Create: `docs/research/README.md`
- Create: `docs/research/current-status.md`
- Keep while active: `docs/specs/2026-09-08-lean-package-boundaries-design.md`
- Keep while active: `docs/plans/2026-09-08-lean-package-boundaries.md`
- Delete after canonical content is transferred: `docs/trade_rl_lean_redesign_20260908.md`
- Modify: `README.md`
- Test: `tests/architecture/test_current_docs.py`

**Interfaces:** Agent entry is root `AGENTS.md` -> `docs/AGENTS.md` -> current architecture/research docs.

- [ ] **Step 1: Write RED docs routing test**

Assert current docs exist, links resolve locally, forbidden `docs/history`/`docs/archive` directories do not exist, and README points to `docs/README.md` or canonical architecture docs.

- [ ] **Step 2: Write current-only docs**

Move durable content from the dated redesign document into architecture/research roles; do not copy historical implementation narratives.

- [ ] **Step 3: Add agent maintenance contract**

`docs/AGENTS.md` must state where current authority lives, where active specs/plans live, when they are deleted, and that Git history is the archive.

- [ ] **Step 4: Delete the dated monolithic redesign doc after link migration**
- [ ] **Step 5: Run docs/architecture tests**

```bash
uv run pytest -q tests/architecture
```

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md README.md docs tests/architecture
git commit -m "docs: establish current lean documentation routing"
```

---

## Task 10: Final deletion audit, falsification, and exact-head quality gate

**Files:**
- Modify only files found defective by final review.
- Delete: implementation-time migration helpers or stale compatibility files discovered by audit.
- Update: `docs/plans/2026-09-08-lean-package-file-inventory.md` so every entry reflects the final path/action.

**Interfaces:** final repository tree and package public APIs.

- [ ] **Step 1: Verify every inventory entry**

No production file may be unclassified. Every MOVE old path must be absent and new path present. Every DELETE path must be absent with reason retained in the inventory.

- [ ] **Step 2: Search for forbidden legacy paths/names**

Fail if maintained source/tests/docs still import or describe:

```text
trade_rl.domain
trade_rl.integrations.binance (as a .py god module)
old flat private strategy/evaluation/simulation paths
retired residual/U-series architecture as current
```

- [ ] **Step 3: Falsification review**

Attempt to find a refactor that would pass tests while violating behavior. Explicitly inspect:

- canonical JSON/digest identity;
- dataset artifact identity/publication;
- fit symbol scope vs evaluation symbol scope;
- zero-eligible fit-symbol failure;
- deterministic per-symbol replay raw returns;
- MarketExecutor/BookState fill and accounting state transitions;
- Binance cache/retry/metadata/source assembly;
- package-level `__all__` public symbols;
- no hidden forwarding compatibility modules;
- CI collecting all test directories.

Add a regression test before fixing any discovered defect.

- [ ] **Step 4: Run targeted-to-global verification**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Expected: all PASS. Record exact counts, not only “Green”.

- [ ] **Step 5: Inspect final diff and status**

Verify no temporary scripts/workflows/generated artifacts, no unexpected license changes, and no strategy/economic changes beyond import/module ownership.

- [ ] **Step 6: Push exact HEAD and require same-head GitHub Actions success**

Do not use older workflow evidence. Verify the CI run checks out the final PR head SHA and passes Ruff, Format, Mypy, full tests, and package identity.

- [ ] **Step 7: Independent requirements-first review**

Reconstruct the acceptance criteria from the design, then compare them against the final tree/diff/tests without relying on implementation conclusions. Record unverified items and remaining risks.

- [ ] **Step 8: Update PR body with evidence and keep Draft until all quality gates pass**

Do not merge to `main` without explicit user authorization.
