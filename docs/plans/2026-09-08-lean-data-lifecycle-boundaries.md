# Lean Data Lifecycle Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Reorganize `trade_rl.data` into explicit artifact, build, feature, and view responsibilities while preserving exact dataset semantics, artifact bytes/digests, package-level public APIs, and all causal-data contracts.

**Architecture:** Keep the central dataset model/contracts/identity/source at `data/` root. Move artifact codec/publication into `data/artifacts/`, build config/builder into `data/build/`, feature implementations into `data/features/`, and move `MarketDatasetView` to `data/view.py`. Delete deprecated/ambiguous compatibility paths instead of forwarding them.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md` + Amendment 1.

## Global Constraints

- Base is exact verified Phase-1 head `e347f9838e3b1ed972bedabd322d09dee2a902ce`.
- Preserve `MarketDataset`, `MarketCalendarKind`, `InstrumentExecutionRule`, `PublishedDatasetArtifact`, `DatasetArtifactFiles`, `inspect_published_market_dataset_artifact`, `load_market_dataset_artifact`, `publish_market_dataset_artifact`, and `write_market_dataset_files` as package-level `trade_rl.data` exports.
- Remove deprecated `write_market_dataset_artifact()` entirely; do not keep a warning shim.
- Dataset manifest JSON, deterministic NPZ bytes, artifact digest, `dataset_id`, view identity, feature values/availability, and builder outputs must not change because of module relocation.
- No Binance behavior change in this sub-project; only its imports may be updated to the new `data` owners.
- No strategy/risk/simulation/evaluation numerical behavior change.
- Same-head full CI is required before this phase is considered verified.

---

## Task 1: Establish data-layout RED contracts

**Files:**
- Create: `tests/architecture/test_lean_data_layout.py`
- Modify: `tests/data/test_market_artifact.py`
- Modify: `tests/data/test_market_dataset_artifact.py`

- [ ] **Step 1: Require the final folders before production moves**

`tests/architecture/test_lean_data_layout.py` must assert:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "trade_rl" / "data"


def test_data_lifecycle_packages_exist() -> None:
    for relative in (
        "artifacts/__init__.py",
        "artifacts/codec.py",
        "artifacts/publication.py",
        "build/__init__.py",
        "build/config.py",
        "build/builder.py",
        "features/__init__.py",
        "features/core.py",
        "features/cross_asset.py",
        "features/economic.py",
        "features/multitimeframe.py",
        "view.py",
    ):
        assert (DATA / relative).is_file(), relative


def test_flat_legacy_data_modules_are_absent() -> None:
    for name in (
        "artifact.py",
        "artifact_codec.py",
        "artifacts.py",
        "builder.py",
        "config.py",
        "features.py",
        "cross_asset_features.py",
        "economic_semantics.py",
        "multitimeframe.py",
    ):
        assert not (DATA / name).exists(), name
```

- [ ] **Step 2: Change the deprecated writer oracle from warning to absence**

In `tests/data/test_market_artifact.py`, remove the warning-based call to `write_market_dataset_artifact()` and add:

```python
def test_deprecated_direct_dataset_writer_is_removed() -> None:
    import trade_rl.data as data

    assert not hasattr(data, "write_market_dataset_artifact")
```

Any private-module compatibility test must be updated to assert the legacy module is absent rather than import it.

- [ ] **Step 3: Run RED**

```bash
uv run pytest -q tests/architecture/test_lean_data_layout.py tests/data/test_market_artifact.py tests/data/test_market_dataset_artifact.py
```

Expected: FAIL because new packages do not yet exist and the old flat modules still exist.

- [ ] **Step 4: Commit RED only**

```bash
git add tests/architecture/test_lean_data_layout.py tests/data/test_market_artifact.py tests/data/test_market_dataset_artifact.py
git commit -m "test: define lean data lifecycle layout"
```

---

## Task 2: Move dataset artifact/view ownership without byte drift

**Files:**
- Create: `trade_rl/data/artifacts/__init__.py`
- Move: `trade_rl/data/artifact_codec.py` -> `trade_rl/data/artifacts/codec.py`
- Move: `trade_rl/data/artifact.py` -> `trade_rl/data/artifacts/publication.py`
- Create: `trade_rl/data/view.py` from the `MarketDatasetView` responsibility in `trade_rl/data/artifacts.py`
- Delete: `trade_rl/data/artifacts.py`
- Modify: `trade_rl/data/__init__.py`
- Modify all repository imports of the old paths

**Public interfaces:** package-level exports listed in Global Constraints remain unchanged. `MarketDatasetView` is importable from `trade_rl.data.view`; it is not required to become a new top-level export.

- [ ] **Step 1: Move codec with no implementation edits beyond imports**

The exact existing `DATASET_*` constants, deterministic zip timestamp, manifest construction, digest verification, array shape/dtype checks, and write/load functions move to `data/artifacts/codec.py`.

- [ ] **Step 2: Move publication and remove deprecated writer**

`publication.py` retains `PublishedDatasetArtifact`, inspect/load/publish behavior and the `_write_arrays` test seam. Delete `warnings` import and `write_market_dataset_artifact()` completely.

- [ ] **Step 3: Extract `MarketDatasetView`**

Move `MarketDatasetView` and `DATASET_VIEW_SCHEMA` to `data/view.py`; do not duplicate dataset-loading helpers there.

- [ ] **Step 4: Rewrite all old artifact/view imports, then delete old paths**

Before deleting, require:

```bash
rg -n 'trade_rl\.data\.(artifact|artifact_codec|artifacts)' trade_rl tests
```

Every maintained result must be intentionally migrated. After deletion, the only appearances of old names may be architecture assertions or historical design text, not executable imports.

- [ ] **Step 5: Run artifact identity oracles**

```bash
uv run pytest -q tests/data/test_market_artifact.py tests/data/test_market_dataset_artifact.py tests/data/test_market_dataset_identity_v2.py tests/data/test_dataset_content_identity.py tests/artifacts tests/architecture
uv run mypy trade_rl/data
```

Expected: all PASS and existing byte/digest/identity assertions unchanged.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/data tests/data tests/architecture
git commit -m "refactor: colocate dataset artifact ownership"
```

---

## Task 3: Move build ownership

**Files:**
- Move: `trade_rl/data/config.py` -> `trade_rl/data/build/config.py`
- Move: `trade_rl/data/builder.py` -> `trade_rl/data/build/builder.py`
- Create: `trade_rl/data/build/__init__.py`
- Modify internal imports and affected tests/integrations

- [ ] **Step 1: Move code with behavior unchanged**
- [ ] **Step 2: Update all old `data.config` / `data.builder` imports**
- [ ] **Step 3: Require no executable old imports remain**

```bash
rg -n 'trade_rl\.data\.(config|builder)' trade_rl tests
```

- [ ] **Step 4: Run builder/config tests**

```bash
uv run pytest -q tests/data/test_market_builder.py tests/data/test_market_build_config_session.py tests/data/test_market_contracts.py tests/integrations tests/architecture/test_lean_data_layout.py
uv run mypy trade_rl/data trade_rl/integrations
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/data trade_rl/integrations tests
git commit -m "refactor: isolate dataset build responsibilities"
```

---

## Task 4: Move feature families

**Files:**
- Move: `trade_rl/data/features.py` -> `trade_rl/data/features/core.py`
- Move: `trade_rl/data/cross_asset_features.py` -> `trade_rl/data/features/cross_asset.py`
- Move: `trade_rl/data/economic_semantics.py` -> `trade_rl/data/features/economic.py`
- Move: `trade_rl/data/multitimeframe.py` -> `trade_rl/data/features/multitimeframe.py`
- Create: `trade_rl/data/features/__init__.py`
- Modify all internal/test imports

- [ ] **Step 1: Move each implementation wholesale before refactoring internals**
- [ ] **Step 2: Update imports and delete all old flat feature paths**
- [ ] **Step 3: Run causality and feature-value regression layers**

```bash
uv run pytest -q tests/data/test_indicator_features.py tests/data/test_extended_indicator_features_v2.py tests/data/test_cross_asset_features.py tests/data/test_extended_prefix_causality.py tests/data/test_information_availability.py tests/data/test_feature_availability_v2.py tests/data/test_economic_semantics.py tests/architecture/test_lean_data_layout.py
uv run mypy trade_rl/data
```

- [ ] **Step 4: Commit**

```bash
git add trade_rl/data tests/data tests/architecture
git commit -m "refactor: group causal feature families"
```

---

## Task 5: Phase 2A full falsification and exact-head gate

- [ ] **Step 1: Verify final layout and absence of all nine old flat files**
- [ ] **Step 2: Compare exact Phase-1 base to Phase-2A head**

Production changes outside `trade_rl/data` must be import-only, except architecture/tests/docs. No strategy model implementation or economic execution code may change.

- [ ] **Step 3: Run full verification**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

- [ ] **Step 4: Falsification review**

Specifically try to break:

- deterministic dataset NPZ bytes and manifest digest;
- canonical `dataset_id` reload;
- atomic publication failure cleanup;
- immutable destination rejection;
- `MarketDatasetView` half-open range and derived identity;
- feature availability/cutoff causality;
- package-level `trade_rl.data` public exports;
- Binance tests after import-only updates.

Any defect gets a failing regression before its fix.

- [ ] **Step 5: Require normal same-head GitHub Actions success**

Record exact Ruff/Format/Mypy/test/package-identity evidence. Keep the PR Draft and do not merge.
