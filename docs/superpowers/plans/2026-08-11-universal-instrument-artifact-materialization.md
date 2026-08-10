# Universal Instrument Artifact Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize one immutable, cross-bound bundle containing `stored-instruments.json`, `symbol-disjoint.json`, and `universal-instrument-partition.json` from a verified PostgreSQL indicator inventory and execution-metadata digests.

**Architecture:** Keep evidence DTO ownership in `trade_rl.catalog`, partition composition in `trade_rl.workflows`, and PostgreSQL access in `trade_rl.integrations`. A focused core workflow builds all three in memory, validates cross-digest closure, writes them into a sibling staging directory, reloads them, and atomically publishes the dedicated bundle directory. A thin PostgreSQL orchestration workflow loads `StoredIndicatorSourceInventory` through the existing metadata-only adapter and delegates to the core workflow.

**Tech Stack:** Python 3.12, standard-library dataclasses/pathlib/os/uuid, existing canonical JSON and content-digest contracts, pytest, Ruff, MyPy, Import Linter, GitHub Actions PostgreSQL service workflow.

## Global Constraints

- The approved source design is `docs/architecture/universal-single-instrument-zero-shot-design.md`.
- The bundle contains exactly three files: `stored-instruments.json`, `symbol-disjoint.json`, and `universal-instrument-partition.json`.
- The output path is a dedicated bundle directory; unrelated files are not allowed inside it.
- A new bundle is published with one same-filesystem directory rename only after all three files reload successfully.
- Existing output may be reused only when all file bytes and decoded contracts match the requested bundle exactly.
- Any mismatch, partial directory, symlink, unknown file, catalog/manifest/partition digest mismatch, or fewer than 15 eligible symbols fails closed.
- The workflow never reads `npz_payload`; PostgreSQL discovery remains metadata-only.
- Execution metadata is supplied as a symbol-to-SHA-256 mapping; obtaining or approving that metadata is outside this change.
- The current full-training generation, checkpoint, PPO/BC behavior, serving path, and Docker runtime remain untouched.
- No `main` merge, production authorization, episode router, normalizer, dataset loader, or training generation is part of this plan.

---

### Task 1: Cross-bound artifact bundle contract

**Files:**
- Create: `trade_rl/workflows/universal_instrument_artifacts.py`
- Create: `tests/workflows/test_universal_instrument_artifacts.py`

**Interfaces:**
- Consumes: `StoredInstrumentCatalog`, `SymbolDisjointManifest`, and `UniversalInstrumentPartition`.
- Produces: `UniversalInstrumentArtifactBundle`, `UniversalInstrumentArtifactPaths`, and strict filename constants.

- [ ] **Step 1: Write the failing cross-binding tests**

Create fixtures for two distinct valid catalogs and partitions. Pin these behaviors:

```python
def test_bundle_requires_catalog_manifest_partition_cross_binding() -> None:
    catalog = _catalog(seed_source="catalog-a")
    partition = build_universal_instrument_partition(catalog, seed=17)

    bundle = UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        partition=partition,
    )

    assert bundle.partition.catalog_digest == bundle.catalog.digest
    assert bundle.partition.symbol_disjoint_manifest == bundle.symbol_disjoint_manifest


def test_bundle_rejects_manifest_from_another_partition() -> None:
    first = _bundle(seed=17)
    second = _bundle(seed=23)

    with pytest.raises(ValueError, match="manifest|partition"):
        UniversalInstrumentArtifactBundle(
            catalog=first.catalog,
            symbol_disjoint_manifest=second.symbol_disjoint_manifest,
            partition=first.partition,
        )
```

Also require the catalog eligible set to equal the manifest source universe exactly and expose paths ending in the three required filenames.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/workflows/test_universal_instrument_artifacts.py::test_bundle_requires_catalog_manifest_partition_cross_binding \
  tests/workflows/test_universal_instrument_artifacts.py::test_bundle_rejects_manifest_from_another_partition
```

Expected: collection fails because `trade_rl.workflows.universal_instrument_artifacts` does not exist.

- [ ] **Step 3: Implement the minimal immutable contracts**

Create:

```python
STORED_INSTRUMENTS_FILENAME = "stored-instruments.json"
SYMBOL_DISJOINT_FILENAME = "symbol-disjoint.json"
UNIVERSAL_INSTRUMENT_PARTITION_FILENAME = "universal-instrument-partition.json"

@dataclass(frozen=True, slots=True)
class UniversalInstrumentArtifactBundle:
    catalog: StoredInstrumentCatalog
    symbol_disjoint_manifest: SymbolDisjointManifest
    partition: UniversalInstrumentPartition

@dataclass(frozen=True, slots=True)
class UniversalInstrumentArtifactPaths:
    root: Path
    stored_instruments: Path
    symbol_disjoint: Path
    universal_partition: Path
```

`UniversalInstrumentArtifactBundle.__post_init__` must verify concrete types, exact catalog digest binding, exact manifest equality/digest binding, and exact eligible-universe closure. `UniversalInstrumentArtifactPaths.for_root(root)` must derive all paths without touching the filesystem.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```bash
uv run pytest -q tests/workflows/test_universal_instrument_artifacts.py
uv run ruff check trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
uv run ruff format --check trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
uv run mypy trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
```

Expected: bundle tests pass; static checks pass.

- [ ] **Step 5: Commit the contract**

```bash
git add trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
git commit -m "feat(workflows): define universal instrument artifact bundle"
```

---

### Task 2: Deterministic bundle construction

**Files:**
- Modify: `trade_rl/workflows/universal_instrument_artifacts.py`
- Modify: `tests/workflows/test_universal_instrument_artifacts.py`

**Interfaces:**
- Consumes: `StoredIndicatorSourceInventory`, research interval, execution metadata digests, and partition seed.
- Produces: `build_universal_instrument_artifact_bundle(...) -> UniversalInstrumentArtifactBundle`.

- [ ] **Step 1: Write failing construction tests**

Pin exact composition:

```python
def test_builds_catalog_manifest_and_partition_from_one_source() -> None:
    source = _source(symbol_count=16)
    metadata = _metadata(source.symbols)
    metadata.pop(source.symbols[0])

    bundle = build_universal_instrument_artifact_bundle(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=metadata,
        seed=17,
    )

    assert bundle.catalog.excluded_symbols[0].symbol == source.symbols[0]
    assert bundle.symbol_disjoint_manifest.source_universe == tuple(
        sorted(bundle.catalog.eligible_symbols)
    )
    assert bundle.partition.catalog_digest == bundle.catalog.digest
```

Pin fail-closed behavior when exclusions leave fewer than 15 eligible symbols, unknown metadata is supplied, the interval exceeds source coverage, seed is invalid, and repeated construction with the same inputs produces equal objects and digests.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
uv run pytest -q tests/workflows/test_universal_instrument_artifacts.py -k "builds or fewer or deterministic"
```

Expected: failures because `build_universal_instrument_artifact_bundle` is missing.

- [ ] **Step 3: Implement construction by composing existing contracts**

Implement:

```python
def build_universal_instrument_artifact_bundle(
    source: StoredIndicatorSourceInventory,
    *,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    seed: int,
) -> UniversalInstrumentArtifactBundle:
    catalog = build_stored_instrument_catalog(
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
    )
    partition = build_universal_instrument_partition(catalog, seed=seed)
    return UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        partition=partition,
    )
```

Do not duplicate split math, catalog eligibility rules, digest calculation, or symbol ranking.

- [ ] **Step 4: Run focused tests and static checks**

Run the same focused pytest/Ruff/format/MyPy commands from Task 1.

Expected: all construction and contract tests pass.

- [ ] **Step 5: Commit deterministic construction**

```bash
git add trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
git commit -m "feat(workflows): build universal instrument artifact bundle"
```

---

### Task 3: Directory-level atomic publication and strict loading

**Files:**
- Modify: `trade_rl/workflows/universal_instrument_artifacts.py`
- Modify: `tests/workflows/test_universal_instrument_artifacts.py`

**Interfaces:**
- Consumes: `UniversalInstrumentArtifactBundle` and a dedicated output directory.
- Produces:
  - `write_universal_instrument_artifact_bundle(root, bundle) -> UniversalInstrumentArtifactPaths`
  - `load_universal_instrument_artifact_bundle(root) -> UniversalInstrumentArtifactBundle`
  - `materialize_universal_instrument_artifacts(...) -> UniversalInstrumentArtifactPaths`

- [ ] **Step 1: Write failing publication tests**

Pin these behaviors:

```python
def test_materializes_exact_file_closure_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "universal-instruments"
    expected = _bundle(seed=17)

    paths = write_universal_instrument_artifact_bundle(output, expected)

    assert {path.name for path in output.iterdir()} == {
        "stored-instruments.json",
        "symbol-disjoint.json",
        "universal-instrument-partition.json",
    }
    assert load_universal_instrument_artifact_bundle(output) == expected
    assert paths.root == output


def test_exact_existing_bundle_is_reused_but_mismatch_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "universal-instruments"
    first = _bundle(seed=17)
    second = _bundle(seed=23)
    write_universal_instrument_artifact_bundle(output, first)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    write_universal_instrument_artifact_bundle(output, first)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    with pytest.raises(FileExistsError, match="different|mismatch"):
        write_universal_instrument_artifact_bundle(output, second)
```

Also test a partial existing directory, extra file, symlink root, tampered dependency, staging writer failure, and concurrent exact reuse.

- [ ] **Step 2: Run publication tests and confirm RED**

Run:

```bash
uv run pytest -q tests/workflows/test_universal_instrument_artifacts.py -k "materializes or existing or partial or staged or tampered"
```

Expected: failures because publication/loading functions are missing.

- [ ] **Step 3: Implement strict load and exact reuse**

`load_universal_instrument_artifact_bundle(root)` must reject a missing path, non-directory, symlink, or non-exact filename set; load catalog, manifest, and partition in dependency order; and construct the bundle again to repeat cross-binding validation.

- [ ] **Step 4: Implement staged atomic publication**

`write_universal_instrument_artifact_bundle` must validate in memory; exact-load existing output; otherwise write all three files to a unique sibling `.<root>.staging-<pid>-<uuid>` directory; reload it; atomically rename it to the final root; fsync the parent on POSIX; clean staging on failure; and never delete or replace existing output. A concurrent winner is accepted only when strict loading proves exact equality.

- [ ] **Step 5: Implement the one-call materializer**

`materialize_universal_instrument_artifacts(...)` must build the complete bundle before any filesystem operation, then publish it. Validation failures such as fewer than 15 eligible symbols must leave no output.

- [ ] **Step 6: Run focused and related tests**

```bash
uv run pytest -q tests/workflows/test_universal_instrument_artifacts.py
uv run pytest -q tests/catalog/test_stored_instrument_catalog.py tests/workflows/test_symbol_disjoint_manifest.py tests/workflows/test_universal_instrument_partition.py
uv run ruff check trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
uv run ruff format --check trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
uv run mypy trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
uv run lint-imports
```

Expected: all tests and architecture contracts pass.

- [ ] **Step 7: Commit atomic publication**

```bash
git add trade_rl/workflows/universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py
git commit -m "feat(workflows): atomically materialize universal instrument artifacts"
```

---

### Task 4: PostgreSQL orchestration boundary and dedicated CI

**Files:**
- Create: `trade_rl/workflows/postgres_universal_instrument_artifacts.py`
- Create: `tests/workflows/test_postgres_universal_instrument_artifacts.py`
- Modify: `.github/workflows/postgres-catalog.yml`

**Interfaces:**
- Consumes: `IndicatorArtifactConnection`, cache ID, research interval, execution metadata digests, seed, and output directory.
- Produces: `materialize_postgres_universal_instrument_artifacts(...) -> UniversalInstrumentArtifactPaths`.

- [ ] **Step 1: Write the failing orchestration test**

Use a DB-API fake returning one valid indicator manifest and 15×4 metadata rows. Require the orchestration function to create the exact bundle, pass the cache ID to both metadata-only queries, never select `npz_payload`, and fail before publication when missing metadata reduces eligibility below 15.

- [ ] **Step 2: Run the test and confirm RED**

```bash
uv run pytest -q tests/workflows/test_postgres_universal_instrument_artifacts.py
```

Expected: collection failure because the orchestration module is missing.

- [ ] **Step 3: Implement thin composition only**

```python
def materialize_postgres_universal_instrument_artifacts(
    connection: IndicatorArtifactConnection,
    *,
    output_dir: str | Path,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    seed: int,
    cache_id: str = INDICATOR_CACHE_ID,
) -> UniversalInstrumentArtifactPaths:
    source = load_postgres_indicator_source_inventory(connection, cache_id=cache_id)
    return materialize_universal_instrument_artifacts(
        output_dir,
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
        seed=seed,
    )
```

Do not add SQL, digest logic, or artifact-writing logic to this file.

- [ ] **Step 4: Extend PostgreSQL specialist CI**

Add both workflow modules and tests to the pull-request path filter and specialist pytest command while retaining exact-head checkout, PostgreSQL startup, and migrations.

- [ ] **Step 5: Run focused verification**

```bash
uv run pytest -q tests/workflows/test_universal_instrument_artifacts.py tests/workflows/test_postgres_universal_instrument_artifacts.py tests/integrations/test_postgres_indicator_inventory.py
uv run ruff check trade_rl/workflows/universal_instrument_artifacts.py trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py tests/workflows/test_postgres_universal_instrument_artifacts.py
uv run ruff format --check trade_rl/workflows/universal_instrument_artifacts.py trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py tests/workflows/test_postgres_universal_instrument_artifacts.py
uv run mypy trade_rl/workflows/universal_instrument_artifacts.py trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/workflows/test_universal_instrument_artifacts.py tests/workflows/test_postgres_universal_instrument_artifacts.py
uv run lint-imports
```

Expected: focused tests and static/architecture checks pass.

- [ ] **Step 6: Commit orchestration and CI**

```bash
git add trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/workflows/test_postgres_universal_instrument_artifacts.py .github/workflows/postgres-catalog.yml
git commit -m "feat(workflows): materialize PostgreSQL universal instrument bundle"
```

---

### Task 5: Exact-head repository verification and review

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes: final PR head.
- Produces: verified Ready PR; no merge.

- [ ] **Step 1: Self-review the complete diff**

Check single responsibility, duplicate logic, path traversal/symlink behavior, concurrent publication, staging cleanup, Windows directory rename behavior, error messages, digest binding, unknown-file rejection, and that no current runtime or training configuration changed.

- [ ] **Step 2: Run repository verification on one exact head**

Require the common CI and PostgreSQL Catalog workflow on the same SHA. The common CI must pass frontend tests/build/layout, workflow security, Ruff, format, MyPy, Import Linter, Vulture, recovery/structured-serving smoke, full pytest with branch coverage, critical ratchets, package identity, Windows/Ubuntu compatibility, and complete training-image probe.

- [ ] **Step 3: Confirm PR scope and review state**

Require only the plan, two workflow modules, two workflow tests, and PostgreSQL workflow. Reject temporary workflows, generated artifacts, secrets, debug output, or unrelated refactoring. Require zero unresolved review threads.

- [ ] **Step 4: Update PR description and mark Ready**

Document What, Why, design decisions, exact scope, RED/GREEN evidence, non-goals, residual risks, and Production **NO-GO**. Do not merge.
