# Universal Instrument Artifact Materialization Plan

## Goal

Materialize one immutable, cross-bound bundle from a verified stored-indicator inventory and execution-metadata digests. The bundle contains exactly:

- `stored-instruments.json`
- `symbol-disjoint.json`
- `universal-instrument-partition.json`

This implements the next U1 slice of `docs/architecture/universal-single-instrument-zero-shot-design.md`. Episode routing, normalization, dataset loading, training, serving, and production authorization are explicitly outside this change.

## Responsibility boundaries

- `trade_rl.catalog` continues to own stored-market evidence DTOs and catalog validation.
- `trade_rl.integrations` continues to own PostgreSQL metadata access and must never deserialize `npz_payload` in this flow.
- `trade_rl.workflows` composes catalog, symbol split, partition, and immutable artifact publication.
- The current full-training generation, checkpoints, PPO/BC behavior, Docker runtime, and serving path remain untouched.

## Core invariants

1. The catalog digest in the partition equals the exact stored-catalog digest.
2. The partition embeds the exact `SymbolDisjointManifest` written beside it.
3. The manifest source universe equals the catalog eligible-symbol closure.
4. Fewer than 15 eligible symbols fails before any filesystem write.
5. The published directory contains exactly the three required filenames.
6. Existing output is reusable only when strict decoding proves exact object equality.
7. Partial directories, unknown files, symlink roots, tampered digests, or cross-bound mismatches fail closed.
8. New publication occurs only after all staged files reload successfully.
9. Publication uses one same-filesystem directory rename; no existing root is deleted or replaced.
10. Every failure removes the unique staging directory and leaves no partially published root.

## Task 1: Bundle contracts

Create:

- `trade_rl/workflows/universal_instrument_artifacts.py`
- `tests/workflows/test_universal_instrument_artifacts.py`

Define filename constants plus immutable:

```python
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

`UniversalInstrumentArtifactBundle` validates concrete types and all catalog/manifest/partition cross-bindings. `UniversalInstrumentArtifactPaths.for_root()` derives paths without touching the filesystem.

## Task 2: Deterministic construction

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
    catalog = build_stored_instrument_catalog(...)
    partition = build_universal_instrument_partition(catalog, seed=seed)
    return UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        partition=partition,
    )
```

Reuse existing catalog and partition rules. Do not duplicate split math, eligibility logic, or digest calculation.

Required tests cover deterministic identity, one explicit exclusion while retaining 15 eligible symbols, invalid seed/input, interval outside source coverage, unknown metadata, and fail-closed behavior below the 15-symbol minimum.

## Task 3: Strict loading and atomic publication

Implement:

```python
load_universal_instrument_artifact_bundle(root)
write_universal_instrument_artifact_bundle(root, bundle)
materialize_universal_instrument_artifacts(root, source, ...)
```

Strict loading must reject missing paths, non-directories, symlinks, and non-exact filename closure. It loads catalog, manifest, and partition in dependency order, then constructs the bundle again to repeat cross-binding validation.

For a new root, publication must:

1. validate the complete in-memory bundle;
2. create a unique sibling staging directory named `.<root>.staging-<pid>-<uuid>`;
3. write the three canonical files using existing writers;
4. strict-load the staged directory and require equality;
5. atomically rename staging to the final root;
6. fsync the parent directory on POSIX;
7. remove staging content in `finally` on every failure.

If a concurrent writer wins, accept the existing root only when strict loading proves exact equality. Never delete or replace a pre-existing root.

Required tests cover exact round-trip, exact immutable reuse, mismatch rejection, partial/extra/non-directory/symlink rejection, tampering, injected staged-write failure, and validation before filesystem access.

## Task 4: PostgreSQL orchestration

Create:

- `trade_rl/workflows/postgres_universal_instrument_artifacts.py`
- `tests/workflows/test_postgres_universal_instrument_artifacts.py`

The orchestration function is intentionally thin:

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
    source = load_postgres_indicator_source_inventory(
        connection,
        cache_id=cache_id,
    )
    return materialize_universal_instrument_artifacts(
        output_dir,
        source,
        research_start=research_start,
        research_end=research_end,
        metadata_digests=metadata_digests,
        seed=seed,
    )
```

No SQL, digest logic, or artifact-writing logic belongs in this wrapper. Update `.github/workflows/postgres-catalog.yml` so its path filters and migrated-PostgreSQL test command include both new workflow modules and tests.

## TDD and verification sequence

1. Commit tests first and capture an exact-head RED caused only by the missing workflow module.
2. Implement the smallest core module that satisfies the bundle tests.
3. Add the PostgreSQL wrapper test, capture its RED, then implement the thin wrapper.
4. Run focused pytest, Ruff, Ruff format, MyPy, and Import Linter.
5. Self-review path/symlink/concurrency/staging cleanup and cross-digest behavior.
6. On one final head, require common CI plus PostgreSQL Catalog to pass:
   - frontend tests, typecheck, build, and layout checks;
   - workflow security;
   - Ruff and format;
   - MyPy;
   - Import Linter;
   - Vulture;
   - recovery and structured-serving smoke;
   - full pytest with branch coverage and critical ratchets;
   - Windows and Ubuntu compatibility;
   - complete training-image and non-root runtime probe;
   - migrated PostgreSQL specialist tests.
7. Confirm the PR contains only this plan, the two workflow modules, the two workflow test modules, and the PostgreSQL workflow update.
8. Mark Ready after exact-head verification; do not merge without an explicit owner decision.

Production remains **NO-GO** until later data, routing, training, sealed-test, and runtime gates are implemented and verified.