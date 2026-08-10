# Stored Instrument Catalog and Zero-Shot Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover the variable set of verified Binance USDS-M symbols stored in Docker/PostgreSQL, freeze that inventory into an immutable catalog, and derive a catalog-bound train/validation/sealed-test symbol partition for mandatory zero-shot research.

**Architecture:** Keep source validation, durable contracts, and orchestration separate. A PostgreSQL adapter reads only manifest and artifact metadata, never NPZ payloads. A framework-independent workflow contract validates catalog eligibility and canonical serialization. A partition contract reuses `SymbolDisjointManifest` but binds it explicitly to the catalog digest and provides fail-closed split access. Training behavior remains unchanged until U2.

**Tech Stack:** Python 3.12, frozen dataclasses, DB-API protocol, canonical JSON/content digests, PostgreSQL indicator manifest, pytest, Ruff, MyPy, Import Linter, GitHub Actions.

## Global Constraints

- Start from the exact verified U0 reward/boundary head; do not duplicate or alter reward semantics.
- Do not query Binance or mutate PostgreSQL.
- Do not read or deserialize NPZ payload bytes while discovering the inventory.
- The source of truth is the verified PostgreSQL indicator manifest and artifact metadata closure.
- Active generations never observe symbols added after catalog materialization.
- Concrete ticker values may appear in catalog/split evidence, but not in policy observations or architecture identity.
- Train, validation, and test symbols must be pairwise disjoint and close exactly over the eligible catalog universe.
- Universal research requires at least 15 eligible symbols and at least 9 training symbols.
- Validation/test symbols are not exposed to training accessors.
- Missing or inconsistent source rows fail closed; they are not silently skipped.
- A symbol may be excluded only for predeclared eligibility reasons, recorded immutably.
- Existing Stage A triplet artifacts remain readable and unchanged.
- No environment, PPO, BC, model, normalizer, checkpoint, Docker, or serving behavior changes in U1.
- Production remains `NO-GO`.

---

### Task 1: Define immutable source and catalog contracts

**Files:**
- Create: `trade_rl/workflows/stored_instrument_catalog.py`
- Create: `tests/workflows/test_stored_instrument_catalog.py`

**Interfaces:**
- Produces: source-inventory dataclasses, `StoredInstrumentCatalog`, exclusion evidence, strict JSON reader/writer, and catalog builder.
- Consumes: only standard library plus existing artifact hashing/codec/domain validators.

- [ ] **Step 1: Write RED tests for source and catalog invariants**

Cover:

```text
ordered unique source symbols
ordered required timeframes
one artifact row per symbol/timeframe
lowercase SHA-256 evidence
common feature-config digest
research interval contained in source interval
metadata digest required for eligibility
zero-availability artifacts recorded as exclusions
stable canonical ordering and digest
strict JSON field closure
round trip
payload tamper rejection
```

Use deterministic fixtures with at least 15 symbols and four timeframes.

- [ ] **Step 2: Define source evidence dataclasses**

```python
@dataclass(frozen=True, slots=True)
class StoredIndicatorArtifactEvidence:
    symbol: str
    timeframe: str
    row_count: int
    feature_count: int
    available_value_count: int
    first_event_time_ms: int
    last_event_time_ms: int
    payload_schema: str
    payload_sha256: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class StoredIndicatorSourceInventory:
    cache_id: str
    source_manifest_digest: str
    market: str
    symbols: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    feature_config_digest: str
    required_timeframes: tuple[str, ...]
    artifacts: tuple[StoredIndicatorArtifactEvidence, ...]
```

The source inventory validates exact symbol × timeframe closure and exposes `artifact_for(symbol, timeframe)`.

- [ ] **Step 3: Define exclusion and catalog contracts**

```python
@dataclass(frozen=True, slots=True)
class StoredInstrumentExclusion:
    symbol: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoredInstrumentCatalog:
    source_cache_id: str
    source_manifest_digest: str
    market: str
    feature_config_digest: str
    required_timeframes: tuple[str, ...]
    research_start: datetime
    research_end: datetime
    eligible_symbols: tuple[str, ...]
    excluded_symbols: tuple[StoredInstrumentExclusion, ...]
    per_symbol_artifact_digests: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    per_symbol_metadata_digests: tuple[tuple[str, str], ...]
    schema_version: str = "stored_instrument_catalog_v1"
    digest: str = ""
```

- [ ] **Step 4: Implement deterministic catalog building**

```python
def build_stored_instrument_catalog(
    source: StoredIndicatorSourceInventory,
    *,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
) -> StoredInstrumentCatalog:
```

Eligibility reasons in v1:

```text
missing_execution_metadata
no_available_values:<timeframe>
```

Unknown metadata symbols are rejected. Missing artifact rows, schema drift, invalid source coverage, unsupported market, or malformed digests fail the entire build.

- [ ] **Step 5: Implement strict JSON I/O**

```python
write_stored_instrument_catalog(path, catalog)
load_stored_instrument_catalog(path)
```

Writes are canonical; loader requires exact field closure and validates digest.

- [ ] **Step 6: Verify GREEN**

```bash
pytest tests/workflows/test_stored_instrument_catalog.py -q
ruff check trade_rl/workflows/stored_instrument_catalog.py tests/workflows/test_stored_instrument_catalog.py
ruff format --check trade_rl/workflows/stored_instrument_catalog.py tests/workflows/test_stored_instrument_catalog.py
mypy trade_rl/workflows/stored_instrument_catalog.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/workflows/stored_instrument_catalog.py tests/workflows/test_stored_instrument_catalog.py
git commit -m "feat: define stored instrument catalog"
```

---

### Task 2: Add a read-only PostgreSQL inventory adapter

**Files:**
- Create: `trade_rl/integrations/postgres_stored_instrument_catalog.py`
- Create: `tests/integrations/test_postgres_stored_instrument_catalog.py`

**Interfaces:**
- Produces: `load_postgres_stored_indicator_inventory(connection, cache_id=...)`.
- Consumes: existing indicator table constants, DB-API connection protocol, and Task 1 source dataclasses.

- [ ] **Step 1: Write RED adapter tests with a recording fake connection**

Pin:

```text
manifest row validation
feature-config digest validation
ordered symbol inventory
required timeframe derivation from feature specs
metadata-only artifact query
query never selects npz_payload
exact artifact closure
artifact_count validation
duplicate/missing/extra row rejection
payload schema and SHA validation
source manifest digest stability
```

- [ ] **Step 2: Query the manifest metadata**

Read the existing manifest columns:

```sql
cache_id, schema_version, market, symbols, start_time, end_time,
feature_config_digest, feature_specs, artifact_count
```

Compute `source_manifest_digest` from the validated canonical row payload.

- [ ] **Step 3: Query artifact metadata only**

Read:

```sql
symbol, timeframe, row_count, feature_count, available_value_count,
first_event_time_ms, last_event_time_ms, payload_schema,
payload_sha256, payload_bytes
```

Do not select `npz_payload`.

- [ ] **Step 4: Reuse public constants, not private loader helpers**

Import only stable table/cache constants from `postgres_indicator_artifacts.py`. Keep parsing and validation local to the new adapter so the existing binary loader remains unchanged.

- [ ] **Step 5: Verify focused GREEN**

```bash
pytest \
  tests/integrations/test_postgres_stored_instrument_catalog.py \
  tests/integrations/test_postgres_indicator_artifacts.py -q
ruff check trade_rl/integrations/postgres_stored_instrument_catalog.py tests/integrations/test_postgres_stored_instrument_catalog.py
ruff format --check trade_rl/integrations/postgres_stored_instrument_catalog.py tests/integrations/test_postgres_stored_instrument_catalog.py
mypy trade_rl/integrations/postgres_stored_instrument_catalog.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/integrations/postgres_stored_instrument_catalog.py tests/integrations/test_postgres_stored_instrument_catalog.py
git commit -m "feat: discover stored PostgreSQL instruments"
```

---

### Task 3: Bind a zero-shot symbol partition to the catalog

**Files:**
- Create: `trade_rl/workflows/universal_instrument_partition.py`
- Create: `tests/workflows/test_universal_instrument_partition.py`

**Interfaces:**
- Produces: catalog-bound partition, split-count policy, access guards, strict JSON I/O.
- Consumes: `StoredInstrumentCatalog` and existing `SymbolDisjointManifest`.

- [ ] **Step 1: Write RED partition tests**

Cover:

```text
15 symbols -> 9 train / 3 validation / 3 test
20 symbols -> 12 train / 4 validation / 4 test
N < 15 rejected
train count < 9 rejected
same catalog+seed -> same partition
catalog digest changes partition digest
split closure and disjointness
training accessor rejects validation/test symbols
validation accessor rejects train/test symbols
test accessor rejects train/validation symbols
JSON round trip and tamper rejection
```

- [ ] **Step 2: Implement split counts**

```python
def universal_split_counts(symbol_count: int) -> tuple[int, int, int]:
    validation = max(3, symbol_count // 5)
    test = max(3, symbol_count // 5)
    train = symbol_count - validation - test
```

Fail closed when `symbol_count < 15` or `train < 9`.

- [ ] **Step 3: Build via existing symbol-disjoint contract**

```python
def build_universal_instrument_partition(
    catalog: StoredInstrumentCatalog,
    *,
    seed: int,
) -> UniversalInstrumentPartition:
```

Call `build_symbol_disjoint_manifest()` with the eligible symbols and derived counts. Bind both catalog and manifest digests.

- [ ] **Step 4: Define access guards**

```python
partition.require_symbol("BTCUSDT", split="train")
partition.require_symbols((...), split="validation")
```

The methods return the validated input and raise on cross-split access. They do not expose validation/test datasets through a training-specific API.

- [ ] **Step 5: Implement strict JSON I/O**

The partition serializes:

```text
catalog_digest
symbol_disjoint_manifest_digest
seed
train_symbols
validation_symbols
test_symbols
split_counts
schema_version
digest
```

Loader validates against supplied catalog and symbol-disjoint manifest.

- [ ] **Step 6: Verify GREEN**

```bash
pytest \
  tests/workflows/test_universal_instrument_partition.py \
  tests/workflows/test_symbol_disjoint_manifest.py -q
ruff check trade_rl/workflows/universal_instrument_partition.py tests/workflows/test_universal_instrument_partition.py
ruff format --check trade_rl/workflows/universal_instrument_partition.py tests/workflows/test_universal_instrument_partition.py
mypy trade_rl/workflows/universal_instrument_partition.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/workflows/universal_instrument_partition.py tests/workflows/test_universal_instrument_partition.py
git commit -m "feat: bind zero-shot symbol partitions"
```

---

### Task 4: Materialize the immutable research universe

**Files:**
- Create: `trade_rl/workflows/stored_instrument_research_universe.py`
- Create: `tests/workflows/test_stored_instrument_research_universe.py`

**Interfaces:**
- Produces: one idempotent materialization function and three immutable artifacts.
- Consumes: Tasks 1–3 and an already-open PostgreSQL connection.

- [ ] **Step 1: Write RED orchestration tests**

Pin:

```text
adapter -> catalog -> symbol manifest -> partition data flow
output paths and exact content
idempotent same-content rerun
different-content overwrite rejection
catalog digest propagated to partition
validation/test never sent to a training callback
no training environment or model construction
```

- [ ] **Step 2: Implement materialization**

```python
def materialize_stored_instrument_research_universe(
    connection: IndicatorArtifactConnection,
    *,
    output_root: Path,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
    split_seed: int,
    cache_id: str = INDICATOR_CACHE_ID,
) -> StoredInstrumentResearchUniverse:
```

Write:

```text
stored-instruments.json
symbol-disjoint.json
universal-instrument-partition.json
```

Use immutable same-content-or-error writes.

- [ ] **Step 3: Add an optional training-symbol callback only for guard verification**

The orchestration may accept a test-only/protocol callback that receives exactly `partition.train_symbols`; production U1 does not start training.

- [ ] **Step 4: Verify GREEN**

```bash
pytest \
  tests/workflows/test_stored_instrument_research_universe.py \
  tests/workflows/test_stored_instrument_catalog.py \
  tests/workflows/test_universal_instrument_partition.py \
  tests/integrations/test_postgres_stored_instrument_catalog.py -q
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/stored_instrument_research_universe.py tests/workflows/test_stored_instrument_research_universe.py
git commit -m "feat: materialize universal research universe"
```

---

### Task 5: Documentation, architecture checks, and exact-head verification

**Files:**
- Modify: `docs/SINGLE_SYMBOL.md`
- Create: `docs/UNIVERSAL_SINGLE_INSTRUMENT.md`
- Modify: architecture tests only where required by new import boundaries.
- Update: U1 PR body after evidence exists.

- [ ] **Step 1: Document the frozen-universe lifecycle**

State:

```text
Docker/PostgreSQL storage may change
active catalog cannot change
new storage state -> new catalog -> new partition -> new generation
```

Clarify that U1 does not yet train across symbols and does not alter the BTC maintained runtime.

- [ ] **Step 2: Verify dependency direction**

Required direction:

```text
integrations/postgres_stored_instrument_catalog
  -> workflows/stored_instrument_catalog contracts
workflows/universal_instrument_partition
  -> workflows/stored_instrument_catalog
  -> workflows/symbol_disjoint_manifest
```

No workflow imports psycopg or concrete model frameworks.

- [ ] **Step 3: Run focused suites**

```bash
pytest tests/workflows tests/integrations/test_postgres_stored_instrument_catalog.py -q
```

- [ ] **Step 4: Run complete exact-head CI**

Require the same repository-wide checks as U0 plus PostgreSQL Catalog and Training image on one head.

- [ ] **Step 5: Self-review**

Verify:

```text
no hard-coded symbol pool in the catalog builder
no DB writes
no NPZ reads
no ticker in policy files
no training behavior change
no validation/test leakage accessor
strict artifact closure
catalog and partition digests in every artifact
```

- [ ] **Step 6: Mark the U1 PR Ready only after exact-head verification**

Do not merge without explicit user authorization.
