# PostgreSQL Native Indicator Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the fixed 15-symbol 2024-11-13 through 2026-07-05 PostgreSQL source into an immutable, causally correct native-timeframe cache with auditable intermediate-data reports.

**Architecture:** Add an immutable table-set value that routes existing loaders without weakening the legacy defaults. A source reader validates the old `public.rl_*` rows, a pure native-bar/feature builder produces deterministic arrays, and a transactional publisher writes the new `market_raw` generation only after every artifact and report passes validation.

**Tech Stack:** Python 3.12+, NumPy, psycopg 3, PostgreSQL 16, pytest, existing `trade_rl.data.features` and `trade_rl.integrations.binance_universal` contracts.

## Global Constraints

- Source database: existing `trade_rl_db`; all `public.rl_*` tables are read-only.
- Symbols: `ADAUSDT, APTUSDT, ARBUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, LINKUSDT, LTCUSDT, OPUSDT, SOLUSDT, SUIUSDT, XRPUSDT` in that order.
- Half-open UTC interval: `[2024-11-13T00:00:00Z, 2026-07-05T00:00:00Z)`.
- Cache ID: `binance-usds-m-native-indicators-15x-20241113-20260705-v1`.
- Native timeframes: `15m, 1h, 4h, 1d`; bars become available at close and may use completed source rows only.
- Feature contract: exactly 206 target-local features from `binance_universal_feature_specs(base_timeframe="15m", feature_timeframes=("1h", "4h", "1d"))`.
- Publication is atomic and immutable; an identity collision with different content fails.
- Never drop, truncate, update, or delete old source/cache tables, containers, volumes, or artifacts.

---

### Task 1: Period-Correct PostgreSQL Table Routing

**Files:**
- Create: `trade_rl/integrations/postgres_market_tables.py`
- Modify: `trade_rl/integrations/postgres_indicator_artifacts.py:17-22,232-267`
- Modify: `trade_rl/integrations/postgres_indicator_inventory.py:254-287`
- Modify: `trade_rl/integrations/postgres_market_dataset.py:29-30,119-224,342-570`
- Modify: `trade_rl/workflows/postgres_universal_instrument_artifacts.py:22-45`
- Test: `tests/integrations/test_postgres_market_tables.py`
- Test: `tests/integrations/test_postgres_indicator_artifacts.py`
- Test: `tests/integrations/test_postgres_indicator_inventory.py`
- Test: `tests/integrations/test_postgres_market_dataset.py`

**Interfaces:**
- Produces: `PostgresMarketTableSet`, `LEGACY_MARKET_TABLES`,
  `UNIVERSAL_202411_202607_TABLES`, and
  `UNIVERSAL_202411_202607_CACHE_ID`.
- Changes: loaders accept keyword-only `tables: PostgresMarketTableSet = LEGACY_MARKET_TABLES`.
- Consumed by: Tasks 3-4 and the runtime/preflight plan.

- [ ] **Step 1: Write failing routing and injection tests**

```python
def test_period_correct_table_set_is_exact_and_immutable() -> None:
    from trade_rl.integrations.postgres_market_tables import (
        UNIVERSAL_202411_202607_TABLES,
    )

    assert UNIVERSAL_202411_202607_TABLES.kline == (
        "market_raw.binance_usds_m_klines_202411_202607"
    )
    assert UNIVERSAL_202411_202607_TABLES.funding == (
        "market_raw.binance_usds_m_funding_202411_202607"
    )
    assert UNIVERSAL_202411_202607_TABLES.indicator_manifest.endswith(
        "indicator_manifests_202411_202607"
    )
    assert UNIVERSAL_202411_202607_TABLES.indicator_artifact.endswith(
        "indicator_artifacts_202411_202607"
    )


def test_table_set_rejects_sql_identifier_injection() -> None:
    from trade_rl.integrations.postgres_market_tables import PostgresMarketTableSet

    with pytest.raises(ValueError, match="table identifier"):
        PostgresMarketTableSet(
            kline="market_raw.safe; DROP TABLE public.rl_klines",
            funding="market_raw.funding",
            indicator_manifest="market_raw.manifest",
            indicator_artifact="market_raw.artifact",
        )
```

Also extend fake-cursor tests to call each loader with
`UNIVERSAL_202411_202607_TABLES` and assert every emitted query contains only the
new table name while existing default tests still contain the legacy names.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
uv run pytest tests/integrations/test_postgres_market_tables.py tests/integrations/test_postgres_indicator_artifacts.py tests/integrations/test_postgres_indicator_inventory.py tests/integrations/test_postgres_market_dataset.py -q
```

Expected: collection fails because `postgres_market_tables` and the `tables=`
arguments do not exist.

- [ ] **Step 3: Add the immutable table-set contract and route all SQL through it**

```python
from dataclasses import dataclass, fields
import re

_TABLE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PostgresMarketTableSet:
    kline: str
    funding: str
    indicator_manifest: str
    indicator_artifact: str

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if _TABLE.fullmatch(value) is None:
                raise ValueError(f"{name} table identifier is invalid")


LEGACY_MARKET_TABLES = PostgresMarketTableSet(
    kline="market_raw.binance_usds_m_klines_202101_202606",
    funding="market_raw.binance_usds_m_funding_202101_202606",
    indicator_manifest="market_raw.binance_usds_m_indicator_manifests_202101_202606",
    indicator_artifact="market_raw.binance_usds_m_indicator_artifacts_202101_202606",
)
UNIVERSAL_202411_202607_TABLES = PostgresMarketTableSet(
    kline="market_raw.binance_usds_m_klines_202411_202607",
    funding="market_raw.binance_usds_m_funding_202411_202607",
    indicator_manifest="market_raw.binance_usds_m_indicator_manifests_202411_202607",
    indicator_artifact="market_raw.binance_usds_m_indicator_artifacts_202411_202607",
)
UNIVERSAL_202411_202607_CACHE_ID = (
    "binance-usds-m-native-indicators-15x-20241113-20260705-v1"
)
```

Use `tables.indicator_manifest`, `tables.indicator_artifact`, `tables.kline`, and
`tables.funding` in the existing f-strings. Include every selected table name in
dataset content identity. Pass `tables` through
`materialize_postgres_universal_instrument_artifacts` to inventory loading.

- [ ] **Step 4: Run focused tests and static validation**

```powershell
uv run pytest tests/integrations/test_postgres_market_tables.py tests/integrations/test_postgres_indicator_artifacts.py tests/integrations/test_postgres_indicator_inventory.py tests/integrations/test_postgres_market_dataset.py tests/workflows/test_postgres_universal_instrument_artifacts.py -q
uv run ruff check trade_rl/integrations/postgres_market_tables.py trade_rl/integrations/postgres_indicator_artifacts.py trade_rl/integrations/postgres_indicator_inventory.py trade_rl/integrations/postgres_market_dataset.py trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/integrations/test_postgres_market_tables.py
uv run mypy trade_rl/integrations/postgres_market_tables.py trade_rl/integrations/postgres_indicator_artifacts.py trade_rl/integrations/postgres_indicator_inventory.py trade_rl/integrations/postgres_market_dataset.py
```

Expected: all commands pass; old loader tests prove backward compatibility and
new tests prove the explicit period-correct route.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/integrations/postgres_market_tables.py trade_rl/integrations/postgres_indicator_artifacts.py trade_rl/integrations/postgres_indicator_inventory.py trade_rl/integrations/postgres_market_dataset.py trade_rl/workflows/postgres_universal_instrument_artifacts.py tests/integrations/test_postgres_market_tables.py tests/integrations/test_postgres_indicator_artifacts.py tests/integrations/test_postgres_indicator_inventory.py tests/integrations/test_postgres_market_dataset.py tests/workflows/test_postgres_universal_instrument_artifacts.py
git commit -m "feat: route postgres market cache generations"
```

### Task 2: Read and Validate the Legacy Raw Source

**Files:**
- Create: `trade_rl/integrations/postgres_universal_source.py`
- Test: `tests/integrations/test_postgres_universal_source.py`

**Interfaces:**
- Produces: `UniversalSourceScope`, `RawSymbolSource`,
  `load_postgres_universal_source(connection, *, scope) -> dict[str, RawSymbolSource]`.
- Consumed by: Task 3.
- Test helpers in the same test file: `FakeSourceDatabase.complete(scope)` emits
  three contiguous minutes for each scoped table; `.mutated(scope, mutation)`
  changes exactly one kline row to create the named gap, duplicate, OHLC, or NaN
  defect while retaining the same DB-API cursor surface.

- [ ] **Step 1: Write failing source-closure tests**

```python
def test_source_loader_uses_half_open_interval_and_declared_symbol_order() -> None:
    scope = UniversalSourceScope.maintained()
    rows = load_postgres_universal_source(FakeSourceDatabase.complete(scope), scope=scope)
    assert tuple(rows) == scope.symbols
    assert scope.start == datetime(2024, 11, 13, tzinfo=UTC)
    assert scope.end == datetime(2026, 7, 5, tzinfo=UTC)
    assert all(item.timestamps[0] == np.datetime64("2024-11-13T00:00:00") for item in rows.values())
    assert all(item.timestamps[-1] < np.datetime64("2026-07-05T00:00:00") for item in rows.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("gap", "contiguous"), ("duplicate", "unique"), ("bad_ohlc", "OHLCV"), ("nan", "finite")),
)
def test_source_loader_rejects_invalid_raw_rows(mutation: str, message: str) -> None:
    scope = UniversalSourceScope.maintained()
    with pytest.raises(ValueError, match=message):
        load_postgres_universal_source(FakeSourceDatabase.mutated(scope, mutation), scope=scope)
```

Assert all queries select `source = 'binance'`, `timeframe = '1m'` where relevant,
`timestamp >= %s`, and `timestamp < %s`, with no mutation SQL.

- [ ] **Step 2: Run the focused test and confirm RED**

```powershell
uv run pytest tests/integrations/test_postgres_universal_source.py -q
```

Expected: import failure for `postgres_universal_source`.

- [ ] **Step 3: Implement strict scope, read models, and closure validation**

```python
@dataclass(frozen=True, slots=True)
class UniversalSourceScope:
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    source: str = "binance"

    @classmethod
    def maintained(cls) -> "UniversalSourceScope":
        return cls(
            symbols=MAINTAINED_SYMBOLS,
            start=datetime(2024, 11, 13, tzinfo=UTC),
            end=datetime(2026, 7, 5, tzinfo=UTC),
        )


@dataclass(frozen=True, slots=True)
class RawSymbolSource:
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    base_volume: np.ndarray
    funding_timestamps: np.ndarray
    funding_rate: np.ndarray
    derivative_values: np.ndarray
    orderflow_values: np.ndarray
```

Load rows in batches per symbol, cast numeric arrays to `float64`, and enforce:

```python
expected = np.arange(start_ns, end_ns, 60_000_000_000, dtype=np.int64)
if not np.array_equal(timestamps.astype("datetime64[ns]").astype(np.int64), expected):
    raise ValueError(f"raw one-minute timestamps are not contiguous for {symbol}")
if not np.isfinite(np.column_stack((open_, high, low, close, base_volume))).all():
    raise ValueError(f"raw OHLCV is not finite for {symbol}")
if np.any(high < np.maximum.reduce((open_, close, low))) or np.any(low > np.minimum.reduce((open_, close, high))):
    raise ValueError(f"raw OHLCV invariant failed for {symbol}")
if np.any(base_volume < 0.0):
    raise ValueError(f"raw OHLCV volume is negative for {symbol}")
```

Validate auxiliary timestamps are strictly increasing; represent absent values
with explicit boolean availability arrays rather than forward-looking fills.

- [ ] **Step 4: Run focused tests and validation**

```powershell
uv run pytest tests/integrations/test_postgres_universal_source.py -q
uv run ruff check trade_rl/integrations/postgres_universal_source.py tests/integrations/test_postgres_universal_source.py
uv run mypy trade_rl/integrations/postgres_universal_source.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/integrations/postgres_universal_source.py tests/integrations/test_postgres_universal_source.py
git commit -m "feat: validate universal postgres source closure"
```

### Task 3: Build Native Bars, Features, and Audit Reports

**Files:**
- Create: `trade_rl/workflows/native_indicator_materializer.py`
- Test: `tests/workflows/test_native_indicator_materializer.py`

**Interfaces:**
- Consumes: `RawSymbolSource`, `UniversalSourceScope` from Task 2.
- Produces: `NativeCacheBuild`, `NativeArtifactPayload`, `IntermediateDataReport`,
  `build_native_indicator_cache(source, *, scope) -> NativeCacheBuild`.
- Consumed by: Task 4.
- Test helpers in the same file: `one_symbol_minutes(start, count)` creates
  strictly increasing OHLC and unique closes; `scope_for(symbol, minutes)` closes
  exactly on the requested minute count; `complete_fixture()` returns all arrays
  required by `RawSymbolSource`; `fixture_scope()` matches that fixture exactly.

- [ ] **Step 1: Write failing causality, boundary, schema, and digest tests**

```python
def test_native_bars_close_on_boundary_without_lookahead() -> None:
    source = one_symbol_minutes("2024-11-13T00:00:00Z", count=61)
    build = build_native_indicator_cache({"BTCUSDT": source}, scope=scope_for("BTCUSDT", minutes=60))
    bars = build.market_bars[("BTCUSDT", "15m")]
    assert bars.open_time_ms.tolist() == [
        1731456000000,
        1731456900000,
        1731457800000,
        1731458700000,
    ]
    assert bars.close[-1] == source.close[59]
    assert source.close[60] not in bars.close


def test_indicator_payload_is_deterministic_and_has_206_features() -> None:
    first = build_native_indicator_cache(complete_fixture(), scope=fixture_scope())
    second = build_native_indicator_cache(complete_fixture(), scope=fixture_scope())
    assert first.manifest.feature_count == 206
    assert first.manifest.digest == second.manifest.digest
    assert [item.payload_sha256 for item in first.artifacts] == [
        item.payload_sha256 for item in second.artifacts
    ]


def test_report_exposes_missing_nonfinite_ohlcv_and_feature_counts() -> None:
    report = build_native_indicator_cache(complete_fixture(), scope=fixture_scope()).report
    item = report.members[0]
    assert item.duplicate_timestamps == 0
    assert item.missing_timestamps == 0
    assert item.nonfinite_available_features == 0
    assert item.ohlcv_violations == 0
```

- [ ] **Step 2: Run the focused test and confirm RED**

```powershell
uv run pytest tests/workflows/test_native_indicator_materializer.py -q
```

Expected: import failure for `native_indicator_materializer`.

- [ ] **Step 3: Implement deterministic causal construction**

Use one pure resampler per timeframe:

```python
def resample_completed_bars(raw: RawSymbolSource, *, minutes: int) -> NativeBars:
    size = minutes
    if len(raw.timestamps) % size:
        raise ValueError("source interval does not close the native timeframe")
    shape = (-1, size)
    return NativeBars(
        open_time_ms=raw.timestamps.reshape(shape)[:, 0].astype("datetime64[ms]").astype(np.int64),
        event_time_ms=(raw.timestamps.reshape(shape)[:, -1] + np.timedelta64(1, "m")).astype("datetime64[ms]").astype(np.int64),
        open=raw.open.reshape(shape)[:, 0],
        high=raw.high.reshape(shape).max(axis=1),
        low=raw.low.reshape(shape).min(axis=1),
        close=raw.close.reshape(shape)[:, -1],
        quote_volume=(raw.base_volume * raw.close).reshape(shape).sum(axis=1),
    )
```

The legacy `public.rl_klines.volume` column is Binance base-asset volume (for
example BTC units), while the maintained Universal contract requires quote
notional. Convert each completed one-minute row causally with
`base_volume * close`, then sum those minute notionals into the native bar. Bind
the method string `base_volume_times_minute_close_v1` into the cache manifest and
report it explicitly; do not label the source column itself as quote volume.

For each timeframe, group only its `FeatureSpec` values and call
`calculate_feature_events` with the native arrays. Set `available_at` to the bar
close/event time. Write NPZ payloads in canonical array-name order with
`allow_pickle=False`, `float32` values, `bool` availability, and `int64`
event-time milliseconds. Hash the exact bytes with SHA-256. Build feature specs
from:

```python
specs = binance_universal_feature_specs(
    base_timeframe="15m",
    feature_timeframes=("1h", "4h", "1d"),
)
if len(specs) != 206:
    raise RuntimeError("Universal feature contract is not 206 channels")
```

Funding events are assigned to the first native bar whose close time is greater
than or equal to the event time, never to an earlier bar. Derivatives and
order-flow rows are aligned with the same backward-looking as-of rule and their
coverage/non-finite/staleness statistics are included in the intermediate-data
report. They do not become extra policy features because the canonical 206
`FeatureSpec` values do not name derivative or order-flow channels.

The report records row count, first/last time, missing/duplicate timestamps,
incomplete bars, OHLCV violations, per-feature available count, non-finite count,
minimum, maximum, mean, standard deviation, extreme count above 20 standard
deviations, and all identity digests. Reject any available non-finite feature.

- [ ] **Step 4: Run focused tests and validation**

```powershell
uv run pytest tests/workflows/test_native_indicator_materializer.py tests/integrations/test_universal_data_training_contract.py -q
uv run ruff check trade_rl/workflows/native_indicator_materializer.py tests/workflows/test_native_indicator_materializer.py
uv run mypy trade_rl/workflows/native_indicator_materializer.py
```

Expected: all pass and the maintained feature-contract test still proves 206
target-local channels.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/native_indicator_materializer.py tests/workflows/test_native_indicator_materializer.py
git commit -m "feat: build causal native indicator cache"
```

### Task 4: Transactional Publisher and Materialization CLI

**Files:**
- Create: `trade_rl/integrations/postgres_native_cache_publisher.py`
- Create: `scripts/materialize_universal_postgres_cache.py`
- Test: `tests/integrations/test_postgres_native_cache_publisher.py`
- Test: `tests/scripts/test_materialize_universal_postgres_cache.py`

**Interfaces:**
- Consumes: `NativeCacheBuild` and `UNIVERSAL_202411_202607_TABLES`.
- Produces: `publish_native_cache(connection, build, *, tables) -> PublishedNativeCache`.
- CLI prints JSON with `cache_id`, `manifest_digest`, `report_path`, row counts,
  and table identities.
- Test helpers in the same files: `TransactionRecordingConnection` implements
  cursor/commit/rollback and stores inserted rows by table/key;
  `native_build_fixture()` returns one symbol × four timeframes with valid NPZ
  bytes; `mutated_build(build)` changes one payload and recomputes only its local
  digest so the existing cache identity collides.

- [ ] **Step 1: Write failing atomicity and CLI tests**

```python
def test_publish_is_atomic_idempotent_and_rejects_drift() -> None:
    connection = TransactionRecordingConnection()
    build = native_build_fixture()
    first = publish_native_cache(connection, build, tables=UNIVERSAL_202411_202607_TABLES)
    second = publish_native_cache(connection, build, tables=UNIVERSAL_202411_202607_TABLES)
    assert first == second
    assert connection.commit_count == 2
    with pytest.raises(FileExistsError, match="different content"):
        publish_native_cache(connection, mutated_build(build), tables=UNIVERSAL_202411_202607_TABLES)
    assert connection.rollback_count == 1


def test_cli_defaults_are_the_maintained_real_data_scope() -> None:
    args = _parser().parse_args(["--postgres-url", "postgresql://db", "--report-root", "artifacts/report"])
    assert args.cache_id == "binance-usds-m-native-indicators-15x-20241113-20260705-v1"
    assert args.start == "2024-11-13T00:00:00Z"
    assert args.end == "2026-07-05T00:00:00Z"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```powershell
uv run pytest tests/integrations/test_postgres_native_cache_publisher.py tests/scripts/test_materialize_universal_postgres_cache.py -q
```

Expected: import failures for publisher and script.

- [ ] **Step 3: Implement schema creation, immutable upsert checks, and CLI**

Create all four tables inside one transaction. Use primary keys
`(cache_id, symbol, interval, open_time_ms)` for klines,
`(cache_id, symbol, calc_time_ms)` for funding, `cache_id` for manifests, and
`(cache_id, symbol, timeframe)` for artifacts. Before inserts, lock the manifest
identity:

```python
cursor.execute(
    f"SELECT manifest_digest FROM {tables.indicator_manifest} WHERE cache_id = %s FOR UPDATE",
    (build.manifest.cache_id,),
)
existing = cursor.fetchone()
if existing is not None:
    if existing[0] != build.manifest.digest:
        raise FileExistsError("cache identity already exists with different content")
    verify_published_cache(cursor, build=build, tables=tables)
    return PublishedNativeCache.from_build(build, tables=tables)
insert_market_rows(cursor, build=build, tables=tables)
insert_indicator_rows(cursor, build=build, tables=tables)
verify_published_cache(cursor, build=build, tables=tables)
```

The CLI connects with `psycopg.connect`, loads Task 2 source, builds Task 3
artifacts, publishes them, and atomically writes
`intermediate-data-report.json` under a cache-ID-specific report directory. It
prints only non-secret JSON and redacts the PostgreSQL URL.

- [ ] **Step 4: Run focused, live, and repository validation**

```powershell
uv run pytest tests/integrations/test_postgres_native_cache_publisher.py tests/scripts/test_materialize_universal_postgres_cache.py -q
uv run ruff check trade_rl/integrations/postgres_native_cache_publisher.py scripts/materialize_universal_postgres_cache.py tests/integrations/test_postgres_native_cache_publisher.py tests/scripts/test_materialize_universal_postgres_cache.py
uv run mypy trade_rl/integrations/postgres_native_cache_publisher.py scripts/materialize_universal_postgres_cache.py
uv run python scripts/materialize_universal_postgres_cache.py --postgres-url "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl" --report-root artifacts/universal/cache-reports
```

Expected: tests and static checks pass; the live command publishes or verifies
exactly 60 indicator artifacts, prints the maintained cache ID, and reports zero
missing/duplicate/non-finite/invalid rows. Query the live closure:

```powershell
docker exec trade_rl_db psql -U trade_rl -d trade_rl -c "SELECT cache_id, artifact_count, manifest_digest FROM market_raw.binance_usds_m_indicator_manifests_202411_202607;"
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/integrations/postgres_native_cache_publisher.py scripts/materialize_universal_postgres_cache.py tests/integrations/test_postgres_native_cache_publisher.py tests/scripts/test_materialize_universal_postgres_cache.py
git commit -m "feat: publish universal postgres indicator cache"
```

## Plan Completion Gate

Run serially because the repository's Windows `uv` environment must not be
mutated concurrently:

```powershell
uv run pytest tests/integrations/test_postgres_market_tables.py tests/integrations/test_postgres_universal_source.py tests/workflows/test_native_indicator_materializer.py tests/integrations/test_postgres_native_cache_publisher.py tests/scripts/test_materialize_universal_postgres_cache.py tests/integrations/test_postgres_indicator_artifacts.py tests/integrations/test_postgres_indicator_inventory.py tests/integrations/test_postgres_market_dataset.py tests/workflows/test_postgres_universal_instrument_artifacts.py -q
uv run ruff check trade_rl scripts tests
uv run mypy trade_rl scripts
```

Archive the exact command output and the generated intermediate-data report. Do
not start runtime/preflight work until the live manifest and all 60 artifact
payload digests round-trip through `load_postgres_indicator_artifacts`.
