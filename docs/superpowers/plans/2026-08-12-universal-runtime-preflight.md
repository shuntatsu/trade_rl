# Universal Runtime and Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified runtime manifest from the new PostgreSQL cache and implement the concrete Universal U3-U6 runtime factory required by the maintained training entrypoint.

**Architecture:** Promote frozen metadata loading into maintained code, materialize the immutable instrument/dataset/normalizer inputs once, and bind their non-secret identities into a canonical preflight manifest. The runtime factory loads only that manifest and recomposes existing U3-U6 helpers while rejecting any live identity drift.

**Tech Stack:** Python 3.12+, psycopg 3, NumPy, canonical JSON and SHA-256 artifacts, pytest, existing Universal workflows and Gymnasium/SB3 factories.

## Global Constraints

- This plan starts only after the period-correct PostgreSQL cache plan passes its live 60-artifact round-trip gate.
- Cache ID and table set must be `binance-usds-m-native-indicators-15x-20241113-20260705-v1` and `UNIVERSAL_202411_202607_TABLES`.
- Frozen metadata root: `/workspace/var/cache/frozen-metadata/usds-m` in Docker; local tests use an explicit path.
- Partition seed 17 and the maintained 15-symbol 9/3/3 train/validation/test split are immutable.
- Only the 9 train symbols may enter dataset materialization, normalizer fitting, Oracle teacher generation, or critic warm start.
- Fold train range is `[0, min(dataset.n_bars for train symbols))` after the materialized dataset has applied its own causal availability/warm-up semantics.
- Runtime manifests contain no PostgreSQL URL, password, token, or environment secret.
- Explicit CLI values and manifest identities must match when both are supplied.

---

### Task 1: Maintained Frozen Metadata Cache Transport

**Files:**
- Create: `trade_rl/integrations/frozen_binance_metadata.py`
- Test: `tests/integrations/test_frozen_binance_metadata.py`
- Modify: `examples/binance-multitimeframe/full_research_pipeline_legacy.py:96-153`

**Interfaces:**
- Produces: `FrozenBinanceExchangeInfoTransport(root: Path)` implementing
  `load_exchange_information_snapshot(...) -> BinanceExchangeInfoSnapshot`.
- Consumed by: Task 3 preflight.
- Private helpers in the new module: `require_mapping(value)` rejects non-object
  JSON, `require_non_empty(value, field)` rejects blank strings, and
  `parse_utc(value)` returns a timezone-aware UTC datetime.
- Test helpers: `exchange_info_fixture()` contains the 15 requested USD-M symbols;
  `write_frozen_cache` writes matching raw/manifest bytes; the mutated writer
  applies exactly the named missing-file, digest, or market defect.

- [ ] **Step 1: Write failing cache integrity and offline-loading tests**

```python
def test_frozen_transport_loads_verified_snapshot_without_network(tmp_path: Path) -> None:
    raw = json.dumps(exchange_info_fixture(), sort_keys=True).encode()
    write_frozen_cache(tmp_path, raw=raw, market="usds-m")
    transport = FrozenBinanceExchangeInfoTransport(tmp_path)
    snapshot = transport.load_exchange_information_snapshot(market=BinanceMarket.USDS_M)
    assert snapshot.raw_payload == raw
    assert snapshot.raw_payload_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", ("missing_manifest", "bad_digest", "wrong_market"))
def test_frozen_transport_fails_closed_on_incomplete_or_drifted_cache(tmp_path: Path, mutation: str) -> None:
    write_mutated_frozen_cache(tmp_path, mutation)
    with pytest.raises((RuntimeError, ValueError)):
        FrozenBinanceExchangeInfoTransport(tmp_path).load_exchange_information_snapshot(
            market=BinanceMarket.USDS_M
        )
```

- [ ] **Step 2: Run the test and confirm RED**

```powershell
uv run pytest tests/integrations/test_frozen_binance_metadata.py -q
```

Expected: import failure for `frozen_binance_metadata`.

- [ ] **Step 3: Promote the existing cache reader into maintained code**

```python
@dataclass(frozen=True, slots=True)
class FrozenBinanceExchangeInfoTransport:
    root: Path

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> BinanceExchangeInfoSnapshot:
        del mode
        resolved_market = BinanceMarket(market)
        raw_path = self.root / "exchange-info.raw.json"
        manifest_path = self.root / "manifest.json"
        if not raw_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("frozen metadata cache is incomplete")
        raw = raw_path.read_bytes()
        manifest = require_mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        digest = hashlib.sha256(raw).hexdigest()
        if manifest.get("schema_version") != "frozen_metadata_cache_v1":
            raise ValueError("frozen metadata cache schema mismatch")
        if manifest.get("market") != resolved_market.value:
            raise ValueError("frozen metadata cache market mismatch")
        if manifest.get("raw_payload_sha256") != digest:
            raise ValueError("frozen metadata cache digest mismatch")
        payload = require_mapping(json.loads(raw))
        return BinanceExchangeInfoSnapshot(
            payload=payload,
            raw_payload=raw,
            source_uri=require_non_empty(str(manifest["source_uri"]), field="source_uri"),
            retrieved_at=parse_utc(str(manifest["retrieved_at"])),
            raw_payload_sha256=digest,
        )
```

Replace the private legacy class with an import alias to the maintained class so
existing flows retain behavior without duplicate implementations.

- [ ] **Step 4: Run focused and legacy tests**

```powershell
uv run pytest tests/integrations/test_frozen_binance_metadata.py tests/examples/test_binance_metadata_mode_runner.py -q
uv run ruff check trade_rl/integrations/frozen_binance_metadata.py examples/binance-multitimeframe/full_research_pipeline_legacy.py tests/integrations/test_frozen_binance_metadata.py
uv run mypy trade_rl/integrations/frozen_binance_metadata.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/integrations/frozen_binance_metadata.py examples/binance-multitimeframe/full_research_pipeline_legacy.py tests/integrations/test_frozen_binance_metadata.py tests/examples/test_binance_metadata_mode_runner.py
git commit -m "refactor: promote frozen binance metadata transport"
```

### Task 2: Canonical Runtime Manifest and Shared Normalizer Artifact

**Files:**
- Create: `trade_rl/workflows/universal_runtime_manifest.py`
- Create: `trade_rl/workflows/universal_normalizer_artifact.py`
- Test: `tests/workflows/test_universal_runtime_manifest.py`
- Test: `tests/workflows/test_universal_normalizer_artifact.py`

**Interfaces:**
- Produces: `UniversalRuntimeManifest`, `write_universal_runtime_manifest`,
  `load_universal_runtime_manifest`, `write_universal_shared_normalizer`, and
  `load_universal_shared_normalizer`.
- Consumed by: Tasks 3-4 and the launch plan.
- Test helpers construct a concrete 9/3/3 manifest with 64-character digests and
  a two-feature fitted `SymbolBalancedStandardNormalizer`; `mutate_json_array`
  rewrites one numeric JSON element without updating its bound digest.

- [ ] **Step 1: Write failing round-trip, secret-rejection, and drift tests**

```python
def test_runtime_manifest_round_trip_closes_all_static_identities(tmp_path: Path) -> None:
    expected = runtime_manifest_fixture()
    path = write_universal_runtime_manifest(tmp_path / "runtime-manifest.json", expected)
    actual = load_universal_runtime_manifest(path)
    assert actual == expected
    assert actual.train_symbols == expected.partition.train_symbols
    assert actual.fold_train_range == (0, expected.shared_complete_row_count)


@pytest.mark.parametrize("secret", ("postgresql://user:password@db/x", "password", "token"))
def test_runtime_manifest_rejects_secret_material(secret: str) -> None:
    payload = runtime_manifest_fixture().to_json_dict()
    payload["unexpected"] = secret
    with pytest.raises(ValueError, match="unknown|secret"):
        UniversalRuntimeManifest.from_json_dict(payload)


def test_shared_normalizer_artifact_rejects_statistics_drift(tmp_path: Path) -> None:
    normalizer = shared_normalizer_fixture()
    root = write_universal_shared_normalizer(tmp_path, normalizer)
    mutate_json_array(root / "universal-normalizer.json", "mean", 0, 1.0)
    with pytest.raises(ValueError, match="statistics digest"):
        load_universal_shared_normalizer(root)
```

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/workflows/test_universal_runtime_manifest.py tests/workflows/test_universal_normalizer_artifact.py -q
```

Expected: import failures for both new workflow modules.

- [ ] **Step 3: Implement strict manifest and normalizer serialization**

```python
@dataclass(frozen=True, slots=True)
class UniversalRuntimeManifest:
    cache_id: str
    tables: PostgresMarketTableSet
    research_start: datetime
    research_end: datetime
    instrument_artifact_relpath: Path
    dataset_artifact_relpath: Path
    normalizer_artifact_relpath: Path
    train_symbols: tuple[str, ...]
    validation_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    fold_train_range: tuple[int, int]
    shared_complete_row_count: int
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    metadata_evidence_digest: str
    source_manifest_digest: str
    dataset_digests: tuple[tuple[str, str], ...]
    schema_version: str = "universal_runtime_manifest_v1"
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.cache_id != UNIVERSAL_202411_202607_CACHE_ID:
            raise ValueError("runtime manifest cache identity mismatch")
        if len(self.train_symbols) != 9 or len(self.validation_symbols) != 3 or len(self.test_symbols) != 3:
            raise ValueError("runtime manifest partition must be 9/3/3")
        if set(self.train_symbols) & set(self.validation_symbols) or set(self.train_symbols) & set(self.test_symbols):
            raise ValueError("runtime manifest partition overlaps")
        if self.fold_train_range != (0, self.shared_complete_row_count):
            raise ValueError("runtime manifest train range is not maximal shared closure")
        require_manifest_digests(self)
        expected = content_digest(self.digest_payload())
        if self.manifest_digest and self.manifest_digest != expected:
            raise ValueError("runtime manifest digest mismatch")
        object.__setattr__(self, "manifest_digest", expected)
```

Serialize every normalizer field (`mean`, `std`, `constant_mask`, symbols,
feature/catalog/split digests, fold range, sample counts, clip value, version) as
canonical JSON. Reconstruct `SymbolBalancedStandardNormalizer` and require its
computed `statistics_digest` to match before returning. Writes are immutable:
same bytes reuse, different bytes fail.

Artifact paths in the manifest must be relative, slash-normalized, non-empty,
and contain neither an absolute root nor `..`. Loaders resolve them against the
manifest file's parent directory. Thus the same identity resolves under the host
artifact root and `/workspace/var/universal` inside Docker without embedding a
machine-specific path in the digest.

- [ ] **Step 4: Run focused tests and validation**

```powershell
uv run pytest tests/workflows/test_universal_runtime_manifest.py tests/workflows/test_universal_normalizer_artifact.py tests/rl/test_universal_research_u3_u6.py -q
uv run ruff check trade_rl/workflows/universal_runtime_manifest.py trade_rl/workflows/universal_normalizer_artifact.py tests/workflows/test_universal_runtime_manifest.py tests/workflows/test_universal_normalizer_artifact.py
uv run mypy trade_rl/workflows/universal_runtime_manifest.py trade_rl/workflows/universal_normalizer_artifact.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_runtime_manifest.py trade_rl/workflows/universal_normalizer_artifact.py tests/workflows/test_universal_runtime_manifest.py tests/workflows/test_universal_normalizer_artifact.py
git commit -m "feat: bind universal runtime manifest identities"
```

### Task 3: Preflight Materialization Command

**Files:**
- Create: `trade_rl/workflows/universal_runtime_preflight.py`
- Create: `scripts/materialize_universal_runtime.py`
- Test: `tests/workflows/test_universal_runtime_preflight.py`
- Test: `tests/scripts/test_materialize_universal_runtime.py`

**Interfaces:**
- Consumes: new cache, frozen transport, existing instrument/dataset/normalizer helpers.
- Produces: `materialize_universal_runtime_inputs(...) -> UniversalRuntimeManifest`.
- CLI writes the instrument bundle, nine train datasets, shared normalizer, and runtime manifest.
- Test helpers use the existing fake indicator/database rows to record requested
  symbol tuples, write the same valid frozen 15-symbol snapshot from Task 1, and
  rerun the workflow against one temporary artifact root; the drift helper
  rewrites a dataset manifest ID while leaving its directory name unchanged.

- [ ] **Step 1: Write failing train-only and identity-closure tests**

```python
def test_preflight_materializes_only_train_symbols(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    manifest = materialize_universal_runtime_inputs(
        connection=fake_database(calls),
        frozen_metadata_root=frozen_metadata_fixture(tmp_path),
        instrument_artifact_root=tmp_path / "instruments",
        dataset_artifact_root=tmp_path / "datasets",
        normalizer_artifact_root=tmp_path / "normalizer",
        runtime_manifest_path=tmp_path / "runtime.json",
    )
    assert set(manifest.train_symbols) == {call[0] for call in calls if call}
    assert not set(manifest.validation_symbols) & set().union(*map(set, calls))
    assert not set(manifest.test_symbols) & set().union(*map(set, calls))
    assert manifest.fold_train_range == (0, manifest.shared_complete_row_count)


def test_preflight_reuses_identical_artifacts_and_rejects_drift(tmp_path: Path) -> None:
    first = run_preflight_fixture(tmp_path)
    second = run_preflight_fixture(tmp_path)
    assert first.manifest_digest == second.manifest_digest
    mutate_dataset_identity(tmp_path / "datasets" / first.train_symbols[0])
    with pytest.raises(ValueError, match="dataset artifact identity"):
        run_preflight_fixture(tmp_path)
```

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/workflows/test_universal_runtime_preflight.py tests/scripts/test_materialize_universal_runtime.py -q
```

Expected: import failures for preflight workflow and script.

- [ ] **Step 3: Implement the preflight composition**

The workflow performs this exact sequence:

```python
resolution = resolve_frozen_snapshot(
    transport=FrozenBinanceExchangeInfoTransport(frozen_metadata_root),
    market=BinanceMarket.USDS_M,
    symbols=MAINTAINED_SYMBOLS,
    start_time=RESEARCH_START,
    end_time=RESEARCH_END,
)
metadata_digests = {
    symbol: content_digest(dict(resolution.metadata[symbol]))
    for symbol in MAINTAINED_SYMBOLS
}
materialize_postgres_universal_instrument_artifacts(
    connection,
    output_dir=instrument_artifact_root,
    research_start=RESEARCH_START,
    research_end=RESEARCH_END,
    metadata_digests=metadata_digests,
    seed=17,
    cache_id=UNIVERSAL_202411_202607_CACHE_ID,
    tables=UNIVERSAL_202411_202607_TABLES,
)
instrument_bundle = load_universal_instrument_artifact_bundle(instrument_artifact_root)
datasets = materialize_universal_train_datasets(
    connection,
    instrument_bundle=instrument_bundle,
    metadata_resolution=resolution,
    feature_specs=binance_universal_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    ),
    indicator_loader=partial(
        load_postgres_indicator_artifacts,
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        tables=UNIVERSAL_202411_202607_TABLES,
    ),
    dataset_builder=partial(
        build_postgres_market_dataset,
        tables=UNIVERSAL_202411_202607_TABLES,
    ),
)
shared_count = min(dataset.n_bars for dataset in datasets.values())
fold_range = (0, shared_count)
normalizer = fit_universal_shared_normalizer(
    datasets,
    train_symbols=instrument_bundle.partition.train_symbols,
    catalog_digest=instrument_bundle.catalog.digest,
    split_manifest_digest=instrument_bundle.partition.symbol_disjoint_manifest_digest,
    fold_train_range=fold_range,
)
paths = publish_universal_train_dataset_artifacts(
    datasets,
    train_symbols=instrument_bundle.partition.train_symbols,
    artifact_root=dataset_artifact_root,
)
```

Persist the normalizer, compute dataset digests from the loaded artifact
identities, construct the manifest, write it immutably, reload it, and compare
equality before returning. The CLI requires explicit paths and `--postgres-url`,
but prints a redacted JSON result without the URL.

- [ ] **Step 4: Run focused and live preflight validation**

```powershell
uv run pytest tests/workflows/test_universal_runtime_preflight.py tests/scripts/test_materialize_universal_runtime.py tests/workflows/test_universal_training_runner.py tests/workflows/test_postgres_universal_instrument_artifacts.py -q
uv run ruff check trade_rl/workflows/universal_runtime_preflight.py scripts/materialize_universal_runtime.py tests/workflows/test_universal_runtime_preflight.py tests/scripts/test_materialize_universal_runtime.py
uv run mypy trade_rl/workflows/universal_runtime_preflight.py scripts/materialize_universal_runtime.py
uv run python scripts/materialize_universal_runtime.py --postgres-url "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl" --frozen-metadata-root data/runtime/frozen-metadata/usds-m --instrument-artifact-root artifacts/universal/instruments --dataset-artifact-root artifacts/universal/datasets --normalizer-artifact-root artifacts/universal/normalizer --runtime-manifest artifacts/universal/runtime-manifest.json
```

Expected: unit/static checks pass; the live command prints 9 train, 3 validation,
3 test symbols, a nonzero shared row count, and valid 64-character digests. Verify
the manifest contains neither `postgresql://` nor `password`.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_runtime_preflight.py scripts/materialize_universal_runtime.py tests/workflows/test_universal_runtime_preflight.py tests/scripts/test_materialize_universal_runtime.py
git commit -m "feat: materialize universal runtime preflight"
```

### Task 4: Concrete Runtime Factory and Manifest-Driven Entrypoint

**Files:**
- Create: `trade_rl/integrations/binance_universal_runtime.py`
- Modify: `trade_rl/workflows/universal_full_research_entrypoint.py:29-62`
- Modify: `scripts/run_universal_full_research.py:17-99`
- Modify: `START.md:272-317`
- Test: `tests/integrations/test_binance_universal_runtime.py`
- Test: `tests/workflows/test_universal_full_research_entrypoint.py`
- Test: `tests/scripts/test_run_universal_full_research.py`

**Interfaces:**
- Produces: `build_runtime(*, algorithm, run_config, context) -> UniversalTrainingRuntime`.
- Changes: `UniversalRuntimeFactoryContext` carries `runtime_manifest_path` and verifies legacy explicit fields against it.
- CLI adds required `--runtime-manifest`; existing identity flags become optional compatibility assertions.
- Test helpers load all three canonical U6 JSON files, create a real preflight
  artifact tree under `tmp_path`, and return a context whose manifest/base roots
  point at that tree; the drift helper mutates the first dataset identity.

- [ ] **Step 1: Write failing runtime composition and drift tests**

```python
def test_concrete_factory_returns_runtime_for_all_algorithms_with_shared_static_identity(tmp_path: Path) -> None:
    context = runtime_context_fixture(tmp_path)
    runtimes = [
        build_runtime(algorithm=algorithm, run_config=config_for(algorithm), context=context)
        for algorithm in FullResearchAlgorithm
    ]
    assert all(isinstance(runtime, UniversalTrainingRuntime) for runtime in runtimes)
    assert len({runtime.catalog_digest for runtime in runtimes}) == 1
    assert len({runtime.statistics_digest for runtime in runtimes}) == 1
    assert len({runtime.feature_schema_digest for runtime in runtimes}) == 1
    assert len({runtime.training_contract_digest for runtime in runtimes}) == 3


def test_concrete_factory_rejects_manifest_or_dataset_drift(tmp_path: Path) -> None:
    context = runtime_context_fixture(tmp_path)
    mutate_dataset_identity(context.dataset_artifact_root / context.manifest.train_symbols[0])
    with pytest.raises(ValueError, match="dataset.*identity"):
        build_runtime(
            algorithm=FullResearchAlgorithm.PPO,
            run_config=config_for(FullResearchAlgorithm.PPO),
            context=context,
        )
```

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/integrations/test_binance_universal_runtime.py tests/workflows/test_universal_full_research_entrypoint.py tests/scripts/test_run_universal_full_research.py -q
```

Expected: concrete factory import and `--runtime-manifest` assertions fail.

- [ ] **Step 3: Implement manifest verification and runtime composition**

Load the instrument bundle, normalizer, and each dataset artifact; verify every
identity against the manifest. Build the maintained runtime as follows:

```python
contracts = build_universal_instrument_contracts(
    metadata_resolution,
    train_symbols=manifest.train_symbols,
)
bindings = build_universal_bindings(
    datasets=datasets,
    contracts=contracts,
    catalog=instrument_bundle.catalog,
    train_symbols=manifest.train_symbols,
)
provider = CausalInstrumentContextProvider(contracts=contracts)
normalizers = {
    symbol: bind_universal_normalizers(
        datasets[symbol],
        shared=shared,
        action_spec_digest=concrete_action_spec_digest(run_config.action, symbol),
        action_size=1,
        n_factors=0,
        finite_horizon=True,
        candidate_config_digest=content_digest(run_config.candidate_digest_payload()),
    )
    for symbol in manifest.train_symbols
}
concrete = UniversalDatasetArtifactEnvironmentFactory(
    dataset_artifact_paths=dataset_paths,
    run_config=run_config,
    normalizers=normalizers,
)
routed = UniversalRoutedEnvironmentFactory(
    train_symbols=manifest.train_symbols,
    partition_digest=manifest.partition_digest,
    bindings=bindings,
    concrete_environment_factory=concrete,
    instrument_context_provider=provider,
    training_contract_digest=content_digest({"phase": "runtime-rebind"}),
    run_seed=min(run_config.training.seeds),
)
return build_universal_training_runtime(
    train_symbols=manifest.train_symbols,
    catalog_digest=manifest.catalog_digest,
    partition_digest=manifest.partition_digest,
    split_manifest_digest=manifest.split_manifest_digest,
    feature_schema_digest=manifest.feature_schema_digest,
    statistics_digest=manifest.statistics_digest,
    instrument_context_schema_digest=provider.schema_digest,
    routed_environment_factory=routed,
    training=run_config.training,
)
```

Resolve frozen metadata again through the manifest's metadata root supplied by
context and require its evidence digest to match. The entrypoint loads the
manifest once, constructs context from it, and passes the manifest's fold range,
statistics digest, and feature digest to U6. Compatibility arguments fail if
they disagree.

- [ ] **Step 4: Run the Universal contract suite and CLI help seam**

```powershell
uv run pytest tests/integrations/test_binance_universal_runtime.py tests/workflows/test_universal_full_research_entrypoint.py tests/scripts/test_run_universal_full_research.py tests/workflows/test_universal_full_research_training.py tests/workflows/test_universal_training_runtime.py tests/workflows/test_universal_training_execution.py tests/workflows/test_universal_sb3_training_assembly.py -q
uv run ruff check trade_rl/integrations/binance_universal_runtime.py trade_rl/workflows/universal_full_research_entrypoint.py scripts/run_universal_full_research.py tests/integrations/test_binance_universal_runtime.py tests/workflows/test_universal_full_research_entrypoint.py tests/scripts/test_run_universal_full_research.py
uv run mypy trade_rl/integrations/binance_universal_runtime.py trade_rl/workflows/universal_full_research_entrypoint.py scripts/run_universal_full_research.py
uv run python scripts/run_universal_full_research.py --help
```

Expected: all tests and static checks pass; help names
`trade_rl.integrations.binance_universal_runtime:build_runtime` and requires the
runtime manifest.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/integrations/binance_universal_runtime.py trade_rl/workflows/universal_full_research_entrypoint.py scripts/run_universal_full_research.py START.md tests/integrations/test_binance_universal_runtime.py tests/workflows/test_universal_full_research_entrypoint.py tests/scripts/test_run_universal_full_research.py
git commit -m "feat: add universal binance runtime factory"
```

## Plan Completion Gate

```powershell
uv run pytest tests/integrations/test_frozen_binance_metadata.py tests/workflows/test_universal_runtime_manifest.py tests/workflows/test_universal_normalizer_artifact.py tests/workflows/test_universal_runtime_preflight.py tests/scripts/test_materialize_universal_runtime.py tests/integrations/test_binance_universal_runtime.py tests/workflows/test_universal_full_research_entrypoint.py tests/workflows/test_universal_full_research_training.py tests/workflows/test_universal_training_runtime.py tests/workflows/test_universal_training_execution.py tests/workflows/test_universal_sb3_training_assembly.py tests/scripts/test_run_universal_full_research.py -q
uv run ruff check trade_rl scripts tests
uv run mypy trade_rl scripts
```

Then run a CPU end-to-end smoke using a temporary copy of each canonical config
with one seed, one environment, three PPO updates, and an output path containing
`smoke` in its generation ID. The smoke manifest must retain
`research_success=false`; it is not full-training completion evidence.
