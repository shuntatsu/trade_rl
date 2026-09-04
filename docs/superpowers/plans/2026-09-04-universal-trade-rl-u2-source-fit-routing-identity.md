# Universal Trade RL U2 Source / FIT / Routing Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the U2 V1 source/FIT/routing provenance chain from frozen U0 source artifacts through exact FIT views, U2-owned bindings, deterministic 8-worker routing, SB3 seed translation, timeout semantics, and minimal fixed-PPO orchestration without changing U1 economics.

**Architecture:** Keep metadata-only preregistration in `universal_trade_rl_u2_preflight.py`, add one focused source-artifact-to-FIT-view module, and keep U1 economics inside existing U1 constructors/validators. Build one high-level U2 environment factory that materializes each immutable FIT dataset once, derives bindings and a shared run-level environment-generation digest, then creates fresh mutable U1 environments for worker indices 0..7. Reuse the maintained SB3 `DummyVecEnv`, policy assembly, checkpoint machinery, and timeout bootstrap behavior rather than implementing parallel infrastructure.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.3.2, PyTorch, dataclasses, `MarketDataset`, `MarketDatasetView`, canonical SHA-256 content digests, pytest, Ruff, MyPy, Import Linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity-amendment.md`

## Global Constraints

- Real U2 training remains **NO-GO** throughout this plan.
- Admission remains **SEALED** and Production remains **NO-GO**.
- PPO recipe remains exact U2 V1: PPO, seeds `(0, 1, 2)`, seed 0 primary, `524_288` timesteps, `n_envs=8`, `n_steps=128`, batch 256, 10 epochs, deterministic CUDA, `vector_environment_mode="in_process"`.
- U1 remains the sole Risk / Execution / Accounting authority; do not duplicate U1 economic configuration in U2.
- U1 action/reward/normalizer/context semantics do not change.
- U1 normal horizon remains `terminated=False`, `truncated=True`, `liquidate_on_end=False`.
- Do not change generic `EpisodeRoutedSingleInstrumentEnv`, generic checkpoint schemas, or generic SB3 vectorization unless a new independently demonstrated generic defect requires it.
- Source artifact filesystem paths are locators only and must not enter research identity.
- FIT is an in-memory `MarketDatasetView` materialization, not a new persistent market-data artifact.
- Each symbol's FIT `MarketDataset` is materialized once per U2 factory and may be shared across workers; mutable U1 environments/state must be fresh per worker.
- U2 member seed namespace is exact: `member_seed == PPO seed == router run_seed`; worker `i` externally receives `member_seed + i`.
- A shared run-level U2 `environment_digest` must bind worker set `0..7`; per-worker router digests may differ.
- Ordinary checkpoint environment compatibility is not an exact mid-episode trajectory-resume proof. Do not claim exact trajectory resume in this plan.
- Every production change follows Red -> Green -> Refactor. Do not weaken an assertion, skip a failing test, or modify an oracle merely to obtain Green.

## Quality Contract

**Objective:** Make source/FIT data provenance, U2 routing randomness, vector-worker identity, and checkpoint environment compatibility observably deterministic and fail-closed before any real PPO or Development evaluation.

**Non-goals:** No Development B/C/D evaluation, Selection, Admission authorization, Production promotion, hyperparameter change, alternate algorithm/architecture, persistent FIT artifact, or exact mid-episode resume implementation.

**Acceptance Criteria:**

1. FIT metadata is phase-aligned to the source 15-minute grid before numeric loading.
2. Only a canonical source artifact matching the exact U0 source dataset identity can be sliced.
3. FIT child identity equals exact maintained `MarketDatasetView.identity`.
4. Production U2 bindings are derived internally from verified source/FIT/U1 identities.
5. Unrelated execution/descriptor binding metadata cannot perturb U2 episode sampling.
6. Workers 0..7 share one member run seed and one environment-generation digest while retaining exact worker indices.
7. Mutable U1 runtime state is isolated across workers.
8. SB3 worker reset seeds `member_seed + i` are accepted without changing router run seed.
9. Checkpoint environment identity changes after source/FIT/binding/member-seed/vector-generation drift.
10. Actual U2 vectorization preserves timeout metadata and exact terminal observation.
11. PPO applies exactly one timeout bootstrap and economic reward/wealth remains unmodified.
12. Minimal U2 orchestration binds one seed plan to one U2 environment factory and the maintained PPO backend without starting any training in tests.

**Test Oracle:** source dataset IDs/timestamps/counts, FIT view ID and arrays, binding payloads, episode seed/binding, router run seed/index/cycle, object identity/state isolation, environment-generation digest, SB3 reset seed vector, timeout info/terminal observation, rollout-buffer reward target, `PolicyTrainingResult.environment_digest`, checkpoint manifest identity.

**Required Test Layers:** Unit, contract/falsification, canonical artifact integration, U1/U2 integration, real `DummyVecEnv` integration, controlled SB3 PPO timeout-bootstrap integration, static analysis, architecture/import checks, related suite, full suite, package/build, exact-final-HEAD CI.

**Quality Gate:** Do not call this source/FIT/routing work complete until all Acceptance Criteria have executable assertions, all required checks run on one exact final HEAD, falsification review finds no unresolved Critical/High issue, and remaining limitations are recorded.

---

## File Map

**Create**

- `trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py` — canonical U0 source artifact loading and deterministic FIT `MarketDatasetView` materialization only.
- `tests/workflows/test_universal_trade_rl_u2_fit_dataset.py` — source artifact/FIT provenance and locator-independence tests.
- `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py` — indexed eight-worker factory, SB3 reset seeding, mutable-state isolation, and timeout metadata integration.
- `tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py` — controlled SB3 timeout bootstrap oracle.

**Modify**

- `trade_rl/workflows/universal_trade_rl_u2_preflight.py` — source/FIT grid-phase arithmetic and exposed FIT start/stop index properties.
- `trade_rl/workflows/universal_trade_rl_u2_environment.py` — U2-owned binding derivation, U2 episode seed/reset translation, shared environment-generation identity, and high-level indexed factory.
- `trade_rl/workflows/universal_trade_rl_u2_training.py` — minimal per-seed orchestration only after environment/vector gates are closed.
- `tests/workflows/test_universal_trade_rl_u2_preflight.py` — aligned/misaligned FIT metadata contracts.
- `tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py` — metadata-only fail-closed cases.
- `tests/rl/test_universal_trade_u2_environment.py` — binding derivation, sampling identity, generation digest, and low-level U2 adapter tests.
- `tests/workflows/test_universal_trade_rl_u2_training_runtime.py` — per-seed orchestration/lineage assertions.
- `docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md` — append a short pointer that Task 4-6 mechanics are superseded by the 2026-09-04 amendment/plan; do not rewrite historical preregistration text.

**Explicitly Reuse**

- `trade_rl.data.artifact.load_market_dataset_artifact`
- `trade_rl.data.artifacts.MarketDatasetView`
- `trade_rl.rl.universal_single_instrument_env.EpisodeRoutedSingleInstrumentEnv`
- `trade_rl.workflows.universal_trade_rl_u1_contract.require_universal_trade_rl_u1_environment_contract`
- `trade_rl.integrations.sb3_environment._filtered_environment_factory`
- `trade_rl.integrations.sb3_environment._build_training_environment`
- `trade_rl.integrations.sb3_training.StableBaselines3PPOBackend`
- existing `CheckpointManifest` / checkpoint loader validation

---

### Task 1: Enforce source/FIT grid-phase alignment in metadata preflight

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_preflight.py`
- Test: `tests/workflows/test_universal_trade_rl_u2_preflight.py`
- Test: `tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py`

**Interfaces:**
- Consumes: existing `U2TrainingSource` fields and `U2_DECISION_STEP_NS`.
- Produces: `U2TrainingSource.fit_start_index: int` and `U2TrainingSource.fit_stop_index: int`, both derived solely from metadata.

- [ ] **Step 1: Write the phase-shift RED test**

Add a test that creates a source on `00:00, 00:15, ...` and a FIT interval shifted by five minutes while keeping both grids independently dense:

```python
def test_u2_training_source_rejects_fit_grid_not_aligned_to_source_grid() -> None:
    step = U2_DECISION_STEP_NS
    source_first = 100 * step
    source_rows = 200
    source_last = source_first + (source_rows - 1) * step
    fit_first = source_first + 5 * 60 * 1_000_000_000
    fit_rows = 32
    fit_last = fit_first + (fit_rows - 1) * step

    with pytest.raises(ValueError, match="FIT|source|align|grid"):
        U2TrainingSource(
            symbol="BTCUSDT",
            dataset_digest="a" * 64,
            source_first_timestamp_ns=source_first,
            source_last_timestamp_ns=source_last,
            source_row_count=source_rows,
            fit_first_timestamp_ns=fit_first,
            fit_last_timestamp_ns=fit_last,
            fit_stop_timestamp_ns_exclusive=fit_last + step,
            fit_bar_count=fit_rows,
        )
```

- [ ] **Step 2: Run the RED test**

Run:

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_preflight.py::test_u2_training_source_rejects_fit_grid_not_aligned_to_source_grid
```

Expected: **FAIL** because the current constructor validates source/FIT dense grids independently but does not reject phase shift.

- [ ] **Step 3: Add aligned-index positive and overflow falsification tests**

Add:

```python
def test_u2_training_source_exposes_exact_fit_indices() -> None:
    step = U2_DECISION_STEP_NS
    source_first = 100 * step
    source_rows = 200
    source_last = source_first + (source_rows - 1) * step
    fit_start_index = 40
    fit_rows = 32
    fit_first = source_first + fit_start_index * step
    fit_last = fit_first + (fit_rows - 1) * step
    source = U2TrainingSource(
        symbol="BTCUSDT",
        dataset_digest="a" * 64,
        source_first_timestamp_ns=source_first,
        source_last_timestamp_ns=source_last,
        source_row_count=source_rows,
        fit_first_timestamp_ns=fit_first,
        fit_last_timestamp_ns=fit_last,
        fit_stop_timestamp_ns_exclusive=fit_last + step,
        fit_bar_count=fit_rows,
    )
    assert source.fit_start_index == 40
    assert source.fit_stop_index == 72


def test_u2_training_source_rejects_fit_stop_beyond_source_count() -> None:
    step = U2_DECISION_STEP_NS
    source_first = 100 * step
    source_rows = 50
    source_last = source_first + (source_rows - 1) * step
    fit_first = source_first + 30 * step
    fit_rows = 25
    fit_last = fit_first + (fit_rows - 1) * step
    with pytest.raises(ValueError, match="FIT|source|range|count"):
        U2TrainingSource(
            symbol="BTCUSDT",
            dataset_digest="a" * 64,
            source_first_timestamp_ns=source_first,
            source_last_timestamp_ns=source_last,
            source_row_count=source_rows,
            fit_first_timestamp_ns=fit_first,
            fit_last_timestamp_ns=fit_last,
            fit_stop_timestamp_ns_exclusive=fit_last + step,
            fit_bar_count=fit_rows,
        )
```

- [ ] **Step 4: Implement minimal metadata arithmetic**

In `U2TrainingSource.__post_init__`, after existing dense-grid validation:

```python
fit_offset_ns = fit_first - source_first
if fit_offset_ns < 0 or fit_offset_ns % U2_DECISION_STEP_NS != 0:
    raise ValueError("U2 FIT interval must align to the source 15m grid")
fit_start_index = fit_offset_ns // U2_DECISION_STEP_NS
fit_stop_index = fit_start_index + fit_bars
if not 0 <= fit_start_index < fit_stop_index <= source_rows:
    raise ValueError("U2 FIT interval is outside the source row range")
if source_first + (fit_stop_index - 1) * U2_DECISION_STEP_NS != fit_last:
    raise ValueError("U2 FIT metadata does not match source-grid indices")
```

Expose properties that recompute the same validated arithmetic without stored mutable state:

```python
@property
def fit_start_index(self) -> int:
    return (
        self.fit_first_timestamp_ns - self.source_first_timestamp_ns
    ) // U2_DECISION_STEP_NS

@property
def fit_stop_index(self) -> int:
    return self.fit_start_index + self.fit_bar_count
```

- [ ] **Step 5: Run focused Green verification**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_preflight.py
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
git commit -m "fix: align U2 FIT metadata to source grid"
```

---

### Task 2: Load canonical U0 artifacts and materialize exact FIT views

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py`
- Create: `tests/workflows/test_universal_trade_rl_u2_fit_dataset.py`

**Interfaces:**
- Consumes: `U2TrainingSourceClosure`, exact Train-symbol locator mapping, canonical `load_market_dataset_artifact`.
- Produces:

```python
U2SourceArtifactLocator = str | Path
U2SourceArtifactLoader = Callable[[U2SourceArtifactLocator], MarketDataset]

def load_universal_trade_rl_u2_fit_datasets(
    *,
    closure: U2TrainingSourceClosure,
    artifact_locators: Mapping[str, U2SourceArtifactLocator],
    loader: U2SourceArtifactLoader = load_market_dataset_artifact,
) -> dict[str, MarketDataset]: ...
```

The returned dict must preserve `closure.sources` order.

- [ ] **Step 1: Write the canonical full-source -> FIT RED test**

Use `make_u1_market(symbol="BTCUSDT", n_bars=10_000)` and publish it as a canonical source artifact. Build one `U2TrainingSource` whose FIT is rows `[1_000, 9_000)` and assert:

```python
def test_u2_fit_loader_materializes_exact_market_dataset_view(tmp_path: Path) -> None:
    source_dataset = make_u1_market(symbol="BTCUSDT", n_bars=10_000)
    artifact_root = tmp_path / "BTCUSDT"
    publish_market_dataset_artifact(artifact_root, source_dataset)
    step = U2_DECISION_STEP_NS
    first_ns = _timestamp_ns(source_dataset.timestamps[0])
    fit_start = 1_000
    fit_stop = 9_000
    fit_first = first_ns + fit_start * step
    fit_last = first_ns + (fit_stop - 1) * step
    source = U2TrainingSource(
        symbol="BTCUSDT",
        dataset_digest=source_dataset.dataset_id,
        source_first_timestamp_ns=first_ns,
        source_last_timestamp_ns=_timestamp_ns(source_dataset.timestamps[-1]),
        source_row_count=source_dataset.n_bars,
        fit_first_timestamp_ns=fit_first,
        fit_last_timestamp_ns=fit_last,
        fit_stop_timestamp_ns_exclusive=fit_last + step,
        fit_bar_count=fit_stop - fit_start,
    )
    closure = _single_source_closure(source)

    loaded = load_universal_trade_rl_u2_fit_datasets(
        closure=closure,
        artifact_locators={"BTCUSDT": artifact_root},
    )

    expected_view = MarketDatasetView(source_dataset, fit_start, fit_stop)
    fit = loaded["BTCUSDT"]
    assert fit.dataset_id == expected_view.identity
    assert fit.n_bars == 8_000
    np.testing.assert_array_equal(fit.close, source_dataset.close[fit_start:fit_stop])
```

Implement `_single_source_closure` in the test with fixed valid SHA-256 fixture digests and the source's exact FIT bounds; do not use Development/Admission sources.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_fit_dataset.py::test_u2_fit_loader_materializes_exact_market_dataset_view
```

Expected: **FAIL** with module/function missing.

- [ ] **Step 3: Add source-provenance falsification tests**

Add tests for:

```python
# locator closure must be exact
with pytest.raises(ValueError, match="locator|Train|closure"):
    load_universal_trade_rl_u2_fit_datasets(
        closure=closure,
        artifact_locators={"BTCUSDT": valid_root, "XRPUSDT": valid_root},
    )

# wrong canonical artifact with same symbol/timestamps/count but changed prices
with pytest.raises(ValueError, match="dataset|identity|source"):
    load_universal_trade_rl_u2_fit_datasets(
        closure=closure,
        artifact_locators={"BTCUSDT": wrong_content_root},
    )

# path independence: copy the exact artifact directory, then IDs must match
first = load_universal_trade_rl_u2_fit_datasets(
    closure=closure,
    artifact_locators={"BTCUSDT": root_a},
)
second = load_universal_trade_rl_u2_fit_datasets(
    closure=closure,
    artifact_locators={"BTCUSDT": root_b},
)
assert first["BTCUSDT"].dataset_id == second["BTCUSDT"].dataset_id
```

Also add a test with an injected loader returning a `MarketDataset` whose `identity_payload_json` is `None`; it must fail before `MarketDatasetView.materialize()` is called.

- [ ] **Step 4: Implement the focused loader**

Create helpers with no environment imports:

```python
def _require_exact_locator_closure(
    closure: U2TrainingSourceClosure,
    locators: Mapping[str, U2SourceArtifactLocator],
) -> dict[str, U2SourceArtifactLocator]: ...


def _require_source_dataset(
    *, source: U2TrainingSource, dataset: MarketDataset
) -> MarketDataset:
    if not dataset.identity_verified:
        raise ValueError("U2 source dataset must have verified canonical identity")
    if dataset.dataset_id != source.dataset_digest:
        raise ValueError("U2 source dataset identity mismatch")
    if dataset.symbols != (source.symbol,):
        raise ValueError("U2 source dataset symbol mismatch")
    if dataset.n_bars != source.source_row_count:
        raise ValueError("U2 source dataset row count mismatch")
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    expected = source.source_first_timestamp_ns + np.arange(
        source.source_row_count, dtype=np.int64
    ) * np.int64(U2_DECISION_STEP_NS)
    if not np.array_equal(timestamps_ns, expected):
        raise ValueError("U2 source dataset timestamps differ from frozen source grid")
    return dataset
```

Materialize once per symbol:

```python
view = MarketDatasetView(dataset, source.fit_start_index, source.fit_stop_index)
fit = view.materialize()
if fit.dataset_id != view.identity:
    raise ValueError("U2 FIT dataset view identity mismatch")
if fit.n_bars != source.fit_bar_count:
    raise ValueError("U2 FIT dataset bar count mismatch")
```

- [ ] **Step 5: Verify focused artifact integration**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_fit_dataset.py tests/data/test_market_dataset_artifact.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
git commit -m "feat: derive U2 FIT datasets from frozen source artifacts"
```

---

### Task 3: Derive U2 instrument bindings internally

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- Test: `tests/rl/test_universal_trade_u2_environment.py`

**Interfaces:**
- Consumes: closure, FIT dataset mapping, frozen U1 contract.
- Produces:

```python
def build_universal_trade_rl_u2_instrument_bindings(
    *,
    closure: U2TrainingSourceClosure,
    fit_datasets: Mapping[str, MarketDataset],
    u1_contract: UniversalTradeRLU1Contract,
) -> tuple[InstrumentDatasetBinding, ...]: ...
```

- [ ] **Step 1: Write RED for exact binding payloads**

For every source, assert:

```python
bindings = build_universal_trade_rl_u2_instrument_bindings(
    closure=fixture.closure,
    fit_datasets=fit_datasets,
    u1_contract=fixture.u1_contract,
)
by_symbol = {binding.concrete_symbol: binding for binding in bindings}
binding = by_symbol["BTCUSDT"]
assert binding.source_dataset_id == fit_datasets["BTCUSDT"].dataset_id
assert binding.symbol_dataset_digest == source.dataset_digest
assert binding.split == "train"
assert binding.execution_metadata_digest == content_digest(
    {
        "schema_version": "universal_trade_rl_u2_execution_binding_v1",
        "fit_dataset_id": fit_datasets["BTCUSDT"].dataset_id,
        "u1_execution_policy_digest": fixture.u1_contract.execution_policy_digest,
        "u1_pretrade_risk_digest": fixture.u1_contract.pretrade_risk_digest,
        "u1_portfolio_risk_digest": fixture.u1_contract.portfolio_risk_digest,
    }
)
assert binding.instrument_descriptor_digest == content_digest(
    {
        "schema_version": "universal_trade_rl_u2_instrument_descriptor_disabled_v1",
        "instrument_context_enabled": False,
        "v4_context_enabled": False,
    }
)
```

Add negative tests for missing/extra FIT symbol and FIT dataset whose first/last/count do not equal the source FIT contract.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py -k "derives_binding or binding_fit_closure"
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement exact digest helpers and binding derivation**

Use exact payloads from the spec. The function must iterate in `closure.sources` order and must validate `fit_datasets` exact symbol closure before constructing bindings.

- [ ] **Step 4: Keep the existing low-level builder but mark its responsibility**

Do not delete `build_universal_trade_rl_u2_environment(...)`. Keep it as a low-level prevalidated-environment validator used by focused tests. Add a docstring sentence that production U2 training must use the high-level factory introduced in Task 6, which supplies internally derived bindings.

- [ ] **Step 5: Verify**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_environment.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: derive U2 training bindings internally"
```

---

### Task 4: Make U2 episode sampling independent of unrelated binding metadata and accept SB3 worker seed offsets

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- Test: `tests/rl/test_universal_trade_u2_environment.py`

**Interfaces:**
- Consumes: existing `_OwnedU2RoutedEnvironment` and internal bindings.
- Produces U2-only overrides:

```python
@property
def run_seed(self) -> int: ...

@property
def environment_index(self) -> int: ...

@property
def canonical_probe_seed(self) -> int: ...

def _episode_seed(...)->int: ...

def reset(...): ...
```

- [ ] **Step 1: Write RED proving unrelated binding metadata currently changes sampling**

Build two low-level U2 environments with identical source/FIT/run seed/index but different valid `execution_metadata_digest` and `instrument_descriptor_digest`. Reset both and assert equal `active_episode_binding.episode_seed` and equal episode start/stop.

```python
assert first.active_episode_binding.episode_seed == second.active_episode_binding.episode_seed
assert first.active_episode_binding.episode_start == second.active_episode_binding.episode_start
assert first.active_episode_binding.episode_stop == second.active_episode_binding.episode_stop
```

Expected current behavior: FAIL because the generic `_episode_seed()` hashes complete `binding.digest`.

- [ ] **Step 2: Write RED for worker reset namespace**

```python
environment = _build(fixture, environment_index=3)
assert environment.canonical_probe_seed == _RUN_SEED + 3
environment.reset(seed=_RUN_SEED + 3)

for invalid in (_RUN_SEED, _RUN_SEED + 2, _RUN_SEED + 4):
    other = _build(fixture, environment_index=3)
    try:
        with pytest.raises(ValueError, match="seed"):
            other.reset(seed=invalid)
    finally:
        other.close()
```

Expected current behavior: FAIL because canonical seed is currently the immutable run seed.

- [ ] **Step 3: Implement U2-specific episode seed**

Override `_episode_seed()` only in `_OwnedU2RoutedEnvironment`:

```python
payload = {
    "schema_version": "universal_trade_rl_u2_episode_seed_v1",
    "run_seed": self._run_seed,
    "partition_digest": self._router.partition_digest,
    "environment_index": self._environment_index,
    "completed_episode_count": route.completed_episode_count,
    "fit_dataset_id": binding.source_dataset_id,
}
return int(content_digest(payload)[:8], 16)
```

- [ ] **Step 4: Implement U2 external seed translation**

```python
@property
def canonical_probe_seed(self) -> int:
    return self._run_seed + self._environment_index

@property
def run_seed(self) -> int:
    return self._run_seed

@property
def environment_index(self) -> int:
    return self._environment_index


def reset(self, *, seed=None, options=None):
    if seed is not None:
        resolved = _non_negative_integer(seed, field="U2 reset seed")
        if resolved != self.canonical_probe_seed:
            raise ValueError("U2 reset seed must equal member_seed + environment_index")
    return super().reset(seed=self._run_seed, options=options)
```

Do not change the generic routed environment.

- [ ] **Step 5: Verify focused routing regressions**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/rl/test_universal_episode_router.py tests/rl/test_universal_training_contract_binding.py
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "fix: isolate U2 episode and worker seed namespaces"
```

---

### Task 5: Define the shared run-level U2 environment-generation digest

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- Test: `tests/rl/test_universal_trade_u2_environment.py`

**Interfaces:**
- Produces:

```python
def build_universal_trade_rl_u2_environment_generation_digest(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    source_closure: U2TrainingSourceClosure,
    bindings: tuple[InstrumentDatasetBinding, ...],
    run_seed: int,
) -> str: ...
```

Exact digest payload is the one in spec Section 10.

- [ ] **Step 1: Write RED for exact generation identity**

Construct the expected payload explicitly in the test and require exact equality with `content_digest(expected_payload)`. Assert that changing each of these independently changes the digest: `run_seed`, source closure, one FIT binding digest. Also assert the payload binds `environment_indices == (0,1,2,3,4,5,6,7)`.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py -k "environment_generation"
```

- [ ] **Step 3: Implement strict contract/config extraction**

Read only from `UniversalTradeRLU2Contract`:

```python
payload = dict(u2_contract.training_config_payload)
if payload.get("n_envs") != 8:
    raise ValueError("U2 environment generation requires n_envs=8")
if payload.get("vector_environment_mode") != "in_process":
    raise ValueError("U2 environment generation requires in_process vector mode")
if run_seed not in u2_contract.training_seeds:
    raise ValueError("U2 environment generation seed is not preregistered")
```

Then hash the exact spec payload; do not add locator paths or scalar `environment_index`.

- [ ] **Step 4: Allow `_OwnedU2RoutedEnvironment` to expose a supplied shared generation digest**

Add a required `environment_generation_digest` argument on the production construction path and override:

```python
@property
def environment_digest(self) -> str:
    return self._u2_environment_generation_digest
```

For low-level tests that intentionally bypass the high-level production factory, preserve an explicit low-level mode by requiring callers to pass a digest rather than silently falling back to the generic worker-specific digest. Update existing `_build()` test helper to use a deterministic fixture generation digest.

- [ ] **Step 5: Verify identity tests**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/rl/test_training_environment_identity.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: bind U2 run-level environment generation"
```

---

### Task 6: Build the production indexed U2 environment factory and enforce worker-state isolation

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- Create: `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`
- Modify: `tests/rl/test_universal_trade_u2_environment.py`

**Interfaces:**
- Consumes: U2 contract, source closure, source locators, U1 contract/policy/normalizer, `Callable[[MarketDataset], UniversalTradeEnvironment]`, member run seed.
- Produces:

```python
class UniversalTradeRLU2EnvironmentFactory:
    @property
    def environment_generation_digest(self) -> str: ...
    @property
    def source_closure_digest(self) -> str: ...
    @property
    def run_seed(self) -> int: ...
    def __call__(self) -> EpisodeRoutedSingleInstrumentEnv: ...
    def for_environment_index(
        self, index: int
    ) -> Callable[[], EpisodeRoutedSingleInstrumentEnv]: ...


def build_universal_trade_rl_u2_environment_factory(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    source_closure: U2TrainingSourceClosure,
    source_artifact_locators: Mapping[str, str | Path],
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    u1_environment_factory: Callable[[MarketDataset], UniversalTradeEnvironment],
    run_seed: int,
) -> UniversalTradeRLU2EnvironmentFactory: ...
```

- [ ] **Step 1: Write RED for one-time FIT materialization and indexed workers**

Use an injected source loader spy and assert the factory loads each Train source exactly once at construction, then creates workers without reloading sources:

```python
factory = build_universal_trade_rl_u2_environment_factory(...)
assert loader_calls == list(fixture.train_symbols)
workers = [factory.for_environment_index(i)() for i in range(8)]
assert loader_calls == list(fixture.train_symbols)
try:
    assert [worker.environment_index for worker in workers] == list(range(8))
    assert {worker.run_seed for worker in workers} == {member_seed}
    assert {worker.environment_digest for worker in workers} == {
        factory.environment_generation_digest
    }
finally:
    for worker in workers:
        worker.close()
```

Expose a loader injection only on the lower-level constructor used in tests if needed; the public builder must default to the canonical loader.

- [ ] **Step 2: Write RED for mutable-state isolation**

Reset workers 0 and 1, obtain active U1 children, and assert:

```python
assert worker0._active_environment is not worker1._active_environment
assert worker0._active_environment.base_env is not worker1._active_environment.base_env
assert worker0._active_environment.dataset.dataset_id == worker1._active_environment.dataset.dataset_id
assert worker0._active_environment.sequence_normalizer is worker1._active_environment.sequence_normalizer
```

Trade only in worker 0 and assert worker 1 runtime remains cash/zero-pending using the existing `_assert_cash_runtime` oracle.

- [ ] **Step 3: Write RED for invalid worker indices and caller reuse**

Require `for_environment_index(-1)` and `for_environment_index(8)` to fail. Add a deliberately bad `u1_environment_factory` that returns the same `UniversalTradeEnvironment` object twice; the high-level U2 factory must reject runtime reuse rather than allow cross-worker state aliasing.

- [ ] **Step 4: Implement the high-level factory**

At construction:

1. validate U2/source/U1/normalizer identity closure;
2. call `load_universal_trade_rl_u2_fit_datasets()` once;
3. derive internal bindings once;
4. compute shared generation digest once;
5. retain immutable FIT datasets/normalizer/contracts.

For each worker creation, call `u1_environment_factory(fit_dataset)` afresh for each symbol, validate with `require_universal_trade_rl_u1_environment_contract`, and pass only internal bindings into the low-level U2 builder. Track issued mutable U1 object IDs for the factory lifetime and reject a repeated object ID.

- [ ] **Step 5: Verify unit + integration layer**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py -k "factory or isolation or generation"
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_environment.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "feat: add isolated eight-worker U2 environment factory"
```

---

### Task 7: Prove actual maintained `DummyVecEnv` seed/reset integration

**Files:**
- Modify: `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`
- Production change: only `trade_rl/workflows/universal_trade_rl_u2_environment.py` if the new U2 adapter still fails; do not change generic SB3 vectorization when the U2 adapter can satisfy the contract.

**Interfaces:**
- Consumes: `UniversalTradeRLU2EnvironmentFactory`, `_filtered_environment_factory`, `_build_training_environment`.
- Produces: executable evidence for exact eight-worker external seed translation.

- [ ] **Step 1: Write the actual maintained vector-path test**

```python
def test_u2_in_process_vector_accepts_sb3_member_seed_offsets(u2_factory) -> None:
    from trade_rl.integrations.sb3_environment import (
        _build_training_environment,
        _filtered_environment_factory,
    )

    member_seed = u2_factory.run_seed
    vector = _build_training_environment(
        _filtered_environment_factory(u2_factory),
        8,
        subprocesses=False,
    )
    try:
        assert vector.seed(member_seed) == [member_seed + i for i in range(8)]
        observation = vector.reset()
        assert observation is not None
        for index, filtered in enumerate(vector.envs):
            worker = filtered.unwrapped
            assert worker.run_seed == member_seed
            assert worker.environment_index == index
            assert worker.canonical_probe_seed == member_seed + index
            assert worker.environment_digest == u2_factory.environment_generation_digest
    finally:
        vector.close()
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_vector_runtime.py::test_u2_in_process_vector_accepts_sb3_member_seed_offsets
```

Expected after Tasks 4-6: PASS. If it fails, record the exact failing boundary before modifying code.

- [ ] **Step 3: Run generic indexed-factory regression**

```bash
uv run pytest -q tests/integrations/test_sb3_indexed_environment_factory.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
```

- [ ] **Step 4: Commit test evidence**

```bash
git add tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "test: prove U2 eight-worker SB3 seed integration"
```

---

### Task 8: Prove U2 timeout metadata and exact terminal observation through `DummyVecEnv`

**Files:**
- Modify: `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`

**Interfaces:**
- Consumes: actual high-level U2 factory and actual in-process vector path.
- Produces: timeout metadata/terminal observation evidence only; no custom bootstrap implementation.

- [ ] **Step 1: Write synchronized direct-vs-vector terminal observation integration test**

Create one direct worker index 0 and one eight-worker vector from two equivalent factories with the same frozen identity/member seed. Reset both using the same canonical external seed for worker 0. Step zero action in lockstep until the direct worker truncates. At the terminal step:

```python
next_direct, raw_reward, terminated, truncated, direct_info = direct.step(zero_action)
vector_obs, vector_rewards, vector_dones, vector_infos = vector.step(
    np.zeros((8, 1), dtype=np.float32)
)
assert terminated is False
assert truncated is True
assert vector_dones[0]
assert vector_infos[0]["TimeLimit.truncated"] is True
terminal = vector_infos[0]["terminal_observation"]
assert set(terminal) == set(next_direct)
for key in next_direct:
    np.testing.assert_allclose(terminal[key], next_direct[key], rtol=0.0, atol=0.0)
```

The vector observation at index 0 after that step is the next episode reset observation and must not be substituted for `terminal_observation`.

- [ ] **Step 2: Preserve raw economic reward oracle**

Before the terminal direct step, record direct `hybrid.portfolio_value`; after the step assert:

```python
expected_reward = 100.0 * math.log(after_value / before_value)
assert raw_reward == pytest.approx(expected_reward, abs=1e-10)
assert vector_rewards[0] == pytest.approx(raw_reward, abs=1e-10)
```

No `gamma * V_terminal` term is allowed in the environment/vector economic reward.

- [ ] **Step 3: Run the timeout integration test**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_vector_runtime.py -k "timeout or terminal_observation"
```

Expected: PASS using maintained SB3 `DummyVecEnv`; if it fails, fix only the U2/filter boundary that loses semantics.

- [ ] **Step 4: Commit evidence**

```bash
git add tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "test: prove U2 timeout metadata through vectorization"
```

---

### Task 9: Add a controlled exactly-once PPO timeout-bootstrap oracle

**Files:**
- Create: `tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py`
- Production changes: none expected.

**Interfaces:**
- Consumes: maintained Stable-Baselines3 PPO 2.3.2 `OnPolicyAlgorithm.collect_rollouts()` semantics.
- Produces: executable evidence that timeout target reward is `raw_reward + gamma * V_terminal` exactly once and the raw environment reward remains unchanged.

- [ ] **Step 1: Create a one-step truncating Gym environment**

```python
class _OneStepTimeoutEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.raw_rewards: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        del action
        reward = 1.25
        self.raw_rewards.append(reward)
        return np.asarray([2.0], dtype=np.float32), reward, False, True, {}
```

Wrap with `DummyVecEnv([lambda: env])` so SB3 creates `TimeLimit.truncated` and `terminal_observation` through its maintained path.

- [ ] **Step 2: Instantiate a tiny PPO and patch only terminal value prediction**

Use `PPO("MlpPolicy", vec, n_steps=2, batch_size=2, gamma=0.9, seed=0, device="cpu")`. Patch `model.policy.predict_values` so the terminal observation returns a known scalar tensor `2.0`; leave rollout-buffer arithmetic intact.

Use a minimal callback subclass whose `_on_step()` returns `True`, initialize learning state with `model._setup_learn(total_timesteps=2, callback=callback, reset_num_timesteps=True, tb_log_name="u2-timeout", progress_bar=False)`, then call one `collect_rollouts()` into the model's real rollout buffer.

- [ ] **Step 3: Assert exactly one bootstrap**

The first stored training reward must equal:

```python
expected = 1.25 + 0.9 * 2.0
assert model.rollout_buffer.rewards[0, 0] == pytest.approx(expected)
assert env.raw_rewards[0] == pytest.approx(1.25)
assert model.rollout_buffer.rewards[0, 0] != pytest.approx(1.25 + 2 * 0.9 * 2.0)
```

Also assert the environment has no wealth/reward mutation API invoked by the patch; this test is training-target-only.

- [ ] **Step 4: Run oracle**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py
```

Expected: PASS on pinned SB3 2.3.2. Do not add custom timeout bootstrap production code when this passes.

- [ ] **Step 5: Commit**

```bash
git add tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py
git commit -m "test: prove exactly-once PPO timeout bootstrap"
```

---

### Task 10: Add minimal fixed-PPO per-seed U2 orchestration without real training in tests

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_training.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_training_runtime.py`

**Interfaces:**
- Consumes: `UniversalTradeRLU2SeedTrainingPlan`, `UniversalTradeRLU2EnvironmentFactory`, maintained `StableBaselines3PPOBackend`.
- Produces:

```python
class U2TrainingBackend(Protocol):
    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult: ...

U2TrainingBackendFactory = Callable[
    [UniversalTradeRLU2EnvironmentFactory], U2TrainingBackend
]


def train_universal_trade_rl_u2_seed(
    *,
    plan: UniversalTradeRLU2SeedTrainingPlan,
    environment_factory: UniversalTradeRLU2EnvironmentFactory,
    output_path: Path,
    backend_factory: U2TrainingBackendFactory = StableBaselines3PPOBackend,
) -> PolicyTrainingResult: ...
```

No resume argument is exposed in this U2 V1 orchestration task.

- [ ] **Step 1: Write RED for seed/source/environment lineage before backend call**

Create a fake backend factory that records constructor environment factory and `train()` arguments. Assert:

```python
result = train_universal_trade_rl_u2_seed(
    plan=plan,
    environment_factory=factory,
    output_path=tmp_path / "policy.zip",
    backend_factory=fake_backend_factory,
)
assert calls.seed == plan.seed
assert content_digest(calls.config.digest_payload()) == plan.training_config_digest
assert calls.environment_factory.environment_generation_digest == expected_environment_digest
```

Add pre-backend rejection tests for:

```python
factory.run_seed != plan.seed
factory.source_closure_digest != plan.source_closure_digest
```

and assert fake backend call count remains zero.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_training_runtime.py -k "train_universal_trade_rl_u2_seed"
```

- [ ] **Step 3: Implement fail-closed orchestration**

Implementation order:

```python
if environment_factory.run_seed != plan.seed:
    raise ValueError("U2 training environment member seed mismatch")
if environment_factory.source_closure_digest != plan.source_closure_digest:
    raise ValueError("U2 training environment source closure mismatch")
config = build_universal_trade_rl_u2_training_config()
if content_digest(config.digest_payload()) != plan.training_config_digest:
    raise ValueError("U2 training plan configuration mismatch")
probe = environment_factory()
try:
    if probe.environment_digest != environment_factory.environment_generation_digest:
        raise ValueError("U2 training environment generation mismatch")
finally:
    probe.close()
backend = backend_factory(environment_factory)
return backend.train(seed=plan.seed, config=config, output_path=output_path)
```

Do not loop over seeds here. The caller creates one plan/factory/backend run per preregistered seed.

- [ ] **Step 4: Verify existing anti-cherry-picking gates still pass**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/workflows/universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
git commit -m "feat: orchestrate one fixed U2 PPO member"
```

---

### Task 11: Documentation pointer and architecture/self-review gate

**Files:**
- Modify: `docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`
- Test/Review: architecture/import contracts and complete diff.

**Interfaces:** No runtime API changes.

- [ ] **Step 1: Append a historical supersession pointer**

Add a short section near the top of the historical implementation plan:

```markdown
> **2026-09-04 mechanics amendment:** Task 4-6 source/FIT/routing/vector identity details are superseded by `../specs/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity-amendment.md` and the executable plan `../../../superpowers/plans/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity.md`. Economic preregistration and Selection thresholds are unchanged.
```

Do not edit historical economic thresholds or claim training readiness.

- [ ] **Step 2: Run architecture/static checks**

```bash
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_preflight.py trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py trade_rl/workflows/universal_trade_rl_u2_environment.py trade_rl/workflows/universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
uv run ruff format --check trade_rl tests
uv run mypy trade_rl/workflows/universal_trade_rl_u2_preflight.py trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py trade_rl/workflows/universal_trade_rl_u2_environment.py trade_rl/workflows/universal_trade_rl_u2_training.py
uv run pytest -q tests/test_architecture_contract.py tests/architecture
```

- [ ] **Step 3: Run related U0/U1/U2 regression suite**

```bash
uv run pytest -q \
  tests/workflows/test_universal_trade_rl_u2_time_partition.py \
  tests/workflows/test_universal_trade_rl_u2_contract.py \
  tests/workflows/test_universal_trade_rl_u2_preflight.py \
  tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py \
  tests/workflows/test_universal_trade_rl_u2_fit_dataset.py \
  tests/rl/test_universal_trade_u2_environment.py \
  tests/workflows/test_universal_trade_rl_u2_training.py \
  tests/workflows/test_universal_trade_rl_u2_training_runtime.py \
  tests/integrations/test_sb3_indexed_environment_factory.py \
  tests/integrations/test_universal_trade_rl_u2_vector_runtime.py \
  tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py
```

- [ ] **Step 4: Perform explicit falsification review before full suite**

Reconstruct from the spec, not from implementation intent, and try to prove each of the following can still slip through:

- same U0 metadata with different numeric source content;
- same source/FIT identity under different filesystem path;
- FIT off-by-one start/stop;
- arbitrary caller binding entering production path;
- execution/descriptor metadata perturbing episode seed;
- worker index collapse;
- shared mutable U1 object across workers;
- worker 1..7 reset seed rejection;
- worker-specific drift hidden behind worker-0 digest;
- timeout terminal observation replaced by reset observation;
- double bootstrap;
- raw economic reward mutated by bootstrap;
- seed 1/2 run reusing seed-0 environment generation.

Any observed loophole is fixed with a new failing regression test before proceeding.

- [ ] **Step 5: Commit documentation/review updates**

```bash
git add docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md
git commit -m "docs: link U2 source routing mechanics amendment"
```

---

### Task 12: Full verification and exact-HEAD readiness evidence

**Files:** No planned production changes. If a failure reveals a defect, return to the smallest affected task, add a regression test, fix, and rerun this task from the start.

- [ ] **Step 1: Inspect final diff/status/HEAD before verification**

```bash
git status --short
git diff --check
git diff --stat main...HEAD
git log -1 --oneline
```

Require no accidental generated files, debug code, temporary workflows, secrets, or unrelated refactors.

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -q
```

Record exact pass/fail/skip counts; do not summarize as merely "Green".

- [ ] **Step 3: Run full static/build/package gates used by repository CI**

At minimum:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run pytest -q tests/test_architecture_contract.py tests/architecture
uv build
```

Also run the repository's maintained training-capability audit command exactly as defined by `.github/workflows/ci.yml` on the final HEAD; do not substitute a narrower local smoke test.

- [ ] **Step 4: Verify changed-line execution and assertion quality**

Inspect coverage for the changed U2 modules and confirm tests execute:

- source phase-mismatch branch;
- wrong/tampered artifact path;
- exact FIT view path;
- binding derivation;
- U2 episode seed override;
- external seed rejection/acceptance;
- generation digest construction;
- all indexed workers;
- mutable-state isolation;
- timeout metadata;
- exactly-once bootstrap;
- per-seed orchestration fail-closed branches.

Coverage percentage alone is not sufficient; inspect assertions and failure-mode coverage.

- [ ] **Step 5: Independent review from original requirements**

Review final diff against:

1. `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`;
2. `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-robustness-timeout-amendment.md`;
3. `docs/implementation-plans/specs/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity-amendment.md`.

Do not inherit implementer conclusions. Explicitly list any unmet acceptance criterion, weakened invariant, untested failure mode, or hidden dependency.

- [ ] **Step 6: Verify exact-HEAD CI**

Push only after local verification. On the exact pushed HEAD, require the repository's required CI jobs to complete successfully. Verify the run checkout SHA equals the final `git rev-parse HEAD`. Do not reuse CI success from an older commit.

- [ ] **Step 7: Final readiness report**

Report separately:

- what changed;
- why the design uses `MarketDatasetView` rather than a second FIT artifact;
- mapping from each Acceptance Criterion to test evidence;
- failure modes exercised;
- exact local test/static/build results;
- independent/falsification findings;
- exact-head CI status;
- unverified items;
- remaining risk.

The final report must state explicitly that this work proves **mechanics/provenance readiness only**. It does not prove profitability, Development generalization, Admission success, Production readiness, or exact mid-episode resume.
