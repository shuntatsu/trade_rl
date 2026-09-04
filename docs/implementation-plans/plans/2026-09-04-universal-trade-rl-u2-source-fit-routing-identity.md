# Universal Trade RL U2 Source / FIT / Routing Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the U2 V1 source/FIT/routing provenance chain from frozen U0 source artifacts through exact FIT views, U2-owned bindings, deterministic 8-worker routing, SB3 seed translation, timeout semantics, and minimal fixed-PPO orchestration without changing U1 economics.

**Architecture:** Keep metadata-only preregistration in `universal_trade_rl_u2_preflight.py`; put canonical source-artifact loading and `MarketDatasetView` materialization in a focused `universal_trade_rl_u2_fit_dataset.py`; keep U1 Risk/Execution/Accounting construction injected and validated. A high-level `UniversalTradeRLU2EnvironmentFactory` owns immutable FIT datasets, internally derived bindings, the member seed, and one shared run-level environment-generation digest, while constructing fresh mutable U1 environments for worker indices 0..7. Reuse maintained `DummyVecEnv`, SB3 PPO assembly, timeout handling, and generic checkpoint validation.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.3.2, PyTorch, dataclasses, `MarketDataset`, `MarketDatasetView`, canonical SHA-256 digests, pytest, Ruff, MyPy, Import Linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity-amendment.md`

## Global Constraints

- Real U2 PPO training remains **NO-GO** during implementation/tests.
- Admission remains **SEALED** and Production remains **NO-GO**.
- Exact U2 recipe remains PPO; seeds `(0,1,2)`; seed 0 primary; `524_288` timesteps; `n_envs=8`; `n_steps=128`; batch 256; 10 epochs; deterministic CUDA; `vector_environment_mode="in_process"`.
- U1 remains the sole Risk / Execution / Accounting authority. U2 must not reproduce or silently default U1 economic configuration.
- U1 action/reward/normalizer/context semantics remain unchanged; normal horizon remains `terminated=False`, `truncated=True`, `liquidate_on_end=False`.
- Do not modify generic `EpisodeRoutedSingleInstrumentEnv`, generic checkpoint schemas, or generic SB3 vectorization unless an independently reproduced generic defect requires it.
- Source paths are locators, not research identity.
- FIT is an in-memory `MarketDatasetView` materialization, never a new persistent dataset artifact.
- One factory materializes each symbol FIT dataset once; immutable FIT datasets/normalizer/contracts may be shared, mutable U1 environments/state may not.
- U2 seed namespace is exact: `member_seed == PPO seed == router run_seed`; worker `i` externally receives `member_seed + i`.
- All workers 0..7 expose the same run-level U2 `environment_digest`; their router digests may differ.
- Environment/checkpoint identity equality proves compatibility, not exact mid-episode trajectory continuation.
- Every production change follows Red -> Green -> Refactor. Do not weaken tests, skip failures, or change an oracle to match implementation output.

## Quality Contract

**Objective:** Make U0 source provenance, FIT derivation, U2 routing randomness, eight-worker identity/state isolation, SB3 timeout semantics, and checkpoint environment compatibility deterministic and fail-closed before real PPO/Development evaluation.

**Non-goals:** No Development B/C/D evaluation, Selection, Admission authorization, Production promotion, hyperparameter/architecture change, persistent FIT artifact, or exact mid-episode resume implementation.

**Acceptance Criteria:**

1. FIT metadata is phase-aligned to the source 15-minute grid before numeric loading.
2. Only a canonical source artifact matching exact U0 source identity can be sliced.
3. FIT child ID equals exact `MarketDatasetView.identity` and contains only the preregistered FIT range.
4. Production U2 bindings are internally derived from source/FIT/U1 identities.
5. Unrelated execution/descriptor metadata cannot perturb episode sampling.
6. Workers 0..7 share member run seed and generation digest while retaining exact indices.
7. Mutable U1 state is isolated across workers.
8. SB3 reset seeds `member_seed + i` are accepted without redefining router run seed.
9. Source/FIT/binding/member-seed/vector-generation drift changes checkpoint environment identity.
10. Actual U2 vectorization preserves timeout metadata and exact terminal observation.
11. PPO applies exactly one timeout bootstrap while raw economic reward/wealth is unchanged.
12. Minimal per-seed orchestration validates plan/factory/backend result identity without running real training in tests.

**Test Oracle:** exact source IDs/timestamps/counts; FIT indices/view ID/arrays; binding payloads; episode seed/start/stop; router seed/index/cycle; object identity/runtime snapshots; generation digest; SB3 seed vector; timeout info and terminal observation; rollout-buffer reward; `PolicyTrainingResult.environment_digest` and `actual_timesteps`; checkpoint environment identity.

**Required Test Layers:** Unit; contract/falsification; canonical artifact integration; U1/U2 integration; actual `DummyVecEnv`; controlled PPO timeout bootstrap; static/typing/architecture checks; related/full suite; package/build; exact-final-HEAD CI.

**Quality Gate:** Do not report mechanics/provenance completion until every Acceptance Criterion has an executable assertion, required layers pass on one exact final HEAD, falsification/independent review finds no unresolved Critical/High mechanics issue, and remaining limitations are recorded.

---

## File Map

**Create**

- `trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py` — source artifact verification + FIT `MarketDatasetView` only.
- `tests/workflows/test_universal_trade_rl_u2_fit_dataset.py` — artifact/FIT provenance and locator independence.
- `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py` — 8-worker factory, state isolation, SB3 seeds, timeout metadata.
- `tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py` — controlled exactly-once PPO bootstrap oracle.

**Modify**

- `trade_rl/workflows/universal_trade_rl_u2_preflight.py`
- `trade_rl/workflows/universal_trade_rl_u2_environment.py`
- `trade_rl/workflows/universal_trade_rl_u2_training.py`
- `tests/workflows/test_universal_trade_rl_u2_preflight.py`
- `tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py`
- `tests/rl/test_universal_trade_u2_environment.py`
- `tests/workflows/test_universal_trade_rl_u2_training_runtime.py`
- `docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md` — historical supersession pointer only.

**Reuse without redesign**

- `trade_rl.data.artifact.load_market_dataset_artifact`
- `trade_rl.data.artifacts.MarketDatasetView`
- `trade_rl.rl.universal_single_instrument_env.EpisodeRoutedSingleInstrumentEnv`
- `trade_rl.workflows.universal_trade_rl_u1_contract.require_universal_trade_rl_u1_environment_contract`
- `trade_rl.integrations.sb3_environment._filtered_environment_factory`
- `trade_rl.integrations.sb3_environment._build_training_environment`
- `trade_rl.integrations.sb3_training.StableBaselines3PPOBackend`
- existing checkpoint manifest/loader validation.

---

### Task 1: Enforce source/FIT grid-phase alignment

**Files:** `trade_rl/workflows/universal_trade_rl_u2_preflight.py`, `tests/workflows/test_universal_trade_rl_u2_preflight.py`, `tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py`

**Produces:** `U2TrainingSource.fit_start_index: int`, `U2TrainingSource.fit_stop_index: int`.

- [ ] **Step 1: Write the RED phase-shift test**

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

- [ ] **Step 2: Observe RED**

`uv run pytest -q tests/workflows/test_universal_trade_rl_u2_preflight.py::test_u2_training_source_rejects_fit_grid_not_aligned_to_source_grid`

Expected current result: FAIL because independent dense-grid validation does not enforce shared phase.

- [ ] **Step 3: Add positive index oracle**

Construct a source whose FIT begins exactly 40 bars after source start and assert `fit_start_index == 40`, `fit_stop_index == 72` for 32 FIT bars. Add a negative constructor case where FIT start is aligned but its computed stop exceeds `source_row_count`.

- [ ] **Step 4: Implement metadata arithmetic**

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

Properties recompute the validated values from frozen metadata.

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_preflight.py
git add trade_rl/workflows/universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py
git commit -m "fix: align U2 FIT metadata to source grid"
```

---

### Task 2: Verify canonical source artifacts and materialize exact FIT views

**Files:** Create `trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py`, `tests/workflows/test_universal_trade_rl_u2_fit_dataset.py`.

**Public interfaces:**

- `U2SourceArtifactLocator = str | Path`
- `U2SourceArtifactLoader = Callable[[U2SourceArtifactLocator], MarketDataset]`
- `load_universal_trade_rl_u2_fit_datasets(*, closure, artifact_locators, loader=load_market_dataset_artifact) -> dict[str, MarketDataset]`

Export all three through this module's `__all__`.

- [ ] **Step 1: Write canonical full-source -> FIT RED**

Publish `make_u1_market(symbol="BTCUSDT", n_bars=10_000)` with `publish_market_dataset_artifact`. Create one `U2TrainingSource` whose FIT is `[1_000, 9_000)`. A test-only `_single_source_closure(source)` must construct `U2TrainingSourceClosure` using fixed SHA-256 fixtures (`"1"*64` through `"6"*64`) and copy all FIT bounds from `source`. Then assert:

```python
loaded = load_universal_trade_rl_u2_fit_datasets(
    closure=closure,
    artifact_locators={"BTCUSDT": artifact_root},
)
expected_view = MarketDatasetView(source_dataset, 1_000, 9_000)
fit = loaded["BTCUSDT"]
assert fit.dataset_id == expected_view.identity
assert fit.n_bars == 8_000
np.testing.assert_array_equal(fit.close, source_dataset.close[1_000:9_000])
```

- [ ] **Step 2: Observe RED**

`uv run pytest -q tests/workflows/test_universal_trade_rl_u2_fit_dataset.py::test_u2_fit_loader_materializes_exact_market_dataset_view`

Expected: module/function missing.

- [ ] **Step 3: Add falsification tests before implementation**

Require rejection for: missing locator; extra locator; wrong canonical artifact with same symbol/timestamps/count but changed numeric arrays; injected loader returning an unverified `MarketDataset`; tampered canonical artifact. Copy a byte-identical artifact to another directory and assert both locator choices produce the same FIT dataset ID.

- [ ] **Step 4: Implement exact loader**

`_require_exact_locator_closure()` must compare locator keys to `tuple(source.symbol for source in closure.sources)` exactly. `_require_source_dataset()` requires `identity_verified`, exact dataset ID, exact one-symbol tuple, exact row count, and full timestamp array equality to `source_first + arange(source_row_count) * U2_DECISION_STEP_NS`.

For each source in closure order:

```python
view = MarketDatasetView(dataset, source.fit_start_index, source.fit_stop_index)
fit = view.materialize()
if fit.dataset_id != view.identity:
    raise ValueError("U2 FIT dataset view identity mismatch")
if fit.n_bars != source.fit_bar_count:
    raise ValueError("U2 FIT dataset bar count mismatch")
if _timestamp_ns(fit.timestamps[0]) != source.fit_first_timestamp_ns:
    raise ValueError("U2 FIT dataset first timestamp mismatch")
if _timestamp_ns(fit.timestamps[-1]) != source.fit_last_timestamp_ns:
    raise ValueError("U2 FIT dataset last timestamp mismatch")
```

Return a dict inserted in closure source order. No environment imports and no persistent FIT write.

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_fit_dataset.py tests/data/test_market_dataset_artifact.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py
git add trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py
git commit -m "feat: derive U2 FIT datasets from frozen source artifacts"
```

---

### Task 3: Derive U2 bindings internally

**Files:** `trade_rl/workflows/universal_trade_rl_u2_environment.py`, `tests/rl/test_universal_trade_u2_environment.py`.

**Produces:** `build_universal_trade_rl_u2_instrument_bindings(*, closure, fit_datasets, u1_contract) -> tuple[InstrumentDatasetBinding, ...]` and exports it in `__all__`.

- [ ] **Step 1: RED exact binding payloads**

For each source, assert `source_dataset_id == fit_dataset.dataset_id`, `symbol_dataset_digest == source.dataset_digest`, `split == "train"`, and exact digests:

```python
expected_execution = content_digest({
    "schema_version": "universal_trade_rl_u2_execution_binding_v1",
    "fit_dataset_id": fit.dataset_id,
    "u1_execution_policy_digest": u1_contract.execution_policy_digest,
    "u1_pretrade_risk_digest": u1_contract.pretrade_risk_digest,
    "u1_portfolio_risk_digest": u1_contract.portfolio_risk_digest,
})
expected_descriptor = content_digest({
    "schema_version": "universal_trade_rl_u2_instrument_descriptor_disabled_v1",
    "instrument_context_enabled": False,
    "v4_context_enabled": False,
})
```

Add missing/extra FIT symbol and wrong FIT first/last/count rejection tests.

- [ ] **Step 2: Observe RED**

`uv run pytest -q tests/rl/test_universal_trade_u2_environment.py -k "derives_binding or binding_fit_closure"`

- [ ] **Step 3: Implement minimal pure derivation**

Validate exact FIT symbol closure and FIT bounds before constructing `InstrumentDatasetBinding` in `closure.sources` order. Keep the existing low-level `build_universal_trade_rl_u2_environment` as a validator for prevalidated inputs, but document that production training uses Task 6's high-level factory.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_environment.py
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: derive U2 training bindings internally"
```

---

### Task 4: Isolate U2 episode seed and external worker reset seed namespaces

**Files:** `trade_rl/workflows/universal_trade_rl_u2_environment.py`, `tests/rl/test_universal_trade_u2_environment.py`.

- [ ] **Step 1: RED unrelated metadata sampling drift**

Build two low-level U2 environments with identical source/FIT/run seed/index but different valid execution/descriptor binding digests. Reset both and require equal `active_episode_binding.episode_seed`, `episode_start`, and `episode_stop`. Current generic binding-digest seed should fail this oracle.

- [ ] **Step 2: RED worker reset contract**

For `run_seed=17`, `environment_index=3`, require `canonical_probe_seed == 20`, accept `reset(seed=20)`, and reject 17, 19, and 21.

- [ ] **Step 3: Implement U2-only episode seed override**

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

- [ ] **Step 4: Implement U2-only external reset translation**

Expose read-only `run_seed`, `environment_index`, `canonical_probe_seed`. Validate the external seed against `run_seed + environment_index`, then delegate to `super().reset(seed=self._run_seed, options=options)`. Do not modify the generic routed class.

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/rl/test_universal_episode_router.py tests/rl/test_universal_training_contract_binding.py
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "fix: isolate U2 episode and worker seed namespaces"
```

---

### Task 5: Bind the shared run-level environment generation

**Files:** `trade_rl/workflows/universal_trade_rl_u2_environment.py`, `tests/rl/test_universal_trade_u2_environment.py`.

**Produces:** `build_universal_trade_rl_u2_environment_generation_digest(*, u2_contract, source_closure, bindings, run_seed) -> str`, exported in `__all__`.

- [ ] **Step 1: RED exact digest**

The test must construct the exact spec Section 10 payload and compare against `content_digest(expected_payload)`. Independently mutate member seed, source closure, and one binding and require different digests. Require `environment_indices == (0,1,2,3,4,5,6,7)`.

- [ ] **Step 2: Implement exact generation digest**

Validate bindings with `validate_training_instrument_bindings(tuple(source.symbol for source in source_closure.sources), bindings)` and canonicalize back into source order before hashing. Read `n_envs` and `vector_environment_mode` only from `u2_contract.training_config_payload`; require exact `8` and `"in_process"`; require `run_seed in u2_contract.training_seeds`. Hash only the exact spec payload; never include locator paths or one scalar worker index.

- [ ] **Step 3: Make `_OwnedU2RoutedEnvironment.environment_digest` explicitly supplied**

Add required `environment_generation_digest` to the U2 low-level builder/subclass path, validate it as SHA-256, store it, and expose it via the property. Update all existing low-level U2 tests to pass a deterministic fixture digest; never silently fall back to generic worker-specific digest.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/rl/test_training_environment_identity.py
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py
git commit -m "feat: bind U2 run-level environment generation"
```

---

### Task 6: Build the high-level indexed factory and enforce mutable-state isolation

**Files:** `trade_rl/workflows/universal_trade_rl_u2_environment.py`, create `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`, modify `tests/rl/test_universal_trade_u2_environment.py`.

**Exact builder signature:**

`build_universal_trade_rl_u2_environment_factory(*, u2_contract, source_closure, source_artifact_locators, u1_contract, policy_contract, normalizer, u1_environment_factory, run_seed, source_loader=load_market_dataset_artifact) -> UniversalTradeRLU2EnvironmentFactory`

`source_loader` is an explicit testing seam; production callers omit it and therefore use the canonical loader.

**Factory surface:** read-only `environment_generation_digest`, `source_closure_digest`, `run_seed`; `__call__()` equivalent to worker 0; `for_environment_index(index)` returns a zero-argument worker constructor and rejects indices outside 0..7.

- [ ] **Step 1: RED one-time source/FIT materialization**

With a source-loader spy, build the factory and require exactly one loader call per Train symbol during factory construction. Create all eight workers and require no further source loader calls.

- [ ] **Step 2: RED worker identity/state isolation**

Require indices `[0..7]`, common member run seed, common generation digest, distinct active `UniversalTradeEnvironment` and `base_env` objects for workers 0/1, identical FIT dataset IDs/normalizer generation, and cash/zero-pending state in worker 1 after trading only worker 0.

- [ ] **Step 3: RED invalid index and U1 object reuse**

Reject `for_environment_index(-1)` and `(8)`. A bad `u1_environment_factory` returning the same environment object on a later call must be rejected before two workers can share state. Track issued mutable environment object IDs for the high-level factory lifetime solely for this fail-closed guard.

- [ ] **Step 4: Implement construction order**

At factory construction: validate U2/source/U1/normalizer identities; load FIT datasets once; derive bindings once; compute generation digest once. Per worker: create fresh U1 child for each symbol from shared FIT dataset; run `require_universal_trade_rl_u1_environment_contract`; pass only internal bindings and shared generation digest to the low-level U2 builder.

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest -q tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py -k "factory or isolation or generation"
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
uv run ruff format --check trade_rl/workflows/universal_trade_rl_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
uv run mypy trade_rl/workflows/universal_trade_rl_u2_environment.py
git add trade_rl/workflows/universal_trade_rl_u2_environment.py tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "feat: add isolated eight-worker U2 environment factory"
```

---

### Task 7: Prove maintained `DummyVecEnv` seed/reset integration

**Files:** `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`; change U2 adapter production code only if this proves a remaining U2 defect.

- [ ] **Step 1: Add actual maintained vector-path oracle**

```python
vector = _build_training_environment(
    _filtered_environment_factory(u2_factory),
    8,
    subprocesses=False,
)
try:
    member_seed = u2_factory.run_seed
    assert vector.seed(member_seed) == [member_seed + i for i in range(8)]
    vector.reset()
    for index, filtered in enumerate(vector.envs):
        worker = filtered.unwrapped
        assert worker.run_seed == member_seed
        assert worker.environment_index == index
        assert worker.canonical_probe_seed == member_seed + index
        assert worker.environment_digest == u2_factory.environment_generation_digest
finally:
    vector.close()
```

- [ ] **Step 2: Run actual U2 + generic indexed factory regressions**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_vector_runtime.py::test_u2_in_process_vector_accepts_sb3_member_seed_offsets
uv run pytest -q tests/integrations/test_sb3_indexed_environment_factory.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
```

If the first test fails, record the exact boundary and add/fix only the smallest U2-specific adapter contract; do not preemptively change generic vectorization.

- [ ] **Step 3: Commit evidence**

```bash
git add tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "test: prove U2 eight-worker SB3 seed integration"
```

---

### Task 8: Prove timeout metadata and exact terminal observation on actual U2 vectorization

**Files:** `tests/integrations/test_universal_trade_rl_u2_vector_runtime.py`.

- [ ] **Step 1: Build synchronized direct-vs-vector oracle**

Create two equivalent high-level factories with identical source closure/FIT/member seed. From factory A create direct worker 0. From factory B create the actual 8-worker in-process vector. Reset direct with its canonical seed; call `vector.seed(member_seed)` and `vector.reset()`. Step all environments with zero actions in lockstep until direct worker 0 truncates.

At the terminal step require:

```python
assert terminated is False
assert truncated is True
assert vector_dones[0]
assert vector_infos[0]["TimeLimit.truncated"] is True
terminal = vector_infos[0]["terminal_observation"]
assert set(terminal) == set(next_direct)
for key in next_direct:
    np.testing.assert_allclose(terminal[key], next_direct[key], rtol=0.0, atol=0.0)
```

Also require the post-step vector observation at worker 0 is the next reset observation and is not used as `terminal_observation`.

- [ ] **Step 2: Preserve economic reward oracle**

Record direct wealth immediately before/after terminal step. Require `raw_reward == 100 * log(after/before)` and `vector_rewards[0] == raw_reward` to `1e-10` absolute tolerance. There must be no value-function term in environment/vector reward.

- [ ] **Step 3: Verify + commit**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_vector_runtime.py -k "timeout or terminal_observation"
git add tests/integrations/test_universal_trade_rl_u2_vector_runtime.py
git commit -m "test: prove U2 timeout metadata through vectorization"
```

---

### Task 9: Prove exactly-once PPO timeout bootstrap with controlled value

**Files:** create `tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py`; no production change expected.

- [ ] **Step 1: Create one-step timeout environment**

Use a one-dimensional Gym environment returning reward `1.25`, terminal observation `[2.0]`, `terminated=False`, `truncated=True` on every step; retain each raw reward in `raw_rewards`.

- [ ] **Step 2: Run real PPO rollout collection with known terminal value**

Construct `DummyVecEnv([lambda: env])`, then:

```python
model = PPO(
    "MlpPolicy",
    vec,
    n_steps=2,
    batch_size=2,
    gamma=0.9,
    seed=0,
    device="cpu",
)

class _Callback(BaseCallback):
    def _on_step(self) -> bool:
        return True

callback = _Callback()
_, callback = model._setup_learn(
    total_timesteps=2,
    callback=callback,
    reset_num_timesteps=True,
    tb_log_name="u2-timeout",
    progress_bar=False,
)
```

Monkeypatch `model.policy.predict_values` to return `torch.full((observations.shape[0], 1), 2.0, device=observations.device)` for the terminal-value call. Then execute:

```python
assert model.collect_rollouts(
    model.env,
    callback,
    model.rollout_buffer,
    n_rollout_steps=2,
)
```

- [ ] **Step 3: Exact oracle**

```python
assert env.raw_rewards[0] == pytest.approx(1.25)
assert model.rollout_buffer.rewards[0, 0] == pytest.approx(1.25 + 0.9 * 2.0)
assert model.rollout_buffer.rewards[0, 0] != pytest.approx(1.25 + 2 * 0.9 * 2.0)
```

This is a training-target test only. Do not add custom U2 bootstrap production code when maintained SB3 passes.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py
git add tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py
git commit -m "test: prove exactly-once PPO timeout bootstrap"
```

---

### Task 10: Add minimal fixed-PPO per-seed orchestration

**Files:** `trade_rl/workflows/universal_trade_rl_u2_training.py`, `tests/workflows/test_universal_trade_rl_u2_training_runtime.py`.

**Interfaces:** define a `U2TrainingBackend` Protocol with the existing `train(*, seed, config, output_path) -> PolicyTrainingResult` method; define `U2TrainingBackendFactory` as a callable receiving `UniversalTradeRLU2EnvironmentFactory`; export `train_universal_trade_rl_u2_seed(*, plan, environment_factory, output_path, backend_factory=StableBaselines3PPOBackend) -> PolicyTrainingResult`. Do not expose resume arguments in this U2 V1 orchestration task.

- [ ] **Step 1: RED pre-backend lineage**

A fake backend records constructor factory and train arguments. Require seed/config/generation identity on the happy path. Create fake factory variants with wrong `run_seed` and wrong `source_closure_digest`; both must fail with backend call count zero.

- [ ] **Step 2: RED post-backend result identity**

Fake backend result cases must be rejected when `result.environment_digest != environment_factory.environment_generation_digest` or `result.actual_timesteps != plan.final_timesteps`. The valid fake result uses the exact generation digest and `524_288` actual timesteps.

- [ ] **Step 3: Implement fail-closed orchestration**

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
result = backend.train(seed=plan.seed, config=config, output_path=output_path)
if result.environment_digest != environment_factory.environment_generation_digest:
    raise ValueError("U2 training result environment mismatch")
if result.actual_timesteps != plan.final_timesteps:
    raise ValueError("U2 training result timestep mismatch")
return result
```

No seed loop: each preregistered seed receives its own plan/factory/backend call because generation digest includes member seed.

- [ ] **Step 4: Verify + commit**

```bash
uv run pytest -q tests/workflows/test_universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
git add trade_rl/workflows/universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
git commit -m "feat: orchestrate one fixed U2 PPO member"
```

---

### Task 11: Documentation pointer, architecture review, and falsification pass

**Files:** `docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`; review all changed files.

- [ ] **Step 1: Append historical supersession pointer**

Add near the top without rewriting economic preregistration:

```markdown
> **2026-09-04 mechanics amendment:** Task 4-6 source/FIT/routing/vector identity details are superseded by `../specs/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity-amendment.md` and `../../../superpowers/plans/2026-09-04-universal-trade-rl-u2-source-fit-routing-identity.md`. Economic preregistration and Selection thresholds are unchanged.
```

- [ ] **Step 2: Static/architecture checks**

```bash
uv run ruff check trade_rl/workflows/universal_trade_rl_u2_preflight.py trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py trade_rl/workflows/universal_trade_rl_u2_environment.py trade_rl/workflows/universal_trade_rl_u2_training.py tests/workflows/test_universal_trade_rl_u2_preflight.py tests/workflows/test_universal_trade_rl_u2_preflight_falsification.py tests/workflows/test_universal_trade_rl_u2_fit_dataset.py tests/rl/test_universal_trade_u2_environment.py tests/integrations/test_universal_trade_rl_u2_vector_runtime.py tests/integrations/test_universal_trade_rl_u2_timeout_bootstrap.py tests/workflows/test_universal_trade_rl_u2_training_runtime.py
uv run ruff format --check trade_rl tests
uv run mypy trade_rl/workflows/universal_trade_rl_u2_preflight.py trade_rl/workflows/universal_trade_rl_u2_fit_dataset.py trade_rl/workflows/universal_trade_rl_u2_environment.py trade_rl/workflows/universal_trade_rl_u2_training.py
uv run pytest -q tests/test_architecture_contract.py tests/architecture
```

- [ ] **Step 3: Related regression suite**

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

- [ ] **Step 4: Falsification review from spec, not implementation**

Attempt to construct: same U0 metadata/different numeric content; same identity/different locator; FIT off-by-one; caller binding entering production path; execution/descriptor metadata perturbing episode seed; worker index collapse; shared mutable U1 object; worker 1..7 reset rejection; worker-specific drift hidden by probe digest; timeout terminal observation replaced by reset observation; double bootstrap; raw economic reward mutated; seed 1/2 reusing seed-0 generation. Every successful counterexample gets a failing regression test and fix before proceeding.

- [ ] **Step 5: Commit documentation pointer**

```bash
git add docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md
git commit -m "docs: link U2 source routing mechanics amendment"
```

---

### Task 12: Full verification and exact-HEAD readiness evidence

No planned production changes. Any defect returns to the smallest affected task with a new regression test; rerun Task 12 from the beginning after the fix.

- [ ] **Step 1: Inspect final tree**

```bash
git status --short
git diff --check
git diff --stat main...HEAD
git log -1 --oneline
```

No generated/debug/temp workflow/secrets/unrelated refactor may remain.

- [ ] **Step 2: Full suite**

`uv run pytest -q`

Record exact passed/failed/skipped counts.

- [ ] **Step 3: Full static/build/package gates**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run pytest -q tests/test_architecture_contract.py tests/architecture
uv build
```

Also execute the exact maintained training-capability audit command from `.github/workflows/ci.yml`; do not replace it with a narrower smoke test.

- [ ] **Step 4: Changed-line/assertion review**

Confirm tests execute and assert source phase rejection, artifact tampering/wrong content, exact FIT view, binding derivation, episode-seed override, external seed acceptance/rejection, generation digest, all worker indices, state isolation, timeout metadata, one bootstrap, and orchestration pre/post-backend failure branches. Coverage percentage alone is insufficient.

- [ ] **Step 5: Independent review**

Reconstruct requirements from all three U2 specs: base design, robustness/timeout amendment, and 2026-09-04 source/FIT/routing amendment. List unmet Acceptance Criteria, weakened invariants, untested failure modes, and hidden dependencies without inheriting implementer conclusions.

- [ ] **Step 6: Exact-HEAD CI**

After local verification, push and require required CI on the exact final `git rev-parse HEAD`. Confirm CI checkout SHA equals that HEAD; older successful runs do not count.

- [ ] **Step 7: Final report**

Report: changes; design rationale (`MarketDatasetView` vs second artifact); Acceptance Criteria -> evidence mapping; failure modes; exact test/static/build results; falsification/independent review; exact-head CI; unverified items; remaining risk. State explicitly that this proves mechanics/provenance readiness only, not profitability, Development generalization, Admission success, Production readiness, or exact mid-episode resume.
