# Train-Only Causal Scenario Library C2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a frozen, deterministic, train-only and query-past-only conditioned historical block library that emits replayable market scenarios compatible with the C1 causal scenario evaluator.

**Architecture:** C2 is split into four focused modules. `causal_scenario_conditions.py` owns causal condition layout and train-only robust normalization. `causal_scenario_library.py` stores a compact immutable anchor/condition index and materializes only the selected 64 relative historical blocks during past-only deterministic kNN selection, avoiding duplication of every overlapping 96-step future window. `causal_scenario_replay.py` anchors a selected relative block to a query state and validates it against a freshly reconstructed source block before constructing a `MarketDataset`. `causal_scenario_library_artifact.py` persists only the compact frozen index with exact file closure and digest-bound arrays. C2 is a research-only workflow adapter above the data and strategy layers; it is not imported by maintained training, Serving, promotion, release, or production paths.

**Tech Stack:** Python 3.12, NumPy, immutable dataclasses, `MarketDataset`, `MarketDatasetView`, `TrendStrategy`, canonical content digests, deterministic NPZ/JSON artifacts, Pytest, Ruff, MyPy, import-linter.

## Global Constraints

- Use only rows inside the supplied `MarketDatasetView` when building the train library.
- Every source block contains exactly 96 future decisions by default and remains wholly inside the train view.
- Query selection may use only anchors whose materialized block final source bar is no later than the query cutoff.
- A query inside the train interval additionally requires `source_stop <= query_index - horizon_decisions`.
- The version-one condition vector uses Trend fast/base/slow targets, 24-hour realized volatility, 7-day pairwise correlations, spread, log market notional, funding rate/due, and tradable/buy/sell/borrow/active masks.
- Continuous condition components use train-anchor median/MAD normalization with an epsilon floor; binary masks remain unscaled.
- Version one selects exactly 64 nearest scenarios by squared Euclidean distance and anchor-index tie-break, with uniform probability.
- Fewer than 64 eligible past anchors is a hard error; there is no unconditional fallback.
- The frozen library and its artifact store anchor indices and condition matrices only; relative future blocks are materialized on demand for the selected 64 anchors.
- Replayed price, volume, market-notional, timestamps, execution constraints, lifecycle flags, and information delays must be deterministic and validated through `MarketDataset`.
- C2 must not read checkpoint, selection, test, outer, fresh-confirmation, or query-future rows.
- C2 must not be imported by maintained training, Serving, promotion, release, or production modules.
- Production status remains `NO-GO`.

---

## File Structure

- Create `trade_rl/workflows/causal_scenario/conditions.py`: condition layout, raw causal feature calculation, robust train normalizer.
- Create `trade_rl/workflows/causal_scenario/library.py`: compact anchor-index library, relative block/selection contracts, train-only builder, past-only kNN selector, and selected-block materialization.
- Create `trade_rl/workflows/causal_scenario/replay.py`: query-anchored deterministic `MarketDataset` reconstruction.
- Create `trade_rl/workflows/causal_scenario/library_artifact.py`: deterministic library artifact writer/loader.
- Modify `trade_rl/evaluation/__init__.py`: publish only stable C2 APIs.
- Create `tests/evaluation/test_causal_scenario_conditions.py`.
- Create `tests/evaluation/test_causal_scenario_library.py`.
- Create `tests/evaluation/test_causal_scenario_replay.py`.
- Create `tests/evaluation/test_causal_scenario_library_artifact.py`.
- Create `tests/evaluation/conftest.py`: pytest-only deterministic `MarketDataset` factory fixture; tests consume it by fixture injection and never import `tests.*` modules.
- Create `tests/architecture/test_causal_scenario_library_boundary.py`.
- Create `docs/verification/2026-07-25-causal-scenario-library-c2.md` after exact-head verification.

### Task 1: Add causal condition layout and train-only robust normalization

**Files:**
- Create: `trade_rl/workflows/causal_scenario/conditions.py`
- Create: `tests/evaluation/conftest.py`
- Test: `tests/evaluation/test_causal_scenario_conditions.py`

**Interfaces:**
- Produces `CausalConditionConfig`, `CausalConditionLayout`, `TrainRobustConditionNormalizer`, `build_causal_condition_layout`, `compute_raw_causal_condition`, and `fit_train_condition_normalizer`.
- Consumes `MarketDataset`, `MarketDatasetView`, `TrendConfig`, and `TrendStrategy`.

- [x] **Step 1: Write RED tests for the exact feature order and immutable arrays**

Use the `market_dataset_factory` pytest fixture from `tests/evaluation/conftest.py` to construct a 15-minute, three-symbol `MarketDataset` with deterministic prices and execution arrays. The fixture returns a callable and is never imported from another test module. Assert the layout order is exactly:

```text
trend_fast:<symbol>
trend_base:<symbol>
trend_slow:<symbol>
realized_vol_24h:<symbol>
corr_7d:<left>|<right>
spread_rate:<symbol>
log_market_notional:<symbol>
funding_rate:<symbol>
funding_due:<symbol>
tradable:<symbol>
buy_allowed:<symbol>
sell_allowed:<symbol>
borrow_available:<symbol>
asset_active:<symbol>
```

Assert the correlation names use symbol order and the final six groups are marked binary only for `funding_due` and the five masks. Assert all returned arrays are read-only.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_conditions.py -q
```

Expected: import failure because the module does not exist.

- [x] **Step 3: Implement immutable configuration and layout contracts**

Implement:

```python
@dataclass(frozen=True, slots=True)
class CausalConditionConfig:
    volatility_hours: float = 24.0
    correlation_hours: float = 168.0
    scale_epsilon: float = 1e-9
    liquidity_floor: float = 1e-12
    schema_version: str = "causal_scenario_condition_v1"

@dataclass(frozen=True, slots=True)
class CausalConditionLayout:
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    continuous_mask: np.ndarray
    schema_version: str = "causal_scenario_condition_layout_v1"
```

Validate finite positive hours/floors, unique names, exact mask shape, and digest every field. Normalize signed zero and copy arrays to C-contiguous read-only `float64`/`bool` arrays.

- [x] **Step 4: Implement the raw condition calculation**

`compute_raw_causal_condition(dataset, index, trend_strategy, config)` must:

1. reject indices outside the dataset or without 7-day/Trend history;
2. obtain `TrendTargets.fast/base/slow` at the closed row;
3. calculate close-to-close log returns over the exact 24-hour and 168-hour timestamp lookbacks;
4. calculate 24-hour realized volatility as `sqrt(sum(return**2))` per symbol;
5. calculate pairwise 7-day correlations, returning `0.0` when either series has standard deviation at or below `scale_epsilon`;
6. append spread, `log(max(market_notional_at_close, liquidity_floor))`, funding rate/due, and masks;
7. reject non-finite results and return a read-only vector matching the deterministic layout.

The function must never read row `index + 1` or later.

- [x] **Step 5: Write RED prefix-causality and normalization tests**

Add tests that mutate every row after the query index, including prices, funding, liquidity, and masks, and prove the raw vector is bit-identical. Add a train-view test where post-train rows are changed and prove fitted medians/scales/digest are unchanged.

- [x] **Step 6: Implement the train-only normalizer**

Implement:

```python
@dataclass(frozen=True, slots=True)
class TrainRobustConditionNormalizer:
    feature_names: tuple[str, ...]
    continuous_mask: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    train_view_digest: str
    schema_version: str = "train_robust_condition_normalizer_v1"

    def transform(self, raw: np.ndarray) -> np.ndarray: ...
```

For continuous dimensions, use `median(raw_anchor_values)` and `maximum(median(abs(x - median)), scale_epsilon)`. Binary dimensions must have median `0` and scale `1` in the stored contract and pass through unchanged. Reject binary inputs outside `{0, 1}`. `fit_train_condition_normalizer` accepts the already eligible raw anchor matrix plus train-view identity so it cannot inspect other ranges.

- [x] **Step 7: Run Task 1 tests GREEN and commit**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_conditions.py -q
ruff check trade_rl/workflows/causal_scenario/conditions.py tests/evaluation/test_causal_scenario_conditions.py
mypy trade_rl/workflows/causal_scenario/conditions.py
```

Commit message: `feat: add train-only causal condition normalization`.

### Task 2: Build a compact immutable train-only anchor library

**Files:**
- Create: `trade_rl/workflows/causal_scenario/library.py`
- Test: `tests/evaluation/test_causal_scenario_library.py`

**Interfaces:**
- Consumes Task 1 condition APIs and `MarketDatasetView`.
- Produces `CausalScenarioLibraryConfig`, `RelativeScenarioBlock`, `FrozenCausalScenarioLibrary`, and `build_causal_scenario_library`.

- [x] **Step 1: Write RED library-boundary and compactness tests**

Use a train view with at least `7 days + 96 decisions + 64 anchors`. Assert every stored anchor can yield a source range satisfying:

```python
source_start == anchor_index + 1
source_stop == source_start + horizon_decisions
train_view.start <= anchor_index
source_stop <= train_view.stop
```

Mutate rows outside the train view and prove library digest, anchor indices, and condition matrices are unchanged. Assert the frozen library has no stored `blocks` collection and that a representative artifact remains below the declared compactness threshold.

- [x] **Step 2: Define exact immutable contracts**

`RelativeScenarioBlock` keeps the complete selected future path contract, but `FrozenCausalScenarioLibrary` stores only:

```python
@dataclass(frozen=True, slots=True)
class FrozenCausalScenarioLibrary:
    dataset_id: str
    train_view_digest: str
    train_start: int
    train_stop: int
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    config: CausalScenarioLibraryConfig
    trend_config_payload: Mapping[str, object]
    layout: CausalConditionLayout
    normalizer: TrainRobustConditionNormalizer
    anchor_indices: np.ndarray
    raw_conditions: np.ndarray
    normalized_conditions: np.ndarray
    library_digest: str
```

Validate at least 64 strictly increasing unique anchors, complete future ranges inside train, exact condition shapes, train-view-bound normalizer identity, and exact normalized-condition reproduction. All arrays are copied, finite, C-contiguous, and read-only.

- [x] **Step 3: Implement deterministic selected-block extraction**

`_extract_block` materializes future rows `anchor + 1 : anchor + 1 + horizon` only after an anchor is selected. Prices use anchor close/mark/index denominators; volume and market notional use source close values; elapsed timestamps and all execution/information arrays are preserved. No overlapping future arrays are duplicated in the frozen library.

- [x] **Step 4: Implement train-only builder ordering and digest**

The builder derives the first anchor relative to `train_view.start`, so a non-zero train start cannot read pre-train history. It computes all raw conditions with `history_start=train_view.start`, fits the normalizer only on eligible train anchors, stores compact anchor/condition arrays, requires at least 64 complete anchors, and digests dataset/view/config/trend/layout/normalizer plus those arrays.

- [x] **Step 5: Add malformed-data, leak, scalability, and signed-zero tests**

Reject invalid ranges, duplicate/non-ascending/out-of-train anchors, condition shape mismatches, inconsistent normalization, non-finite values, mutable arrays, and digest mismatches. Prove non-zero train-view prefix causality, close-price notional semantics, deterministic compact artifacts, and positive-zero normalization.

- [x] **Step 6: Run Task 2 tests GREEN and commit**

Run focused tests with statement and branch coverage at 100% for the new module. Commit message: `feat: build compact train-only scenario library`.

### Task 3: Add query-past-only deterministic conditioned selection

**Files:**
- Modify: `trade_rl/workflows/causal_scenario/library.py`
- Test: `tests/evaluation/test_causal_scenario_library.py`

**Interfaces:**
- Produces `CausalScenarioSelection` and `select_causal_scenarios`.
- Returns the C1 `CausalScenarioSet` plus selected blocks in matching order.

- [x] **Step 1: Write RED kNN, tie-break, and embargo tests**

Construct a frozen compact library with controlled normalized conditions. Assert squared Euclidean distance, ascending anchor-index tie-break, exactly 64 selections, uniform probabilities, and on-demand block materialization in matching order.

For a query inside train, assert blocks are eligible only when:

```python
block.source_stop <= query_index - config.horizon_decisions
```

For a query after train, all train-contained blocks that ended before the query are eligible. Fewer than 64 eligible blocks must raise.

- [x] **Step 2: Implement immutable selection contract**

```python
@dataclass(frozen=True, slots=True)
class CausalScenarioSelection:
    library_digest: str
    query_index: int
    query_timestamp_ns: int
    raw_query_condition: np.ndarray
    normalized_query_condition: np.ndarray
    scenario_set: CausalScenarioSet
    blocks: tuple[RelativeScenarioBlock, ...]
    selection_digest: str
```

Validate scenario IDs, anchors, distances, normalized anchor conditions, probabilities, and block ordering against the embedded `CausalScenarioSet`.

- [x] **Step 3: Implement prefix-scoped public selection**

`select_causal_scenarios(library, query_view, query_index, trend_strategy)` must require:

- matching dataset ID and symbols;
- `query_view.stop == query_index + 1` so the supplied view is a causal prefix;
- query index after required history and not before train start;
- condition layout/trend/config identity match.

Compute the raw and normalized query condition, filter eligible anchor positions, sort `(distance, anchor_index)`, choose exactly `scenario_count`, materialize only those selected blocks, build scenario IDs from library/block/query identities, and return `CausalScenarioSelection`.

- [x] **Step 4: Add prefix-mutation and insufficient-history tests**

Change every row at and after `query_index + 1` and prove selection digest is unchanged. Reject a query view that extends beyond the query or cannot provide required history.

- [x] **Step 5: Run Task 3 tests GREEN and commit**

Commit message: `feat: select past-only conditioned scenarios`.

### Task 4: Reconstruct query-anchored replay datasets

**Files:**
- Create: `trade_rl/workflows/causal_scenario/replay.py`
- Test: `tests/evaluation/test_causal_scenario_replay.py`

**Interfaces:**
- Consumes `CausalScenarioSelection`, one selected rank, causal query prefix, and query index.
- Produces `materialize_causal_scenario_dataset(...) -> MarketDataset` and `CausalScenarioReplayIdentity`.

- [x] **Step 1: Write RED controlled-path reconstruction tests**

For a source block with known gap/OHLC/volume paths and a different query price scale, assert:

- row 0 equals the query row;
- rows 1..96 equal query anchors multiplied by stored relatives;
- high/low contain open/close after transformation;
- volume and reconstructed market-notional ratios match stored relatives;
- source information delays are shifted to synthetic timestamps;
- fees, spread, funding, borrow, masks, corporate actions, and lifecycle ordering match the source block;
- repeated materialization has the same dataset ID and arrays.

- [x] **Step 2: Implement replay identity and deterministic reconstruction**

Construct a `horizon + 1` row `MarketDataset`:

1. row 0 copies the exact query row and query metadata;
2. synthetic future timestamps are query timestamp plus stored source elapsed nanoseconds;
3. future price and dividend fields use stored relatives and query anchors;
4. future volume uses query volume and stored relative volume;
5. future source arrays copy from the selected block;
6. future `available_at` equals synthetic event time plus stored source availability delay;
7. static symbols, feature names, volume units, multipliers, calendar, and configuration digests come from the query dataset;
8. call `MarketDataset(...).with_content_identity(provenance)` and verify `market_notional_relative` within `1e-10` absolute tolerance.

Never repair invalid OHLC, negative liquidity, invalid masks, or non-monotonic timestamps; `MarketDataset` validation must fail closed.

- [x] **Step 3: Add invalid replay and no-alias tests**

Reject out-of-range selected rank, library/query identity mismatch, zero/non-finite query anchors, incompatible symbol/feature schemas, inconsistent market-notional ratios, and source arrays with invalid dtype/shape. Prove returned arrays do not alias the query dataset or library block arrays.

- [x] **Step 4: Run Task 4 tests GREEN and commit**

Commit message: `feat: materialize query-anchored scenario replays`.

### Task 5: Add deterministic frozen-library artifact persistence

**Files:**
- Create: `trade_rl/workflows/causal_scenario/library_artifact.py`
- Test: `tests/evaluation/test_causal_scenario_library_artifact.py`

**Interfaces:**
- Produces `write_causal_scenario_library_artifact` and `load_causal_scenario_library_artifact`.

- [x] **Step 1: Write RED deterministic round-trip and tamper tests**

Assert byte-identical compact artifacts across repeated writes, complete object equality after load, read-only arrays, a representative artifact below 1 MiB, and rejection of extra/missing/symlink files, manifest field changes, NPZ byte changes, dtype/shape changes, reordered anchors, config/trend/layout/normalizer mismatch, and library digest mismatch.

- [x] **Step 2: Implement deterministic artifact format**

Use exact files:

```text
manifest.json
arrays.npz
```

Store scalar/string/config/layout metadata in canonical JSON. Store only anchor indices, raw/normalized condition matrices, the layout mask, and normalizer median/scale in deterministic ZIP_STORED NPZ with sorted names, fixed timestamp, Unix mode, and `allow_pickle=False`.

The manifest binds source dataset ID, train view/range, symbols/features, config/trend/layout/normalizer metadata, anchor count, array dtypes/shapes, and library digest. Writer uses arrays-first/manifest-second atomic replacement. Loader enforces exact field/file/array closure and reconstructs the compact immutable index before accepting the library digest.

- [x] **Step 3: Run Task 5 tests GREEN and commit**

Commit message: `feat: persist frozen causal scenario libraries`.

### Task 6: Publish API, enforce architecture boundaries, and verify C2

**Files:**
- Modify: `trade_rl/evaluation/__init__.py`
- Create: `tests/architecture/test_causal_scenario_library_boundary.py`
- Create: `docs/verification/2026-07-25-causal-scenario-library-c2.md`
- Remove: `.github/workflows/temporary-materialize-c1.yml`
- Remove: `.github/workflows/temporary-materialize-c2.yml`

- [x] **Step 1: Add public-import and prohibited-dependency tests**

Assert all stable C2 symbols import from the dedicated `trade_rl.workflows.causal_scenario` subpackage without entering the top-level workflow facade. Scan maintained training, RL, integrations, Serving, promotion, release, and workflow runtime modules and reject imports containing `causal_scenario_conditions`, `causal_scenario_library`, `causal_scenario_replay`, or `causal_scenario_library_artifact`.

- [x] **Step 2: Run focused C2 tests with strict branch coverage**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_conditions.py \
       tests/evaluation/test_causal_scenario_library.py \
       tests/evaluation/test_causal_scenario_replay.py \
       tests/evaluation/test_causal_scenario_library_artifact.py \
       tests/architecture/test_causal_scenario_library_boundary.py \
       --cov=trade_rl.workflows.causal_scenario.conditions \
       --cov=trade_rl.workflows.causal_scenario.library \
       --cov=trade_rl.workflows.causal_scenario.replay \
       --cov=trade_rl.workflows.causal_scenario.library_artifact \
       --cov-branch --cov-report=term-missing --cov-fail-under=100 -q
```

- [ ] **Step 3: Run complete static and repository gates**

Run:

```bash
ruff check .
ruff format --check --diff .
mypy .
lint-imports
python -m compileall -q trade_rl tests
pytest -q
```

- [ ] **Step 4: Run standard packaged and cross-platform CI**

Require exact-head success for Rebuilt Core, Ubuntu/Windows compatibility, PostgreSQL Catalog, training image, workflow security, Serving smoke, critical branch coverage, and CLI smoke.

- [ ] **Step 5: Review leakage and replay invariants**

Verify from the final diff that:

- builder accepts only `MarketDatasetView` and never an unrestricted range argument;
- no source block crosses train stop;
- query selection is prefix-scoped and applies the train-query embargo;
- mutation at or after the query cutoff cannot alter selection;
- replay uses selected historical blocks only and never realized query future;
- no C2 import enters maintained runtime paths;
- no temporary workflow or transfer file remains.

- [ ] **Step 6: Record verification, update PR body, mark ready, and squash merge**

The verification document records exact head SHA, workflow run IDs, focused test count, branch coverage, and production `NO-GO`. Merge only with the verified exact head.
