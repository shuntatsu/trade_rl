# Oracle Bellman Batched Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated per-episode Oracle dynamic-programming work with a path-independent market tape, a rolling-buffer NumPy reference solver, and a batched float64 PyTorch CUDA solver without changing maintained execution and risk semantics.

**Architecture:** Market-dependent inputs are validated and precomputed once in `OracleMarketTape`. Backend-neutral contracts drive NumPy and CUDA implementations through one orchestrator; episode batching is the primary CUDA parallel axis, while deterministic target-state blocking bounds memory. Existing public teacher functions remain compatibility entry points, and reusable artifacts include truthful solver provenance through cache identity v2.

**Tech Stack:** Python 3.12, NumPy 1.26.x, PyTorch 2.4.1 CUDA, pytest, Hypothesis, Ruff, MyPy, Import Linter, PostgreSQL artifact catalog.

## Global Constraints

- Preserve `OracleTeacherConfig` execution, risk, state-grid, signal-delay, train-range, output-shape, and submitted-target semantics.
- Use float64 for every contract-sensitive calculation. Float32 and mixed precision are outside this implementation.
- Keep NumPy as the runtime default. A CUDA-default change requires a separate reviewed configuration change backed by benchmark evidence.
- Require exact equality for validity masks, direction/borrow/tradability interpretation, fill classification, and margin classification.
- Compare factors, weights, accumulated scores, and replayed portfolio values with field-specific float64 tolerances.
- Permit a CUDA path difference only when its final score is within `tie_tolerance=1e-12` log-score units of the NumPy optimum and deterministic executor replay succeeds.
- Resolve ties by choosing the lowest prior-state index among candidates within tolerance of the maximum.
- Never materialize an unconditional full `T × S × S × N` history. Retain rolling forward state and integer backpointers; use deterministic contiguous target-state blocks.
- Keep CPU-only installations functional. Backend-neutral modules must not import PyTorch.
- Permit one deterministic smaller-block retry after CUDA OOM. A second failure follows the configured hard-fail or pre-publication NumPy fallback policy.
- Never fall back or publish after partial artifact registration.
- Keep cache identity v1 entries readable. Do not relabel them as v2 without truthful solver provenance.
- At execution start, use an isolated worktree and re-check current `main`, open PRs, and ancestry. PR #347 presently does not overlap Oracle code, but its documentation-history rule must be rechecked before finalizing documentation placement.

---

## File Structure

**Create:**

- `trade_rl/learning/oracle_bellman_contracts.py` — backend-neutral configuration, inputs, results, provenance, and typed failures.
- `trade_rl/learning/oracle_market_tape.py` — path-independent market preprocessing and validation.
- `trade_rl/learning/oracle_transition_numpy.py` — maintained NumPy one-step transition semantics.
- `trade_rl/learning/oracle_bellman_numpy.py` — rolling-buffer batched NumPy DP and backtracking.
- `trade_rl/learning/oracle_bellman_torch.py` — float64 CUDA transitions, reduction, blocking, OOM retry, and optional compilation.
- `trade_rl/learning/oracle_solver.py` — backend selection, batching, fallback, validation, and aggregation.
- `trade_rl/operations/oracle_teacher_benchmark.py` — reproducible CPU/CUDA benchmark and JSON evidence writer.
- `tests/learning/test_oracle_bellman_contracts.py`
- `tests/learning/test_oracle_market_tape.py`
- `tests/learning/test_oracle_transition_numpy.py`
- `tests/learning/test_oracle_bellman_numpy.py`
- `tests/learning/test_oracle_bellman_torch.py`
- `tests/learning/test_oracle_solver.py`
- `tests/operations/test_oracle_teacher_benchmark.py`
- `docs/operations/oracle-bellman-solver-benchmark.md`

**Modify:**

- `trade_rl/learning/oracle_teacher.py`
- `trade_rl/learning/episode_oracle_teacher.py`
- `trade_rl/learning/episode_teacher_artifact.py`
- `trade_rl/catalog/reusable_artifacts.py`
- `trade_rl/integrations/sb3_training.py`
- `compose.training.yaml`
- `tests/learning/test_oracle_teacher.py`
- `tests/learning/test_episode_oracle_teacher.py`
- `tests/learning/test_episode_teacher_integration.py`
- `tests/catalog/test_reusable_artifacts.py`
- `tests/integrations/test_sb3_training.py`
- `tests/test_training_compose_contract.py`
- `docs/operations/docker-gpu-full-training.md`

---

### Task 1: Freeze Legacy Behavior and Record a CPU Baseline

**Files:**
- Modify: `tests/learning/test_oracle_teacher.py`
- Modify: `tests/learning/test_episode_oracle_teacher.py`
- Create: `trade_rl/operations/oracle_teacher_benchmark.py`
- Create: `tests/operations/test_oracle_teacher_benchmark.py`

**Interfaces:**
- Produces: `OracleBenchmarkCase`, `OracleBenchmarkResult`, `run_oracle_teacher_benchmark(...)`.
- Preserves: current `oracle_target_path(...)` and `episode_oracle_target_path(...)` behavior before refactoring.

- [ ] **Step 1: Add characterization tests using the existing `_market` fixture helper**

Add explicit cases for full fill, partial fill, minimum-notional no-op, blocked buy, blocked sell, unavailable borrow, weight drift, open/close margin boundaries, both signal-delay modes, non-cash initial state, and future-data isolation. For minimum-notional behavior, construct the market directly instead of introducing an undefined helper:

```python
def test_oracle_minimum_notional_noop_remains_valid() -> None:
    from dataclasses import replace

    market = _market(np.array([100.0, 101.0, 101.0]))
    constrained = replace(
        market,
        minimum_notional=np.full_like(market.close, 500.0),
    )
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        reference_portfolio_value=1_000.0,
    )
    _, open_weights, open_equity, _ = _open_state_matrix(
        constrained,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, _, _, effective = _transition_matrices(
        constrained,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=np.array([[0.45]], dtype=np.float64),
    )
    assert valid[0, 0]
    np.testing.assert_array_equal(effective[0, 0], open_weights[0])
```

- [ ] **Step 2: Run the characterization suite before production changes**

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
```

Expected: all tests pass on the legacy solver.

- [ ] **Step 3: Add benchmark data contracts and a real CLI**

Implement:

```python
@dataclass(frozen=True, slots=True)
class OracleBenchmarkCase:
    episode_count: int
    episode_bars: int
    state_count: int
    symbol_count: int
    repetitions: int

@dataclass(frozen=True, slots=True)
class OracleBenchmarkResult:
    backend: str
    cold_seconds: float
    steady_seconds: tuple[float, ...]
    peak_host_bytes: int | None
    peak_device_bytes: int | None
    metadata: dict[str, object]

def run_oracle_teacher_benchmark(
    dataset: MarketDataset,
    contracts: tuple[OracleEpisodeContract, ...],
    teacher_config: OracleTeacherConfig,
    *,
    backend: str,
    repetitions: int,
) -> OracleBenchmarkResult: ...
```

The module's `main()` must accept `--backend`, `--episode-count`, `--episode-bars`, `--repetitions`, and `--output`, create a deterministic synthetic dataset when no external dataset is supplied, and write canonical JSON. Task 1 supports only `legacy_numpy`; later tasks extend the accepted backend values.

- [ ] **Step 4: Test schema, CLI parsing, and deterministic case identity**

```bash
uv run pytest -q tests/operations/test_oracle_teacher_benchmark.py
```

Expected: pass; timing values are checked only for finiteness and non-negativity.

- [ ] **Step 5: Record the baseline outside version control and commit**

```bash
uv run python -m trade_rl.operations.oracle_teacher_benchmark --backend legacy_numpy --episode-count 4 --episode-bars 128 --repetitions 3 --output var/benchmarks/oracle-legacy-small.json
git add trade_rl/operations/oracle_teacher_benchmark.py tests/operations/test_oracle_teacher_benchmark.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
git commit -m "test: characterize Oracle solver and add baseline benchmark"
```

---

### Task 2: Add Backend-Neutral Contracts

**Files:**
- Create: `trade_rl/learning/oracle_bellman_contracts.py`
- Create: `tests/learning/test_oracle_bellman_contracts.py`
- Modify: `trade_rl/learning/oracle_teacher.py`

**Interfaces:**
- Produces: `OracleSolverConfig`, `OracleBellmanParameters`, `OracleEpisodeInputs`, `OracleSolverProvenance`, `OracleSolveResult`, `OracleBackendFailure`.
- `OracleTeacherConfig.bellman_parameters` returns `OracleBellmanParameters` without creating a solver-to-teacher import cycle.

- [ ] **Step 1: Write failing configuration and digest tests**

```python
def test_solver_config_digest_includes_tie_tolerance() -> None:
    first = OracleSolverConfig(selection="numpy", tie_tolerance=1e-12)
    second = OracleSolverConfig(selection="numpy", tie_tolerance=1e-11)
    assert first.digest != second.digest

@pytest.mark.parametrize("selection", ["numpy", "cuda", "cuda_or_numpy"])
def test_solver_config_accepts_maintained_selection(selection: str) -> None:
    assert OracleSolverConfig(selection=selection).selection == selection
```

Also reject non-float64 dtype, non-positive batch/block/chunk values, memory fractions outside `(0, 1]`, unsupported compile modes, and non-positive tie tolerance.

- [ ] **Step 2: Confirm the test fails for the absent module**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_contracts.py
```

Expected: import failure.

- [ ] **Step 3: Implement immutable contracts with exact names**

```python
SolverSelection = Literal["numpy", "cuda", "cuda_or_numpy"]
CompileMode = Literal["disabled", "reduce_overhead"]

@dataclass(frozen=True, slots=True)
class OracleSolverConfig:
    selection: SolverSelection = "numpy"
    numeric_dtype: str = "float64"
    tie_tolerance: float = 1e-12
    episode_batch_size: int = 8
    target_state_block_size: int | None = None
    cuda_memory_fraction: float = 0.65
    compile_mode: CompileMode = "disabled"
    compile_chunk_size: int = 16
    schema_version: str = "oracle_solver_config_v1"
```

`OracleBellmanParameters` copies the immutable execution/risk contracts and the scalar controls required by transition equations. Neutral modules must not import `OracleTeacherConfig` or PyTorch.

- [ ] **Step 4: Run unit, type, and lint checks**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_contracts.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_bellman_contracts.py trade_rl/learning/oracle_teacher.py
uv run ruff check trade_rl/learning/oracle_bellman_contracts.py tests/learning/test_oracle_bellman_contracts.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/learning/oracle_bellman_contracts.py trade_rl/learning/oracle_teacher.py tests/learning/test_oracle_bellman_contracts.py
git commit -m "feat: define Oracle Bellman solver contracts"
```

---

### Task 3: Precompute the Path-Independent Market Tape

**Files:**
- Create: `trade_rl/learning/oracle_market_tape.py`
- Create: `tests/learning/test_oracle_market_tape.py`

**Interfaces:**
- Consumes: `MarketDataset`, `OracleBellmanParameters`, `(start, stop)`.
- Produces: `build_oracle_market_tape(...) -> OracleMarketTape` with read-only contiguous arrays and a digest.

- [ ] **Step 1: Write direct-versus-tape tests**

Compare each tape step with the current direct expressions for raw/equity position factors, mark/open ratios, active/tradable/direction masks, market notional, capacity, minimum notional, unit cost inputs, funding, borrow, dividend, cash rate, and elapsed year fraction.

```python
def test_market_tape_market_notional_matches_dataset() -> None:
    tape = build_oracle_market_tape(market, (1, 7), parameters)
    expected = market.market_notional(2, market.open[2], volume=market.volume[1])
    np.testing.assert_allclose(tape.market_notional[0], expected, rtol=0.0, atol=0.0)
```

- [ ] **Step 2: Confirm the tests fail before implementation**

```bash
uv run pytest -q tests/learning/test_oracle_market_tape.py
```

- [ ] **Step 3: Implement construction and fail-closed validation**

Store `start`, `stop`, `steps`, symbol count, schema `oracle_market_tape_v1`, and digest. Make arrays C-contiguous and read-only. Reject shape drift, negative liquidity/notional, non-finite required values, and timeline misalignment before device transfer.

- [ ] **Step 4: Prove train-range isolation**

Mutate data at index `stop` and later, rebuild, and assert identical tape arrays and digest for the original range.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/learning/test_oracle_market_tape.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_market_tape.py
uv run ruff check trade_rl/learning/oracle_market_tape.py tests/learning/test_oracle_market_tape.py
git add trade_rl/learning/oracle_market_tape.py tests/learning/test_oracle_market_tape.py
git commit -m "feat: precompute Oracle market tape"
```

---

### Task 4: Centralize the NumPy Transition Kernel

**Files:**
- Create: `trade_rl/learning/oracle_transition_numpy.py`
- Create: `tests/learning/test_oracle_transition_numpy.py`
- Modify: `trade_rl/learning/oracle_teacher.py`

**Interfaces:**
- Produces: `NumPyTransitionBatch` and `numpy_transition_step(...)`.
- Input shapes: scores `[B,S]`, prior close weights `[B,S,N]`, targets `[K,N]`.
- Output shapes: valid/factors/classifications `[B,S,K]`, candidate weights/targets `[B,S,K,N]`.

- [ ] **Step 1: Write differential tests against current private helpers**

Call the legacy `_open_state_matrix` and `_transition_matrices`, then call `numpy_transition_step` for an equivalent one-episode tape. Assert exact mask/classification equality and `rtol=1e-10`, `atol=1e-12` for float64 outputs.

- [ ] **Step 2: Confirm the new module is absent**

```bash
uv run pytest -q tests/learning/test_oracle_transition_numpy.py
```

- [ ] **Step 3: Implement the batched one-step kernel**

Use only tape data for market-side inputs. Preserve no-op versus invalid distinction, partial fills, minimum notional, capacity, costs, collateral, maintenance margin, funding, borrowing, dividends, and closing-weight drift. Allocate only tensors required by reduction or contract evidence.

- [ ] **Step 4: Convert old private helpers into compatibility wrappers**

Retain their current callable signatures for tests and downstream imports, but delegate to the shared NumPy equations. After this step, there is one maintained NumPy equation set.

- [ ] **Step 5: Verify executor parity and commit**

```bash
uv run pytest -q tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_transition_numpy.py
uv run ruff check trade_rl/learning/oracle_transition_numpy.py tests/learning/test_oracle_transition_numpy.py
git add trade_rl/learning/oracle_transition_numpy.py trade_rl/learning/oracle_teacher.py tests/learning/test_oracle_transition_numpy.py
git commit -m "refactor: centralize Oracle NumPy transitions"
```

---

### Task 5: Implement Rolling-Buffer Batched NumPy DP

**Files:**
- Create: `trade_rl/learning/oracle_bellman_numpy.py`
- Create: `tests/learning/test_oracle_bellman_numpy.py`

**Interfaces:**
- Produces: `reduce_candidates_numpy(...)` and `solve_numpy_oracle_batch(...) -> OracleSolveResult`.
- Consumes: tape, states, episode inputs, Bellman parameters, and solver configuration.

- [ ] **Step 1: Write deterministic reduction tests**

```python
def test_lowest_prior_index_wins_within_tolerance() -> None:
    scores = np.array([[[1.0], [1.0 + 5e-13], [0.0]]], dtype=np.float64)
    best, pointers = reduce_candidates_numpy(scores, tie_tolerance=1e-12)
    assert pointers.tolist() == [[0]]
    np.testing.assert_allclose(best, [[1.0 + 5e-13]])
```

Also test pointer dtype (`int16` through 32767 states, else `int32`), invalid pointer detection, reverse reconstruction, blocked/unblocked equivalence, and both signal-delay modes.

- [ ] **Step 2: Confirm tests fail**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_numpy.py
```

- [ ] **Step 3: Implement rolling buffers and backpointers**

Keep only previous/next scores and close weights. Retain `pointers [B,T,S]`. Process target states in deterministic contiguous blocks and merge them in global target order.

- [ ] **Step 4: Add deterministic Hypothesis differential tests**

Generate one-to-three-symbol markets across long-only/short-enabled, zero/non-zero costs, liquidity limits, permissions, initial weights, and both signal delays. Separate exact mask/classification assertions from numerical tolerances. For path differences, enforce tie-score and replay equivalence.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_bellman_numpy.py
uv run ruff check trade_rl/learning/oracle_bellman_numpy.py tests/learning/test_oracle_bellman_numpy.py
git add trade_rl/learning/oracle_bellman_numpy.py tests/learning/test_oracle_bellman_numpy.py
git commit -m "feat: add rolling batched NumPy Bellman solver"
```

---

### Task 6: Route Public Teacher APIs Through One Orchestrator

**Files:**
- Create: `trade_rl/learning/oracle_solver.py`
- Create: `tests/learning/test_oracle_solver.py`
- Modify: `trade_rl/learning/oracle_teacher.py`
- Modify: `trade_rl/learning/episode_oracle_teacher.py`
- Modify: `tests/learning/test_episode_teacher_integration.py`

**Interfaces:**
- Produces: `solve_oracle_episodes(...) -> OracleSolveResult`.
- Preserves: public dtypes, read-only outputs, episode ordering, contract digests, and NumPy default behavior.

- [ ] **Step 1: Write integration tests with explicit NumPy selection**

Require full-range and episode outputs to match captured legacy behavior. Require batch contract ordering and artifact digest determinism.

- [ ] **Step 2: Implement orchestration**

Validate contracts, group equal-horizon episodes, build/reuse the relevant tape, batch according to `episode_batch_size`, call the backend, and split results back into contract order.

- [ ] **Step 3: Remove duplicate public generation routes**

Keep `max_workers` as a compatibility parameter for NumPy callers during migration. Route all numerical work through the new solver; do not retain a second implementation hidden behind multiprocessing.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest -q tests/learning/test_oracle_solver.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_episode_teacher_integration.py
git add trade_rl/learning/oracle_solver.py trade_rl/learning/oracle_teacher.py trade_rl/learning/episode_oracle_teacher.py tests/learning/test_oracle_solver.py tests/learning/test_episode_teacher_integration.py
git commit -m "refactor: route Oracle teachers through batched solver"
```

---

### Task 7: Add Cache Identity v2 and Solver Provenance

**Files:**
- Modify: `trade_rl/learning/episode_teacher_artifact.py`
- Modify: `trade_rl/catalog/reusable_artifacts.py`
- Modify: `tests/catalog/test_reusable_artifacts.py`
- Modify: `tests/learning/test_episode_teacher_integration.py`

**Interfaces:**
- Produces: `teacher_cache_identity_v2(...)` and serialized `OracleSolverProvenance`.
- Preserves: v1 loading and backfill without fabricated fields.

- [ ] **Step 1: Write backend-separation and legacy-backfill tests**

```python
def test_cache_identity_v2_separates_actual_backends() -> None:
    numpy_key = teacher_cache_identity_v2(base_identity, solver_backend="numpy")
    cuda_key = teacher_cache_identity_v2(base_identity, solver_backend="torch_cuda")
    assert numpy_key != cuda_key
```

Add a v1 backfill test asserting the retained schema is `teacher_cache_identity_v1` and no CUDA claim appears.

- [ ] **Step 2: Implement v2 fields**

Include solver contract, actual backend, float64 dtype, tie contract/tolerance, tape schema/digest, episode batch size, target block size, compile mode/chunk, fallback reason, OOM retry, solver wall time, peak memory, and CUDA/PyTorch/device details when applicable.

- [ ] **Step 3: Verify content and catalog behavior**

Backend differences change cache identity. Equal target payloads may retain equal payload digests without causing catalog conflict because their cache keys differ.

- [ ] **Step 4: Test and commit**

```bash
uv run pytest -q tests/catalog/test_reusable_artifacts.py tests/learning/test_episode_teacher_integration.py
git add trade_rl/learning/episode_teacher_artifact.py trade_rl/catalog/reusable_artifacts.py tests/catalog/test_reusable_artifacts.py tests/learning/test_episode_teacher_integration.py
git commit -m "feat: version Oracle solver provenance and cache identity"
```

---

### Task 8: Implement the Float64 PyTorch CUDA Backend

**Files:**
- Create: `trade_rl/learning/oracle_bellman_torch.py`
- Create: `tests/learning/test_oracle_bellman_torch.py`
- Modify: `trade_rl/learning/oracle_solver.py`

**Interfaces:**
- Produces: `solve_torch_cuda_oracle_batch(...) -> OracleSolveResult` with actual-backend provenance.
- Uses `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`; no custom pytest option is introduced.

- [ ] **Step 1: Write CUDA-conditional parity tests**

Compare tape transfer, one-step exact masks/classifications, tie reduction, blocked/unblocked results, backtracking, and executor replay with NumPy.

- [ ] **Step 2: Implement eager CUDA execution**

Transfer the tape once per episode batch. Keep forward state on-device. Use `torch.float64`, `torch.bool`, and matching pointer integers. Do not perform per-step host transfers.

- [ ] **Step 3: Implement bounded target-state blocks**

Estimate candidate memory before execution. Respect explicit block size or derive one from `cuda_memory_fraction` and free memory. Merge blocks with the same deterministic tie rule.

- [ ] **Step 4: Implement one OOM retry**

On OOM, release backend-owned candidate references, synchronize, clear allocator cache for the retry path, halve block size deterministically, and retry once. Record both sizes. Raise `OracleBackendFailure(kind="cuda_oom")` after the second failure.

- [ ] **Step 5: Add optional fixed-chunk compilation**

Keep eager mode authoritative. Permit `reduce_overhead` only with chunk sizes `{8,16,32,64}`. Report compile warm-up separately and retain eager fallback inside the CUDA backend when compilation itself is unsupported before solving begins.

- [ ] **Step 6: Run CPU-safe and maintained-GPU tests**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_torch.py
uv run pytest -q tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py
```

On CPU, CUDA cases skip. On the RTX 4070 Ti SUPER host, all CUDA cases execute and pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/learning/oracle_bellman_torch.py trade_rl/learning/oracle_solver.py tests/learning/test_oracle_bellman_torch.py
git commit -m "feat: add batched float64 CUDA Bellman solver"
```

---

### Task 9: Expose Explicit Runtime Selection and Safe Fallback

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `compose.training.yaml`
- Modify: `tests/integrations/test_sb3_training.py`
- Modify: `tests/test_training_compose_contract.py`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Parses and supplies `OracleSolverConfig`.
- Supports `numpy`, `cuda`, and `cuda_or_numpy`; default is `numpy`.

- [ ] **Step 1: Write environment parsing tests**

Cover:

```text
TRADE_RL_ORACLE_SOLVER
TRADE_RL_ORACLE_EPISODE_BATCH_SIZE
TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE
TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION
TRADE_RL_ORACLE_COMPILE_MODE
TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE
```

Reject invalid values before generation. Missing values preserve NumPy defaults.

- [ ] **Step 2: Implement hard-fail and fallback contracts**

`cuda` fails on unavailable CUDA or backend failure. `cuda_or_numpy` may fall back only before artifact promotion and records the typed reason. Failed CUDA output never receives a ready catalog registration.

- [ ] **Step 3: Enforce one CUDA owner**

For `cuda` and `cuda_or_numpy`, require `TRADE_RL_TEACHER_WORKERS=1`. Episode concurrency is the CUDA batch dimension, not multiple device-owning processes. Fail before allocation when the values conflict.

- [ ] **Step 4: Update compose and operational documentation**

Set checked defaults to NumPy and one teacher worker. Document backend selection, fallback, batching, memory fraction, blocking, compilation, and provenance.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py tests/learning/test_episode_teacher_integration.py
git add trade_rl/integrations/sb3_training.py compose.training.yaml tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py docs/operations/docker-gpu-full-training.md
git commit -m "feat: configure Oracle solver backend and fallback"
```

---

### Task 10: Produce Differential, Replay, and Performance Evidence

**Files:**
- Modify: `trade_rl/operations/oracle_teacher_benchmark.py`
- Modify: `tests/operations/test_oracle_teacher_benchmark.py`
- Modify: `tests/learning/test_oracle_bellman_torch.py`
- Modify: `tests/learning/test_oracle_solver.py`
- Create: `docs/operations/oracle-bellman-solver-benchmark.md`

**Interfaces:**
- Produces reproducible JSON evidence and a human-readable report.
- Does not alter the default backend.

- [ ] **Step 1: Add synchronized CUDA timing and peak-memory reporting**

Synchronize immediately before and after each measured CUDA region. Separate transfer, compile/warm-up, steady solve, backtracking, replay, and artifact I/O. Record allocated and reserved peaks.

- [ ] **Step 2: Encode the maintained benchmark matrix**

Use backends `legacy_numpy`, maintained multi-worker CPU, new NumPy, eager CUDA, and compiled CUDA; episode batches `1,4,8,16,32`; compile chunks `8,16,32,64`; horizons `128` and `2880`; current state count plus one larger bounded synthetic case; one cold and five steady repetitions.

- [ ] **Step 3: Run the complete correctness corpus first**

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py tests/learning/test_episode_teacher_integration.py
```

On the maintained GPU host, CUDA tests execute rather than skip.

- [ ] **Step 4: Run the RTX 4070 Ti SUPER benchmark**

```bash
uv run python -m trade_rl.operations.oracle_teacher_benchmark --backend all --episode-count 32 --episode-bars 2880 --repetitions 5 --output var/benchmarks/oracle-bellman-4070ti-super.json
```

Do not claim a speedup unless synchronized steady-state results complete.

- [ ] **Step 5: Write the report**

Include medians, ranges, cold-start cost, peak host/device memory, exact workload identity, and comparison with maintained multi-worker CPU. Retain NumPy default or recommend a separate CUDA-default PR.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/operations/oracle_teacher_benchmark.py tests/operations/test_oracle_teacher_benchmark.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py docs/operations/oracle-bellman-solver-benchmark.md
git commit -m "perf: validate Oracle Bellman CPU and CUDA backends"
```

---

### Task 11: Self-Review and Final Verification

**Files:**
- Modify only files identified by the review, staged interactively with `git add -p`.

**Interfaces:**
- Produces one verified final head suitable for a draft PR.

- [ ] **Step 1: Review the entire diff as a reviewer**

Check responsibility boundaries, circular imports, duplicate equations, hidden host/device transfers, unbounded temporaries, threshold direction, partial publication, cache truthfulness, dead compatibility paths, and unrelated changes.

- [ ] **Step 2: Run focused suites after review fixes**

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_episode_teacher_integration.py tests/learning/test_oracle_bellman_contracts.py tests/learning/test_oracle_market_tape.py tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py tests/catalog/test_reusable_artifacts.py tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py tests/operations/test_oracle_teacher_benchmark.py
```

- [ ] **Step 3: Run static and architecture checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports
```

- [ ] **Step 4: Run the complete test and coverage suite on the same head**

```bash
uv run pytest --cov=trade_rl --cov-report=term-missing
```

Require all tests to pass, branch coverage at least 80%, and all critical ratchets to pass.

- [ ] **Step 5: Run changed-surface compatibility checks**

Run the repository's maintained CPU training smoke, CUDA training-image smoke, PostgreSQL catalog workflow, Windows compatibility workflow, and Ubuntu compatibility workflow on the final head. Record exact final-head results in the PR body.

- [ ] **Step 6: Verify Git and secret hygiene**

```bash
git status --short
git diff --check
git diff main...HEAD --stat
git log --oneline main..HEAD
```

Confirm no benchmark payloads, local datasets, CUDA dumps, generated artifacts, credentials, or temporary workflows are tracked.

- [ ] **Step 7: Commit review fixes only when present**

```bash
git add -p
git diff --cached --check
git commit -m "fix: address Oracle Bellman solver self-review"
```

When `git diff --cached --quiet` reports no staged changes, skip the commit.

- [ ] **Step 8: Open a draft PR without merging**

Include What, Why, architecture, numerical contract, cache migration, fallback behavior, benchmark results, tests, risks, non-goals, and the reason NumPy remains the default. Keep Draft status until final-head CI and maintained-hardware evidence succeed.
