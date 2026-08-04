# Oracle Bellman Batched Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated per-episode Oracle dynamic-programming work with a path-independent market tape, a rolling-buffer NumPy reference solver, and a batched float64 PyTorch CUDA solver without changing the maintained execution and risk semantics.

**Architecture:** Market-dependent inputs are validated and precomputed once in `OracleMarketTape`. Backend-neutral contracts drive NumPy and CUDA implementations through one orchestrator; episode batching is the primary CUDA parallel axis, while target-state blocking bounds memory. Existing public teacher functions remain compatibility entry points, and reusable artifacts include truthful solver provenance through cache identity v2.

**Tech Stack:** Python 3.12, NumPy 1.26.x, PyTorch 2.4.1 CUDA, pytest, Hypothesis, Ruff, MyPy, Import Linter, PostgreSQL artifact catalog.

## Global Constraints

- Preserve `OracleTeacherConfig` execution, risk, state-grid, signal-delay, and train-range semantics.
- Use float64 for all contract-sensitive numerical calculations; do not introduce float32 or mixed precision.
- Keep the initial runtime default on NumPy. CUDA becomes the default only through a later reviewed configuration change supported by benchmark evidence.
- Exact equality remains mandatory for validity masks and transition classifications; numerical values use field-specific tolerances.
- A CUDA path may differ only when its final score is within `tie_tolerance=1e-12` log-score units of the NumPy optimum and executor replay succeeds.
- Tie resolution selects the lowest prior-state index among candidates within tolerance of the maximum.
- Do not materialize an unconditional full `T × S × S × N` tensor. Use rolling state buffers and deterministic target-state blocks.
- CPU-only installations remain supported; neutral modules must not import PyTorch.
- CUDA OOM may trigger one deterministic smaller-block retry. A second failure follows the configured hard-fail or pre-publication NumPy fallback policy.
- No fallback or artifact publication is allowed after partial catalog registration.
- Legacy cache identity v1 entries remain readable and must not be rewritten as v2 without truthful solver provenance.
- At execution start, create an isolated worktree and re-check `main`, open PRs, and branch ancestry. PR #347 currently does not overlap Oracle code, but its documentation-history policy must be rechecked before retaining `docs/superpowers` files.

---

## File Structure

**Create:**

- `trade_rl/learning/oracle_bellman_contracts.py` — backend-neutral solver configuration, inputs, results, provenance, and typed failures.
- `trade_rl/learning/oracle_market_tape.py` — path-independent market preprocessing and validation.
- `trade_rl/learning/oracle_transition_numpy.py` — maintained NumPy one-step transition semantics.
- `trade_rl/learning/oracle_bellman_numpy.py` — rolling-buffer batched NumPy dynamic programming and backtracking.
- `trade_rl/learning/oracle_bellman_torch.py` — float64 CUDA transition, reduction, state blocking, and optional compilation.
- `trade_rl/learning/oracle_solver.py` — backend selection, batching, fallback, validation, and result aggregation.
- `trade_rl/operations/oracle_teacher_benchmark.py` — reproducible CPU/CUDA benchmark and JSON evidence writer.
- `tests/learning/test_oracle_bellman_contracts.py`
- `tests/learning/test_oracle_market_tape.py`
- `tests/learning/test_oracle_transition_numpy.py`
- `tests/learning/test_oracle_bellman_numpy.py`
- `tests/learning/test_oracle_bellman_torch.py`
- `tests/learning/test_oracle_solver.py`
- `tests/operations/test_oracle_teacher_benchmark.py`

**Modify:**

- `trade_rl/learning/oracle_teacher.py` — retain public configuration and entry point; delegate transition/DP work.
- `trade_rl/learning/episode_oracle_teacher.py` — replace process-oriented CUDA generation with backend-neutral episode batching.
- `trade_rl/learning/episode_teacher_artifact.py` — record solver provenance in artifact metadata.
- `trade_rl/catalog/reusable_artifacts.py` — add cache identity v2 while preserving v1 backfill.
- `trade_rl/integrations/sb3_training.py` — parse solver configuration and pass it into teacher generation.
- `compose.training.yaml` — expose explicit backend and bounded-memory settings, retaining NumPy defaults.
- `tests/learning/test_oracle_teacher.py`
- `tests/learning/test_episode_oracle_teacher.py`
- `tests/learning/test_episode_teacher_integration.py`
- `tests/catalog/test_reusable_artifacts.py`
- `tests/integrations/test_sb3_training.py`
- `tests/test_training_compose_contract.py`
- `docs/operations/docker-gpu-full-training.md`

---

### Task 1: Establish the Legacy Characterization and Performance Baseline

**Files:**
- Modify: `tests/learning/test_oracle_teacher.py`
- Modify: `tests/learning/test_episode_oracle_teacher.py`
- Create: `trade_rl/operations/oracle_teacher_benchmark.py`
- Create: `tests/operations/test_oracle_teacher_benchmark.py`

**Interfaces:**
- Produces: `OracleBenchmarkCase`, `OracleBenchmarkResult`, and `run_oracle_teacher_benchmark(...)` used again in Task 10.
- Preserves: current `oracle_target_path(...)` and `episode_oracle_target_path(...)` behavior before refactoring.

- [ ] **Step 1: Add characterization cases for every maintained transition class**

Add focused tests covering full fill, partial fill, minimum-notional no-op, blocked buy, blocked sell, unavailable borrow, weight drift, open and close margin boundaries, `signal_delay_decisions` zero and one, non-cash initial weights, and future-data isolation. Assertions must distinguish boolean classification from numerical tolerance.

```python
def test_oracle_minimum_notional_noop_keeps_realized_weight_and_validity() -> None:
    market = constrained_market(minimum_notional=500.0)
    result = characterize_single_transition(market, initial_weight=0.0, target=0.45)
    assert result.valid
    assert result.fill_class == "noop_minimum_notional"
    np.testing.assert_array_equal(result.effective_target, result.open_weight)
```

- [ ] **Step 2: Run the characterization tests before refactoring**

Run:

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
```

Expected: all tests pass on the legacy implementation.

- [ ] **Step 3: Add a benchmark module that does not alter production behavior**

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

The initial backend value is `legacy_numpy`. CUDA synchronization and device metrics are added only after the CUDA backend exists.

- [ ] **Step 4: Test benchmark schema and deterministic case identity**

Run:

```bash
uv run pytest -q tests/operations/test_oracle_teacher_benchmark.py
```

Expected: pass; timing values are only checked for finiteness and non-negativity.

- [ ] **Step 5: Record the baseline command and commit**

Run one small local baseline and save its JSON outside version control under `var/benchmarks/`.

```bash
uv run python -m trade_rl.operations.oracle_teacher_benchmark --backend legacy_numpy --episode-count 4 --episode-bars 128 --repetitions 3 --output var/benchmarks/oracle-legacy-small.json
git add trade_rl/operations/oracle_teacher_benchmark.py tests/operations/test_oracle_teacher_benchmark.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
git commit -m "test: characterize Oracle solver and add baseline benchmark"
```

---

### Task 2: Introduce Backend-Neutral Solver Contracts

**Files:**
- Create: `trade_rl/learning/oracle_bellman_contracts.py`
- Create: `tests/learning/test_oracle_bellman_contracts.py`
- Modify: `trade_rl/learning/oracle_teacher.py`

**Interfaces:**
- Produces: `OracleSolverConfig`, `OracleBellmanParameters`, `OracleEpisodeInputs`, `OracleSolverProvenance`, `OracleSolveResult`, `OracleBackendFailure`.
- Consumes: existing `ExecutionCostConfig`, `PortfolioRiskConfig`, and scalar fields from `OracleTeacherConfig`.

- [ ] **Step 1: Write failing validation and identity tests**

```python
def test_solver_config_identity_changes_with_tie_tolerance() -> None:
    first = OracleSolverConfig(selection="numpy", tie_tolerance=1e-12)
    second = OracleSolverConfig(selection="numpy", tie_tolerance=1e-11)
    assert first.digest != second.digest

@pytest.mark.parametrize("selection", ["numpy", "cuda", "cuda_or_numpy"])
def test_solver_config_accepts_maintained_selections(selection: str) -> None:
    assert OracleSolverConfig(selection=selection).selection == selection
```

Also reject non-float64 dtype, invalid batch sizes, memory fractions outside `(0, 1]`, unsupported compile modes, and non-positive tolerances.

- [ ] **Step 2: Verify the tests fail for missing contracts**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_contracts.py
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement immutable contracts with explicit schema versions**

Use these maintained names:

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

`OracleBellmanParameters` receives copied scalar and immutable execution/risk contracts so solver modules never import `OracleTeacherConfig`. Add `OracleTeacherConfig.bellman_parameters` as a read-only property.

- [ ] **Step 4: Run unit, type, and import checks**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_contracts.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_bellman_contracts.py trade_rl/learning/oracle_teacher.py
uv run ruff check trade_rl/learning/oracle_bellman_contracts.py tests/learning/test_oracle_bellman_contracts.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/learning/oracle_bellman_contracts.py trade_rl/learning/oracle_teacher.py tests/learning/test_oracle_bellman_contracts.py
git commit -m "feat: define Oracle Bellman solver contracts"
```

---

### Task 3: Build and Validate the Path-Independent Market Tape

**Files:**
- Create: `trade_rl/learning/oracle_market_tape.py`
- Create: `tests/learning/test_oracle_market_tape.py`

**Interfaces:**
- Consumes: `MarketDataset`, `OracleBellmanParameters`, and `(start, stop)`.
- Produces: `build_oracle_market_tape(...) -> OracleMarketTape` with read-only contiguous NumPy arrays and a content digest.

- [ ] **Step 1: Write direct-versus-precomputed field tests**

For each step, compare tape fields with the current direct expressions for raw/equity position factors, mark/open ratios, active/tradable/direction masks, market notional, participation capacity, minimum notional, base unit cost, funding, borrow, dividend, cash rate, and elapsed year fraction.

```python
def test_market_tape_matches_direct_market_notional() -> None:
    tape = build_oracle_market_tape(market, (1, 7), parameters)
    expected = market.market_notional(2, market.open[2], volume=market.volume[1])
    np.testing.assert_allclose(tape.market_notional[0], expected, rtol=0.0, atol=0.0)
```

- [ ] **Step 2: Verify tests fail before implementation**

```bash
uv run pytest -q tests/learning/test_oracle_market_tape.py
```

Expected: import failure.

- [ ] **Step 3: Implement the builder and fail-closed validation**

The tape must store `start`, `stop`, `steps=stop-start-1`, symbol count, schema version `oracle_market_tape_v1`, and digest. Every array must be C-contiguous and read-only. Reject shape drift, negative market notional, non-finite required prices/rates, or timeline misalignment before any backend receives the tape.

- [ ] **Step 4: Prove future-data isolation**

Modify values at index `stop` and beyond, rebuild the tape, and assert identical digest and arrays for the supplied range.

- [ ] **Step 5: Run focused checks and commit**

```bash
uv run pytest -q tests/learning/test_oracle_market_tape.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_market_tape.py
uv run ruff check trade_rl/learning/oracle_market_tape.py tests/learning/test_oracle_market_tape.py
git add trade_rl/learning/oracle_market_tape.py tests/learning/test_oracle_market_tape.py
git commit -m "feat: precompute Oracle market tape"
```

---

### Task 4: Extract One-Step NumPy Transition Semantics

**Files:**
- Create: `trade_rl/learning/oracle_transition_numpy.py`
- Create: `tests/learning/test_oracle_transition_numpy.py`
- Modify: `trade_rl/learning/oracle_teacher.py`

**Interfaces:**
- Consumes: one `OracleMarketTape` step, `prior_scores [B,S]`, `prior_close_weights [B,S,N]`, `targets [K,N]`, and `OracleBellmanParameters`.
- Produces: `NumPyTransitionBatch` containing exact masks and float64 candidate factors, close weights, and effective targets.

- [ ] **Step 1: Write differential tests against legacy private helpers**

Use one-episode inputs first and compare `_open_state_matrix` plus `_transition_matrices` with:

```python
numpy_transition_step(
    tape=tape,
    step=0,
    prior_scores=prior_scores[None, :],
    prior_close_weights=prior_close_weights[None, :, :],
    targets=targets,
    parameters=parameters,
)
```

Require exact equality for masks and fill classes; use `rtol=1e-10`, `atol=1e-12` for factors and weights.

- [ ] **Step 2: Verify the new tests fail**

```bash
uv run pytest -q tests/learning/test_oracle_transition_numpy.py
```

Expected: import failure.

- [ ] **Step 3: Implement the vectorized batch kernel**

Keep all arithmetic float64. Avoid repeated `resolved_array()` calls and market-notional calculation. Return typed classifications for full fill, partial fill, no requested trade, minimum-notional no-op, capacity-zero no-op, and invalid transition. Use broadcast views where safe and allocate only outputs that survive reduction.

- [ ] **Step 4: Replace legacy helper bodies with compatibility wrappers**

`oracle_teacher._open_state_matrix` and `_transition_matrices` remain importable for current tests, but delegate to the shared NumPy implementation. There must be only one maintained NumPy equation set after this step.

- [ ] **Step 5: Run executor-contract tests and commit**

```bash
uv run pytest -q tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_teacher.py
uv run mypy trade_rl/learning/oracle_transition_numpy.py
uv run ruff check trade_rl/learning/oracle_transition_numpy.py tests/learning/test_oracle_transition_numpy.py
git add trade_rl/learning/oracle_transition_numpy.py trade_rl/learning/oracle_teacher.py tests/learning/test_oracle_transition_numpy.py
git commit -m "refactor: centralize Oracle NumPy transitions"
```

---

### Task 5: Implement the Rolling-Buffer Batched NumPy Solver

**Files:**
- Create: `trade_rl/learning/oracle_bellman_numpy.py`
- Create: `tests/learning/test_oracle_bellman_numpy.py`

**Interfaces:**
- Consumes: `OracleMarketTape`, `states [S,N]`, `OracleEpisodeInputs`, `OracleBellmanParameters`, and `OracleSolverConfig`.
- Produces: `solve_numpy_oracle_batch(...) -> OracleSolveResult`.

- [ ] **Step 1: Write tests for deterministic reduction and backtracking**

```python
def test_lowest_prior_index_wins_within_tolerance() -> None:
    scores = np.array([[[1.0], [1.0 + 5e-13], [0.0]]], dtype=np.float64)
    best_scores, pointers = reduce_candidates_numpy(scores, tie_tolerance=1e-12)
    assert pointers.tolist() == [[0]]
    np.testing.assert_allclose(best_scores, [[1.0 + 5e-13]])
```

Also test pointer dtype selection (`int16` through state count 32767, otherwise `int32`), missing-pointer failure, blocked versus unblocked target reductions, and both signal-delay modes.

- [ ] **Step 2: Verify tests fail**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_numpy.py
```

Expected: import failure.

- [ ] **Step 3: Implement rolling forward buffers**

Retain only previous/next scores and close weights. Retain `pointers [B,T,S]`; reconstruct paths in reverse. Process target states in deterministic contiguous blocks when requested and preserve global target order.

- [ ] **Step 4: Add randomized legacy differential tests**

Use deterministic Hypothesis cases for one to three symbols, long-only/short-enabled, zero/non-zero costs, liquidity limits, permissions, initial weights, and both signal delays. Compare exact masks separately from tolerance values. When paths differ, verify the declared score-equivalence and executor-replay conditions.

- [ ] **Step 5: Run focused and randomized suites**

```bash
uv run pytest -q tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/learning/oracle_bellman_numpy.py tests/learning/test_oracle_bellman_numpy.py
git commit -m "feat: add rolling batched NumPy Bellman solver"
```

---

### Task 6: Route Existing Teacher Entry Points Through the New NumPy Solver

**Files:**
- Create: `trade_rl/learning/oracle_solver.py`
- Create: `tests/learning/test_oracle_solver.py`
- Modify: `trade_rl/learning/oracle_teacher.py`
- Modify: `trade_rl/learning/episode_oracle_teacher.py`
- Modify: `tests/learning/test_episode_teacher_integration.py`

**Interfaces:**
- Produces: `solve_oracle_episodes(...) -> OracleSolveResult` and one orchestration path shared by full-range and episode teachers.
- Preserves: public return dtypes, read-only arrays, episode ordering, contract digests, and current default behavior.

- [ ] **Step 1: Write integration tests that require the NumPy backend explicitly**

Assert that `oracle_target_path(...)` and `episode_oracle_target_path(...)` produce the same targets as the captured legacy fixtures and that `build_episode_oracle_batch(..., solver_config=OracleSolverConfig(selection="numpy"))` preserves contract order and digest determinism.

- [ ] **Step 2: Implement the orchestrator and compatibility defaults**

`solve_oracle_episodes` validates inputs, builds one tape per required timeline, batches contracts with identical horizon, calls the selected backend, and returns per-episode targets plus provenance. The default `OracleSolverConfig()` remains NumPy.

- [ ] **Step 3: Remove NumPy episode multiprocessing from the default path only after differential parity**

The new NumPy batched solver may still use bounded CPU batching, but must not retain two independent public generation routes. Preserve `max_workers` as a compatibility input until callers migrate; reject contradictory `max_workers>1` with CUDA selection in Task 8.

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest -q tests/learning/test_oracle_solver.py tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_episode_teacher_integration.py
```

Expected: all pass and existing public tests require no weakened assertions.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/learning/oracle_solver.py trade_rl/learning/oracle_teacher.py trade_rl/learning/episode_oracle_teacher.py tests/learning/test_oracle_solver.py tests/learning/test_episode_teacher_integration.py
git commit -m "refactor: route Oracle teachers through batched solver"
```

---

### Task 7: Version Solver Provenance, Teacher Artifacts, and Cache Identity

**Files:**
- Modify: `trade_rl/learning/episode_teacher_artifact.py`
- Modify: `trade_rl/catalog/reusable_artifacts.py`
- Modify: `tests/catalog/test_reusable_artifacts.py`
- Modify: `tests/learning/test_episode_teacher_integration.py`

**Interfaces:**
- Produces: `teacher_cache_identity_v2(...)` and serialized `OracleSolverProvenance`.
- Preserves: loading and backfilling legacy v1 artifacts without fabricating provenance.

- [ ] **Step 1: Write cache collision and legacy backfill tests**

```python
def test_teacher_cache_identity_v2_separates_numpy_and_cuda() -> None:
    numpy_key = teacher_cache_identity_v2(..., solver_backend="numpy")
    cuda_key = teacher_cache_identity_v2(..., solver_backend="torch_cuda")
    assert numpy_key != cuda_key


def test_v1_backfill_remains_legacy_without_solver_claims() -> None:
    registered = backfill_teacher_cache(index)
    assert registered == 1
    record = catalog.only_record()
    assert record.registration.cache_key["schema_version"] == "teacher_cache_identity_v1"
```

- [ ] **Step 2: Add v2 identity and artifact metadata fields**

Include solver contract, backend actually used, float64 dtype, tie-break contract, tie tolerance, market-tape schema/digest, batch size, block size, compile mode/chunk size, fallback reason, OOM retry, wall time, peak memory, and CUDA/PyTorch/device details when applicable.

- [ ] **Step 3: Ensure artifact digest covers provenance consistently**

A backend change must change cache identity. Payload equality may still produce the same target-array digest; catalog registration must not report a false immutable-content conflict because cache keys differ.

- [ ] **Step 4: Run catalog and artifact tests**

```bash
uv run pytest -q tests/catalog/test_reusable_artifacts.py tests/learning/test_episode_teacher_integration.py
```

Expected: all pass, including legacy fixtures.

- [ ] **Step 5: Commit**

```bash
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
- Consumes the same contracts as the NumPy backend.
- Produces exact classification masks, tolerance-comparable float64 outputs, deterministic pointers, and complete CUDA provenance.

- [ ] **Step 1: Write CUDA-conditional primitive tests**

Mark tests with `pytest.skipif(not torch.cuda.is_available(), reason="CUDA required")`. Compare tape transfer, one-step masks, fill classifications, tie reduction, blocked/unblocked output, backtracking, and executor replay against NumPy.

- [ ] **Step 2: Implement device transfer and eager one-step equations**

Convert the complete tape once per batch and keep all forward state resident on CUDA. Use `torch.float64`, `torch.bool`, and appropriate pointer integer types. Do not transfer per-step tensors back to the host.

- [ ] **Step 3: Add deterministic target-state blocking**

Estimate candidate bytes before execution. Respect explicit block size when supplied; otherwise derive a contiguous block size from `cuda_memory_fraction` and free-memory inspection. Merge block reductions using the same lowest-index-within-tolerance rule.

- [ ] **Step 4: Add one deterministic OOM retry**

On CUDA OOM, clear only backend-owned references, synchronize, empty the allocator cache for this retry path, halve the block size deterministically, and retry once. Record both attempted sizes. Raise `OracleBackendFailure(kind="cuda_oom")` after the second failure.

- [ ] **Step 5: Add optional fixed-chunk compilation behind configuration**

Support eager mode first. For `compile_mode="reduce_overhead"`, compile a stable fixed-size chunk with chunk sizes restricted to `{8,16,32,64}`. Report compile warm-up separately; never make compilation required for correctness.

- [ ] **Step 6: Run CPU-safe tests, then RTX 4070 Ti SUPER tests**

CPU-safe:

```bash
uv run pytest -q tests/learning/test_oracle_bellman_torch.py
```

Maintained GPU:

```bash
uv run pytest -q tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py --run-cuda
```

Expected: exact mask/classification parity and tolerance-comparable values.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/learning/oracle_bellman_torch.py trade_rl/learning/oracle_solver.py tests/learning/test_oracle_bellman_torch.py
git commit -m "feat: add batched float64 CUDA Bellman solver"
```

---

### Task 9: Add Explicit Runtime Selection, Fallback, and Training Configuration

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `compose.training.yaml`
- Modify: `tests/integrations/test_sb3_training.py`
- Modify: `tests/test_training_compose_contract.py`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Consumes environment configuration and creates `OracleSolverConfig`.
- Produces explicit `numpy`, `cuda`, or `cuda_or_numpy` behavior with NumPy as default.

- [ ] **Step 1: Write configuration parsing tests**

Cover:

```text
TRADE_RL_ORACLE_SOLVER=numpy|cuda|cuda_or_numpy
TRADE_RL_ORACLE_EPISODE_BATCH_SIZE=8
TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE=
TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION=0.65
TRADE_RL_ORACLE_COMPILE_MODE=disabled|reduce_overhead
TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE=16
```

Reject invalid values before teacher generation starts. Confirm missing variables preserve NumPy behavior.

- [ ] **Step 2: Implement hard-fail and pre-publication fallback semantics**

`cuda` fails on unavailable CUDA or backend failure. `cuda_or_numpy` may fall back only before artifact promotion and records the failure kind and message. Staging directories are removed or retained according to the existing failure-evidence policy; no ready catalog record is created for failed CUDA output.

- [ ] **Step 3: Prevent conflicting worker ownership**

For CUDA selections, one process owns the device and episode contracts use the solver batch dimension. Existing `TRADE_RL_TEACHER_WORKERS` must not create multiple CUDA owners. Either require it to be `1` for explicit CUDA or ignore it only with a recorded compatibility warning and tested behavior; choose the fail-closed requirement unless an existing caller cannot be migrated atomically.

- [ ] **Step 4: Update compose and operations documentation**

Keep `TRADE_RL_ORACLE_SOLVER: numpy` as the checked default. Document the RTX 4070 Ti SUPER acceptance command and the meaning of fallback, batch size, block size, compile mode, and artifact provenance.

- [ ] **Step 5: Run integration and compose tests**

```bash
uv run pytest -q tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py tests/learning/test_episode_teacher_integration.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/integrations/sb3_training.py compose.training.yaml tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py docs/operations/docker-gpu-full-training.md
git commit -m "feat: configure Oracle solver backend and fallback"
```

---

### Task 10: Complete Differential, Replay, and Performance Evidence

**Files:**
- Modify: `trade_rl/operations/oracle_teacher_benchmark.py`
- Modify: `tests/operations/test_oracle_teacher_benchmark.py`
- Modify: `tests/learning/test_oracle_bellman_torch.py`
- Modify: `tests/learning/test_oracle_solver.py`
- Create: `docs/operations/oracle-bellman-solver-benchmark.md`

**Interfaces:**
- Produces reproducible JSON evidence and a human-readable benchmark report.
- Does not change the default backend.

- [ ] **Step 1: Add synchronized CUDA timing and memory metrics**

Call `torch.cuda.synchronize()` immediately before and after measured regions. Reset and record peak allocated/reserved bytes. Separate transfer, warm-up/compile, steady solver, backtracking, replay validation, and artifact I/O durations.

- [ ] **Step 2: Define the maintained benchmark matrix**

Run at least:

```text
Backends: legacy serial CPU, maintained multi-worker CPU, new NumPy, eager CUDA, compiled CUDA candidates
Episode batches: 1, 4, 8, 16, 32
Compile chunks: 8, 16, 32, 64
Horizons: 128 smoke and 2880 maintained
State counts: current default plus one larger bounded synthetic case
Repetitions: 1 cold + 5 steady
```

The JSON identity records commit SHA, dirty state, Python/NumPy/PyTorch/CUDA versions, GPU model, state count, symbols, episode count, horizon, batch/block sizes, and synchronization policy.

- [ ] **Step 3: Run the correctness corpus before performance claims**

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py tests/learning/test_episode_teacher_integration.py --run-cuda
```

Expected: all pass on the maintained GPU host.

- [ ] **Step 4: Run the maintained RTX 4070 Ti SUPER benchmark**

```bash
uv run python -m trade_rl.operations.oracle_teacher_benchmark --backend all --episode-count 32 --episode-bars 2880 --repetitions 5 --output var/benchmarks/oracle-bellman-4070ti-super.json
```

Do not claim a speedup unless the command completes and the report includes synchronized steady-state measurements.

- [ ] **Step 5: Write the benchmark report and default recommendation**

Report raw medians, ranges, peak memory, cold-start cost, and comparison against maintained multi-worker CPU. Recommend retaining NumPy or proposing CUDA default separately; this implementation PR must not silently change the default.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/operations/oracle_teacher_benchmark.py tests/operations/test_oracle_teacher_benchmark.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py docs/operations/oracle-bellman-solver-benchmark.md
git commit -m "perf: validate Oracle Bellman CPU and CUDA backends"
```

---

### Task 11: Final Architecture Review and Repository-Wide Verification

**Files:**
- Modify only files required by findings from this review.

**Interfaces:**
- Produces one verified final head suitable for draft PR review.

- [ ] **Step 1: Review the complete diff as a reviewer**

Check responsibility boundaries, circular imports, duplicate transition equations, hidden device transfers, unbounded temporaries, threshold comparisons, partial artifact publication, cache provenance truthfulness, dead compatibility code, and accidental changes outside Oracle teacher generation.

- [ ] **Step 2: Run the narrow suites after review fixes**

```bash
uv run pytest -q tests/learning/test_oracle_teacher.py tests/learning/test_episode_oracle_teacher.py tests/learning/test_episode_teacher_integration.py tests/learning/test_oracle_bellman_contracts.py tests/learning/test_oracle_market_tape.py tests/learning/test_oracle_transition_numpy.py tests/learning/test_oracle_bellman_numpy.py tests/learning/test_oracle_bellman_torch.py tests/learning/test_oracle_solver.py tests/catalog/test_reusable_artifacts.py tests/integrations/test_sb3_training.py tests/test_training_compose_contract.py tests/operations/test_oracle_teacher_benchmark.py
```

Expected: all pass.

- [ ] **Step 3: Run static and architecture verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports
```

Expected: no errors and all architecture contracts kept.

- [ ] **Step 4: Run the full test and coverage suite on the same head**

```bash
uv run pytest --cov=trade_rl --cov-report=term-missing
```

Expected: full suite passes, branch coverage remains at or above 80%, and all critical-coverage ratchets pass.

- [ ] **Step 5: Run build/runtime compatibility checks required by changed surfaces**

Run the repository's maintained CPU-only training smoke, CUDA training-image smoke, PostgreSQL catalog tests, Windows compatibility tests, and Ubuntu compatibility tests on the final head. Record exact commands and outputs in the PR body.

- [ ] **Step 6: Verify Git and secret hygiene**

```bash
git status --short
git diff --check
git diff main...HEAD --stat
git log --oneline main..HEAD
```

Confirm no benchmark payloads, local datasets, CUDA dumps, generated artifacts, credentials, or temporary workflows are tracked.

- [ ] **Step 7: Commit final review fixes**

```bash
git add <only-reviewed-files>
git commit -m "fix: address Oracle Bellman solver self-review"
```

Skip this commit when the review produces no changes; do not create an empty commit.

- [ ] **Step 8: Prepare a draft PR without merging**

The PR body must include What, Why, architecture, numerical contract, cache migration, fallback behavior, benchmark results, tests, risks, non-goals, and the reason NumPy remains the default. Do not mark Ready until all final-head checks and maintained-hardware evidence succeed.
