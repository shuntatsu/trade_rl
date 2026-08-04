# Oracle Bellman Batched Solver Design

## Status

Approved for implementation planning on 2026-08-04.

## Context

Oracle teacher generation currently evaluates every portfolio-state transition with NumPy inside a Python loop over decision steps. The transition calculation already vectorizes the prior-state and target-state axes, but each step repeatedly resolves market arrays, computes market-side quantities, allocates temporary tensors, and runs one episode at a time. Episode generation is parallelized across CPU workers, which improves throughput but duplicates transient dynamic-programming work and does not use the available RTX 4070 Ti SUPER.

A naive full `T × S × S` transition tensor is not semantically valid for the maintained Oracle contract. The transition at time `t` depends on the selected prior path through realized close weights and accumulated portfolio value. Those values affect minimum-notional eligibility, partial fills, liquidity caps, effective targets, collateral, maintenance margin, and the next close weights. Only market-side inputs that are independent of the chosen path may be safely precomputed for all steps.

## Decision

Introduce a backend-neutral batched Bellman solver with:

1. a precomputed immutable market tape for path-independent inputs;
2. a NumPy float64 reference backend;
3. a PyTorch CUDA float64 backend that batches independent Oracle episodes;
4. rolling forward-state buffers and retained backpointers;
5. explicit deterministic tie handling;
6. solver provenance in teacher cache identity and artifact metadata;
7. differential and executor-replay validation before enabling CUDA by default.

The semantic policy is option B: preserve the execution and risk contract, while allowing bounded floating-point differences and alternate paths only when their scores are within the declared tie tolerance.

## Goals

- Reduce Oracle teacher generation wall-clock time on CPU and RTX 4070 Ti SUPER.
- Preserve train-range isolation and the maintained deterministic execution semantics.
- Preserve transition validity classification for tradability, direction permissions, borrow availability, minimum notional, partial/no fill, and margin constraints.
- Keep the solver usable without CUDA through the NumPy backend.
- Keep large temporary tensors bounded as portfolio-state count grows.
- Produce benchmark and correctness evidence sufficient to choose a default backend.

## Non-goals

- Replacing the bounded-state Oracle approximation with a different optimization problem.
- Discretizing portfolio value or realized weights to make every transition path-independent.
- Changing portfolio states, reward meaning, execution costs, or risk limits.
- Introducing float32 or mixed precision in the initial implementation.
- Implementing Triton or custom CUDA kernels before PyTorch measurements show a justified bottleneck.
- Promising a fixed speedup before benchmark evidence exists.

## Correctness contract

### Exact requirements

The NumPy and CUDA implementations must agree exactly on:

- portfolio-state definitions and ordering;
- train-range bounds and output shape;
- active, tradable, buy-allowed, sell-allowed, and borrow-available interpretation;
- whether a transition is valid;
- minimum-notional no-op classification;
- partial-fill versus full-fill classification;
- margin and maintenance-margin violation classification;
- submitted targets belonging to the maintained bounded state set;
- signal-delay behavior, including the discarded terminal pending action;
- absence of data reads beyond the supplied training range.

### Tolerance requirements

The following may differ within declared absolute and relative tolerances:

- gap factors;
- close-equity factors;
- close weights;
- accumulated log scores;
- replayed portfolio value.

Initial comparison tolerances are `rtol=1e-10` and `atol=1e-12` for float64 arithmetic. Tests may use a larger explicitly justified tolerance for long-horizon accumulated values, but must not use one blanket tolerance to hide classification changes.

### Path-equivalence rule

A CUDA path may differ from the NumPy reference only when:

1. each selected transition satisfies the exact contract;
2. executor replay succeeds;
3. the selected final score is within `tie_tolerance` of the NumPy best score; and
4. the alternate path does not change any exact transition classification.

The initial `tie_tolerance` is `1e-12` in log-score units. It is part of solver provenance and may only change with a solver-contract version change.

## Architecture

```text
MarketDataset
    |
    v
OracleMarketTapeBuilder
    |
    v
OracleMarketTape
    |------------------------------|
    v                              v
NumPyBellmanBackend          TorchCudaBellmanBackend
    |                              |
    |------------------------------|
                   |
                   v
          BatchedOracleSolver
                   |
                   v
      state paths / submitted targets
                   |
                   v
        Executor replay validation
                   |
                   v
      Teacher artifact and cache index
```

### Module boundaries

#### `oracle_market_tape.py`

Owns construction and validation of path-independent arrays. It may depend on `MarketDataset` but not on PyTorch. It returns read-only contiguous NumPy arrays and an identity digest.

#### `oracle_bellman_contracts.py`

Owns backend-neutral dataclasses and protocols:

- `OracleMarketTape`;
- `OracleEpisodeInputs`;
- `OracleSolverConfig`;
- `OracleSolverProvenance`;
- `OracleSolveResult`;
- the backend protocol.

This module must not import CUDA-specific code.

#### `oracle_bellman_numpy.py`

Owns the float64 reference Bellman step and batched solver. It is the correctness oracle for backend differential tests. It uses the same rolling-buffer and batched interfaces as the CUDA backend.

#### `oracle_bellman_torch.py`

Owns PyTorch device transfer, CUDA Bellman operations, optional chunk compilation, synchronization for benchmarks, and conversion of final targets and evidence back to NumPy.

#### Existing teacher modules

`oracle_teacher.py` and `episode_oracle_teacher.py` become orchestration and compatibility layers. Public behavior remains unchanged unless a backend is selected explicitly. Shared transition semantics must not remain duplicated between old and new solvers after migration.

## Market tape

`OracleMarketTape` precomputes arrays over the requested immutable timeline. Expected fields include:

```python
@dataclass(frozen=True, slots=True)
class OracleMarketTape:
    raw_position_factor: np.ndarray
    equity_position_factor: np.ndarray
    mark_open_ratio: np.ndarray
    active: np.ndarray
    tradable: np.ndarray
    buy_allowed: np.ndarray
    sell_allowed: np.ndarray
    borrow_available: np.ndarray
    market_notional: np.ndarray
    participation_capacity: np.ndarray
    minimum_notional: np.ndarray
    venue_and_base_unit_cost: np.ndarray
    funding_due_rate: np.ndarray
    borrow_rate: np.ndarray
    dividend_open_ratio: np.ndarray
    cash_rate: np.ndarray
    elapsed_year_fraction: np.ndarray
    start: int
    stop: int
    digest: str
```

Field names may be adjusted during implementation to match existing terminology, but responsibilities must remain path-independent. The builder must validate shapes, finiteness where required, non-negative liquidity inputs, and exact timeline alignment.

The tape must not contain final `transition_valid`, `close_factor`, `effective_targets`, or `close_weights`, because those depend on prior realized state and accumulated equity.

## Batched dynamic programming

Let:

- `B` be episode batch size;
- `T` be decisions per episode;
- `S` be portfolio-state count;
- `N` be symbol count.

Forward state:

```text
prior_scores         [B, S]
prior_close_weights  [B, S, N]
```

Candidate tensors for one step or target-state block:

```text
transition_valid     [B, S, K]
close_factor         [B, S, K]
candidate_weights    [B, S, K, N]
candidate_targets    [B, S, K, N]
```

`K` equals `S` for small state spaces and is a target-state block for larger spaces.

Reduction output:

```text
next_scores          [B, K]
next_close_weights   [B, K, N]
backpointer          [B, K]
```

For a blocked target-state implementation, global target order must be preserved and the block reductions must produce the same result as an unblocked reduction.

### Rolling storage

Only the previous and next score and close-weight buffers are retained during the forward pass. Full score and close-weight histories are not stored. Backpointers are retained as integer arrays because they are required for reverse path reconstruction.

Use the narrowest safe backpointer dtype:

- `int16` when `S <= 32767`;
- otherwise `int32`.

Invalid pointers use `-1`.

### Deterministic tie handling

The backend must not rely solely on unspecified parallel reduction order. For each target state:

1. compute the maximum candidate score;
2. mark candidates whose score is at least `maximum - tie_tolerance`;
3. select the lowest prior-state index among eligible candidates.

This rule is shared by NumPy and CUDA. Existing control-projection tie penalties remain part of the candidate score and are applied before tolerance-based tie selection.

## GPU execution

### Backend choice

Use PyTorch CUDA rather than adding CuPy initially because PyTorch is already part of the maintained training environment and CUDA image. This avoids a second GPU array dependency and allows optional `torch.compile` evaluation.

### Precision

Use float64 for all contract-sensitive calculations. Boolean masks and integer pointers retain native boolean and integer dtypes. Float32 or mixed precision requires a separate design because it can change threshold and margin classification.

### Episode batching

Independent sampled episodes form the primary GPU batch dimension. Initial benchmark candidates are `B = 1, 4, 8, 16, 32`, constrained by measured memory use.

A single episode with the default small state space may underutilize the GPU. The maintained CPU process-parallel episode generation is therefore replaced, for the CUDA backend, by one process owning the GPU and batching episode contracts. Multiple CPU processes must not concurrently allocate independent copies of the same CUDA workload.

### State blocking

The solver estimates temporary memory before execution. When a full `[B, S, S, N]` candidate tensor exceeds the configured memory budget, it partitions the target-state axis into deterministic contiguous blocks. The default budget is a configurable fraction of currently free CUDA memory, capped to leave headroom for the training process and driver allocations.

Out-of-memory errors are not silently retried with arbitrary behavior. The CUDA backend may retry once with a deterministically smaller block size, recording the retry and final block size in provenance. If the second attempt fails, it returns a typed backend failure so orchestration can apply the configured fallback policy.

### Compilation

The first implementation runs eager PyTorch CUDA. Optional compilation is benchmarked only after correctness is established.

Candidate modes:

- eager one-step execution;
- fixed chunks of 8, 16, 32, or 64 steps;
- `torch.compile(mode="reduce-overhead", fullgraph=True)` when the graph is stable.

Compilation cache warm-up is excluded from steady-state timing but reported separately. The uncompiled CUDA backend remains available as a fallback.

## Backend selection and fallback

Add an explicit solver selection contract, for example:

```text
numpy
cuda
cuda_or_numpy
```

- `numpy` always uses the reference backend.
- `cuda` fails if CUDA is unavailable or the CUDA solver fails.
- `cuda_or_numpy` may fall back to NumPy only before artifact publication. The artifact records the backend actually used and the fallback reason.

No fallback may occur after partial artifact registration. Temporary output is written to staging and promoted only after solver validation and artifact hashing succeed.

The initial default remains `numpy` until the acceptance benchmark and differential suite pass on the maintained CUDA environment. Default changes require a separate reviewed configuration change.

## Solver provenance and cache identity

The current teacher cache identity does not distinguish numerical solver implementations. Introduce `teacher_cache_identity_v2` containing at least:

```json
{
  "schema_version": "teacher_cache_identity_v2",
  "dataset_id": "...",
  "train_range": [0, 1],
  "environment_digest": "...",
  "action_spec_digest": "...",
  "teacher_config_digest": "...",
  "solver_contract": "batched_bellman_v1",
  "solver_backend": "numpy" | "torch_cuda",
  "numeric_dtype": "float64",
  "tie_break_contract": "lowest_prior_within_tolerance_v1",
  "tie_tolerance": 1e-12,
  "market_tape_schema": "oracle_market_tape_v1"
}
```

Artifact metadata additionally records:

- PyTorch and CUDA versions when applicable;
- device name and compute capability;
- episode batch size;
- target-state block size;
- eager or compiled execution mode;
- compile chunk size;
- fallback and OOM-retry information;
- market-tape digest;
- solver wall-clock and peak-memory metrics.

CPU and CUDA artifacts may have identical payload digests, but distinct cache identities prevent an alternate tie-equivalent payload from being treated as an immutable-content conflict.

Backfill of v1 cache entries keeps them readable as legacy NumPy artifacts. It must not rewrite them as v2 without enough provenance to construct a truthful v2 identity.

## Error handling

- Invalid market-tape shapes or values fail before device transfer.
- No executable path raises the existing Oracle failure with episode identity included.
- A missing backpointer raises a solver-integrity error and prevents artifact publication.
- CUDA unavailable under `cuda` is a hard error.
- CUDA unavailable under `cuda_or_numpy` produces a recorded NumPy fallback.
- Non-finite candidate values are masked according to the maintained transition contract; unexpected non-finite forward state fails closed.
- Backend differential validation failures prevent enabling or publishing the CUDA result for that run.

## Testing strategy

### Characterization tests first

Before changing implementation, add tests that capture current behavior for:

- deterministic target paths;
- partial fills;
- minimum-notional no-op;
- blocked buy and sell directions;
- borrow availability;
- weight drift;
- margin boundaries;
- signal delay zero and one;
- explicit non-cash episode initial weights;
- train-range future-data isolation;
- artifact digest and cache conflict behavior.

### Unit tests

- Market-tape fields match direct per-step calculations.
- Rolling-buffer NumPy output matches the legacy solver.
- Blocked and unblocked reductions match.
- Tie selection chooses the lowest eligible prior index.
- Pointer dtype selection and reverse reconstruction are correct.
- Solver identity changes when any numerical contract field changes.
- v1 cache backfill remains legacy and conflict-safe.

### Randomized differential tests

Generate deterministic small markets across combinations of:

- one to three symbols;
- long-only and short-enabled execution;
- zero and non-zero costs;
- low liquidity and partial fills;
- changing tradability and direction permissions;
- varying initial weights;
- signal delay zero and one;
- state counts small enough for exhaustive comparison.

Compare legacy, new NumPy, and CUDA outputs using the exact and tolerance contracts separately. A numerical tolerance must never replace equality assertions for masks or classifications.

### Executor replay tests

Replay selected teacher paths through the maintained deterministic executor and compare:

- effective fills;
- final portfolio value;
- final weights;
- margin validity;
- submitted-action alignment.

### Performance tests

Benchmarks report:

- legacy serial CPU;
- legacy maintained multi-worker CPU;
- new NumPy serial and batched CPU;
- eager CUDA for each episode batch size;
- compiled CUDA candidates;
- cold-start, warm-up, and steady-state times separately;
- peak host and device memory;
- state count, symbol count, episode length, and episode count.

CUDA timing includes synchronization around the measured region. Artifact I/O and cache hits are reported separately from solver compute.

Performance tests are not ordinary CI pass/fail tests. CI runs a small smoke benchmark to detect catastrophic regression; maintained-hardware acceptance uses a recorded benchmark command and evidence artifact.

## Acceptance criteria

### Correctness

- All existing Oracle and teacher tests pass unchanged or are strengthened without weakening their contract.
- New NumPy solver matches the legacy solver under exact and tolerance rules.
- CUDA validity and classification masks exactly match NumPy for the differential corpus.
- CUDA paths satisfy the path-equivalence rule.
- Executor replay remains within the declared float64 tolerance.
- Cache identity v2 prevents false immutable-content conflicts.

### Performance

- The new NumPy path does not materially regress the current serial solver on representative workloads.
- CUDA is faster than current serial CPU on the maintained full episode workload.
- CUDA is compared with the maintained multi-worker CPU configuration before any default change.
- No performance claim is made without measured episode count, horizon, state count, batch size, synchronization policy, and hardware details.

### Operability

- CPU-only environments remain supported.
- GPU memory is bounded through state blocking.
- Backend and fallback behavior are explicit in configuration and artifacts.
- A failed CUDA attempt cannot publish a partial or mislabeled teacher artifact.

## Implementation sequence

1. Add baseline benchmark and characterization tests around the legacy solver.
2. Introduce backend-neutral contracts and `OracleMarketTape`.
3. Implement and test market-tape construction against direct calculations.
4. Implement rolling-buffer NumPy Bellman solver.
5. Differentially compare the new NumPy solver with the legacy implementation.
6. Route existing Oracle entry points through the NumPy solver without changing public behavior.
7. Add episode batching to backend-neutral orchestration.
8. Implement eager PyTorch CUDA backend in float64.
9. Add blocked target-state execution and deterministic OOM handling.
10. Add CUDA differential and executor-replay tests.
11. Introduce solver provenance and cache identity v2 with legacy v1 compatibility.
12. Benchmark eager CUDA and select initial batch and block defaults.
13. Evaluate optional compiled chunks and retain only measured improvements.
14. Run focused tests, architecture checks, type checks, full tests, and maintained-hardware benchmark from the same final commit.
15. Review evidence before considering a default backend change.

## Risks and mitigations

### GPU underutilization

Default state count may be small. Batch independent episodes and measure before claiming benefit.

### Memory growth

Candidate tensors scale with `B × S × K × N`. Estimate memory and block the target axis deterministically.

### Numerical boundary changes

Keep float64 and assert classification equality separately from numerical tolerance.

### Cache conflicts

Version the cache identity with solver provenance rather than assuming equivalent semantics imply byte-identical arrays.

### Duplicate transition logic

Use the NumPy backend as the shared semantic implementation and prevent long-term coexistence of separate legacy and new transition formulas. During migration, characterization tests guard the temporary duplication.

### Compilation complexity

Treat `torch.compile` as an optional measured optimization. Eager CUDA remains the maintained base implementation.

## Final design choice

Proceed with a NumPy reference solver plus a PyTorch CUDA batched solver, path-independent market-tape precomputation, rolling forward buffers, deterministic tie handling, target-state blocking, executor replay validation, and solver-aware artifact identity. Do not implement a fixed full `T × S × S` transition tensor because it would either violate the maintained path-dependent execution contract or require a new approximate Oracle design.
