# Direct Maintained Contract Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make simulation, telemetry, and Studio public modules directly own the hardened maintained behavior while preserving every current import path and runtime contract.

**Architecture:** Canonical public classes/functions are declared in `execution.py`, `telemetry/training.py`, and `studio/telemetry.py`. Private telemetry index/storage mechanics move to a standard-library-only private module. Former adapter/strict modules become behavior-free compatibility aliases to the canonical objects.

**Tech Stack:** Python 3.12, NumPy, dataclasses, pathlib, OS file locking, Pydantic, Pytest, Ruff, Mypy, Import Linter, GitHub Actions.

## Global Constraints

- Preserve all package-level and direct-module import paths listed in the design.
- Preserve public class identity across compatibility aliases.
- Preserve simulation result/evidence values and exception messages.
- Preserve telemetry JSONL `training_telemetry_v1` and index `training_telemetry_index_v2`.
- Preserve cross-process locking on Linux and Windows.
- Preserve Studio API schemas and fail-closed duplicate-seed behavior.
- Compatibility modules must contain no behavior-bearing class or function definitions.
- No Import Linter boundary may be weakened.
- No existing coverage threshold may be lowered merely to pass the migration.
- Production remains `NO-GO`; no direct exchange routing is introduced.

---

### Task 1: Add RED public-ownership contracts

**Files:**
- Create: `tests/architecture/test_direct_maintained_contract_exports.py`
- Modify: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- Consumes: current public package facades and direct modules.
- Produces: failing contracts for canonical module ownership, alias identity, compatibility-module thinness, and absence of superseded public bodies.

- [ ] **Step 1: Write canonical ownership assertions**

Assert:

```python
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import MarketExecutor as DirectMarketExecutor
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor
from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter
from trade_rl.telemetry.training import (
    TrainingTelemetryRecord as DirectRecord,
    TrainingTelemetryWriter as DirectWriter,
)
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
)
from trade_rl.studio.telemetry import StudioTelemetryReader
from trade_rl.studio.strict_telemetry import StrictStudioTelemetryReader

assert MarketExecutor is DirectMarketExecutor
assert StatefulCompatibilityMarketExecutor is DirectMarketExecutor
assert DirectMarketExecutor.__module__ == "trade_rl.simulation.execution"
assert TrainingTelemetryRecord is DirectRecord is StrictTrainingTelemetryRecord
assert TrainingTelemetryWriter is DirectWriter is IndexedTrainingTelemetryWriter
assert DirectRecord.__module__ == "trade_rl.telemetry.training"
assert DirectWriter.__module__ == "trade_rl.telemetry.training"
assert StudioTelemetryReader is StrictStudioTelemetryReader
assert StudioTelemetryReader.__module__ == "trade_rl.studio.telemetry"
```

- [ ] **Step 2: Write source thinness assertions**

Parse the compatibility modules with `ast` and assert they contain no `ClassDef` and no public `FunctionDef`/`AsyncFunctionDef`. Assert `simulation/__init__.py` imports `MarketExecutor` from `execution`, `telemetry/__init__.py` imports public names from `training`, and `studio/api.py` does not import `strict_telemetry`.

- [ ] **Step 3: Write superseded-body assertions**

Inspect `execution.MarketExecutor.execute_interval` and require calls to `execute_target_statefully`, compatibility order-book fields, and target identity generation. Assert `_target_weights` is absent. Inspect `training.py` and require `expected_generation`, indexed storage delegation, and strict boolean validation; forbid the old linear-scan public reader and thread-only text writer fields. Inspect `StudioTelemetryReader._paths` and require duplicate seed rejection.

- [ ] **Step 4: Update the older architecture follow-up contract**

Replace assertions that package objects equal adapter subclasses with assertions that adapter names are aliases to direct public objects.

- [ ] **Step 5: Run RED**

Run:

```bash
uv run pytest -q \
  tests/architecture/test_direct_maintained_contract_exports.py \
  tests/architecture/test_architecture_followup.py
```

Expected: failures showing `MarketExecutor`, telemetry objects, and Studio reader are not yet canonically owned and compatibility modules still define behavior.

- [ ] **Step 6: Commit RED evidence**

```bash
git add tests/architecture/test_direct_maintained_contract_exports.py tests/architecture/test_architecture_followup.py
git commit -m "test: require direct maintained contract ownership"
```

### Task 2: Move maintained simulation behavior to `execution.py`

**Files:**
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/simulation/execution_adapter.py`
- Modify: `trade_rl/simulation/__init__.py`
- Modify: `tests/simulation/test_stateful_execution_adapter.py`
- Modify: `tests/simulation/test_execution_slippage_coverage.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces canonical `trade_rl.simulation.execution.MarketExecutor` with stateful compatibility methods and explicit compatibility alias `StatefulCompatibilityMarketExecutor = MarketExecutor`.

- [ ] **Step 1: Add direct-import parity tests**

Instantiate package, direct, and compatibility imports and verify they are the same class. Run the existing compatibility-chain tests against `trade_rl.simulation.execution.MarketExecutor` directly.

- [ ] **Step 2: Move maintained compatibility state into `MarketExecutor`**

Add in `__init__`:

```python
self._compatibility_order_book = OrderBookState.empty()
self._compatibility_last_book: BookState | None = None
```

Add the `compatibility_order_book` property and `_reset_compatibility_chain()` helper. Extend `reset_random_state()` to reset both RNG and compatibility state.

- [ ] **Step 3: Replace `execute_interval()`**

Use the existing maintained algorithm from `StatefulCompatibilityMarketExecutor` without semantic edits: choose the chained order book only when the exact previous returned book is passed, derive `compatibility_target_execution_v1` identity with `content_digest`, call `execute_target_statefully`, persist returned state, detect attempted event types, and project `StatefulExecutionResult` into `ExecutionResult`.

- [ ] **Step 4: Remove only superseded vector-target code**

Delete `_target_weights()` and the old `execute_interval()` body. Keep `_FillResult`, `_fill_toward_quantities()`, rounding, borrow, capacity, and carry helpers because `liquidate_at_close()` still consumes them. Remove only coverage exclusions that reference deleted source.

- [ ] **Step 5: Convert adapter and package facade to aliases**

`execution_adapter.py` must contain only:

```python
from trade_rl.simulation.execution import MarketExecutor

StatefulCompatibilityMarketExecutor = MarketExecutor

__all__ = ["StatefulCompatibilityMarketExecutor"]
```

`simulation/__init__.py` imports `MarketExecutor` from `execution.py` directly.

- [ ] **Step 6: Run focused simulation verification**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_execution_adapter.py \
  tests/simulation/test_execution_v2.py \
  tests/simulation/test_execution_slippage_coverage.py \
  tests/simulation/test_stateful_execution.py \
  tests/simulation/test_stateful_execution_characterization.py \
  tests/architecture/test_direct_maintained_contract_exports.py
```

Expected: PASS with unchanged result/evidence values.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/simulation tests/simulation pyproject.toml tests/architecture/test_direct_maintained_contract_exports.py
git commit -m "refactor: make execution module own maintained executor"
```

### Task 3: Promote strict indexed telemetry to `training.py`

**Files:**
- Create: `trade_rl/telemetry/_indexed_storage.py`
- Modify: `trade_rl/telemetry/training.py`
- Modify: `trade_rl/telemetry/indexed_training.py`
- Modify: `trade_rl/telemetry/__init__.py`
- Modify: `tests/telemetry/test_training.py`
- Modify: `tests/telemetry/test_indexed_process_concurrency.py`
- Modify: `tests/telemetry/test_training_generation.py` when present
- Modify: `tests/architecture/test_direct_maintained_contract_exports.py`

**Interfaces:**
- Produces canonical `TrainingTelemetryRecord`, `TrainingTelemetryWriter`, `read_training_telemetry()`, and `training_telemetry_status()` in `training.py`; private standard-library-only storage mechanics; behavior-free legacy aliases.

- [ ] **Step 1: Add direct-module strictness tests**

Import directly from `trade_rl.telemetry.training` and verify non-boolean JSON flags are rejected, duplicate/process races are serialized, generation reset semantics are preserved, incomplete tails fail closed, and writer identity replacement is rejected.

- [ ] **Step 2: Make the canonical record strict**

In `TrainingTelemetryRecord.from_json_dict()`, require `emergency_deleverage`, `terminated`, and `truncated` to be present booleans. Remove truthiness coercion.

- [ ] **Step 3: Extract private indexed storage**

Move the current index schema, lock helpers, refresh/snapshot parsing, sparse seek, durable index write, binary append, and writer implementation from `indexed_training.py` into `_indexed_storage.py`. The private module may import the already-defined record/page/status contracts from `training.py`; it must import no non-standard-library project layer other than those contracts.

- [ ] **Step 4: Replace obsolete public storage bodies**

Delete the thread-only text writer and linear scan reader/status from `training.py`. After contract declarations, import the private writer base and define:

```python
class TrainingTelemetryWriter(_IndexedTrainingTelemetryWriter):
    """Canonical process-safe append-only telemetry writer."""
```

Define public wrappers with the maintained signatures, including `expected_generation` for reads.

- [ ] **Step 5: Convert package and legacy module to aliases**

`telemetry/__init__.py` imports all public names from `training.py`. `indexed_training.py` imports canonical names and assigns:

```python
StrictTrainingTelemetryRecord = TrainingTelemetryRecord
IndexedTrainingTelemetryWriter = TrainingTelemetryWriter
read_indexed_training_telemetry = read_training_telemetry
indexed_training_telemetry_status = training_telemetry_status
```

No class or function body remains in `indexed_training.py`.

- [ ] **Step 6: Run focused telemetry verification on Linux-compatible tests**

```bash
uv run pytest -q \
  tests/telemetry/test_training.py \
  tests/telemetry/test_indexed_process_concurrency.py \
  tests/studio/test_telemetry_api.py \
  tests/architecture/test_direct_maintained_contract_exports.py
```

Expected: PASS. Windows-native lock coverage remains required in final CI.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/telemetry tests/telemetry tests/studio tests/architecture/test_direct_maintained_contract_exports.py
git commit -m "refactor: make training module own indexed telemetry"
```

### Task 4: Promote strict Studio discovery to `studio/telemetry.py`

**Files:**
- Modify: `trade_rl/studio/telemetry.py`
- Modify: `trade_rl/studio/strict_telemetry.py`
- Modify: `trade_rl/studio/api.py`
- Modify: `tests/studio/test_telemetry_api.py`
- Modify: `tests/architecture/test_direct_maintained_contract_exports.py`

**Interfaces:**
- Produces canonical `StudioTelemetryReader` with fail-closed duplicate seed discovery and alias `StrictStudioTelemetryReader = StudioTelemetryReader`.

- [ ] **Step 1: Add direct-reader duplicate tests**

Instantiate `trade_rl.studio.telemetry.StudioTelemetryReader` directly. Create two distinct telemetry files claiming the same seed and assert `ArtifactInvalid("multiple telemetry streams claim seed <n>")`. Add a candidate symlink and assert it is ignored/rejected according to the maintained behavior.

- [ ] **Step 2: Integrate strict `_paths()`**

Replace `streams.setdefault(seed, resolved)` with explicit duplicate detection. Check `candidate.is_symlink()` before following the resolved path. Preserve namespace order, root containment, seed inference, and response behavior.

- [ ] **Step 3: Make API and compatibility module direct**

`studio/api.py` imports `StudioTelemetryReader` from `studio.telemetry`. `strict_telemetry.py` contains only an alias and `__all__`.

- [ ] **Step 4: Run focused Studio verification**

```bash
uv run pytest -q \
  tests/studio/test_telemetry_api.py \
  tests/architecture/test_direct_maintained_contract_exports.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/studio tests/studio tests/architecture/test_direct_maintained_contract_exports.py
git commit -m "refactor: make studio telemetry reader canonical"
```

### Task 5: Complete review, coverage, and exact-head evidence

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-23-direct-maintained-contract-exports.md`
- Modify: PR description

**Interfaces:**
- Produces complete `AUD-ARCH-001` closure evidence.

- [ ] **Step 1: Run focused source and behavior contracts**

```bash
uv run pytest -q \
  tests/architecture/test_direct_maintained_contract_exports.py \
  tests/architecture/test_architecture_followup.py \
  tests/simulation/test_stateful_execution_adapter.py \
  tests/telemetry \
  tests/studio/test_telemetry_api.py
```

- [ ] **Step 2: Run full exact-head verification**

Require GitHub Actions success for Studio tests/build/layout, workflow security, Ruff, format, Mypy, Import Linter, dead-code report, Serving smoke, full Pytest/branch coverage, critical coverage, CLI smoke, Ubuntu, Windows, and training-image non-root probe.

- [ ] **Step 3: Run PostgreSQL exact-head verification**

Require Compose validation, PostgreSQL startup/readiness, installation, migration, unit/integration tests, and cleanup on the same final head.

- [ ] **Step 4: Record evidence**

Document RED and GREEN commit SHAs, workflow IDs, test totals, coverage, compatibility identity assertions, and artifact IDs/digests in `docs/verification/2026-07-23-direct-maintained-contract-exports.md`.

- [ ] **Step 5: Review effective diff**

Confirm no compatibility module contains behavior, no obsolete public implementation remains, direct and package imports are identical, and no temporary workflow/script/artifact is included.

- [ ] **Step 6: Ready and merge**

After final exact-head success, mark the PR Ready and Squash merge. Production remains `NO-GO`.
