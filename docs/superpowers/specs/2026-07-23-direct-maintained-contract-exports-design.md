# Direct Maintained Contract Exports Design

Date: 2026-07-23

## Problem

The repository exposes hardened maintained behavior through non-local aliases:

- `trade_rl.simulation.MarketExecutor` points to `execution_adapter.StatefulCompatibilityMarketExecutor`, while `trade_rl.simulation.execution.MarketExecutor` still owns a superseded vector-style `execute_interval()` body.
- `trade_rl.telemetry.TrainingTelemetryRecord`, `TrainingTelemetryWriter`, `read_training_telemetry()`, and `training_telemetry_status()` point to strict/indexed implementations in `indexed_training.py`, while direct imports from `telemetry.training` still expose older parsing, writer, and scan implementations.
- `trade_rl.studio.api` imports `StrictStudioTelemetryReader` from `strict_telemetry.py`, while direct imports from `studio.telemetry` expose a reader that does not reject duplicate seed streams.

This is `AUD-ARCH-001`. Static review of the declared public modules sees obsolete or weaker implementations, direct-module imports can bypass the maintained boundary, and future refactors can accidentally select different behavior according to import path.

## Goal

Make each declared public module the unique maintained owner of its public behavior while preserving all current import paths and runtime semantics.

## Chosen approach

Use direct public ownership with behavior-free compatibility modules.

### Simulation

`trade_rl.simulation.execution.MarketExecutor` becomes the maintained stateful compatibility executor itself.

- Move compatibility order-book state, chain reset, target identity, stateful target execution, and `ExecutionResult` projection from `execution_adapter.py` into `MarketExecutor`.
- Replace the superseded vector-style `execute_interval()` body rather than layering another subclass over it.
- Keep helper methods still required by `liquidate_at_close()` and the stateful order engine.
- Delete only helpers and coverage exclusions whose sole owner was the superseded `execute_interval()` path.
- Make `trade_rl.simulation.__init__` import `MarketExecutor` directly from `execution.py`.
- Retain `execution_adapter.StatefulCompatibilityMarketExecutor` as an explicit compatibility alias to the same `MarketExecutor` object, with no class body or behavior.

### Telemetry

`trade_rl.telemetry.training` becomes the unique public owner of strict records, indexed/process-safe writing, indexed reading, status, and stream-generation semantics.

- Integrate strict boolean validation into `TrainingTelemetryRecord.from_json_dict()`.
- Keep record/page/status contracts in `training.py`.
- Move sparse-index, process-lock, snapshot, and durable append internals into private module `trade_rl.telemetry._indexed_storage`.
- Define public `TrainingTelemetryWriter` in `training.py` as the public class over the private storage implementation.
- Define public `read_training_telemetry()` and `training_telemetry_status()` in `training.py` as the only maintained entry points, including `expected_generation` support.
- Remove the obsolete thread-only writer and linearly scanning public reader/status bodies.
- Make `telemetry.__init__` import only from `training.py`.
- Retain `indexed_training.py` as a behavior-free compatibility module whose old names alias the public objects from `training.py`.

The private storage module remains standard-library-only so the enforced telemetry layer is unchanged.

### Studio

`trade_rl.studio.telemetry.StudioTelemetryReader` becomes the unique maintained reader.

- Integrate duplicate seed-stream rejection and candidate-symlink rejection into `_paths()`.
- Make `studio.api` import the reader directly from `studio.telemetry`.
- Retain `strict_telemetry.StrictStudioTelemetryReader` as an explicit alias to `StudioTelemetryReader`, with no subclass body.
- Do not change the unrelated lazy exports in `studio.__init__`.

## Public compatibility invariants

The following imports remain valid and resolve to identical maintained objects:

```python
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import MarketExecutor as DirectMarketExecutor
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor

assert MarketExecutor is DirectMarketExecutor
assert StatefulCompatibilityMarketExecutor is DirectMarketExecutor
```

```python
from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter
from trade_rl.telemetry.training import (
    TrainingTelemetryRecord as DirectRecord,
    TrainingTelemetryWriter as DirectWriter,
)
from trade_rl.telemetry.indexed_training import (
    StrictTrainingTelemetryRecord,
    IndexedTrainingTelemetryWriter,
)

assert TrainingTelemetryRecord is DirectRecord is StrictTrainingTelemetryRecord
assert TrainingTelemetryWriter is DirectWriter is IndexedTrainingTelemetryWriter
```

```python
from trade_rl.studio.telemetry import StudioTelemetryReader
from trade_rl.studio.strict_telemetry import StrictStudioTelemetryReader

assert StudioTelemetryReader is StrictStudioTelemetryReader
```

The canonical public classes report their declared modules:

- `MarketExecutor.__module__ == "trade_rl.simulation.execution"`
- `TrainingTelemetryRecord.__module__ == "trade_rl.telemetry.training"`
- `TrainingTelemetryWriter.__module__ == "trade_rl.telemetry.training"`
- `StudioTelemetryReader.__module__ == "trade_rl.studio.telemetry"`

## Behavioral invariants

No behavior or evidence contract changes.

- `execute_interval()` keeps the exact stateful compatibility-chain behavior, target identity payload, attempted-order metric projection, reset behavior, exception messages, and `ExecutionResult` fields.
- `execute_orders()` and `liquidate_at_close()` are unchanged in semantics.
- Telemetry JSONL remains `training_telemetry_v1`; index remains `training_telemetry_index_v2`.
- Strict JSON booleans, process serialization, incomplete-tail fail-closed behavior, inode replacement detection, generation reset, sparse seeking, and status values remain unchanged.
- Studio artifact-root containment, seed selection, duplicate-seed rejection, symlink rejection, generation forwarding, and response schemas remain unchanged.
- Existing package-level imports remain source-compatible.

## Architecture constraints

- No public behavior is selected by package import side effects.
- Public modules contain the canonical class/function declarations.
- Compatibility modules may import and alias canonical objects but may not define subclasses, wrappers, storage logic, or alternate method bodies.
- There is one maintained `execute_interval()` implementation.
- There is one maintained telemetry record parser, writer, reader, and status implementation.
- There is one maintained Studio telemetry stream discovery implementation.
- No new dependency crosses an Import Linter boundary.

## Error handling

All existing fail-closed behavior is preserved.

- Invalid telemetry booleans remain errors rather than truthiness coercion.
- Duplicate telemetry sequences, corrupt/incomplete tails, stream replacement, and stale generations remain errors or reset responses as currently specified.
- Duplicate Studio seed identities remain `ArtifactInvalid`.
- Simulation validation and execution exceptions retain exact messages where tests already define them.

## Testing strategy

1. Add an architecture RED contract proving current public objects are not declared by their public modules and compatibility modules still own behavior.
2. Add direct-import parity tests for simulation, telemetry, and Studio.
3. Reuse existing stateful compatibility, telemetry concurrency/generation, Studio API, Windows, and PostgreSQL suites.
4. Add source contracts that compatibility modules contain aliases only and superseded public bodies are absent.
5. Measure branch coverage after obsolete code removal and adjust only thresholds supported by the final exact-head result; no threshold may be lowered merely to make the migration pass.

## Non-goals

- No redesign of the stateful order engine.
- No telemetry schema or index schema migration.
- No Studio API or frontend change.
- No removal of documented compatibility import paths.
- No direct exchange routing, profitability claim, or production-readiness change.

Production remains `NO-GO`.
