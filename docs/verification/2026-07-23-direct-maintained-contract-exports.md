# Direct Maintained Contract Exports Verification

Date: 2026-07-23

## Scope

This verification closes `AUD-ARCH-001`: hardened public behavior was selected through package-level aliases while the declared public modules still exposed superseded or weaker implementations.

The remediation makes these modules the canonical maintained owners:

- `trade_rl.simulation.execution` owns `MarketExecutor` and the stateful compatibility `execute_interval()` behavior;
- `trade_rl.telemetry.training` owns the strict record, process-safe indexed writer, indexed reader, status, and generation contracts;
- `trade_rl.studio.telemetry` owns the fail-closed Studio telemetry reader.

Former adapter/strict modules remain source-compatible but contain behavior-free aliases only.

Design:

- `docs/superpowers/specs/2026-07-23-direct-maintained-contract-exports-design.md`

Implementation plan:

- `docs/superpowers/plans/2026-07-23-direct-maintained-contract-exports.md`

## Canonical identity contract

The final implementation proves:

```python
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import MarketExecutor as DirectMarketExecutor
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor

assert MarketExecutor is DirectMarketExecutor
assert StatefulCompatibilityMarketExecutor is DirectMarketExecutor
assert DirectMarketExecutor.__module__ == "trade_rl.simulation.execution"
```

```python
from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter
from trade_rl.telemetry.training import (
    TrainingTelemetryRecord as DirectRecord,
    TrainingTelemetryWriter as DirectWriter,
)
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
)

assert TrainingTelemetryRecord is DirectRecord is StrictTrainingTelemetryRecord
assert TrainingTelemetryWriter is DirectWriter is IndexedTrainingTelemetryWriter
assert DirectRecord.__module__ == "trade_rl.telemetry.training"
assert DirectWriter.__module__ == "trade_rl.telemetry.training"
```

```python
from trade_rl.studio.telemetry import StudioTelemetryReader
from trade_rl.studio.strict_telemetry import StrictStudioTelemetryReader

assert StudioTelemetryReader is StrictStudioTelemetryReader
assert StudioTelemetryReader.__module__ == "trade_rl.studio.telemetry"
```

## Implemented boundaries

### Simulation

- The maintained stateful compatibility chain now belongs directly to `execution.MarketExecutor`.
- Compatibility order-book state, target identity, stateful target execution, reset behavior, and result projection were moved without semantic changes.
- The superseded vector-style `execute_interval()` and `_target_weights()` helper were removed.
- `liquidate_at_close()` support helpers remain intact.
- `execution_adapter.StatefulCompatibilityMarketExecutor` is now an alias to the canonical class.
- `simulation.__init__` imports the canonical class directly.

### Telemetry

- Strict JSON boolean validation belongs directly to `TrainingTelemetryRecord.from_json_dict()`.
- Public writer/read/status entry points belong to `telemetry.training`.
- Sparse-index, OS-lock, durable append, stream identity, snapshot, and generation mechanics moved to private standard-library-only `_indexed_storage.py`.
- `indexed_training.py` contains compatibility aliases only.
- JSONL remains `training_telemetry_v1`; sparse index remains `training_telemetry_index_v2`.
- Coverage ownership moved from the alias module to `_indexed_storage.py` without lowering the existing `69.0%` threshold.

### Studio

- Duplicate seed-stream rejection and candidate-symlink rejection now belong directly to `StudioTelemetryReader._paths()`.
- `studio.api` imports the canonical reader directly.
- `strict_telemetry.StrictStudioTelemetryReader` is an alias only.

## TDD RED evidence

RED head:

- `cd984c63fef3bcbd823a62ebecd3f4e92e411d37`

GitHub Actions CI run:

- `29963628770`

The run failed only the new direct-ownership architecture contract:

- package and direct `MarketExecutor` identities differed;
- telemetry package and direct record/writer identities differed;
- Studio strict and direct reader identities differed;
- compatibility modules still defined behavior-bearing classes;
- package facades selected adapter modules;
- direct `execution.MarketExecutor` still used the superseded interval body;
- direct `telemetry.training` lacked indexed generation behavior;
- direct Studio reader lacked duplicate seed rejection.

Result:

- `8 failed, 1223 passed, 2 skipped, 11 warnings`.

RED artifacts:

- Pytest diagnostics `8547038367`, digest `sha256:ddda0976504760df9471eaa85ddbceb1e8277fca4d1f5a18d3ce392d0d57cd7e`;
- architecture diagnostics `8547005018`, digest `sha256:a4b659e5e72a1b3c3c86ec890afaebce62736db3a9a7b054f35bbe8220fca67a`;
- static diagnostics `8547004616`, digest `sha256:ebacef9c2983174348c1ddb7b7fe2369db7747580dd66aa79a7afb9961cbb6ad`;
- training-image evidence `8546997428`, digest `sha256:cab8ff97518588a814a59702d9e80b7d9655c998570f77ad979a09fc5a652613`;
- Studio layout diagnostics `8546995871`, digest `sha256:bd2959e3ebd59dfa1de9a39aa1b9ea605c65671c6bee01b201aa523c009ed99b`;
- Windows compatibility `8546981062`, digest `sha256:6db6637a46829d5a59b4d1b28dc49f000f0e22cc5d6c26baba0ea37b4e6b7620`;
- Ubuntu compatibility `8546979293`, digest `sha256:78d9b8e369f22bf922e11079023e57270aac77c23be746f9366746eaed788407`.

## Implementation exact-head verification

Final implementation head before this documentation-only commit:

- `3a72914b136c53683d7f81f9ad715703d79a0d44`

GitHub Actions CI run `29966158073`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed viewport verification: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter and architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1236 passed, 2 skipped, 11 warnings`;
- total coverage: `83.78%`;
- total branch coverage: `4854 / 6862 = 70.74%`;
- `_indexed_storage.py`: `75 / 104 = 72.12% >= 69.0%`;
- `execution.py`: `90 / 100 = 90.0%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility, including native telemetry locking: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29966158104`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

Implementation exact-head artifacts:

- Pytest diagnostics `8547963531`, digest `sha256:f27e24eb603967c952a1000ab262b47d0f3bb784feef47ee35b4ba2608bbf8e4`;
- architecture diagnostics `8547934022`, digest `sha256:1c62988091e88bf89fb52f55564615e22bfd22fb2136b34fdbd2376aa931d520`;
- static diagnostics `8547933488`, digest `sha256:5c56bd83c1c1c66bbb873b26982ad32bc5933a643a708ccfeab0fb85164d2bfa`;
- Studio layout diagnostics `8547925413`, digest `sha256:4a5698c0b4e706f6c74b258c3afa505e4db2699a3e082f8cff92e5a7da4fd051`;
- training-image evidence `8547923441`, digest `sha256:8c9eaab0adcd4edbe83193a28a743abca668031d986c4bc445639c2bba81398d`;
- Windows compatibility `8547919518`, digest `sha256:8f41ddad5931a32b822afc5870a476a82ccff64f356b2e7433ba82a7f1390d53`;
- Ubuntu compatibility `8547912402`, digest `sha256:f5eb007fe6663599b13f58f00c72928c88035e5f4e6b4640720182bfd3558f7a`.

## Review result

The effective implementation diff contains no temporary workflow, trigger, patch payload, source export, or generated status file. Compatibility modules contain aliases only; canonical public modules own the maintained class/function declarations and behavior.

Simulation result/evidence contracts, telemetry JSON/index schemas, process and generation semantics, Studio API schemas, and fail-closed error behavior remain unchanged.

No critical or important review issue remains.

This verification file is documentation-only. The final documentation-complete head must pass exact-head CI before merge.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No profitability or exchange-equivalent fill claim is introduced.
- No primary evidence schema changes.
