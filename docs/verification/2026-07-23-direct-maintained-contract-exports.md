# Direct Maintained Contract Exports Verification — 2026-07-23

## 1. Scope

This verification records the final `AUD-ARCH-001` ownership remediation in PR #111.

PR #79 removed import-time `setattr` mutation and made maintained consumers import explicit facades. A remaining discoverability problem still existed: package imports and direct public-module imports could resolve to different class objects, while compatibility modules continued to own behavior-bearing subclasses or duplicate implementations.

PR #111 establishes one canonical public owner for each maintained contract:

- `trade_rl.simulation.execution.MarketExecutor`;
- `trade_rl.telemetry.training.TrainingTelemetryRecord`;
- `trade_rl.telemetry.training.TrainingTelemetryWriter`;
- `trade_rl.telemetry.training.read_training_telemetry` and `training_telemetry_status`;
- `trade_rl.studio.telemetry.StudioTelemetryReader`.

The former public paths remain valid but are behavior-free aliases:

- `trade_rl.simulation.execution_adapter.StatefulCompatibilityMarketExecutor`;
- `trade_rl.telemetry.indexed_training.StrictTrainingTelemetryRecord`;
- `trade_rl.telemetry.indexed_training.IndexedTrainingTelemetryWriter`;
- `trade_rl.telemetry.indexed_training.read_indexed_training_telemetry`;
- `trade_rl.telemetry.indexed_training.indexed_training_telemetry_status`;
- `trade_rl.studio.strict_telemetry.StrictStudioTelemetryReader`.

Package facades and Studio API consumers import the canonical public modules directly.

Production remains `NO-GO`. This change does not add exchange routing, alter profitability evidence, or authorize deployment.

## 2. Architecture boundary

### Simulation

The maintained stateful compatibility behavior now lives directly in `trade_rl.simulation.execution.MarketExecutor`.

- package import, direct execution import, and the legacy adapter name resolve to the same class object;
- `execute_interval()` delegates to `execute_target_statefully()`;
- compatibility order-book state is kept only when the caller chains the exact returned `BookState` through the same executor;
- unrelated books start a fresh compatibility chain;
- random-state reset also clears the compatibility chain;
- public `ExecutionResult`, stateful evidence, cost, capacity, accounting, and termination behavior are unchanged.

The obsolete private `_target_weights()` implementation and its coverage exclusions were removed rather than retained behind another alias.

### Telemetry

The canonical record and public storage API now live in `trade_rl.telemetry.training`.

- JSONL remains `training_telemetry_v1`;
- the rebuildable sparse index remains `training_telemetry_index_v2`;
- strict boolean parsing remains fail-closed;
- process locks, append-only writes, partial-write handling, incomplete-tail rejection, inode replacement rejection, generation-bound cursor reset, bounded snapshots, and atomic index replacement are preserved;
- the low-level indexed implementation is private in `trade_rl.telemetry._indexed_storage`;
- `indexed_training.py` contains aliases only.

The critical branch-coverage owner was moved from the alias-only module to `_indexed_storage.py`. Public writer lifecycle and `flush_every` validation tests raise measured branch coverage to `75 / 104 = 72.12%`; the configured threshold is `72.1%`.

### Studio

`StudioTelemetryReader` is declared directly in `trade_rl.studio.telemetry`.

- the strict compatibility name resolves to the same object;
- Studio API imports the direct reader;
- telemetry discovery rejects symlink candidates and multiple distinct files claiming the same seed;
- environment, episode, process-lock, and stream-generation protections remain unchanged.

## 3. TDD RED evidence

RED head:

`cd984c63fef3bcbd823a62ebecd3f4e92e411d37`

GitHub Actions CI run `29963628770` failed as intended only after static, compatibility, and training-image checks reached the new architecture contracts.

The full suite result was:

```text
8 failed, 1223 passed, 2 skipped, 11 warnings
coverage: 83.56%
```

The eight failures demonstrated the exact ownership defect:

1. package `MarketExecutor` was not the direct execution class;
2. package telemetry record was not the direct training record;
3. strict Studio reader was not the direct reader;
4. compatibility modules still declared classes or behavior;
5. package facades imported compatibility modules;
6. direct executor still owned the superseded interval body rather than stateful target delegation;
7. direct training telemetry lacked generation/indexed and strict-boolean contracts;
8. direct Studio reader lacked duplicate-seed rejection.

RED artifacts:

- Pytest diagnostics `8547038367`, digest `sha256:ddda0976504760df9471eaa85ddbceb1e8277fca4d1f5a18d3ce392d0d57cd7e`;
- architecture diagnostics `8547005018`, digest `sha256:a4b659e5e72a1b3c3c86ec890afaebce62736db3a9a7b054f35bbe8220fca67a`;
- static diagnostics `8547004616`, digest `sha256:ebacef9c2983174348c1ddb7b7fe2369db7747580dd66aa79a7afb9961cbb6ad`;
- training-image evidence `8546997428`, digest `sha256:cab8ff97518588a814a59702d9e80b7d9655c998570f77ad979a09fc5a652613`;
- Studio layout diagnostics `8546995871`, digest `sha256:bd2959e3ebd59dfa1de9a39aa1b9ea605c65671c6bee01b201aa523c009ed99b`;
- Windows compatibility `8546981062`, digest `sha256:6db6637a46829d5a59b4d1b28dc49f000f0e22cc5d6c26baba0ea37b4e6b7620`;
- Ubuntu compatibility `8546979293`, digest `sha256:78d9b8e369f22bf922e11079023e57270aac77c23be746f9366746eaed788407`.

## 4. Focused GREEN evidence

The migration was first checked against the architecture ownership contracts, the prior architecture follow-up contracts, the complete simulation suite, the complete telemetry suite, Studio telemetry API tests, and telemetry integration tests.

Focused run `29965313710`: success.

```text
193 passed, 1 warning
Ruff: passed
```

Focused diagnostics artifact:

- ID `8547615965`;
- digest `sha256:a0facb5f97d7af4072145a6300b000220b6e7e1f42fb0390c130a4d747d8bd7d`.

The focused runner used the same Studio, development, and training dependency groups as normal CI. Temporary migration and diagnostic workflows deleted themselves and are absent from the effective pull-request diff.

## 5. Code-head exact verification

Strengthened code-and-test head before the ratchet and this verification-note update:

`3a72914b136c53683d7f81f9ad715703d79a0d44`

GitHub Actions CI run `29966158073`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed viewport: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter and architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1236 passed, 2 skipped, 11 warnings`;
- total coverage: `83.78%`;
- total branch coverage: `4854 / 6862 = 70.74%`;
- canonical indexed storage branch coverage: `75 / 104 = 72.12%`;
- direct simulation executor branch coverage: `90 / 100 = 90.0%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29966158104`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup/readiness: passed;
- installation and migrations: passed;
- catalog unit and integration tests: passed;
- shutdown and cleanup: passed.

## 6. Code-head artifacts

- Pytest diagnostics: ID `8547963531`, digest `sha256:f27e24eb603967c952a1000ab262b47d0f3bb784feef47ee35b4ba2608bbf8e4`;
- architecture diagnostics: ID `8547934022`, digest `sha256:1c62988091e88bf89fb52f55564615e22bfd22fb2136b34fdbd2376aa931d520`;
- static diagnostics: ID `8547933488`, digest `sha256:5c56bd83c1c1c66bbb873b26982ad32bc5933a643a708ccfeab0fb85164d2bfa`;
- training-image evidence: ID `8547923441`, digest `sha256:8c9eaab0adcd4edbe83193a28a743abca668031d986c4bc445639c2bba81398d`;
- Studio layout diagnostics: ID `8547925413`, digest `sha256:4a5698c0b4e706f6c74b258c3afa505e4db2699a3e082f8cff92e5a7da4fd051`;
- Windows compatibility: ID `8547919518`, digest `sha256:8f41ddad5931a32b822afc5870a476a82ccff64f356b2e7433ba82a7f1390d53`;
- Ubuntu compatibility: ID `8547912402`, digest `sha256:f5eb007fe6663599b13f58f00c72928c88035e5f4e6b4640720182bfd3558f7a`.

## 7. Review result

- all maintained package, direct, and compatibility imports resolve to one class object per contract;
- compatibility modules contain no behavior-bearing class or public function implementation;
- simulation stateful compatibility behavior is not duplicated;
- telemetry strict parsing and process-safe indexed behavior are not duplicated;
- Studio duplicate-seed behavior is not duplicated;
- existing import paths remain source-compatible;
- telemetry primary and sidecar schemas are unchanged;
- branch coverage follows the private canonical indexed implementation rather than the alias module;
- no temporary workflow, trigger, status file, generated source copy, or obsolete implementation remains in the effective diff;
- no unresolved review thread remains.

The ratchet update and this verification note create a later exact head. Normal CI and PostgreSQL Catalog must pass again on that documentation-complete head before merge.

## 8. Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No model, training, selection, evaluation, promotion, release, Serving, accounting, or exchange behavior is intentionally changed outside the canonical ownership migration described above.
- No profitability or exchange-equivalent fill claim is introduced.
