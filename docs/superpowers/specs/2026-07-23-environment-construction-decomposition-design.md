# Environment Construction Decomposition Design

Date: 2026-07-23

## Problem

`ResidualMarketEnv` is already a bounded Gymnasium facade for step-time behavior, but its constructor still owns provider resolution, artifact identity validation, risk dependency selection, reward timing, action-contract validation, observation-space construction, service assembly, and initial mutable-state creation. This concentration is not a reproduced behavioral defect, but it is the remaining `AUD-RL-001` maintainability risk because future observation, risk, or action features can force unrelated constructor edits.

## Goals

- Preserve the complete public `ResidualMarketEnv(...)` constructor signature.
- Preserve reset, step, observation, information, digest, schema, and error behavior.
- Keep all episode-varying mutable Gymnasium state owned by `ResidualMarketEnv`.
- Make construction responsibilities independently understandable and testable.
- Reduce `ResidualMarketEnv.__init__` to bounded orchestration rather than moving the same monolith into one new file.
- Preserve Linux and Windows behavior and introduce no new dependency.

## Non-goals

- Do not change action semantics, risk projection, reward calculation, execution order, sampling, observation values, or episode state transitions.
- Do not change JSON, artifact, environment, action, observation, execution, or telemetry schemas.
- Do not add direct exchange routing or production authorization.
- Do not replace the existing step-time services extracted by PRs #79, #92, and #107.
- Do not expose a new public builder API.

## Considered approaches

### Private helper methods on `ResidualMarketEnv`

This minimizes new files but leaves provider, observation, and assembly policy inside the same class. The line count improves without improving ownership, so future changes still converge on `environment.py`.

### Mutable builder object

A builder can stage construction, but a long-lived mutable builder introduces ordering constraints and a second mutable lifecycle. It is difficult to prove which intermediate fields are valid and adds little value for a one-shot constructor.

### Typed construction services — selected

Four focused services accept immutable request dataclasses and return explicit result dataclasses. They do not retain cross-construction state and do not mutate the environment. `ResidualMarketEnv` applies the returned values and remains the sole owner of episode-varying mutable state.

## Architecture

### `environment_dependencies.py`

`EnvironmentDependencyResolver` owns construction-time policy that is independent of Gymnasium spaces:

- trend and market-input-resolver reconciliation;
- alpha enablement and artifact digest resolution;
- static or provider factor-basis validation and factor count;
- provider minimum-index validation;
- pre-trade, portfolio-risk, and risk-input-provider selection;
- environment-config validation that links risk and leverage;
- action-spec construction and validation;
- nominal episode/decision timing;
- reward tracker creation and full-preroll minimum index.

It returns `EnvironmentDependencies`, an immutable dataclass containing resolved objects, identities, timing, action names, and minimum start index.

### `environment_observation_contract.py`

`EnvironmentObservationContractFactory` owns flat and structured observation construction:

- `ObservationBuilder` and layout creation;
- flat normalizer identity/schema/passthrough validation;
- sequence builder, sequence-normalizer validation, policy-plane creation, and sequence minimum index;
- sequence component spaces and layout metadata;
- observation schema and contract digest;
- public observation and action spaces.

It returns `EnvironmentObservationContract`. The factory receives the already-resolved action identity rather than importing or mutating the environment.

### `environment_assembly.py`

`EnvironmentServiceAssembler` creates construction-stable collaborators:

- emergency risk monitor;
- hybrid and shadow executors;
- episode sampler;
- execution, observation, decision, risk, reward, information, and termination services.

It returns `EnvironmentServiceAssembly`. It composes the existing maintained services and contains no step-time policy.

### `environment_state.py`

`EnvironmentInitialStateFactory` creates the deterministic initial mutable values:

- initial indices and episode metadata;
- zero book and cloned shadow book;
- pending targets and empty order books;
- previous action, position age, execution state, diagnostics, and reset flag;
- reward-history cache.

It returns `EnvironmentInitialState`. `ResidualMarketEnv` assigns these values explicitly and remains their owner after construction.

## Facade flow

`ResidualMarketEnv.__init__` performs only these phases, in order:

1. store the dataset and raw optional normalizers/stress configuration;
2. resolve construction dependencies;
3. assign resolved dependency fields;
4. build and assign the observation contract;
5. assemble and assign maintained runtime services;
6. compute the unchanged environment digest from the fully assigned fields;
7. create and assign initial mutable state.

The constructor must not directly instantiate the low-level provider, space, executor, or service collaborators owned by the four construction modules.

## Compatibility and invariants

- `ResidualMarketEnv`, `AlphaProvider`, and `FactorBasisProvider` remain importable from `trade_rl.rl.environment`.
- `ResidualMarketEnv.__module__` and the constructor signature remain unchanged.
- `environment_digest`, `action_spec_digest`, and `observation_contract_digest` remain byte-for-byte equal for equivalent inputs.
- Existing `ValueError` and `RuntimeError` messages remain unchanged.
- Flat and structured observation-space keys, shapes, bounds, and dtypes remain unchanged.
- Hybrid and shadow executors remain separate instances with the same policy digest.
- Existing service request/result ordering and all mutable-state application remain unchanged.
- Construction services retain no global, persisted, or cross-environment mutable state.

## Test strategy

### Architecture RED

Add an AST/source contract that initially fails because the four modules do not exist and `ResidualMarketEnv.__init__` exceeds the agreed orchestration span. The final contract requires:

- all four construction modules and owner classes;
- immutable request/result dataclasses;
- `ResidualMarketEnv.__init__` at or below 180 source lines;
- explicit delegation to dependency resolution, observation construction, service assembly, and initial-state creation;
- absence of low-level construction symbols from the constructor body.

### Characterization

A deterministic fixture records constructor-visible contracts before extraction and compares after extraction:

- environment, action, observation, and execution-policy digests;
- minimum start index, episode/decision timing, action names;
- flat and structured observation-space signatures;
- initial book/order/pending/action state;
- first seeded reset observation and reset information;
- one seeded step result and post-step mutable state.

### Focused service tests

Each construction service receives direct tests for its success path and fail-closed validation branches. Existing environment, serving-parity, stateful replay, target action, sequence observation, risk, timing, and reset suites remain the primary behavior regression net.

### Coverage

Add a measured aggregate branch-coverage ratchet for the four construction modules. Set the threshold from exact-head observed coverage and never lower an existing threshold.

## Documentation and safety

Record RED, characterization, focused GREEN, exact-head CI, PostgreSQL integration, coverage, artifact IDs, and review findings in a verification document. The final audit closeout may mark `AUD-RL-001` resolved only after the documentation-complete head passes exact-head CI.

Production remains `NO-GO`; direct exchange routing and profitability claims remain out of scope.