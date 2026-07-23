# Environment Runtime Services Extraction Design

## Status

Approved for implementation from the continuing `AUD-RL-001` remediation workflow.

## Problem

After the provider and observation-contract extractions, `ResidualMarketEnv.__init__()` still constructs eight already-separated runtime services inline:

- `EpisodeContractSampler`;
- `EnvironmentExecutionCoordinator`;
- `EnvironmentObservationAssembler`;
- `EnvironmentDecisionPlanner`;
- `EnvironmentRiskProjector`;
- `EnvironmentRewardCoordinator`;
- `EnvironmentInfoBuilder`;
- `EnvironmentTerminationCoordinator`.

The services themselves are focused and tested, but the environment facade still owns their complete dependency wiring. This keeps constructor density high and makes collaborator identity harder to review as one contract.

No runtime behavior defect has been reproduced. This work is a maintainability refactor only.

## Chosen approach

Create `trade_rl/rl/environment_runtime_services.py` with:

1. a frozen, slotted `EnvironmentRuntimeServices` dataclass containing the eight service instances;
2. an `EnvironmentRuntimeServicesBuilder` that receives only validated construction-time dependencies and builds the services in the existing order;
3. no access to mutable Gymnasium state such as books, order books, current indices, pending targets, episode seeds, diagnostics, or reset state.

`ResidualMarketEnv.__init__()` will invoke the builder once and assign the returned services to the existing private attributes. Attribute names and service types remain unchanged.

## Alternatives considered

### Leave inline wiring

This avoids code movement but leaves the remaining typed service graph hidden inside the facade and does not reduce `AUD-RL-001`.

### Split into multiple smaller builders

Separate episode/execution, decision/risk, and reward/info/termination builders would create more public construction concepts and cross-builder dependencies without a demonstrated need. It would also make the execution coordinator handoff to termination harder to review.

### Single typed bundle — selected

A single bundle reflects one existing responsibility: constructing the environment's already-separated runtime services from validated immutable collaborators. It minimizes API growth and preserves the current order.

## Builder inputs

The builder consumes:

- `MarketDataset` and `ResidualMarketEnvConfig`;
- the resolved minimum start index;
- the complete `EnvironmentObservationContract`;
- optional flat and sequence normalizers;
- resolved `ActionSpec`, composer, pre-trade risk, alpha mode, emergency monitor, portfolio-risk model, and portfolio-risk input provider;
- the shared reward tracker;
- the already-created hybrid and shadow executors.

The builder must not create provider, config, action, reward tracker, executor, observation-contract, or mutable environment state.

## Construction order

The builder preserves the current order exactly:

1. episode sampler;
2. execution coordinator;
3. observation assembler;
4. decision planner;
5. risk projector;
6. reward coordinator;
7. info builder;
8. termination coordinator.

This order is part of the characterization boundary because constructors may validate their inputs.

## Returned contract

`EnvironmentRuntimeServices` exposes these fields:

- `episode_sampler`;
- `execution_coordinator`;
- `observation_assembler`;
- `decision_planner`;
- `risk_projector`;
- `reward_coordinator`;
- `info_builder`;
- `termination_coordinator`.

The facade assigns them to the existing `_episode_sampler`, `_execution_coordinator`, `_observation_assembler`, `_decision_planner`, `_risk_projector`, `_reward_coordinator`, `_info_builder`, and `_termination_coordinator` attributes.

## Behavioral invariants

The extraction must preserve:

- service classes and public/private environment attribute names;
- object identity of every supplied collaborator;
- construction order and validation behavior;
- execution coordinator identity shared with the termination coordinator;
- reward tracker identity shared by reward coordinator, info builder, and termination coordinator;
- hybrid and shadow executor identity in the termination coordinator;
- observation-contract builder/layout/sequence objects used by the observation assembler;
- all existing environment digests, reset behavior, observations, rewards, execution, risk, and terminal accounting.

## Architecture controls

Architecture tests will require the new module and local class ownership, require one builder call in `ResidualMarketEnv.__init__()`, prohibit direct construction of the eight service classes in the facade constructor, and reduce the constructor source-span ceiling from 270 lines to 240 lines.

The extracted module will receive a permanent 100% critical branch-coverage ratchet after complete characterization.

## Testing strategy

TDD starts with tests that import the absent module and therefore fail for the intended reason.

Characterization tests will:

- build a normal environment and assert all eight service types and collaborator identities;
- build the bundle directly and assert observation, decision, risk, reward, information, execution, episode, and termination wiring;
- verify the execution coordinator is the exact object used by termination;
- verify invalid minimum index and invalid initial capital still surface from the same service constructors in the preserved order;
- retain the existing full environment, identity, risk, observation, execution, reward, and terminal suites.

## Non-goals

- no provider, action-spec, config, reward-tracker, executor, observation-contract, or mutable-state extraction;
- no constructor signature change;
- no new policy, validation rule, error message, or digest field;
- no exchange-realism, profitability, operational-readiness, or production claim.

## Audit disposition

`AUD-RL-001` remains `OPEN RISK, FURTHER REDUCED`. After this extraction, the principal remaining constructor density will be config/action/reward/executor construction and mutable Gymnasium-state initialization.

Production remains `NO-GO`.