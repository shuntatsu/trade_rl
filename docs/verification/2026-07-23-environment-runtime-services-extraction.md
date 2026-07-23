# Environment Runtime Services Extraction Verification — 2026-07-23

## 1. Scope

This verification covers the behavior-preserving extraction of the eight existing
runtime-service constructor calls from `ResidualMarketEnv.__init__()` into the typed
`EnvironmentRuntimeServicesBuilder` and frozen `EnvironmentRuntimeServices` bundle.

The extracted boundary constructs:

- `EpisodeContractSampler`;
- `EnvironmentExecutionCoordinator`;
- `EnvironmentObservationAssembler`;
- `EnvironmentDecisionPlanner`;
- `EnvironmentRiskProjector`;
- `EnvironmentRewardCoordinator`;
- `EnvironmentInfoBuilder`;
- `EnvironmentTerminationCoordinator`.

Provider resolution, portfolio-risk provider validation, environment configuration,
action specification, reward-tracker creation, executor creation, observation-contract
construction, and mutable Gymnasium state remain outside this boundary.

Production remains `NO-GO`. This is a maintainability remediation and does not
establish profitability, exchange-equivalent execution, operational authorization,
or direct venue connectivity.

## 2. TDD evidence

The architecture and wiring-characterization tests were committed before production
implementation.

The clean RED commit was:

- commit: `5199118ecb77e1fd08934e8887e1b1a621735fa7`;
- CI run: `29976735374`.

At that commit Ruff, Ruff formatting, Mypy, Import Linter, the maintained serving
smoke checks, Ubuntu compatibility, Windows compatibility, and training-image
validation passed. Complete pytest collection then failed because
`trade_rl.rl.environment_runtime_services` did not exist.

This demonstrated that the new ownership tests detected the missing boundary rather
than merely confirming a completed implementation.

## 3. Implemented boundary

`trade_rl/rl/environment_runtime_services.py` now owns:

- the frozen, slotted `EnvironmentRuntimeServices` contract;
- `EnvironmentRuntimeServicesBuilder`;
- the existing service-construction order;
- explicit collaborator handoff for episode, execution, observation, decision, risk,
  reward, information, and termination services;
- the shared execution-coordinator identity used by termination;
- the shared reward-tracker identity used by reward, information, and termination;
- the hybrid and shadow executor identities used by termination.

The builder receives only already-validated construction dependencies. It does not
receive books, order books, current indices, pending targets, episode seeds,
diagnostics, reset state, or other mutable Gymnasium state.

`ResidualMarketEnv.__init__()` invokes the builder once and assigns the returned
services to the existing private attributes:

- `_episode_sampler`;
- `_execution_coordinator`;
- `_observation_assembler`;
- `_decision_planner`;
- `_risk_projector`;
- `_reward_coordinator`;
- `_info_builder`;
- `_termination_coordinator`.

The environment retains the `EnvironmentExecutionCoordinator` import because two
existing static compatibility helpers delegate to its maintained liquidation
operations. Direct coordinator construction is nevertheless owned by the new builder.

## 4. Constructor reduction and architecture control

Before this extraction, the constructor source span was 262 lines after PR #120.
It is now 232 lines, a reduction of 30 constructor lines.

Across `trade_rl/rl/environment.py`, the extraction adds 19 lines and removes 61
lines, for a net reduction of 42 source lines.

Architecture tests now:

- require local ownership of the runtime-service contract and builder;
- require one builder invocation in the environment constructor;
- prohibit direct construction of the eight services in the facade constructor;
- require the builder to preserve the existing construction order;
- verify the four step-service classes in the new wiring owner rather than requiring
  their names to remain in the facade source;
- preserve step delegation and terminal-info ownership checks;
- limit the constructor source span to 240 lines.

## 5. Characterization coverage

The tests cover:

- all eight service types exposed by a constructed environment;
- dataset, config, minimum-start-index, action-spec, composer, risk, normalizer,
  observation-contract, reward-tracker, and executor identities;
- the exact execution coordinator shared with termination;
- the exact reward tracker shared by reward, information, and termination services;
- the exact hybrid and shadow executors used by termination;
- direct builder construction returning one typed bundle;
- construction-order ownership through the architecture contract;
- the complete maintained `tests/rl` behavior suite.

The extracted module measured:

- 61 / 61 statements covered;
- no executable branch points;
- 100.0% statement and branch coverage by the repository's coverage policy.

A permanent 100.0% critical branch-coverage ratchet is recorded in
`pyproject.toml`.

## 6. Exact-head verification

The implementation and coverage-ratchet exact head was:

- commit: `e0290153acc60fbc1d7b90267e4122b0bc29bcba`;
- CI run: `29978466879`;
- PostgreSQL Catalog run: `29978466883`.

The CI run passed:

- Studio frontend and fixed-viewport verification;
- workflow-security checks;
- Ruff and Ruff formatting;
- Mypy;
- Import Linter;
- dead-code reporting;
- recovery and structured-serving smoke;
- complete pytest and coverage;
- critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- complete training-image build and non-root runtime probe.

The complete test result was:

- 1,285 passed;
- 2 skipped;
- 11 warnings;
- 84.10% total coverage;
- 71.17% total branch coverage.

The PostgreSQL Catalog run passed Compose validation, PostgreSQL startup and
readiness, migrations, catalog/unit/integration tests, and cleanup on the same exact
head.

## 7. Architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral
defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Runtime-service wiring is now typed, independently tested, fully covered, and
protected against returning inline to the environment constructor.

The remaining construction density consists primarily of:

- portfolio-risk provider selection and identity validation;
- environment config, emergency-risk, action-spec, and episode-schedule validation;
- reward-tracker and market-executor construction;
- mutable Gymnasium book, order, observation, and episode-state initialization.

Those concerns should not be mechanically combined or split without another
behavior-preserving seam and characterization evidence.

This item does not block causal research use. Production remains `NO-GO` until the
maintained research, evidence, operational, authorization, and profitability gates
pass independently.
