# Environment Runtime Decomposition Design

## Context

`ResidualMarketEnv` is now a Gymnasium facade over dedicated episode, execution,
observation, and termination services. Its `step()` path still owns four distinct
responsibilities:

1. action parsing, composition, signal-delay planning, and decision-bar selection;
2. emergency, pre-trade, and portfolio risk projection;
3. reward transition calculation;
4. step and terminal `info` construction.

Keeping these responsibilities in the facade makes execution-sensitive changes
hard to review and forces tests to exercise a large mutable object even when only
one policy boundary changes.

## Approaches considered

### A. Private-method extraction only

Split `step()` into more private methods on `ResidualMarketEnv`.

This is the smallest diff, but it leaves the same ownership problem. The methods
would still read and mutate broad environment state and would not be independently
reusable or testable.

### B. Immutable request/result services

Introduce four focused services and immutable request/result dataclasses. Keep
all Gymnasium state mutation in `ResidualMarketEnv`, while each service receives
only the inputs required for one decision.

This is the selected approach. It establishes explicit contracts without changing
public schemas, action semantics, execution ordering, or mutable state ownership.

### C. Full step state machine

Represent the entire step lifecycle as a state machine with persisted phases.

This could support resumable execution later, but it is unnecessarily broad for
the current local research environment and would increase compatibility risk.

## Selected architecture

### `EnvironmentDecisionPlanner`

Owns:

- legacy and maintained action parsing;
- baseline/residual composition;
- submitted hybrid and shadow target creation;
- one-decision signal-delay resolution;
- decision-bar count calculation.

It returns an immutable `EnvironmentDecisionPlan` containing the parsed action,
maintained action vector, diagnostics inputs, submitted targets, executed targets,
next pending targets, warm-up status, and bar count.

It does not mutate environment pending targets or books.

### `EnvironmentRiskProjector`

Owns:

- proposal shape and finiteness validation;
- emergency-risk assessment;
- pre-trade risk constraint;
- causal advanced portfolio-risk input resolution;
- portfolio-risk projection;
- merged reason and projection-distance construction.

It returns the existing `RiskConstrainedTarget`. Dataset and risk models are
constructor dependencies; current index and book are explicit call inputs.

### `EnvironmentRewardCoordinator`

Owns one reward transition. It receives interval returns, current books,
projection distance, and termination flags, and delegates to `RewardTracker`.
It returns the existing reward breakdown. Reward-history pre-roll and reset remain
in the facade because they belong to episode initialization rather than step
transition planning.

### `EnvironmentInfoBuilder`

Owns the stable `info` contract:

- per-step action, execution, risk, reward, delay, and termination fields;
- optional discarded-target and liquidation fields;
- terminal performance metrics and action diagnostics.

It returns a new dictionary on every call and does not mutate books, reward state,
or diagnostics. Existing key names and value types remain unchanged.

### `ResidualMarketEnv`

Remains responsible for:

- Gymnasium `reset()` and `step()` methods;
- mutable books, order books, pending targets, indices, action history, position
  ages, and diagnostics;
- ordering the planner, risk, execution, termination, reward, observation, and
  info services;
- environment identity and compatibility properties.

The facade applies service results in the same order as the current implementation.

## Step data flow

1. Resolve market inputs.
2. Ask `EnvironmentDecisionPlanner` for an immutable decision plan.
3. Project hybrid and shadow executed targets through `EnvironmentRiskProjector`.
4. Execute both constrained targets through `EnvironmentExecutionCoordinator`.
5. Apply book, order-book, index, and pending-target state changes in the facade.
6. Resolve economic and time-limit termination.
7. Calculate reward through `EnvironmentRewardCoordinator`.
8. Update execution observation state, previous action, and diagnostics.
9. Build stable step/terminal information through `EnvironmentInfoBuilder`.
10. Return the next observation and unchanged Gymnasium tuple contract.

## Compatibility constraints

The implementation must not change:

- `ResidualMarketEnv` constructor or public properties;
- action, observation, reward, or environment schema versions;
- environment digest payload;
- target identity construction or execution order;
- signal-delay semantics;
- risk reason ordering;
- reward calculation order;
- terminal accounting behavior;
- any existing `info` key, optional-key condition, or value type;
- direct exchange routing or production readiness status.

Private compatibility methods may remain as thin delegates where tests or internal
callers still use them.

## Error handling

Services fail closed on invalid action vectors, target dimensions, non-finite
values, missing advanced risk inputs, exhausted episode intervals, and invalid
bar counts. No service catches or downgrades existing domain exceptions.

## Testing strategy

1. Add architecture contracts before production code. They require the four new
   services and delegation from `ResidualMarketEnv`.
2. Add focused unit tests for action migration, signal delay, irregular-calendar
   bar calculation, emergency/portfolio risk merging, reward input mapping, and
   optional/terminal `info` fields.
3. Run existing environment, simulation, serving-parity, telemetry, and training
   integration tests as regression evidence.
4. Run Ruff, format, Mypy, Import Linter, dead-code analysis, full Pytest with
   branch coverage, critical coverage, CLI smoke, compatibility jobs, training
   image build, and PostgreSQL Catalog workflow on the exact PR head.
5. Add a new branch-coverage ratchet for the extracted step services using the
   measured post-refactor coverage without reducing the existing environment
   runtime group threshold.

## Pull-request structure

This is a stacked Draft PR based on
`agent/fix-architecture-followup-20260722` because PR #79 is not yet merged. Once
PR #79 lands, this PR can be retargeted to `main` without carrying unrelated
changes.