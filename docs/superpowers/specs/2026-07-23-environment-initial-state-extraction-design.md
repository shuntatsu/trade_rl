# Environment Initial State Extraction Design

## Context

After PR #126, `ResidualMarketEnv.__init__()` is 186 source lines. Provider, portfolio-risk, policy/schedule, observation, and runtime-service construction already have typed owners. The remaining final constructor block directly creates invocation-local mutable Gymnasium state.

This change extracts only that final state creation. It does not reopen the broad dependency or service-assembly design from superseded PRs #115 and #121.

## Goal

Create a typed factory that returns fresh initial mutable values while preserving every existing environment attribute, type, value, identity relationship, and reset behavior.

## Chosen boundary

Create `trade_rl.rl.environment_initial_state` with:

- `EnvironmentInitialStateRequest`, a frozen slotted request containing `dataset`, `config`, `action_spec`, and `minimum_start_index`;
- `EnvironmentInitialState`, a frozen slotted assembly container containing the exact values assigned by the current constructor tail;
- `EnvironmentInitialStateFactory.create(request)`, which constructs a fresh state for one environment instance.

The state contains:

- `start_index`, `end_index`, and `current_index`;
- independent `hybrid` and `shadow` books;
- decision-step index, episode seed, episode hours, and initial-state mode;
- previous action and pending targets;
- independent hybrid and shadow order books;
- position age, observation execution state, action diagnostics, and reset flag.

`_reward_history_cache` remains outside this boundary. It belongs with reward-runtime resources rather than resettable episode state and will not be moved in this PR.

## Preserved behavior

The factory must preserve:

- `start_index == minimum_start_index`;
- `end_index == start_index + 1`;
- `current_index == start_index`;
- zero-quantity hybrid book at `dataset.close[start_index]` with configured initial capital and resolved contract multipliers;
- shadow as an independent clone of hybrid;
- `np.float32` previous-action zeros of `action_spec.size`;
- `np.float64` position-age zeros of `dataset.n_symbols`;
- independent empty order books;
- zero observation execution state;
- a fresh `ActionDiagnosticsAccumulator`;
- `has_reset is False`;
- no public constructor signature change;
- the existing environment digest and reset/step semantics.

Every factory invocation must return fresh mutable objects and arrays. The frozen dataclass freezes only field rebinding inside the assembly container; it does not claim that contained runtime objects are immutable.

## Facade integration

`ResidualMarketEnv.__init__()` calls the factory exactly once after the environment digest is computed, then assigns the returned fields to the existing private and public attributes. No consumer is changed to retain the assembly container.

The constructor must no longer directly call `BookState.zero`, `OrderBookState.empty`, `ObservationExecutionState.zero`, `ActionDiagnosticsAccumulator`, or `np.zeros` for initial-state fields. Its source span must be at most 170 lines.

## Alternatives rejected

1. Broad dependency and service assembly: rejected because it overlaps canonical owners and recreates the stale combined PR design.
2. Reward/executor extraction first: valid later, but less isolated because it changes minimum-index propagation and resource wiring.
3. No extraction: rejected because the final state block is a concrete, characterized responsibility and the user requested continued decomposition.

## Verification

TDD order:

1. Commit direct factory and architecture tests before the module exists.
2. Confirm complete pytest collection fails only because `trade_rl.rl.environment_initial_state` is missing.
3. Implement the minimal factory and one facade delegation.
4. Run focused environment/reset/identity tests, Ruff, formatting, Mypy, and Import Linter.
5. Run complete CI, Ubuntu/Windows compatibility, training-image/non-root probe, and PostgreSQL Catalog verification.
6. Record a permanent 100.0% critical branch-coverage ratchet for the new module.

Production remains `NO-GO`. This is a maintainability refactor and makes no profitability, execution-realism, authorization, or direct-exchange claim.
