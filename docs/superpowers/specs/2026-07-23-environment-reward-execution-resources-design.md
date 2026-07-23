# Environment Reward and Execution Resources Design

## Purpose

Extract the remaining reward-tracker, optional reward pre-roll, market-executor, compatibility-alias, and reward-history-cache construction from `ResidualMarketEnv.__init__()` into one typed, behavior-preserving boundary.

Production remains `NO-GO`. This work does not alter reward mathematics, execution behavior, observation construction, runtime-service wiring, reset behavior, step behavior, public constructor parameters, or exchange capability.

## Alternatives considered

### Leave the construction inline

The current code is behaviorally valid, but it leaves reward history policy and two stateful executors embedded in the Gymnasium facade. The constructor cannot become a bounded orchestration surface while this block remains inline.

### Move the block into the policy/schedule contract

The policy/schedule contract already owns immutable configuration, action layout, and decision timing. Adding mutable `RewardTracker`, `MarketExecutor`, and cache objects would mix deterministic policy resolution with invocation-local runtime resources.

### Move the block into runtime-service assembly

The runtime-service builder consumes an already-created reward tracker and two executors. Making it create those dependencies would hide construction order and couple resource ownership to the eight-service graph.

### Dedicated reward/execution resource boundary — selected

Create `trade_rl.rl.environment_reward_execution_resources` with a frozen, slotted `EnvironmentRewardExecutionResources` result and `EnvironmentRewardExecutionResourcesBuilder`. The builder receives already-resolved configuration values and creates only the existing runtime resources.

## Owned contract

`EnvironmentRewardExecutionResources` contains:

- `reward_tracker: RewardTracker`;
- `minimum_start_index: int`;
- `hybrid_executor: MarketExecutor`;
- `shadow_executor: MarketExecutor`;
- `executor: MarketExecutor`, identity-equal to `hybrid_executor`;
- `reward_history_cache: dict[int, tuple[float, ...]]`.

The builder constructor receives:

- `dataset: MarketDataset`;
- `config: ResidualMarketEnvConfig`;
- `reward_config: RewardConfig`;
- `resolved_decision_hours: float`;
- `minimum_start_index: int`;
- `execution_rule_stress: ExecutionRuleStress | None`.

## Preserved construction order

The builder must preserve the exact existing order:

1. construct `RewardTracker(reward_config, decision_hours=resolved_decision_hours)`;
2. when `config.require_full_reward_preroll` and `reward_config.baseline_underperformance_weight > 0.0`, call `minimum_reward_start_index()` with the incoming signal minimum and reward window;
3. construct the hybrid `MarketExecutor`;
4. construct the shadow `MarketExecutor` independently;
5. set the compatibility executor alias to the hybrid executor;
6. create a fresh empty reward-history cache.

This order preserves current validation and exception behavior. In particular, an invalid reward-tracker decision interval fails before reward pre-roll or executor construction.

## Facade integration

`ResidualMarketEnv.__init__()` invokes the builder once after the policy/schedule contract is resolved and before the observation contract is built. It assigns returned values to the existing attributes:

- `reward_tracker`;
- `_minimum_start_index`;
- `hybrid_executor`;
- `shadow_executor`;
- `executor`;
- `_reward_history_cache`.

The runtime-service builder continues to receive those existing attributes. No public attribute or constructor signature changes.

## Freshness and identity

Each builder call must return:

- a fresh `RewardTracker`;
- two distinct `MarketExecutor` instances with equal execution-policy identity;
- `executor is hybrid_executor`;
- a fresh empty dictionary.

The supplied dataset, environment config, reward config, and execution-rule-stress object are not copied or mutated.

## Architecture controls

Architecture tests prohibit direct `RewardTracker`, `minimum_reward_start_index`, `MarketExecutor`, and reward-cache construction from returning to `ResidualMarketEnv.__init__()`. They require one builder invocation, preserved source order inside the builder, and a constructor source span no greater than 150 lines.

The new module receives a permanent 100.0% critical coverage ratchet.

## Testing

TDD coverage includes:

- no-pre-roll minimum preservation;
- full pre-roll minimum derivation;
- exact decision-hours forwarding;
- independent equivalent executors and compatibility alias identity;
- fresh reward tracker and cache across builds;
- execution-rule-stress identity preservation;
- environment integration under existing attributes;
- reward-tracker validation before downstream construction;
- constructor delegation and 150-line architecture limit;
- full CI, Ubuntu, Windows, training image/non-root probe, and PostgreSQL Catalog verification at the exact final head.
