# Environment Policy and Schedule Contract Design

## 1. Goal

Extract the deterministic environment configuration, action-layout, and episode/decision schedule construction currently embedded in `ResidualMarketEnv.__init__()` into one typed, independently testable contract without changing runtime policy or public behavior.

## 2. Current problem

After the provider, portfolio-risk, observation, and runtime-service extractions, the environment constructor still owns a dense sequence of related static decisions:

1. resolve the supplied or default `ResidualMarketEnvConfig`;
2. construct `CausalEmergencyRiskMonitor` from the config;
3. validate pre-trade gross exposure against execution leverage;
4. validate random initial gross exposure;
5. resolve a supplied or default `ActionSpec`;
6. validate alpha mode, factor count, and target-weight symbol count;
7. derive action names;
8. derive nominal episode and decision bars;
9. reject a decision interval longer than the episode;
10. resolve the reward configuration;
11. derive resolved decision hours;
12. reject episode-hour choices shorter than the resolved decision interval.

These operations are deterministic construction-time policy. They do not need mutable Gymnasium state and should not remain inline in the facade.

## 3. Chosen boundary

Create `trade_rl/rl/environment_policy_schedule_contract.py` with:

- frozen, slotted `EnvironmentPolicyScheduleContract`;
- `EnvironmentPolicyScheduleContractBuilder`.

The contract contains:

- `config: ResidualMarketEnvConfig`;
- `emergency_risk_monitor: CausalEmergencyRiskMonitor`;
- `action_spec: ActionSpec`;
- `action_names: tuple[str, ...]`;
- `nominal_episode_bars: int`;
- `nominal_decision_bars: int`;
- `reward_config: RewardConfig`;
- `resolved_decision_hours: float`.

The builder consumes:

- `dataset: MarketDataset`;
- `pre_trade_risk: PreTradeRisk`;
- `alpha_enabled: bool`;
- `factor_count: int`;
- optional supplied `action_spec`;
- optional supplied `config`.

## 4. Preserved construction and validation order

The builder must preserve the current order exactly:

1. resolve the supplied config or construct `ResidualMarketEnvConfig()`;
2. construct `CausalEmergencyRiskMonitor(config.emergency_risk)`;
3. reject `pre_trade_risk.config.max_gross > config.execution_cost.max_leverage` with `pre-trade max_gross cannot exceed execution max_leverage`;
4. reject `config.random_initial_gross > pre_trade_risk.config.max_gross` with `random_initial_gross cannot exceed pre-trade max_gross`;
5. construct the default `ActionSpec` only when none was supplied, using `alpha_enabled`, `factor_count`, and `config.action_validation_mode`;
6. validate action alpha mode;
7. validate action factor count;
8. validate target-weight symbol count;
9. derive action names from dataset symbols;
10. resolve nominal episode bars;
11. resolve nominal decision bars;
12. reject a decision interval longer than the episode;
13. resolve the reward configuration;
14. derive resolved decision hours;
15. reject any episode-hour choice shorter than the resolved decision interval;
16. return the typed contract.

The existing exception text and the relative order of errors are part of the maintained contract.

## 5. Facade integration

`ResidualMarketEnv.__init__()` will invoke the builder once after the provider and portfolio-risk contracts are assigned and after `pre_trade_risk` exists. It will assign the returned values to the same existing attributes:

- `self.config`;
- `self.emergency_risk_monitor`;
- `self.action_spec`;
- `self._action_names`;
- `self._nominal_episode_bars`;
- `self._nominal_decision_bars`;
- `self._resolved_decision_hours`.

The local `reward_config` variable will receive `contract.reward_config` for the existing `RewardTracker` and reward-preroll code.

The public constructor signature remains unchanged.

## 6. Explicit non-goals

This extraction does not move or change:

- `BaselineResidualComposer` construction;
- `PreTradeRisk` construction;
- portfolio-risk construction;
- `RewardTracker` construction;
- reward-preroll minimum-index calculation;
- `MarketExecutor` construction;
- observation-contract construction;
- runtime-service construction;
- mutable book, order, action, episode, or reset state;
- action semantics, reward semantics, execution semantics, or emergency-risk assessment logic.

## 7. Testing strategy

Direct characterization tests must cover:

- supplied config and action-spec identity preservation;
- default action-spec fields and validation mode;
- emergency monitor configuration identity;
- action-name derivation for residual and target-weight modes;
- nominal episode/decision bars and resolved decision hours;
- leverage validation before random-gross validation;
- alpha validation before factor validation before target-symbol validation;
- decision-duration validation before episode-choice validation;
- exact error messages;
- environment integration and digest payload preservation.

Architecture tests must require:

- local ownership of the contract and builder;
- exactly one builder call in the environment constructor;
- absence of the extracted inline error strings and default action-spec construction from the facade;
- preserved validation ordering in the builder;
- a constructor source span no greater than 190 lines.

The extracted module must achieve 100.0% statement and branch coverage and receive a permanent 100.0% critical branch-coverage ratchet in `pyproject.toml`.

## 8. Verification requirements

The exact implementation head must pass:

- Ruff and Ruff format;
- Mypy;
- Import Linter;
- dead-code reporting;
- serving smoke checks;
- complete pytest and coverage;
- critical branch-coverage checks;
- CLI smoke;
- Ubuntu and Windows compatibility;
- complete training-image and packaged non-root runtime probe;
- PostgreSQL Compose, readiness, migrations, unit/integration tests, and cleanup.

## 9. Architecture status

This change further reduces `AUD-RL-001` but does not resolve it. Reward-tracker/executor construction and mutable Gymnasium-state initialization remain in the facade. Production remains `NO-GO` until independent research, evidence, operational, authorization, and profitability gates pass.