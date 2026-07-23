# Environment Policy and Schedule Contract Extraction Verification — 2026-07-23

## 1. Scope

This verification records the behavior-preserving extraction of deterministic environment policy and schedule construction from `ResidualMarketEnv.__init__()` into `trade_rl.rl.environment_policy_schedule_contract`.

The extracted boundary owns only:

- supplied or default `ResidualMarketEnvConfig` resolution;
- `CausalEmergencyRiskMonitor` construction from the resolved config;
- pre-trade gross exposure versus execution leverage validation;
- random initial gross exposure validation;
- supplied or default `ActionSpec` resolution;
- alpha-mode, factor-count, and target-weight symbol-count validation;
- action-name derivation;
- nominal episode- and decision-bar derivation;
- decision-duration validation;
- reward-config resolution;
- resolved decision-hours derivation;
- episode-hour-choice validation.

The change does not alter action semantics, reward calculations, emergency-risk assessment, portfolio-risk projection, execution, observation construction, reward-tracker construction, market-executor construction, or mutable Gymnasium state.

Production remains `NO-GO`.

## 2. Test-first evidence and limitation

The architecture contract was committed at `dbd68a3161d947d84785d4ff37c8b4bb1afd5f8d` before `trade_rl/rl/environment_policy_schedule_contract.py` existed. That commit requires local ownership of the new contract, one facade delegation, the absence of the extracted inline policy, preserved validation order, and a 190-line constructor bound.

The direct characterization test file was also committed before production implementation. It defines supplied-identity preservation, default action-spec construction, action names, bars and decision hours, exact validation messages and order, environment integration, and digest preservation.

A clean RED GitHub Actions run is not claimed. Connector-authored branch commits did not reliably produce a preserved `pull_request` run during the initial RED phase, and temporary branch-local workflows were removed from the final diff. The test-before-production commit order is preserved, but the maintained verification evidence begins with the exact GREEN/ratchet head below.

## 3. Implemented contract

`EnvironmentPolicyScheduleContract` is a frozen, slotted dataclass containing:

- `config`;
- `emergency_risk_monitor`;
- `action_spec`;
- `action_names`;
- `nominal_episode_bars`;
- `nominal_decision_bars`;
- `reward_config`;
- `resolved_decision_hours`.

`EnvironmentPolicyScheduleContractBuilder` preserves the former inline order:

1. resolve the supplied config or construct the default config;
2. construct the emergency-risk monitor;
3. validate pre-trade gross exposure against execution leverage;
4. validate random initial gross exposure;
5. construct the default action spec only when none was supplied;
6. validate action alpha mode;
7. validate action factor count;
8. validate target-weight symbol count;
9. derive action names;
10. resolve nominal episode bars;
11. resolve nominal decision bars;
12. reject a decision interval longer than the episode;
13. resolve reward configuration;
14. derive resolved decision hours;
15. reject episode-hour choices shorter than that interval;
16. return the typed contract.

The existing exception text and relative error order remain maintained behavior.

`ResidualMarketEnv.__init__()` invokes the builder once and assigns its values to the same existing environment attributes. `RewardTracker`, reward-preroll minimum-index calculation, both `MarketExecutor` instances, observation construction, runtime-service wiring, and mutable state remain in the facade.

The public constructor signature is unchanged.

## 4. Characterization and architecture controls

Direct tests cover:

- supplied config and action-spec identity preservation;
- emergency-monitor config identity;
- default action-spec alpha, factor, and validation-mode fields;
- residual and target-weight action names;
- bar-based and hour-based decision schedules;
- default-config failure preservation;
- leverage validation before random-gross validation;
- alpha validation before factor validation before target-symbol validation;
- decision-duration validation before episode-choice validation;
- exact exception text;
- environment integration and digest payload preservation;
- stress reset peak-value and drawdown preservation.

The stress regression fixes the required relationship:

`peak_value = initial_capital / (1 - stress_drawdown_fraction)`.

Architecture tests require:

- local ownership of the contract and builder;
- exactly one builder invocation in the environment constructor;
- absence of direct emergency-monitor construction, default action-spec construction, and extracted exception policy from the facade;
- preserved validation ordering in the builder;
- a constructor source span no greater than 190 lines.

The measured constructor source span is 186 lines, reduced from 218 after PR #125. Across `trade_rl/rl/environment.py`, this extraction adds 20 lines and removes 50 lines, for a net reduction of 30 source lines.

## 5. Exact-head verification

The exact implementation, stress regression, and coverage-ratchet head was:

- commit: `08d8bdf6b39f00adaaca6d3f65e3183404083447`;
- CI run: `29993016948`;
- PostgreSQL Catalog run: `29993016956`.

The complete test result was:

- 1,315 passed;
- 2 skipped;
- 11 warnings;
- 84.18% total coverage;
- 71.28% total branch coverage.

`trade_rl/rl/environment_policy_schedule_contract.py` measured:

- 53 / 53 statements covered;
- 16 / 16 branches covered;
- 100.0% statement coverage;
- 100.0% branch coverage.

A permanent 100.0% critical branch-coverage ratchet is recorded in `pyproject.toml`.

CI run `29993016948` passed:

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
- complete training-image build and packaged non-root runtime probe.

PostgreSQL Catalog run `29993016956` passed:

- Compose validation;
- PostgreSQL startup and readiness;
- dependency installation;
- migrations;
- catalog unit and integration tests;
- cleanup.

## 6. Final architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Config, emergency-monitor, action-layout, and episode/decision schedule construction are now typed, independently characterized, fully covered, and prohibited from returning inline to the environment facade.

The remaining constructor density consists primarily of:

- reward-tracker and reward-preroll construction;
- hybrid and shadow market-executor construction;
- observation/runtime contract assignment;
- mutable Gymnasium book, order, action, observation, episode, and reset-state initialization.

This item does not block causal research use. Production remains `NO-GO` until the maintained research, evidence, operational, authorization, and profitability gates pass independently.
