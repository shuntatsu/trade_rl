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

## 2. TDD RED evidence

The architecture and direct characterization contracts were committed before production implementation. The clean RED head was:

- commit: `4c52249de08230a4493658a1f23104e3513e1e34`;
- CI run: `29989543655`;
- pytest diagnostic artifact: `8556494515`;
- artifact digest: `sha256:6d522d868f88eb24f88fd50f6047e701787622f0a4fbc68f2f71992ef2e286c8`.

At this head:

- Studio frontend and fixed-viewport verification passed;
- workflow-security checks passed;
- Ruff and Ruff formatting passed;
- Mypy passed;
- Import Linter passed;
- dead-code reporting passed;
- recovery and structured-serving smoke passed;
- Ubuntu compatibility passed;
- Windows compatibility passed;
- the complete training image and packaged non-root runtime probe passed.

Complete pytest collection then failed with exactly two collection errors because `trade_rl.rl.environment_policy_schedule_contract` did not exist. No production implementation for the boundary was present at the clean RED head.

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

## 5. Integration regression detected and repaired

Implementation review detected one unrelated deletion in commit `9e01814d999cd2d0fc9cd0f5ecf25da2aa16f44e`: the denominator expression in the existing stress initial-state peak calculation had been removed. That change was outside the policy/schedule extraction and would have attempted division by an empty tuple during stress reset.

Commit `05efa77a9d183663de7f786a6c27f35116f05347` restored `1.0 - self.config.stress_drawdown_fraction`. The maintained stress-reset test exercises this path. Temporary implementation workflows, triggers, and synchronization markers were removed before final verification.

## 6. Exact-head verification

The exact implementation, repaired stress regression, and coverage-ratchet head was:

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

Final CI evidence artifacts include:

- pytest and coverage: `8557864354`, digest `sha256:ccac19935762ff2695400fddbd0fb359159daf79c25f442687d2c73ed64925c7`;
- training image: `8557816573`, digest `sha256:fd8cb7d41ddca4dabe1c157fb723a90171051cf61a5be30790e31f02fde1de99`;
- architecture diagnostics: `8557812721`, digest `sha256:6a2eeebf71094192b7c409f8894f02609003a7a34e8f1ef1a3835effbad30113`;
- static diagnostics: `8557812197`, digest `sha256:8207d00790d0f996c775e8d4f061708a29a8ae697b88bfbb7b797325c7fd70e9`;
- Windows compatibility: `8557793257`, digest `sha256:b14d31cf706e83b3ba3b9e916ab6960b44d354236bd255060c8385f444b9ca6e`;
- Ubuntu compatibility: `8557786892`, digest `sha256:d26f78e9448a14afaea58565b20f168eae1763c80f41c3211550de1917ff4333`.

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

## 7. Final architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Config, emergency-monitor, action-layout, and episode/decision schedule construction are now typed, independently characterized, fully covered, and prohibited from returning inline to the environment facade.

The remaining constructor density consists primarily of:

- reward-tracker and reward-preroll construction;
- hybrid and shadow market-executor construction;
- observation/runtime contract assignment;
- mutable Gymnasium book, order, action, observation, episode, and reset-state initialization.

This item does not block causal research use. Production remains `NO-GO` until the maintained research, evidence, operational, authorization, and profitability gates pass independently.
