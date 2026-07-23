# Environment Portfolio-Risk Contract Extraction Verification — 2026-07-23

## 1. Scope

This verification records the behavior-preserving extraction of portfolio-risk
construction policy from `ResidualMarketEnv.__init__()` into
`trade_rl.rl.environment_portfolio_risk_contract`.

The extracted boundary owns only:

- default `PortfolioRiskModel()` construction;
- preservation of a supplied portfolio-risk model;
- preservation of a supplied causal inputs provider;
- automatic `RollingPortfolioRiskInputsProvider()` selection when advanced
  covariance, beta, or stress inputs are required and no provider was supplied;
- SHA-256 identity validation for the resolved provider;
- provider minimum-index validation;
- aggregation of the provider minimum index with the existing causal signal
  minimum.

The change does not alter portfolio-risk projection mathematics, rolling input
calculation, pre-trade risk, emergency risk, action policy, reward policy,
execution, observation construction, or mutable Gymnasium state.

Production remains `NO-GO`.

## 2. TDD RED evidence

The clean RED head was:

- commit: `384b9d93b7686cdacd8f6d1dd89b23686e814c21`;
- CI run: `29983063624`.

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

Complete pytest collection then failed because
`trade_rl.rl.environment_portfolio_risk_contract` did not exist. No production
implementation for the boundary was present at the clean RED head.

## 3. Implemented contract

`EnvironmentPortfolioRiskContract` is a frozen, slotted dataclass containing:

- `portfolio_risk`;
- `inputs_provider`;
- `minimum_start_index`.

`EnvironmentPortfolioRiskContractBuilder` preserves the former inline order:

1. resolve the supplied or default `PortfolioRiskModel`;
2. preserve the supplied inputs provider;
3. create `RollingPortfolioRiskInputsProvider` only when advanced inputs are
   required and the provider is absent;
4. validate `identity_digest` with field
   `portfolio_risk_inputs_provider.identity_digest`;
5. read `minimum_index` only after digest validation succeeds;
6. reject boolean, non-integer, negative, or dataset-out-of-range values with
   `portfolio risk inputs minimum_index is invalid`;
7. aggregate the valid provider index with the incoming minimum using `max`.

`ResidualMarketEnv.__init__()` invokes the builder once and assigns the returned
model, provider, and minimum index to the existing environment attributes.
The public constructor signature is unchanged.

## 4. Characterization and architecture controls

Direct tests cover:

- default model construction without an unnecessary provider;
- supplied model and provider identity preservation;
- automatic rolling-provider selection for advanced risk;
- preservation of the larger existing minimum index;
- digest failure before `minimum_index` access;
- boolean, non-integer, negative, and out-of-range minimum indices;
- environment integration and digest payload preservation.

Architecture tests require:

- local ownership of the contract and builder;
- exactly one builder invocation in the environment constructor;
- absence of inline rolling-provider construction, provider digest validation,
  and minimum-index error policy from the facade constructor;
- digest validation before minimum-index access and aggregation;
- a constructor source span no greater than 220 lines.

The measured constructor source span is 218 lines, reduced from 232 after PR
#122. Across `trade_rl/rl/environment.py`, this extraction adds 12 lines and
removes 27 lines, for a net reduction of 15 source lines.

## 5. Coverage controls

The exact implementation and coverage-ratchet head was:

- commit: `4db0f5d59e05e60461058764aca783cfd8692e33`;
- CI run: `29983798488`;
- PostgreSQL Catalog run: `29983798442`.

The complete test result was:

- 1,298 passed;
- 2 skipped;
- 11 warnings;
- 84.12% total coverage;
- 71.19% total branch coverage.

`trade_rl/rl/environment_portfolio_risk_contract.py` measured:

- 29 / 29 statements covered;
- 6 / 6 branches covered;
- 100.0% statement coverage;
- 100.0% branch coverage.

A permanent 100.0% critical branch-coverage ratchet is recorded in
`pyproject.toml`.

## 6. Maintained verification

CI run `29983798488` passed:

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

PostgreSQL Catalog run `29983798442` passed:

- Compose validation;
- PostgreSQL startup and readiness;
- dependency installation;
- migrations;
- catalog unit and integration tests;
- cleanup.

## 7. Final architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral
defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Portfolio-risk input construction is now typed, independently characterized,
fully covered, and prohibited from returning inline to the environment facade.
The remaining constructor density consists primarily of:

- environment config, emergency-risk, action-spec, and episode-schedule
  validation;
- reward-tracker and market-executor construction;
- mutable Gymnasium book, order, observation, and episode-state initialization.

This item does not block causal research use. Production remains `NO-GO` until
the maintained research, evidence, operational, authorization, and profitability
gates pass independently.
