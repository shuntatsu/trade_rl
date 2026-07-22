# Environment Runtime Decomposition Verification

Date: 2026-07-22

Branch: `agent/decompose-environment-runtime-20260722`

Pull request: #80

Dependency base: PR #79 head
`edaa6930ffb0b351ae5a8aa9afa6c80d47ca5e27`

## Scope

This change extracts four responsibilities from `ResidualMarketEnv.step()` while
keeping all Gymnasium mutable state in the facade:

- `EnvironmentDecisionPlanner`: action parsing, residual/target composition,
  signal-delay handling, and decision-bar planning;
- `EnvironmentRiskProjector`: emergency, pre-trade, and portfolio-risk
  projection;
- `EnvironmentRewardCoordinator`: reward-transition input mapping;
- `EnvironmentInfoBuilder`: stable step and terminal information construction.

The constructor, public properties, schema versions, environment digest payload,
target identity, execution order, signal-delay semantics, risk-reason order,
reward order, terminal accounting, and existing information keys remain unchanged.

## TDD RED evidence

Architecture contracts were committed before production services existed.

Initial architecture RED:

- head: `dc49660eb386eabcd5762ec2317c503f48a56dfd`
- workflow run: `29906923608`
- result: failed as intended
- artifact: `8524221206`
- artifact digest:
  `sha256:daa32bbb061a793879adf8a42cd52a8331f2b53e2738f77c6cf4f325abd92d1d`

The failures required the four service modules, facade construction and
delegation, removal of direct composition/reward/info ownership from `step()`,
and terminal information delegation.

Expanded service-contract RED:

- head: `29f2e6468d8e7f53e5c89beb06af3576e32cb766`
- workflow run: `29907379481`
- result: failed as intended with `ModuleNotFoundError` for all four service
  modules
- artifact: `8524403476`
- artifact digest:
  `sha256:2008b8ac18aa45ee08e71e4fdb27bd9578882057a39c5c8d556e467f3fe2fe1e`

The RED failures were caused by missing production modules rather than malformed
test fixtures.

## Focused GREEN evidence

Service implementation verification:

- workflow run: `29908220862`
- job: `88884667478`
- Ruff formatting: passed
- Mypy across the repository: passed
- focused service tests: 18 passed
- source and tests committed and pushed by the verified job

Facade integration verification:

- workflow run: `29909051489`, rerun attempt
- job: `88887782701`
- strict source transformation anchors: passed
- Ruff/static integration: passed
- Mypy: passed
- environment, serving-parity, and stateful-replay regression tests: passed
- integration commit and push: passed

The first integration attempt exposed a pre-existing nondeterministic Torch
gradient comparison outside the modified environment code. The exact same job was
rerun without code changes and passed. No production tolerance or test threshold
was changed.

## Exact-head full verification

Code and coverage-ratchet head:

`549e5b9870931364caabb5d0ad00ffdfcd653c92`

GitHub Actions CI run `29909971144`: success.

- exact-head checkout: passed
- Studio tests, TypeScript checking, production build, and fixed-viewport layout:
  passed
- workflow-security validation: passed
- Ruff and format check: passed
- Mypy: passed
- Import Linter architecture contracts: passed
- dead-code report: passed
- recovery and structured Serving smoke: passed
- full Pytest: `1184 passed, 2 skipped, 11 warnings`
- full-suite duration: 89.61 seconds
- total coverage: `83.55%`
- total branch coverage: `70.46%`
- critical branch-coverage checker: passed
- CLI smoke: passed
- Ubuntu compatibility: passed
- Windows compatibility: passed
- complete training-image build and packaged non-root runtime probe: passed

Pytest diagnostics artifact:

- artifact: `8525525728`
- digest:
  `sha256:ecf7e7f8e362927aca7fd018602328c83865ff260fb6c65121643a7a227ef738`

PostgreSQL Catalog run `29909971171`: success.

- exact-head checkout: passed
- Compose validation: passed
- PostgreSQL startup and readiness: passed
- migration: passed
- unit and integration tests: passed
- volume cleanup: passed

## Coverage ratchet

The new service group contains:

- `trade_rl/rl/environment_decision.py`
- `trade_rl/rl/environment_risk.py`
- `trade_rl/rl/environment_reward.py`
- `trade_rl/rl/environment_info.py`

Observed aggregate branch coverage:

- covered branches: 33
- total branches: 38
- observed: `86.8421%`
- configured minimum: `86.8%`

The existing `environment_runtime` minimum remains `64.0%`; it was not reduced.

## Compatibility review

A commit comparison from PR #79 head to the ratchet head showed the P2-specific
change is limited to the four service modules, `ResidualMarketEnv`, focused tests,
coverage configuration, and design/plan documentation.

Review conclusions:

- books, order books, pending targets, indices, position age, previous action, and
  diagnostics remain mutable only in `ResidualMarketEnv`;
- services return new dataclass results or dictionaries and do not mutate facade
  state;
- legacy two-value residual action migration is preserved;
- zero-delay and one-decision-delay target semantics are preserved;
- emergency, pre-trade, and portfolio-risk reason ordering is preserved;
- reward inputs remain numerically identical to the previous facade code;
- optional discarded-target and liquidation information conditions are
  preserved;
- terminal performance metrics use the same return-series and evaluation
  functions as before;
- no temporary patch script or temporary verification workflow remains in the
  pull-request diff.

No critical or important review issue remained at this checkpoint.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- PR #80 remains Draft and is not merged.
