# Environment Runtime Decomposition Verification

Date: 2026-07-23

Branch: `fix/environment-runtime-decomposition-current-main-20260723`

Pull request: #90

Replacement for stacked pull request #80, rebuilt directly on current `main` after
PR #79 and PR #88 merged.

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

## Original focused GREEN evidence

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
- environment, Serving-parity, and stateful-replay regression tests: passed
- integration commit and push: passed

The first integration attempt exposed a pre-existing nondeterministic Torch
gradient comparison outside the modified environment code. The exact same job was
rerun without code changes and passed. No production tolerance or test threshold
was changed.

## Original PR #80 exact-head verification

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

## Current-main reconstruction

The original PR #80 was stacked on the unsquashed PR #79 history. After PR #79 was
squash-merged, directly merging histories produced add/add conflicts even though
the maintained source content was compatible.

PR #90 was therefore created from current `main`. A one-shot workflow applied only
the audited 14-file delta between PR #79 head
`edaa6930ffb0b351ae5a8aa9afa6c80d47ca5e27` and PR #80 head
`88f7e486b6db56fdc16ab89db5a77ef599c8f48f`. The workflow deleted itself in the
same resulting branch history and is absent from the pull-request diff.

The current PR diff is limited to:

- four environment step service modules;
- `ResidualMarketEnv` orchestration changes;
- five focused/architecture test modules;
- the service branch-coverage ratchet;
- design, implementation-plan, and verification documentation.

Current `main`, including sequence-projection stabilization PR #88, was then merged
into the feature branch before verification.

## Current-main code-head verification

Code head:

`fe64a098d8e5eac5e5dd97034d50111cec039f5f`

GitHub Actions CI run `29955909162`: success.

- exact-head checkout: passed
- Studio tests, TypeScript checking, production build, and fixed-viewport layout:
  passed
- workflow-security validation: passed
- Ruff and format check: passed
- Mypy: passed
- Import Linter architecture contracts: passed
- dead-code report: passed
- recovery and structured Serving smoke: passed
- full Pytest: `1193 passed, 2 skipped, 11 warnings`
- full-suite duration: 91.03 seconds
- total coverage: `83.56%`
- total branch coverage: `70.46%`
- critical branch-coverage checker: passed
- CLI smoke: passed
- Ubuntu compatibility: passed
- Windows compatibility: passed
- complete training-image build and packaged non-root runtime probe: passed

Final code-head artifacts:

- Studio diagnostics: `8544051034`, digest
  `sha256:065408887c9c7ea58cb4da2858954ad9bb5c8b13a66f602293c831c421022338`
- static diagnostics: `8544061895`, digest
  `sha256:48885fe7798500a7ae1d8987f22c7c780b6dd16d570985985ad06a9e673d11e9`
- architecture diagnostics: `8544062585`, digest
  `sha256:3413a7be640267c5427b1e7aa335190aa4ea8f49b508e5df937727a8e1a94bac`
- Pytest diagnostics: `8544106649`, digest
  `sha256:e9c9a0e077d75fb9cab6d750d2f0a63ed8fe1c962ed239e99e8a4187095b357e`
- training-image evidence: `8544050181`, digest
  `sha256:2627321368ab3fc064c07c1a69a04a12a8fe5edcdae052aff1cac81a073177c3`

PostgreSQL Catalog run `29955909173`: success.

- exact-head checkout: passed
- Compose validation: passed
- PostgreSQL startup and readiness: passed
- migration: passed
- unit and integration tests: passed
- cleanup: passed

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
  pull-request diff;
- no unresolved review thread remains.

No critical or important review issue remained at this checkpoint.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- PR #90 remains Draft pending documentation-inclusive exact-head verification.
