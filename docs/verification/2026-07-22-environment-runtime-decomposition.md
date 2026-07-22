# Environment Runtime Decomposition Verification

Date: 2026-07-23

Merged pull request: #92

Merge commit: `6cf23b98698f5d53ec40629dd723efc4bd4cfbb6`

Verified implementation head: `86874d7122beef9247c67f89d41a3266a1164492`

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
- repository-wide Mypy: passed
- focused service tests: 18 passed

Facade integration verification:

- workflow run: `29909051489`, rerun attempt
- job: `88887782701`
- strict source transformation anchors: passed
- Ruff and static integration: passed
- Mypy: passed
- environment, Serving-parity, and stateful-replay regression tests: passed

The first integration attempt exposed a pre-existing nondeterministic Torch
gradient comparison outside the modified environment code. The exact same job was
rerun without code changes and passed. No production tolerance or test threshold
was changed.

## Clean current-main reconstruction

The original PR #80 was stacked on the unsquashed PR #79 history. After PR #79 was
squash-merged, the history could not be synchronized cleanly without repeating
already-merged files.

PR #92 therefore recreated the already-verified decomposition directly from
current `main` at `464c14669bd2355b6922e6813870030bcf6cc745`.

The effective change contained exactly 14 files:

- four new environment runtime service modules;
- `ResidualMarketEnv` orchestration changes;
- one architecture ownership contract;
- four focused service test modules;
- one measured branch-coverage group;
- design, implementation-plan, and verification documentation.

No PR #79 implementation file was repeated. No temporary transplant workflow,
patch script, or generated file remained in the merged diff.

## Exact-head verification

GitHub Actions CI run `29955899116`: success.

- exact-head checkout: passed
- Studio Vitest, TypeScript, production build, and fixed-viewport validation:
  passed
- workflow-security validation: passed
- Ruff and format: passed
- Mypy: passed
- Import Linter architecture contracts: passed
- dead-code report: passed
- recovery and structured Serving smoke: passed
- full Pytest: `1193 passed, 2 skipped, 11 warnings`
- total coverage: `83.55%`
- total branch coverage: `70.46%`
- critical branch-coverage ratchets: passed
- CLI smoke: passed
- Ubuntu compatibility: passed
- Windows compatibility: passed
- complete training-image build and packaged non-root runtime probe: passed

Exact-head artifacts:

- Pytest diagnostics: `8544086297`, digest
  `sha256:f44664ca5384e1d7335e5501b9d6aa84fb58397778d1dca1637f954dd8e1a0e9`
- architecture diagnostics: `8544044684`, digest
  `sha256:08af2ea2c1e9e3ba1d7ad77845ddff083d8e15191979126fe89f41e1886050b2`
- static diagnostics: `8544044262`, digest
  `sha256:fed4e3ed151393f7e2950916761bdbc9a03a37a966f290fbf9c08a2b274d6b91`
- training-image evidence: `8544037302`, digest
  `sha256:e4311e0ef41c348d88b764c40a205b2ee554c4cb3c66f0c6ddf4feeeaa42589c`
- Studio layout diagnostics: `8544033273`, digest
  `sha256:a6c369cd0ffd0617d1154cff2a92f7f23a4a18b8a1e47c3f6e9aae465f4c7cdf`
- Windows compatibility: `8544028938`, digest
  `sha256:9a68f30cc46cf0c84a3d37056c79a9ae43e0df8f5b1b22ee49f3c6160bc6185e`
- Ubuntu compatibility: `8544024734`, digest
  `sha256:6f87cfa29f825a4a6c13c626ae54e1f9dcd2d2b5bf42f44d41c248c97f9cbaa4`

PostgreSQL Catalog run `29955899222`: success.

- exact-head checkout: passed
- Compose validation: passed
- PostgreSQL startup and readiness: passed
- installation and migration: passed
- unit and integration tests: passed
- shutdown and cleanup: passed

## Coverage ratchet

The environment-step service group contains:

- `trade_rl/rl/environment_decision.py`
- `trade_rl/rl/environment_risk.py`
- `trade_rl/rl/environment_reward.py`
- `trade_rl/rl/environment_info.py`

Observed aggregate branch coverage:

- covered branches: 33
- total branches: 38
- observed: `86.8421%`
- configured minimum: `86.8%`

The existing `environment_runtime` minimum remains unchanged at `64.0%`.

## Architecture review

- books, order books, pending targets, indices, position ages, previous action,
  and diagnostics remain mutable only in `ResidualMarketEnv`;
- the extracted services consume explicit request dataclasses and return results
  which the facade applies in the previous order;
- legacy two-value residual action migration is preserved;
- zero-delay and one-decision-delay target semantics are preserved;
- emergency, pre-trade, and portfolio-risk reason ordering is preserved;
- reward inputs remain numerically equivalent to the previous facade mapping;
- optional discarded-target and liquidation information conditions are
  preserved;
- terminal performance metrics retain the same return-series and evaluation
  functions;
- no unresolved critical or important review issue remained before merge.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- this verification records an architecture and regression result, not production
  authorization.
