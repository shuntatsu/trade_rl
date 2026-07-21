# P0 Validation Evidence — 2026-07-21

## Scope

This record binds static analysis, tests, PostgreSQL integration, Serving E2E, accounting oracles, execution-sensitivity infrastructure, multi-seed walk-forward evidence, and the training image to exact pull-request commits for PR #73.

## Baseline main

- Baseline commit: `e295949be29612691a1b62aa998d85709930c389`
- Latest previously verified product PR: #72
- Previous CI run: `29804354107` (`CI`, success)
- Previous PostgreSQL run: `29804354112` (`PostgreSQL Catalog`, success)

## Exact source export

- Source-export head: `045d8e9af24a3cf8e3fa9b26d5255c39ac4e2242`
- Source artifact ID: `8486039455`
- Source artifact digest: `sha256:e654746b41f2f520aaaf2dbe081b2542114d82e5aae8c59f597631113211450c`
- Source-tree digest: `f2b3e0f85889199452347b43f31e29841672eb42e6190ad170e8ad3dd5c604f1`
- `uv.lock` digest: `d2fb04f4bca12cb1b0702033aa46db27dc6a821764aad864f373bc490b012c79`

## Local environment limitation

The execution container could not resolve `github.com`, so `uv` could not download the required CPython 3.12 distribution. Authoritative Python 3.12, Docker, PostgreSQL, SB3, Serving, and image verification therefore runs through GitHub Actions on an explicitly checked-out pull-request head SHA. Focused dependency-independent oracle tests were also run locally.

## P0-1 persistent sealed test

- RED test commit: `805d4b1216a1d1ae4714c97a0fa8d3a5f66104a5`
- RED PostgreSQL workflow: `29807890440`; failed before implementation because the persistent ledger module did not exist.
- Implementation commit: `aa78df94def16fc2efe069f05943cd3b359c1642`
- GREEN PostgreSQL workflow: `29809028352`
- Contract: separate catalog and ledger instances cannot open the same `(experiment_plan_digest, dataset_id, fold_index)` twice.

## P0-2 training-serving observation parity

- RED test commit: `352e77be6217d68576fba1a5d047d8cacecf00ba`
- RED CI workflow: `29811288399`
- RED evidence: 330 compatibility tests passed and the new parity test failed because `ResidualMarketEnv.observation_snapshot()` did not exist.
- First implementation product commit: `9fb891f6d933e307b6b08a24c356552c333df0f6`
- Focused reference-fix product commit: `91cfa0a80729e8c3f1f079d57f9a15aa242cf42d`
- Formatting product commit: `b71d43e0b6d805f5575095006326b9e2c6a8661d`
- Functional GREEN CI: `29812366883`; Ubuntu, Windows, the real non-zero observation parity test, and the complete training image passed.
- Compared state: symbol and feature order, feature/global availability, staleness, hybrid and shadow book state, pending target, previous action, raw observation, normalized observation, each policy-member input/action, and deterministic ensemble mean action.

## P0-3 historical metadata promotion gate

- RED test commit: `276bdf267e75f0d61561f544fc17a8a4e37f4f5d`
- RED head: `ce6d47c1c4a6a5ac96fc5efcf72c802cd08f2a65`
- RED CI workflow: `29812672345`
- RED evidence: 331 compatibility tests passed; only `frozen_snapshot` and `conservative_static` failed because package promotion did not raise an error.
- Implementation product commit: `9ff4167ce105a6ed151f3248c39d9c5b1a5c7d43`
- Promotion now requires dataset-bound `historical_signed`, point-in-time, Ed25519-authenticated, effective-dated full-interval evidence with no unresolved limitation.

## Independent accounting and reward oracle

- Focused verification workflow: `29814005381` (`P0 Verify Validation Suite`, success)
- Focused local result: 9 tests passed.
- Expected values are computed by a separate `ManualBook` implementation and do not call production accounting or execution helpers.
- Covered cases: zero fee, explicit fee, no slippage/funding, partial fill, split, delisting recovery, margin shortfall and termination, cash, quantities, marked position value, total equity, PnL, and exact log-growth reward.

## Execution sensitivity matrix

The workflow-layer sensitivity matrix covers:

- fee: 1x, 2x, 4x
- spread: 1x, 2x
- slippage: 1x, 2x, 4x
- capacity: 100%, 50%, 25%
- signal/order delay proxy: 0, 1, 2 bars
- limit fill: optimistic, neutral, conservative
- tradability delay: 0, 1 bar

The focused verification confirms monotonic cost degradation, lower fills under reduced capacity, delay dependence, conservative limit-fill behavior, and delayed-tradability behavior. The three-seed smoke below produced no positive RL edge before sensitivity stress, so it is recorded as `NO-GO`; no claim is made that stress testing proved profitability.

## Actual three-seed and unused-period walk-forward

- Exact head: `2253b263df7a837d364d039c91474a80a760345f`
- Workflow run: `29814577680` (`P0 Multi-Seed Unused Period`, success)
- Artifact ID: `8488798303`
- Artifact digest: `sha256:b76be5c3312c0a6ff2abe25526cbec4817901ea8ebad0dd9ff067097705d2ce7`
- Dataset ID: `02d3ffc568b23cf5cccce8902dab1098dc3b8a8dfdd6c4044ee2ca1792adf633`
- Train range: `[0, 384)`
- Checkpoint-validation range: `[392, 456)`
- Configuration-selection range: `[464, 528)`
- Sealed unused outer-test range: `[536, 632)`; 96 bars
- Seeds: 0, 1, 2; 128 PPO timesteps per seed; CPU smoke configuration

Configuration-selection results, converting the stored summed log-return score with `expm1`:

| Seed | Selection return | Checkpoint-validation return |
| ---: | ---: | ---: |
| 0 | -1.0458% | -1.8594% |
| 1 | -1.0876% | -1.9328% |
| 2 | -1.1205% | -1.9863% |

- Median seed selection return: -1.0876%
- Worst seed: 2 at -1.1205%
- Candidate score standard deviation: `0.00030897883497740824`
- Candidate maximum selection drawdown: 1.2536%
- Candidate maximum turnover per day: `2.0225130911579754`
- Candidate maximum cost fraction: 0.7509%
- Candidate eligibility: false
- Selection reasons: median seed score below threshold; deployable ensemble score below threshold
- Selected configuration: baseline
- Sealed outer-test evaluations: exactly 1
- Sealed outer-test selected/baseline return: -0.4595%
- Outer-test turnover total: `5.792671892149814`
- Outer-test total cost: `810.72854035693`
- Production status: `NO-GO`

The sealed outer period was not reopened for individual seed tuning. Individual seed evidence belongs to checkpoint-validation and configuration-selection; only the frozen selected configuration and baseline are evaluated once on the unused outer period.

## Training image identity evidence

An intermediate exact CI run `29814577665` successfully built and probed the complete non-root training image and uploaded artifact `8488817492`:

- checked-out commit recorded by that run: `d094fa1e97b0cb812756bcc55ca71448f7aaf529`
- source-tree digest: `cde5d356a6d454d9429672d0385e384176b978310454513c6848599f74801d01`
- lockfile digest: `d2fb04f4bca12cb1b0702033aa46db27dc6a821764aad864f373bc490b012c79`
- image ID: `sha256:936c50509552f0931a98b60dab2861dfb524a1f5f30fdab04e5f1b2737f131c5`

The final CI configuration explicitly checks out `${{ github.event.pull_request.head.sha || github.sha }}` in every job. The final exact-head image ID and workflow IDs are recorded in the PR evidence comment after the unchanged final head passes.

## Final verification contract

The unchanged final pull-request head must pass all of the following:

- `ruff check .`
- `ruff format --check .`
- `mypy .`
- `pytest -q` with branch coverage and the existing minimum coverage gate
- import-linter responsibility contracts
- dead-code report
- recovery and structured Serving smoke
- PostgreSQL 16 migration and integration tests
- Ubuntu and Windows compatibility suites
- complete training-image build, exact-head identity artifact, and non-root runtime probe

Passing these gates establishes code, packaging, artifact, and test integrity. It does not establish profitability or authorize direct exchange execution.
