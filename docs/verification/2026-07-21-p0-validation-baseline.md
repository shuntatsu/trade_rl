# P0 Validation Evidence — 2026-07-21

## Scope

This record binds static analysis, tests, PostgreSQL integration, Serving E2E, and the training image to exact pull-request head commits for PR #73.

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

The execution container could not resolve `github.com`, so `uv` could not download the required CPython 3.12 distribution. Local dependency installation was therefore unavailable. All authoritative Python 3.12, Docker, PostgreSQL, SB3, Serving, and image verification is run through GitHub Actions on the exact recorded PR head.

## P0-1 persistent sealed test

- RED test commit: `805d4b1216a1d1ae4714c97a0fa8d3a5f66104a5`
- RED PostgreSQL workflow: `29807890440` (failed in the unit/integration test step before implementation)
- Implementation commit: `aa78df94def16fc2efe069f05943cd3b359c1642`
- GREEN PostgreSQL workflow: `29809028352` (success)

## P0-2 training-serving observation parity

- RED test commit: `352e77be6217d68576fba1a5d047d8cacecf00ba`
- RED CI workflow: `29811288399`
- RED evidence: 330 compatibility tests passed and the new parity test failed because `ResidualMarketEnv.observation_snapshot()` did not exist.
- First implementation product commit: `9fb891f6d933e307b6b08a24c356552c333df0f6`
- Focused reference-fix product commit: `91cfa0a80729e8c3f1f079d57f9a15aa242cf42d`
- Functional GREEN CI: `29812366883`; Ubuntu and Windows compatibility and the complete training image passed. The core job stopped only at formatting before the final formatting commit.
- Formatting product commit: `b71d43e0b6d805f5575095006326b9e2c6a8661d`

## P0-3 historical metadata promotion gate

- RED test commit: `276bdf267e75f0d61561f544fc17a8a4e37f4f5d`
- RED head: `ce6d47c1c4a6a5ac96fc5efcf72c802cd08f2a65`
- RED CI workflow: `29812672345`
- RED evidence: 331 compatibility tests passed; only the two non-historical modes failed because packaging did not raise an error.
- Implementation product commit: `9ff4167ce105a6ed151f3248c39d9c5b1a5c7d43`
- GREEN workflow IDs: pending on this evidence-refresh head.

## Final verification

The final section will record:

- `ruff check .`
- `ruff format --check .`
- `mypy trade_rl`
- complete `pytest` with branch coverage
- PostgreSQL 16 integration
- Ubuntu and Windows compatibility
- Serving recovery/structured E2E smoke
- complete training-image build and non-root probe
- exact final commit SHA and Docker image identity
