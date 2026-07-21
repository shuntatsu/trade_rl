# Conservative Stateful Order Simulator Verification

## Scope

This record covers the OHLCV-based stateful execution model introduced by PR #75. It does not claim order-book reconstruction, exchange-equivalent fills, live exchange connectivity, or model profitability.

## Required Trust Boundaries

- order quantity is fixed from decision-time information;
- fills use the processing bar's volume and one shared symbol capacity pool;
- partial residuals remain explicit pending orders;
- one deterministic OHLC path is selected per symbol and bar;
- final promotion requires conservative path mode, processing-bar capacity, partial-fill carry, complete order events, and a matching execution-policy digest;
- pending-order state participates in Training–Serving observation parity;
- replay evidence binds dataset, seed, policy, action trace, order events, equity curve, and observation trace.

## Maintained Full-Training Defaults

The maintained Binance multi-timeframe training and walk-forward configurations explicitly select:

```text
path mode: conservative
processing-bar volume capacity: true
partial-fill carry: true
trigger-volume fractions: 1.00 / 0.50 / 0.25 / 0.00
stateful environment time in force: GTC
```

## Focused Verification

The implementation was developed test-first. Focused checks cover order-domain invariants, gap rules, shared capacity, partial carry, target reconciliation, admission, accounting, emergency liquidation, pending-order observations, Training–Serving parity, execution promotion evidence, and deterministic replay.

The evidence modules were moved into `trade_rl.simulation` after Import Linter correctly rejected an `evaluation -> simulation` dependency. The corrected architecture and behavior tests passed before the product commit was retained.

## Integrated Exact-Head Verification

The simulator branch was brought fully up to date with `main` commit `eac5fbf41737de03bde7fa3ab044d8080d3c15cc`, which contains the Live Training replay and checkpoint-evidence work. The integrated product head was:

```text
27a564313f64a4ebbd4001fc77518c9985af78b8
```

Repository-wide CI run `29858620871` completed successfully on that exact head. PostgreSQL run `29858620803` also completed successfully.

Successful checks included:

- Studio tests, TypeScript checking, production build, and fixed-viewport layout verification;
- workflow-security validation;
- `ruff check .` and `ruff format --check --diff .`;
- `mypy .`;
- Import Linter and dead-code reporting;
- checkpoint recovery and structured Serving smoke tests;
- full pytest with branch coverage;
- critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu and Windows compatibility suites;
- PostgreSQL Compose validation, migration, and unit/integration tests;
- complete training-image build and non-root runtime probe.

Full pytest result:

```text
1131 passed, 2 skipped, 11 warnings
Total coverage: 83.43%
```

Pytest diagnostics artifact:

```text
artifact id: 8506431711
artifact digest: sha256:637747b7d94724b3e620c68d0fa3574e8c1978c6d3f2b64b624ccdf06debe367
```

Training-image evidence:

```text
commit_sha=27a564313f64a4ebbd4001fc77518c9985af78b8
source_tree_digest=0db25624fcd1e1145e74991715a0361d8a46ffa5006cafe70541a11677f5e3bb
lockfile_digest=d2fb04f4bca12cb1b0702033aa46db27dc6a821764aad864f373bc490b012c79
image_id=sha256:570a58bb83fd29776e76d1d411c30627fbdb33914c4c622f188681fc18ebb5e3
repo_digests=[]
artifact id: 8506377138
artifact digest: sha256:aa7d855ec8b9be4f4237e9864e231176cab72ae80ca09006b20ae56b2d44ca6a
```

## Three-Seed and Deterministic Replay Smoke

Exact-head smoke run `29847415519` completed successfully on head `f4c7f1c52e5bf3476433525d8c4218fd2dd3c8be`.

```text
dataset_id=0c36da4aeb0f538b5db79b53aa420a925def677d80bbeebc9633b7b20175a677
experiment_plan_digest=9d59cb0f389c4534efded5f74f87f7082673e8b0d3ab4f2437927cdc50dae548
execution_policy_digest=49be2d8b37cde7a7e9cd06aa1c8fcff4ac96fe410eae3bb7a7cda416ed0c46a8
train=[0,220)
checkpoint=[224,244)
selection=[248,268)
sealed_test=[272,292)
```

Seeds `0`, `1`, and `2` each trained and produced checkpoint/selection evidence. The candidate failed the declared promotion requirements because its median seed score and deployable ensemble score were below threshold. The workflow therefore selected `baseline`, opened the sealed test exactly once, and made no promotion claim.

Each seed's stateful episode was replayed twice. The repeated result matched for the full replay evidence, with common order-event digest `54b2f3eca7f1382d29226e4a4ac297c942c5ec5c97e0f76ea74942611a0c4427`, equity-curve digest `e298bc90d6b7d1cefb72fc6812b0bff1292fb7b72c61c91f0c94421d38275e09`, and observation-trace digest `efa9fa101a2493e2d3fff668460ec67cf0f67c2500d029c840c0686b17c8413d`.

Smoke artifact:

```text
artifact id: 8501921380
artifact digest: sha256:91ee2bcba5ad7dc58d1a793e698af62c444719b0edd88743abaceee78f3f01c3
promotion_claim=none-smoke-only
```

## Interpretation

The results demonstrate deterministic stateful simulation, evidence production, fail-closed promotion behavior, Training–Serving pending-state parity, and successful integration with the current Live Training branch. They do not establish trading profitability or exchange-equivalent execution quality.

## Known Limitations

- OHLCV cannot recover true intrabar order, queue position, hidden liquidity, auctions, or L2 depth.
- The deterministic capacity priority is an explicit research convention, not exchange queue priority.
- Optimistic and neutral modes are sensitivity tools and cannot serve as primary promotion evidence.
- A smoke training result is pipeline evidence only and is not profitability evidence.
