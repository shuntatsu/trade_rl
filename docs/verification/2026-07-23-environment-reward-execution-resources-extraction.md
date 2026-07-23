# Environment Reward and Execution Resources Extraction Verification — 2026-07-23

## 1. Scope

This verification records the behavior-preserving extraction of reward and execution resource construction from `ResidualMarketEnv.__init__()` into `trade_rl.rl.environment_reward_execution_resources`.

The extracted boundary owns only:

- `RewardTracker` construction using the resolved reward config and decision hours;
- optional full reward pre-roll minimum-index resolution;
- independent hybrid and shadow `MarketExecutor` construction;
- the maintained `executor is hybrid_executor` compatibility alias;
- a fresh empty reward-history cache.

Policy/schedule resolution, observation contracts, runtime-service wiring, initial mutable state, reset behavior, step behavior, reward mathematics, execution mathematics, and the public constructor signature are unchanged.

Production remains `NO-GO`.

## 2. TDD RED evidence

The clean RED head was:

- commit: `3678530a49f44cbe030397a3bcaf892dfc7038a3`;
- CI run: `30012927157`;
- pytest artifact: `8565892226`;
- artifact digest: `sha256:18355159e454235a5b1597a5a7763483157c7036a5459afce29c4b0336cc8d18`.

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

Complete pytest collection failed with exactly two errors because `trade_rl.rl.environment_reward_execution_resources` did not exist. No production implementation for the boundary was present at the clean RED head.

## 3. Implemented boundary

`EnvironmentRewardExecutionResources` is a frozen, slotted dataclass containing:

- `reward_tracker`;
- `minimum_start_index`;
- `hybrid_executor`;
- `shadow_executor`;
- `executor`;
- `reward_history_cache`.

`EnvironmentRewardExecutionResourcesBuilder` preserves the former inline order:

1. construct `RewardTracker`;
2. preserve the incoming minimum start index;
3. when full reward pre-roll is required and baseline underperformance is enabled, derive the reward-safe minimum index;
4. construct the hybrid executor;
5. construct the shadow executor independently;
6. set the compatibility executor alias to the hybrid executor;
7. return a fresh empty reward-history cache.

This order preserves the existing failure boundary: invalid reward-tracker decision hours fail before reward pre-roll or executor construction.

Each builder call returns a fresh reward tracker, two distinct executors with identical execution-policy identity, and a fresh cache. The supplied dataset, config, reward config, and execution-rule-stress object are preserved by identity.

`ResidualMarketEnv.__init__()` invokes the builder once before observation-contract construction and installs the six returned values under the same maintained attributes through `_install_reward_execution_resources()`.

`MarketExecutor` remains imported by the environment module because constructor-external history reconstruction and annotations still require it.

## 4. Characterization and architecture controls

Direct tests cover:

- no-pre-roll minimum preservation;
- full reward pre-roll minimum derivation;
- exact decision-hours forwarding and reward window sizing;
- fresh reward trackers and caches across builder calls;
- independent hybrid and shadow executors;
- equal execution-policy digests;
- `executor is hybrid_executor`;
- execution-rule-stress identity preservation;
- reward-tracker validation before downstream construction;
- environment integration under existing attributes.

Architecture tests require:

- local ownership of the resource contract and builder;
- exactly one builder invocation and one installer invocation in the constructor;
- builder execution before observation-contract construction;
- absence of direct `RewardTracker`, reward pre-roll, `MarketExecutor`, and reward-cache construction from the constructor;
- preserved builder construction order;
- a constructor source span no greater than 150 lines.

The measured constructor source span is 150 lines, reduced from 168 after PR #140.

## 5. Exact-head verification

The exact implementation and permanent coverage-ratchet head was:

- commit: `5856752aa085e16fd7133059ef3878db072eded5`;
- CI run: `30015551379`;
- PostgreSQL Catalog run: `30015552890`.

The complete test result was:

- 1,343 passed;
- 2 skipped;
- 11 warnings;
- 84.25% total coverage;
- 71.24% total branch coverage.

`trade_rl/rl/environment_reward_execution_resources.py` measured:

- 33 / 33 statements covered;
- 2 / 2 branches covered;
- 100.0% statement coverage;
- 100.0% branch coverage.

A permanent 100.0% critical branch-coverage ratchet is recorded in `pyproject.toml`.

CI run `30015551379` passed:

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

PostgreSQL Catalog run `30015552890` passed:

- Compose validation;
- PostgreSQL startup and readiness;
- dependency installation;
- migrations;
- catalog unit and integration tests;
- cleanup.

Final artifacts from CI run `30015551379` were:

- pytest: `8567063508`, digest `sha256:c976a748b2ecb9fbfbdec7c576f86d99db568d335c07a9eddabfbf4d86037974`;
- architecture: `8567013632`, digest `sha256:59f54e240fcde016239a02305e2b405824e3bfdd95d188464576bbab86a54805`;
- static: `8567013082`, digest `sha256:fc2b2e39561fa0b004a088e276f858bc0739ab3848cda78ecc1a5a5be4cff41c`;
- training image: `8567004787`, digest `sha256:47052f266a483adae4bd2ce32b208733e25acd2069db0a45b6664b434bae3831`;
- Windows: `8566989060`, digest `sha256:9923572953843c424b2d8b2d7297e38ca623b5d75247bd0a2eb3a7fac182794a`;
- Ubuntu: `8566980043`, digest `sha256:783d2880ce400241554e230b165c32ce59deadf9adfa09613b4c02c7f70593e7`.

## 6. Final architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Reward-tracker, full reward pre-roll, dual-executor, compatibility-alias, and reward-cache construction are now typed, independently characterized, fully covered, and prohibited from returning inline to the environment constructor.

The remaining constructor density consists primarily of validated contract assignment and observation/runtime orchestration. A further split requires another concrete behavior-preserving seam rather than mechanical line movement.

This item does not block causal research use. Production remains `NO-GO` until the maintained research, evidence, operational, authorization, and profitability gates pass independently.

## 7. Documentation-head integration gate

The architecture-closeout update is documentation-only and does not alter Python, tests, configuration, or runtime behavior. The exact documentation head must nevertheless pass the normal read-only CI workflow and PostgreSQL Catalog workflow before merge; the earlier code-and-ratchet success is not substituted for that final integration check.
