# Environment Construction Decomposition Verification

Date: 2026-07-23

## Scope

This verification closes the code-remediation portion of `AUD-RL-001`: `ResidualMarketEnv` already delegated step-time policy to maintained services, but its constructor still concentrated dependency resolution, action/risk validation, observation-contract construction, service assembly, and initial mutable-state creation.

The remediation preserves the public `ResidualMarketEnv(...)` signature and Gymnasium facade while introducing four focused, stateless construction owners:

- `trade_rl.rl.environment_dependencies.EnvironmentDependencyResolver`;
- `trade_rl.rl.environment_observation_contract.EnvironmentObservationContractFactory`;
- `trade_rl.rl.environment_assembly.EnvironmentServiceAssembler`;
- `trade_rl.rl.environment_state.EnvironmentInitialStateFactory`.

Design:

- `docs/superpowers/specs/2026-07-23-environment-construction-decomposition-design.md`

Implementation plan:

- `docs/superpowers/plans/2026-07-23-environment-construction-decomposition.md`

## Final ownership boundary

### Environment facade

`ResidualMarketEnv` remains declared by `trade_rl.rl.environment` and remains the sole owner of episode-varying mutable state. Its constructor now performs bounded orchestration only:

1. stores the dataset and raw optional inputs;
2. resolves construction dependencies;
3. builds the observation contract and spaces;
4. assembles the maintained step-time services;
5. computes the unchanged environment identity;
6. creates and explicitly assigns initial mutable state.

The constructor span was reduced from 505 source lines to 167 source lines. The permanent architecture contract limits it to 180 lines and forbids direct low-level construction of observation builders, executors, episode samplers, books, and step services.

### Dependency resolution

`EnvironmentDependencyResolver` owns:

- trend strategy and market-input-resolver reconciliation;
- alpha enablement, provider requirements, and artifact identity;
- static/provider factor-basis validation and minimum-index behavior;
- pre-trade, portfolio-risk, and risk-input-provider resolution;
- action-spec validation and action-name construction;
- nominal episode/decision timing and reward preroll;
- immutable resolved dependency output.

### Observation contract

`EnvironmentObservationContractFactory` owns:

- flat observation layout and normalizer validation;
- structured sequence builder, policy plane, component spaces, and metadata;
- observation schema and contract digest;
- public observation/action spaces;
- sequence-derived minimum start index.

### Service assembly

`EnvironmentServiceAssembler` owns construction of:

- emergency risk monitoring;
- separate hybrid and shadow executors;
- episode sampling;
- execution, observation, decision, risk, reward, information, and termination services.

It does not own step-time policy; it composes the existing maintained services.

### Initial state

`EnvironmentInitialStateFactory` creates deterministic initial values for books, order books, pending targets, previous action, episode indices, execution state, diagnostics, reward history, and reset state. It retains no state. The facade assigns each mutable field explicitly; reflection or `self.__dict__.update()` is prohibited.

## Preserved contracts

The implementation preserves:

- the complete public constructor signature and `ResidualMarketEnv.__module__`;
- environment, action, observation, and execution-policy digests;
- flat and structured observation schemas, keys, shapes, bounds, and dtypes;
- action semantics and action names;
- validation branches and exception messages;
- reset, step, risk, reward, execution, information, and termination ordering;
- hybrid/shadow executor separation;
- episode sampler and mutable-state ownership;
- Linux and Windows behavior;
- existing primary evidence schemas.

## TDD RED evidence

RED head:

- `b9e67a3e6e23e92350bf7a8c4de3ccf85b31dd6e`

GitHub Actions CI run:

- `29968493768`

The run reached the full test suite after Studio verification, workflow security, Ruff, format, Mypy, Import Linter, dead-code analysis, and Serving smoke had passed. Ubuntu compatibility, Windows compatibility, and the complete training-image build also passed.

The new architecture contract failed because:

- the four construction modules did not exist;
- `ResidualMarketEnv.__init__` spanned 505 lines instead of at most 180.

Result:

- `2 failed, 1237 passed, 2 skipped, 11 warnings`.

RED artifacts:

- Pytest diagnostics `8548782443`, digest `sha256:dc87aba46b2833949a751c16871393d0d19ceecf57e3ad53e24035f6ca1edd19`;
- architecture diagnostics `8548755825`, digest `sha256:e6d034de5102770e4772f9b18785f31e4fab095f591e848ee61e9e8545c9e242`;
- static diagnostics `8548755571`, digest `sha256:eece910858ab83dab7fdb18c7b84af4829c836a2020390eefdc5e43fd409df9e`;
- Studio layout diagnostics `8548748835`, digest `sha256:715b6a873ba8cf91530210b9fd1a41cba560d1c9b5a6e64d5e6c4e27aee9a997`;
- training-image evidence `8548747351`, digest `sha256:afd3f63bec74ae41d311eb243c93d503934582cb7e4999c8f058c52b367c2c63`;
- Windows compatibility `8548739466`, digest `sha256:1e7633dfd553b82f57263524d9427ee0d937f846758f1ce9941dc9cb142405ff`;
- Ubuntu compatibility `8548737485`, digest `sha256:0354d85372ce8fe0fc96aa682bfe7544cd184f4fdbfaf46202df5a040ff00d65`.

## Focused GREEN evidence

One-shot implementation verification run:

- run `29969856315`;
- artifact `8549233624`;
- artifact digest `sha256:03f9d974bcc20e74810a4ae1e6e69b21b3b860b958d3cb3824b42859f1799136`.

Result:

- Ruff fix, format, and check passed;
- Mypy passed for 192 source files;
- `37 passed` across the new architecture/service contracts and focused environment identity, timing, signal-artifact, target-action, sequence, and pending-order suites.

The one-shot workflow, trigger, patch chunks, and generated status/log files were deleted before the clean implementation commit was pushed.

## Pre-refactor characterization evidence

A temporary, non-merge PR (`#116`) captured a deterministic payload from unchanged `main`. The fixture used:

- the direct target-weight environment from `tests/rl/test_target_weight_action.py`;
- two symbols;
- cash initialization at `start_idx=10`;
- reset seed `17`;
- one action `[0.40, 0.0]`.

The canonical payload covers:

- environment, action, observation, and execution-policy digests;
- action and observation spaces;
- minimum index and nominal timing;
- initial books, order books, pending targets, previous action, and reset state;
- seeded reset observation, information, indices, and books;
- one seeded step observation, reward, terminal flags, information, books, pending state, order books, execution state, and diagnostics.

Baseline evidence:

- workflow run `29970965228`;
- artifact `8549654163`;
- artifact digest `sha256:9f35ef95538d70280cd8093bed424166bbac0c2c7801be95107758fee8fb6761`;
- canonical payload SHA-256 `9d6540b3e3d3616bbb41caff036c6ef37228af56506adb030229aead86b11de1`.

The permanent test `tests/rl/test_environment_construction_characterization.py` reconstructs the full payload after the refactor and requires the same SHA-256. PR #116 was closed without merge.

## Review finding resolved during full-suite verification

The first complete implementation run detected one obsolete architecture assertion that required step-service class names to appear directly in `environment.py`. This was not a runtime behavior regression. The contract was updated to verify the new ownership boundary instead:

- the facade delegates to `EnvironmentServiceAssembler.assemble()`;
- `environment_assembly.py` owns construction of all maintained step services;
- the facade retains the service attributes;
- `step()` continues to delegate decision, risk, reward, and information behavior.

The corrected full suite then passed.

## Implementation exact-head verification

Implementation head before this documentation-only commit:

- `4e9193d9710a78c9a32ce01c2ff8fd97f492d438`

GitHub Actions CI run `29971052229`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed-viewport verification: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter and architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1245 passed, 2 skipped, 11 warnings`;
- total coverage: `83.95%`;
- total branch coverage: `4855 / 6862 = 70.75%`;
- environment-construction group: `64 / 100 = 64.0%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29971052222`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

Implementation exact-head artifacts:

- Pytest diagnostics `8549726869`, digest `sha256:3be3e0d5f76d64aed45a2a07442efe60d5160c658c7bb9591b5a64adf5841daf`;
- architecture diagnostics `8549700865`, digest `sha256:b014af717ffd49745a75854d5c70aeefe8044c276970b2811078157c0a4863a1`;
- static diagnostics `8549700609`, digest `sha256:dd18538ae0ebcad7afdf1f4bc8b6e64d5b5b3cb1039fd6dd660c955fd592dcf3`;
- training-image evidence `8549697305`, digest `sha256:d9125553cd5f5c90f102a8476a86d1620e440b508cd749c74fde6618cb47d46f`;
- Studio layout diagnostics `8549694456`, digest `sha256:c4aec4379efebb6e1d8ce91a48d747f146d125de94c381a6a43e7284db4512e4`;
- Windows compatibility `8549690706`, digest `sha256:b27c6fcedf8b37ce82da28dd670d7d6ec4faaf0fc83c239633c89684f75504cd`;
- Ubuntu compatibility `8549688380`, digest `sha256:ab6096fc8620c545987050063f219d78b5bcc8e2d88f0fb7aedefb576f60ef0f`.

## Coverage ownership

A new aggregate critical branch-coverage ratchet protects the four construction modules:

```toml
[tool.trade_rl.critical_coverage.groups.environment_construction]
minimum = 64.0
paths = [
    "trade_rl/rl/environment_dependencies.py",
    "trade_rl/rl/environment_observation_contract.py",
    "trade_rl/rl/environment_assembly.py",
    "trade_rl/rl/environment_state.py",
]
```

The exact implementation-head result was `64 / 100 = 64.0%`. Existing critical thresholds were not lowered.

## Final diff review

The implementation diff contains only:

- design and implementation-plan documentation;
- four construction modules;
- the bounded environment-facade integration;
- architecture, service, and pre-refactor characterization tests;
- the environment-construction coverage ratchet.

No temporary workflow, trigger, source archive, patch payload, generated status file, duplicate production implementation, schema change, direct-exchange behavior, or production authorization remains.

No critical or important code-review issue remains in the implementation head.

This verification document is documentation-only. The documentation-complete head must pass exact-head CI and PostgreSQL verification before merge.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- Paper/live execution equivalence has not been established.
- No profitability or exchange-equivalent fill claim is introduced.
- No primary evidence schema changes.