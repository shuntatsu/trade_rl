# Environment Initial State Extraction Verification — 2026-07-23

## 1. Scope

This verification records the behavior-preserving extraction of the final invocation-local mutable-state construction from `ResidualMarketEnv.__init__()` into `trade_rl.rl.environment_initial_state`.

The extracted boundary owns only the fresh initial values for:

- start, end, and current indices;
- independent hybrid and shadow books;
- decision-step index, episode seed, episode hours, and initial-state mode;
- previous action and pending targets;
- independent hybrid and shadow order books;
- position age, observation execution state, action diagnostics, and reset flag.

`_reward_history_cache`, reward tracker and reward-preroll construction, both market executors, observation/runtime contracts, reset logic, and step logic remain outside this boundary.

The public constructor signature is unchanged. Production remains `NO-GO`.

## 2. TDD RED evidence

The clean RED head was:

- commit: `68979db8203244a18d5347e707a90fe97451cb7e`;
- CI run: `30003405592`;
- pytest artifact: `8561986383`;
- artifact digest: `sha256:f62c93f185c382804e3a4137713cc04e35d96beaf79b6e50305d41433de530be`.

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

Complete pytest collection failed with exactly two errors because `trade_rl.rl.environment_initial_state` did not exist. No production implementation for the boundary was present at the clean RED head.

## 3. Implemented contract

`EnvironmentInitialStateRequest` is a frozen, slotted dataclass containing the validated dataset, environment config, action spec, and minimum start index.

`EnvironmentInitialState` is a frozen, slotted assembly container containing the exact 18 values installed into the existing environment attributes. The container prevents field rebinding but does not claim that contained books, arrays, order books, execution state, or diagnostics are immutable.

`EnvironmentInitialStateFactory.create()` preserves the former construction order:

1. resolve the start index and initial marks;
2. create the zero-quantity hybrid book using configured capital and dataset contract multipliers;
3. clone an independent shadow book;
4. create zero decision and episode defaults;
5. create the `float32` previous-action vector;
6. create independent empty order books;
7. create the `float64` position-age vector;
8. create zero observation execution state;
9. create a fresh action diagnostics accumulator;
10. return `has_reset=False`.

Each factory call returns fresh mutable objects and arrays.

`ResidualMarketEnv.__init__()` invokes the factory once after the environment digest is computed, then installs the returned values under the same existing attribute names through `_install_initial_state()`.

## 4. Characterization and architecture controls

Direct tests cover:

- exact indices and episode defaults;
- configured cash, initial marks, and contract multipliers;
- independent hybrid and shadow books;
- independence across two factory calls;
- `float32` previous-action shape and values;
- `float64` position-age shape and values;
- independent empty order books;
- fresh observation execution state and diagnostics;
- environment integration under the maintained attribute names.

Architecture tests require:

- local ownership of the request, state, and factory types;
- exactly one factory invocation and one request construction in the environment constructor;
- factory invocation after environment-digest construction;
- absence of direct book, order-book, execution-state, diagnostics, and zero-array creation from the constructor;
- preservation of creation order inside the factory;
- a constructor source span no greater than 170 lines.

The measured constructor source span is 168 lines, reduced from 186 after PR #126.

## 5. Exact-head implementation verification

The clean implementation head before the permanent coverage ratchet was:

- commit: `54a2d65b32356f07a5aa7ce577dd2cf8e46f6a87`;
- exact-head CI run: `30005702390`.

The complete test result was:

- 1,321 passed;
- 2 skipped;
- 11 warnings;
- 84.21% total coverage;
- 71.28% total branch coverage.

`trade_rl/rl/environment_initial_state.py` measured:

- 45 / 45 statements covered;
- no executable branch points;
- 100.0% statement coverage;
- 100.0% branch coverage under the maintained coverage metric.

The pytest artifact was:

- artifact ID: `8562956203`;
- digest: `sha256:cb15690f72ee8a04a8e40ff290c9a345ffe6f530f2d1645b2740f0eb102b9b14`.

CI run `30005702390` passed:

- Studio frontend and fixed-viewport verification;
- workflow-security checks;
- Ruff and Ruff formatting;
- Mypy;
- Import Linter;
- dead-code reporting;
- recovery and structured-serving smoke;
- complete pytest and coverage;
- existing critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- complete training-image build and packaged non-root runtime probe.

A permanent 100.0% critical coverage ratchet for `trade_rl/rl/environment_initial_state.py` is recorded in `pyproject.toml`.

## 6. Final architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

Initial mutable-state construction is now typed, independently characterized, fully covered, and prohibited from returning inline to the environment constructor.

The remaining constructor density consists primarily of:

- reward-tracker and reward-preroll construction;
- hybrid and shadow market-executor construction;
- observation/runtime contract assignment.

This item does not block causal research use. Production remains `NO-GO` until the maintained research, evidence, operational, authorization, and profitability gates pass independently.
