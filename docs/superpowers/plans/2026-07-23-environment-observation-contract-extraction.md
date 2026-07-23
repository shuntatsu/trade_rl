# Environment Observation Contract Extraction Implementation Plan

Date: 2026-07-23

## Goal

Reduce the remaining `AUD-RL-001` construction-density risk by moving deterministic observation-contract assembly out of `ResidualMarketEnv.__init__()` without changing observations, validation behavior, schemas, digests, spaces, minimum indices, or runtime policy.

## Constraints

- Preserve every existing public environment field and value.
- Preserve validation order and exact error strings.
- Keep mutable episode, reward, risk, execution, and book state in the environment facade.
- Do not change the public `ResidualMarketEnv` constructor.
- Do not claim production readiness or full `AUD-RL-001` closure.
- Follow RED -> GREEN -> REFACTOR and capture the failing architecture contract before production code.

## Task 1: Add the RED architecture contract

Create `tests/architecture/test_environment_observation_contract_decomposition.py` that requires:

- `trade_rl.rl.environment_observation_contract` to exist;
- `EnvironmentObservationContract` and `EnvironmentObservationContractBuilder` to be defined there;
- `ResidualMarketEnv.__init__()` to delegate to the builder;
- the constructor not to directly reference `spaces.Box`, `spaces.Dict`, `SequenceWindowSpec`, `build_sequence_policy_plane`, or `observation_passthrough_indices`;
- the constructor source span to be at most 360 lines.

Run the focused test and retain the expected missing-module/delegation failure.

## Task 2: Add characterization tests

Create direct builder tests covering:

- flat observation schema, digest, layout, observation/action spaces, and minimum index;
- structured sequence schema, digest, layout metadata, component spaces, policy plane, and sequence-derived minimum index;
- normalizer size, dataset, observation schema, schema digest, action identity, alpha artifact, factor artifact, and passthrough errors;
- sequence normalizer dataset and schema errors;
- exact sequence window type/order validation errors.

Use existing dataset/config/normalizer test factories where available. Tests must assert behavior, not implementation details.

## Task 3: Implement the typed observation contract

Create `trade_rl/rl/environment_observation_contract.py` with:

- frozen, slotted `EnvironmentObservationContract`;
- `EnvironmentObservationContractBuilder`;
- private validation/space helpers only where they reduce duplication while preserving validation order.

The builder owns static observation contract construction only. It must not import the environment facade, execution, rewards, risk, mutable books, or episode state.

## Task 4: Delegate from the environment

Replace the inline observation-contract block in `ResidualMarketEnv.__init__()` with one builder call and assignments for:

- `observation_builder`;
- `layout`;
- `asset_active_column`;
- `sequence_observation_builder`;
- `sequence_policy_plane`;
- `sequence_layout_metadata`;
- `_observation_schema`;
- `_observation_contract_digest`;
- `observation_space`;
- `action_space`;
- `_minimum_start_index`.

Remove imports used only by the extracted block. Keep `content_digest` if it remains used elsewhere.

## Task 5: Focused verification

Run:

```bash
pytest -q tests/architecture/test_environment_observation_contract_decomposition.py
pytest -q tests/rl/test_environment_observation_contract.py
pytest -q tests/rl/test_environment.py tests/rl/test_sequence_observations.py tests/rl/test_sequence_normalization.py
ruff check trade_rl/rl/environment.py trade_rl/rl/environment_observation_contract.py tests/architecture/test_environment_observation_contract_decomposition.py tests/rl/test_environment_observation_contract.py
ruff format --check trade_rl/rl/environment.py trade_rl/rl/environment_observation_contract.py tests/architecture/test_environment_observation_contract_decomposition.py tests/rl/test_environment_observation_contract.py
mypy trade_rl/rl/environment.py trade_rl/rl/environment_observation_contract.py
```

## Task 6: Full verification

Run the repository-required checks, including full pytest/coverage, import-linter, PostgreSQL integration, platform matrix, production build/image checks, and exact-head GitHub Actions. Do not lower coverage thresholds. Add a new measured threshold only if the exact-head report supports it.

## Task 7: Closeout

Document the exact commit SHA, test totals, coverage, platform results, and unchanged production `NO-GO` status. Update the architecture closeout as `OPEN RISK, FURTHER REDUCED` unless independent evidence justifies stronger wording. Mark the PR ready only after exact-head verification, then squash merge and verify `main`.
