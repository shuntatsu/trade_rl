# Generation-Bound Telemetry Polling Verification

Date: 2026-07-22

## Scope

This verification covers the generation-bound telemetry polling change implemented in Draft PR #83.

Design:

- `docs/superpowers/specs/2026-07-22-telemetry-cursor-generation-design.md`

Implementation plan:

- `docs/superpowers/plans/2026-07-22-telemetry-cursor-generation.md`

The branch starts from PR #81 exact head:

- `2e56fd7b42f2677dc0440ff9ec3cc03a55e5c786`

## Implemented contract

- The internal sidecar index schema is `training_telemetry_index_v2`.
- The telemetry JSONL record schema remains `training_telemetry_v1`.
- Every valid index has a canonical lower-case UUID stream generation.
- A normal append preserves the generation.
- Stream replacement, truncation, invalid index state, or index loss creates a new generation.
- A request with a stale expected generation receives no records, `next_sequence=0`, and `reset_required=True`.
- No-growth status and event polls do not rewrite or `fsync` the sidecar index.
- The process lock is released after an identity-verified bounded snapshot is captured; page-body parsing occurs outside the append serialization lock.
- Studio API responses expose `streamGeneration` and `resetRequired`.
- The frontend clears its in-memory buffer and sequence cursor before replaying a replacement generation.
- Status and Events responses from different generations are discarded and retried once rather than published as a mixed snapshot.
- The existing frontend `connection` contract, 2,048-record buffer cap, and stale-request cancellation behavior remain intact.

## TDD evidence

### Python telemetry RED

Exact head:

- `6407c5496e402dce36c71375a7f64de05e379771`

Workflow:

- Run `29915043603`
- Conclusion: expected failure

Artifact:

- ID `8527504146`
- Digest `sha256:c6c97b601316a984bfad1092a0f2f2a38a2628e550515d8622ecd60c35871dc6`

Observed failures before implementation:

- telemetry status and pages had no generation fields;
- the reader accepted no expected-generation cursor;
- stream replacement and index loss could not request an explicit reset;
- unchanged polls rewrote the index;
- page parsing held the process lock and blocked an append.

The deterministic near-tail bounded-parse test already passed and was retained as a regression contract.

### Python telemetry GREEN

Implementation head:

- `c04f4084375e6e56b91ffe415ed69026fa4d6389`

Workflow:

- Run `29915319097`
- Conclusion: success

Verification included:

- Ruff formatting and linting;
- Mypy across the Python project;
- telemetry unit tests;
- spawn-based process-concurrency tests;
- training telemetry integration tests.

### Studio backend RED

Workflow:

- Run `29915513524`
- Conclusion: expected failure

Artifact:

- ID `8527690750`
- Digest `sha256:1e08ab17ac5482b763f1dc24777d20058f7a5c9c9f5a2670e13b7cc1a8372896`

Observed failures before implementation:

- status and event responses did not expose a stream generation;
- stale generation queries were ignored;
- invalid generation queries were not rejected by the endpoint contract.

### Studio backend GREEN

Implementation head:

- `ec6f7cdc76bdb355dadcaf6d717ab7d113887ce7`

Workflow:

- Run `29915833168`
- Conclusion: success

Verification included Studio API tests, telemetry tests, process tests, integration tests, Ruff, formatting, and Mypy.

### Frontend RED

Workflow:

- Run `29916129060`
- Conclusion: expected failure

Artifact:

- ID `8527945463`
- Digest `sha256:78b78936309483c08bea9bc37c6b48210860f5456d83a1dd6085495072dd6987`

Observed failures before implementation:

- stale generation responses did not trigger a replay from sequence zero;
- mixed Status and Events generations were published instead of retried;
- malformed generation values were accepted by guards;
- non-boolean reset flags were accepted by guards.

### Frontend GREEN

Implementation head:

- `69ff12afa4672b810bc4a74b775b49596e753967`

Workflow:

- Run `29917535977`
- Conclusion: success

Verification included:

- focused generation-reset hook tests;
- the complete Vitest suite;
- TypeScript type checking;
- production frontend build.

## Cross-platform verification

Focused cross-platform workflow:

- Run `29917703239`
- Conclusion: success

Linux artifact:

- ID `8528622708`
- Digest `sha256:95065be9c178d7942bff925ea464033e33e645729f21505f1fdbad727b695575`

Windows artifact:

- ID `8528621179`
- Digest `sha256:9d36bfb92bc44ae85a20bedc164450e266566702a9d20375af05ce1dbca844bc`

Both platforms passed the telemetry generation, process-concurrency, integration, and Studio API focused suites. Linux additionally passed Ruff, format checking, and Mypy. The Windows run exercised the native `msvcrt.locking` path.

## Full-suite verification and flaky-test disposition

Cleanup head:

- `dc66ac1558a4b317ef3f02b26d7fbc352966ec3b`

The first Core attempt reached the full test suite and produced:

- `1196 passed, 1 failed, 2 skipped, 11 warnings`;
- total coverage `83.45%`.

The single failure was the existing Torch gradient comparison:

- `tests/rl/test_sequence_policy_core.py::test_projection_after_selection_matches_legacy_outputs_and_gradients`
- observed maximum absolute difference `4.112720489501953e-06` against tolerance `1e-06`.

That same test had previously shown the same non-deterministic behavior outside this telemetry change. The failed Core job was rerun without changing source or tests.

Same-head rerun:

- CI run `29917834673`
- Rerun Core job `88916816325`
- Conclusion: success

The rerun produced:

- `1197 passed, 2 skipped, 11 warnings`;
- total coverage `83.45%`;
- total branch coverage `70.35%`;
- `trade_rl/telemetry/indexed_training.py`: `76/110 = 69.09%` branch coverage.

Successful rerun artifact:

- ID `8528843820`
- Digest `sha256:270768dc6ea6e5f4a0d609d34af38d3b56fe9ca2088f187d21c9e31ce95e0496`

Because the failure did not reproduce on the identical head and the telemetry-focused Linux/Windows suites were already green, no unrelated Torch test tolerance change was included in this PR.

## Coverage ratchet verification

Ratchet head:

- `664e8cd956a3d9ef62139a46cbdc5f62376807ec`

The per-file critical branch threshold for `trade_rl/telemetry/indexed_training.py` was raised from `68.0%` to `69.0%`. No existing threshold was reduced.

GitHub Actions:

- CI run `29918540093`: success
- PostgreSQL Catalog run `29918540139`: success

Core checks:

- Studio frontend tests: success
- fixed viewport verification: success
- workflow security: success
- Ruff: success
- Ruff format check: success
- Mypy: success
- import architecture: success
- dead-code report: success
- recovery and structured serving smoke: success
- full tests and coverage: success
- critical branch coverage: success
- CLI smoke: success
- Ubuntu compatibility: success
- Windows compatibility: success
- training image build and non-root packaged runtime probe: success

Full-suite result:

- `1197 passed, 2 skipped, 11 warnings`
- total coverage `83.45%`
- total branch coverage `70.35%`
- indexed telemetry branch coverage `69.09%`
- critical telemetry threshold `69.09% >= 69.0%`

Pytest artifact:

- ID `8528980253`
- Digest `sha256:48d40000ee917171166d47297567af2ded2db20cddc49ba831403b1f2f26949c`

PostgreSQL steps passed:

- Compose validation;
- PostgreSQL startup and readiness;
- dependency installation;
- migrations;
- unit and integration tests;
- clean shutdown.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- The telemetry JSONL evidence schema remains v1.
- The index is still a rebuildable cache and is not treated as primary evidence.
- No malformed or truncated telemetry evidence is automatically repaired.
- No record from a replacement generation is returned against an old generation cursor.
- PR #83 remains Draft and is not merged.

## Final-head requirement

This document commit creates a new exact head. Normal CI and PostgreSQL Catalog must both report `completed/success` for that documentation head before the PR is described as fully verified.
