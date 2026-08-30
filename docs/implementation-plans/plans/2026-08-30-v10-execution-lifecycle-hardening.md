# V10 Execution Lifecycle Hardening Implementation Plan

Base: `main@3274e0caf2e0f1d4ad7a2942133621e9c1cc9a05`

## Task 1 — Characterize and RED-test reduce-only PreTrade

Files:
- `tests/risk/test_pretrade.py`

Add tests proving the desired API and safety properties before production changes:

- same-direction micro reduction bypasses hysteresis/no-trade only with explicit reduce-only mask;
- identical normal reduction is still suppressed;
- reduce-only add fails closed;
- reduce-only sign flip fails closed;
- malformed/non-boolean/shape-mismatched masks fail closed;
- emergency flatten behavior remains unchanged.

Run the focused test file and confirm new tests fail for missing `reduce_only_mask` support.

## Task 2 — Implement reduce-only PreTrade contract

Files:
- `trade_rl/risk/pretrade.py`
- `tests/risk/test_pretrade.py`

Implement the minimal optional mask contract. Keep max-turnover, hard-limit, drawdown, and emergency semantics unchanged. Re-run RED tests to GREEN and the full PreTrade test module.

## Task 3 — RED-test environment risk plumbing and delay alignment

Files:
- `tests/rl/test_environment_risk.py` or existing focused risk-projector tests
- a focused environment/decision-delay test module using the repository's maintained fixtures

Add tests proving:

- `EnvironmentRiskRequest.reduce_only_mask` reaches `PreTradeRisk`;
- normal requests retain the old all-false behavior;
- the reduce-only mask is paired with the same delayed target when signal delay is one decision;
- reset/time-limit clears pending intent state.

Observe RED before production changes.

## Task 4 — Implement environment risk plumbing

Files:
- `trade_rl/rl/environment_risk.py`
- `trade_rl/rl/environment.py`
- any focused initial-state/state-transfer module required to keep reset state explicit

Add the optional request field and the smallest environment-owned pending-intent state needed to mirror target delay. Do not alter ActionSpec or normal `step()` semantics. Add a narrow method for setting the next submission's reduce-only mask; default is all false.

Run Task 3 tests to GREEN plus related environment/decision tests.

## Task 5 — RED-test V10 hierarchy use of reduce-only intent

Files:
- `tests/learning/test_causal_alpha_v10_closed_loop.py`
- `tests/learning/test_causal_alpha_v10_closed_loop_falsification.py`
- `tests/simulation/test_causal_alpha_v10_execution_contract.py`

Add tests proving:

- a tiny over-cap position requests the cap rather than flat;
- the metadata marks that action as reduce-only risk projection;
- `FLATTEN_ON_RISK_BREACH` remains flat;
- normal entry/add/hold does not claim reduce-only;
- no direct flip is introduced.

Observe RED before modifying V10 production code.

## Task 6 — Implement V10 reduce-only wrapper/metadata

Files:
- `trade_rl/learning/causal_alpha_v10_hierarchy.py`
- `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`

Change the risk-cap path so non-flat-on-breach modes always request the same-direction capped reduction. Add an explicit metadata boolean/vector indicating reduce-only intent for `risk_cap_projection`, `risk_cap_flatten`/explicit exit as defined in the spec. In hierarchical replay only, wrap the environment so the metadata is sent into the environment's pending reduce-only channel before `step()`.

Bind the new execution-intent schema marker into the hierarchy/compiler identity. V8/V9 controls continue through ordinary replay.

Run Task 5 tests to GREEN and verify control target paths/digests are unchanged under identical base inputs.

## Task 7 — RED-test hard-risk evidence

Files:
- `tests/learning/test_rollout_evaluation.py`
- focused risk-evidence test if clearer

Add synthetic evaluation cases where `hybrid_risk` exposes final feasible weights, `risk_scale`, `max_abs_weight`, `max_gross`, and fail-closed tolerance:

- valid projection -> false;
- max-abs violation -> true;
- gross violation -> true;
- risk-scale zero with nonzero feasible target -> true;
- valid projection followed by post-step price drift -> false.

Observe RED against the current fixed `False` implementation.

## Task 8 — Implement authoritative hard-risk evidence

Files:
- `trade_rl/risk/pretrade.py`
- `trade_rl/rl/environment_risk.py`
- `trade_rl/learning/rollout_evaluation.py`

Carry `max_abs_weight` and fail-closed tolerance through `RiskConstrainedTarget`. Evaluate the final feasible risk target at each step and aggregate an `any()` boolean into canonical collapse evidence. Do not use post-step realized weight as the safety oracle.

Run Task 7 to GREEN plus V6/V7 Selection/replay contract tests.

## Task 9 — RED-test lifecycle trace boundaries

Files:
- `tests/learning/test_rollout_evaluation.py`
- `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

Add tests for:

- requested policy target != submitted target != delayed execution target where delay is active;
- trace persistence of risk reasons/risk scale;
- transition classification entry/hold/reduce/exit/flat;
- flatten initiator classification;
- V10 validation rejects an unexplained realized non-flat -> flat transition;
- backward reading of existing v1 step traces remains possible where the generic reader is used, while new V10 replay identity rejects stale forensic evidence.

Observe RED.

## Task 10 — Implement lifecycle trace v2 and V10 leaf identity

Files:
- `trade_rl/learning/rollout_evaluation.py`
- `trade_rl/workflows/universal_causal_alpha_v8_replay.py` only if backward trace decoding requires it
- `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- optional focused V10 diagnostics module if that keeps ownership clearer

Persist the submitted and execution-intent boundaries and derived transition evidence. Keep legacy decode explicit; do not silently reinterpret old fields as new evidence. Bump V10 replay-leaf identity so old leaves are not resumed as if they contain the new forensic contract.

Run Task 9 to GREEN.

## Task 11 — Related regression and falsification

Run at least:

- PreTrade tests;
- RL environment risk/decision/execution tests;
- rollout evaluation tests;
- V10 closed-loop and execution-contract tests;
- V5/V6/V7/V8/V9/V10 replay/attribution/Selection tests;
- hard-risk falsification tests;
- reduce-only add/flip attempts;
- V8/V9 unchanged-control tests.

Any new failure is investigated before broadening verification.

## Task 12 — Static, architecture, full verification

Run on the exact final source-bearing HEAD:

- Ruff check affected surface;
- Ruff format check affected surface;
- Mypy affected source files;
- import-linter;
- architecture tests;
- full training capability audit;
- `uv build`;
- full pytest suite with the repository's normal coverage settings where practical.

If baseline failures remain on `main`, run the same suite at exact base `3274e0caf2e0f1d4ad7a2942133621e9c1cc9a05` and compare by test identity rather than assumption.

## Task 13 — Self-review, independent falsification, Draft PR

Review the complete diff against the specification, especially:

- no V10 constants or numerical gates changed;
- no reduce-only path available to add/flip;
- V8/V9 controls use normal environment semantics;
- no temporary verification workflow/helper remains in the PR;
- schema migrations are explicit;
- stale artifacts fail closed;
- all CI/results refer to the same final HEAD.

Create a Draft PR against current `main`. If it fully supersedes the stale Draft PR #424, document the supersession and close #424 without merging it. Do not merge the new PR without explicit user permission.

## Economic follow-up (outside code-completion gate)

After code quality gates pass, build a new image and run a fresh DB-backed V10 Selection into a new immutable output root. Compare lifecycle distributions, holding bars, forced-flat initiators, costs, minimum/median symbol net wealth, positive-scope fraction, and CVaR. Do not reuse r20 as a fresh holdout or claim profitability from code-level tests.
