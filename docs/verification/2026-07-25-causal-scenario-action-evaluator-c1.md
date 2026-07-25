# Causal Scenario Action Evaluator C1 Verification

Date: 2026-07-25

Scope: evaluation-only C1 causal scenario action evaluator. This change does not connect the evaluator to training, Serving, promotion, release, or production execution paths.

## Implemented surface

- immutable causal query, scenario, projected-candidate, rollout-evidence, and result contracts
- deterministic residual candidate generation and semantic deduplication
- scenario-relative log-return values, downside-only CVaR, score, regret, and deterministic confidence intervals
- deterministic selection and tie-breaking
- fail-closed artifact writer and loader with file-closure and digest validation
- architecture boundary test prohibiting maintained runtime paths from importing C1

## Exact implementation verification

The initial implementation was reconstructed from hash-verified bounded chunks and tested before its source commit was created.

Initial verification workflow run: `30148623380`
Initial verified source commit: `a936ad38bd7285f7501d7327647d46239c4127df`

A review regression test then proved that positive baseline-relative advantages must contribute zero downside loss. The implementation was corrected to compute `max(-advantage, 0)` before CVaR aggregation.

Final correction workflow run: `30148993028`
Final corrected source commit: `f1e23e26d1f89710a1d37ac6a4d2a1e643edb672`

The following gates passed in Python 3.12 for the corrected implementation:

- `ruff check .`
- `ruff format --check --diff .`
- `mypy .`
- `lint-imports`
- `python -m compileall -q trade_rl tests`
- 66 focused C1 and architecture tests
- 100% statement and branch coverage for `causal_scenario_values` and `causal_scenario_artifact`
- complete repository `pytest -q`

The final review head contains no temporary patch, chunk, or verification-workflow files. Standard repository CI must pass again on the exact final review head before merge.

Production status remains `NO-GO`; C1 is an evaluation-only research component.
