# RL Observation v2 Implementation Plan

Issue: #489
Spec: `docs/specs/2026-09-12-rl-observation-v2.md`
Status: Active

## Goal

Implement and freeze the minimal causal PPO Observation v2 before real-data M2, preserving historical v1 evidence readers and keeping execution economics / sequence models out of scope.

## Quality contract

### Objective

Use local selected values + availability + normalized staleness, fixed global regime values + availability, current intent and current weight in both PPO fit and replay inference.

### Non-goals

No execution-economics observation, recurrent model, PPO hyperparameter/reward/action change, Ridge/LightGBM behavior change, or real M2 execution.

### Invariants

- policy tensor contains no symbol identity;
- local/global data are point-in-time only;
- same encoder semantics are used by fit and inference;
- fit-symbol round-robin remains unchanged;
- `MarketExecutor + BookState` economics stay authoritative;
- old Candidate/Study artifacts remain readable;
- new observation roster is fixed code, not a tunable run-config field.

### Failure modes

- staleness mis-indexing;
- global order drift;
- unavailable placeholder leakage;
- unknown global feature silently ignored;
- training/inference observation divergence;
- evidence identity omits observation contract;
- v1 reader regression;
- accidental economics/model scope creep.

### Test oracle

Exact encoded vector, metamorphic symbol/future invariance, PPO replay parity, artifact payload/schema, Study payload/digest, and historical-reader fixtures.

### Required layers

Unit/contract, metamorphic, PPO environment integration, candidate artifact compatibility, experiment codec compatibility, Ruff/Format/Mypy, full tests, build/distribution/clean install/package identity.

### Quality gate

Formal RED before production edits; targeted GREEN; falsification; exact-head final CI containing current main; no unresolved Critical/High finding.

## Task 1 — Define RED observation contract

Files:
- modify `tests/strategies/test_ppo_intent.py`
- add/modify strategy-interface tests as needed

RED assertions:

- fixed `PPO_OBSERVATION_SCHEMA == "ppo_observation_v2"`;
- fixed ordered global roster;
- exact tensor order: local values, local available, local staleness, global values, global available, intent, weight;
- tensor shape `3*F + 2*G + 2`;
- dataset global storage order does not change contract order;
- missing global name fails closed;
- unavailable local/global raw values encode to zero while masks/staleness remain explicit;
- symbol rename leaves encoded vector equal;
- future-row mutation leaves an earlier vector equal.

Run normal CI and require failure only because v2 contract is not implemented.

## Task 2 — Add shared observation state

Files:
- modify `trade_rl/strategies/interface.py`
- modify `trade_rl/evaluation/replay.py`
- modify `trade_rl/strategies/rl/ppo.py`
- update direct `StrategyObservation` test fixtures

Implementation:

- add required read-only `feature_staleness` to `StrategyObservation` and shape-check it against local features;
- both production observation producers populate it from `MarketDataset.feature_staleness`;
- introduce fixed PPO observation schema/global roster in the RL owner;
- resolve global indices by exact name in contract order;
- have one encoder consume `StrategyObservation` plus resolved global indices;
- mask unavailable/non-finite local/global values to zero;
- retain bounded normalized staleness supplied by `MarketDataset`;
- keep MLP, action/reward/risk/execution unchanged.

Targeted tests must prove RED→GREEN before identity work.

## Task 3 — Bind new Candidate artifact identity without breaking v1

Files:
- modify `trade_rl/evaluation/runs/artifact.py`
- update candidate artifact/identity tests

Implementation:

- new writes use `lean_candidate_result_v2`;
- include `ppo_observation` contract payload in the summary;
- loader accepts both v1 and v2;
- semantic artifact identity reports the actual loaded result schema;
- v2 validation requires the exact fixed observation contract;
- v1 remains readable and semantically unchanged.

Falsification:

- mutate global roster/order/schema in v2 summary and require rejection or identity mismatch according to the maintained validation boundary;
- historical v1 fixture remains loadable.

## Task 4 — Bind Controlled Study identity with versioned Run config

Files:
- modify `trade_rl/evaluation/experiments/contracts/run.py`
- modify `trade_rl/evaluation/experiments/codec.py`
- update experiment contract/codec/architecture tests

Implementation:

- new `ResolvedRunConfig.from_candidate_spec()` produces `resolved_run_config_v2`;
- v2 payload includes fixed `ppo_observation_schema` and ordered `ppo_global_feature_names`;
- constructor validation requires the fixed v2 contract for v2 payloads;
- historical v1 payload remains accepted and re-serialized exactly as v1, without synthetic v2 fields;
- StudyPlan does not need a new top-level field because its digest already embeds `baseline_config.to_payload()`; prove this directly in tests.

Falsification:

- changing observation roster changes new Run digest and Study digest;
- v1 historical payload digest/round-trip remains stable.

## Task 5 — Candidate execution integration and anti-scope-creep checks

Files:
- modify `trade_rl/evaluation/runs/candidate_suite.py` only as required to pass the fixed contract into PPO fit;
- tests in `tests/evaluation/` / `tests/architecture/`.

Verify:

- Ridge/LightGBM continue receiving exactly existing local `feature_indices`;
- only PPO uses staleness/global representation;
- PPO fit-symbol round-robin unchanged;
- MLP architecture stays `[64,64]` actor/critic;
- action SHORT/FLAT/LONG and reward/economic parity tests unchanged;
- no fee/spread/borrow/participation input appears in PPO encoder;
- no sequence/recurrent dependency appears.

## Task 6 — Durable docs and current-only cleanup

Files:
- update `docs/research/current-status.md` with the frozen Observation v2 pre-M2 contract and blocker relationship with #481/#484;
- update `docs/architecture/lean-core.md` if the durable strategy observation contract needs routing;
- update `docs/README.md` while work is active;
- after verified completion, delete this completed spec and plan and remove Active routing, preserving durable contract in architecture/research docs.

Do not preserve implementation-history prose in current docs.

## Task 7 — Final verification and integration

Before Ready/merge:

1. refresh current `main`, open PRs and #484 overlap;
2. synchronize exact current main into the feature branch if it moved;
3. run exact-head permanent CI;
4. require Ruff, Format, production Mypy, architecture/tooling Mypy as applicable, full tests, Build, distribution source closure/sdist rebuild, clean installed smoke, package identity;
5. review final diff for economics/recurrent/hyperparameter scope creep;
6. independently reconstruct Issue #489 acceptance criteria against final source/tests;
7. inspect review threads/comments;
8. update PR body with exact SHA and evidence;
9. merge only when exact tested head still contains then-current main and no blocking issue remains.

If #484 changes overlapping files before integration, do not reuse old Green; reconcile and rerun the same quality gate.
