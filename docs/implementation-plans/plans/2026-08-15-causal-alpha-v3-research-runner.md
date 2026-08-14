# Causal Alpha V3 Research Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing V3 causal-alpha fit/forecast/target primitives to a strict, artifact-bound, resumable real-data workflow ending at deterministic teacher admission and a research-only teacher package.

**Architecture:** Add a V3-specific research lane rather than generalizing the maintained U6 teacher package. Reuse existing Universal runtime artifact closure, causal sample/partition builders, execution cost/liquidity estimators, production episode evaluator, and maintained teacher-admission aggregate while giving all V3 config, signal, freeze, replay, selection, admission, and package artifacts separate schemas and identities.

**Tech Stack:** Python 3.12, NumPy, existing Universal/Gymnasium environment contracts, pytest, Ruff, Mypy, import-linter, GitHub Actions.

## Global Constraints

- Scalar reward remains pure net log growth.
- Hard `max_position_to_market_notional=0.02` remains authoritative.
- One-decision execution delay remains unchanged.
- No validation/test/teacher-admission holdout leakage into fit, signal gate, candidate freeze, or selection.
- All V3 artifacts are research-only and non-promotable.
- DAgger/BC/critic/PPO/Lagrangian execution is out of scope until deterministic V3 teacher admission passes.
- Existing V3 primitive behavior is preserved when new optional arguments are omitted.
- Canonical U6 example configs and maintained teacher package semantics remain unchanged.

---

### Task 1: Strict runner configuration and nested partition contract

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_config.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_config.py`

**Interfaces:**
- Produces `CausalAlphaV3Candidate`, `CausalAlphaV3SignalGate`, `CausalAlphaV3SelectionGate`, `CausalAlphaV3NestedSelectionConfig`, `CausalAlphaV3ResearchConfig`.
- Produces `split_causal_alpha_v3_partitions(...) -> dict[str, CausalAlphaV3NestedPartition]`.

- [ ] Run the authored tests before implementation and record failure due missing V3 runner modules.
- [ ] Implement field-closed JSON parsing using `require_exact_fields` and `require_dataclass_fields`.
- [ ] Reject candidate semantic duplicates, duplicate names, empty grids, and more than eight candidates.
- [ ] Implement chronological signal/economic/holdout split and bind contract digests.
- [ ] Run focused tests and static checks.

### Task 2: Signal cohort and gate evidence

**Files:**
- Extend: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_signal.py`

**Interfaces:**
- Produces `non_overlapping_causal_alpha_v3_rows(...) -> np.ndarray`.
- Produces `CausalAlphaV3SignalScopeMetric`, `CausalAlphaV3SignalGateEvidence`, `evaluate_causal_alpha_v3_signal_gate(...)`.

- [ ] Implement greedy non-overlap selection from decision and realized label-end indices.
- [ ] Implement immutable scope metrics with finite rank correlation, direction accuracy, realized top-bottom spread, cohort indices, and fit/forecast identities.
- [ ] Aggregate scope-level rank IC, top-bottom spread, and direction-accuracy excess with deterministic moving-block bootstrap.
- [ ] Fail closed on insufficient scope count, incomplete coverage, or lower confidence bounds below authored thresholds.
- [ ] Run focused tests and static checks.

### Task 3: Replay/selection evidence contracts and atomic record store

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v3_store.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_store.py`

**Interfaces:**
- Produces `CausalAlphaV3ReplayMetric`, immutable freeze/selection/admission/package evidence contracts, and `CausalAlphaV3RecordStore`.

- [ ] Implement replay metric identity/digest validation and irreversible-rejection helper.
- [ ] Implement atomic per-scope replay writes using canonical JSON and atomic replacement.
- [ ] Re-read every persisted record on resume; reject unknown scope, wrong contract, run/freeze drift, duplicate scope, or digest mismatch.
- [ ] Make identical repeat writes idempotent and conflicting writes fail closed.
- [ ] Run focused tests and static checks.

### Task 4: V3 economic ranking

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_selection.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_selection.py`

**Interfaces:**
- Produces `rank_causal_alpha_v3_candidates(...) -> CausalAlphaV3SelectionEvidence`.
- Raises `CausalAlphaV3SelectionRejected` with complete candidate evidence.

- [ ] Calculate candidate mean gross/net, lower-tail net, turnover, cost, positive-gross fraction, trades, rejection counts, and hard-risk flag.
- [ ] Apply authored economic thresholds without relaxing the maintained -5% lower-tail concept.
- [ ] Rank admissible candidates by lower-tail net, mean net, mean gross, lower turnover, lower cost, deterministic digest.
- [ ] Run focused tests and static checks.

### Task 5: Actionable target compiler contract

**Files:**
- Modify: `trade_rl/learning/causal_alpha_v3.py`
- Test: `tests/learning/test_causal_alpha_v3_actionable_mask.py`

**Interfaces:**
- Extend `causal_alpha_v3_target_path(..., actionable_mask: object | None = None)`.

- [ ] Validate mask shape when supplied.
- [ ] Preserve existing behavior when omitted.
- [ ] On false rows, hold the previous target with reason `unactionable_hold`, no submitted change, and no alpha-driven sign flip.
- [ ] Keep liquidity forced deleveraging authoritative even when forecast is unactionable only if the current position exceeds the causal cap; record `liquidity_deleverage`.
- [ ] Run existing V3 target tests plus the new regression.

### Task 6: Production V3 signal and replay engine

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Extend: `trade_rl/workflows/universal_causal_alpha_v3.py` only for reusable V3 fit cache if needed.
- Tests: add focused integration tests under `tests/workflows/`.

**Interfaces:**
- Build partitions/samples from train bindings and concrete environment factory.
- Fit V3 at exact contract start.
- Generate forecast, production one-way costs, causal liquidity caps, actionable target paths, and replay metrics.

- [ ] Add failing fake-environment integration tests before each production behavior.
- [ ] Build signal-scope evidence from real `CausalAlphaSymbolSamples` with strict realization bounds.
- [ ] Persist signal evidence and freeze candidate set before economic replay.
- [ ] Resume selection from verified per-scope records and prune only irreversible rejection states.
- [ ] Always close concrete environments in `finally`.
- [ ] Run focused integration tests and static checks.

### Task 7: Exactly-once admission and research teacher package

**Files:**
- Extend: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Extend: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Tests: focused admission/package tests.

**Interfaces:**
- Build selected-candidate `EpisodeOracleBatch` targets for train partitions.
- Persist one verified holdout metric per symbol before aggregate admission.
- Produce `UniversalCausalAlphaV3TeacherPackage` only when maintained admission passes.

- [ ] Add failing tests proving holdout is inaccessible before selection and admission failure cannot create a package.
- [ ] Persist/read per-symbol admission records with run/freeze/selection identity closure.
- [ ] Resume by evaluating only absent verified symbol records.
- [ ] Aggregate with maintained `evaluate_causal_alpha_teacher_admission` without threshold changes.
- [ ] Build research-only, non-promotable package on pass.
- [ ] Run focused integration tests and static checks.

### Task 8: CLI, example config, and maintained documentation

**Files:**
- Create: `scripts/run_universal_causal_alpha_v3_research.py`
- Create: `examples/binance/universal-causal-alpha-v3-research.json`
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Test: `tests/scripts/test_run_universal_causal_alpha_v3_research.py`
- Extend canonical V3 documentation/config contract tests.

**Interfaces:**
- CLI exits `0` admitted, `2` signal reject, `3` selection reject, `4` admission reject, `1` invalid/corrupt/unexpected failure.

- [ ] Add CLI tests first.
- [ ] Load maintained `TrainingRunConfig` and `UniversalRuntimeFactoryContext`, then use the existing runtime factory to resolve artifact-bound train runtime inputs.
- [ ] Add a bounded six-candidate research example config; do not alter canonical U6 configs.
- [ ] Document terminal artifacts, resume semantics, and explicit research-only boundary.
- [ ] Run CLI/documentation/canonical contract tests.

### Task 9: Falsification, full verification, and integration

**Files:** no behavior changes unless a defect is reproduced first.

- [ ] Reconstruct acceptance criteria from the spec and review the full diff.
- [ ] Falsify config order, semantic duplicate, source/runtime identity drift, signal/economic/holdout overlap, future-label inclusion, overlapping cohort inflation, partial/tampered record, unknown resume scope, actionable false row, liquidity contraction, hard-risk/tail pruning, admission interruption, and package-after-rejection.
- [ ] Add failing regression tests before every discovered fix.
- [ ] Run Ruff, format, Mypy, import architecture, full pytest/coverage ratchet, Linux/Windows compatibility, training image/package identity, PostgreSQL Catalog, Nautilus Capability when triggered, and exact-head GitHub Actions.
- [ ] Verify final branch diff and exact HEAD. Keep PR Draft until the exact-head software gate is green.
- [ ] Merge only after the user-authorized integration decision and then verify the resulting main HEAD workflows; empirical/economic admission remains a separate result from software quality.
