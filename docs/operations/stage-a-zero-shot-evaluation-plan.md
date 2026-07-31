# Stage A Zero-Shot Evaluation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable, content-addressed Stage A zero-shot evaluation evidence, fold-level paired-bootstrap aggregation, validation-only candidate selection, and selected-candidate-only sealed-test gating.

**Architecture:** Keep artifact validation and strict JSON transport in `stage_a_zero_shot_contracts.py`. Keep statistical aggregation and gate decisions in `stage_a_zero_shot_gate.py`, so the later checkpoint runner can depend on a small pure interface without coupling statistics to filesystem or training code.

**Tech Stack:** Python 3.12, frozen dataclasses, canonical JSON/SHA-256 identities, NumPy deterministic bootstrap, pytest, Ruff, MyPy.

## Global Constraints

- Validation and test symbols never appear in training identities.
- Every declared candidate × fold × seed × split-triplet observation is required exactly once.
- Checkpoint, dataset, execution, and plan identities are revalidated before aggregation.
- Seeds are averaged inside folds; bootstrap resampling units are folds.
- Candidate selection consumes validation evidence only.
- Sealed test accepts exactly one previously selected candidate.
- Bootstrap confidence, resample count, seed, and both thresholds are explicit and content-addressed.
- No PPO, reward, execution simulator, serving, or market-data behavior changes.

---

### Task 1: Immutable evaluation plan and evidence

**Files:**
- Create: `trade_rl/evaluation/stage_a_zero_shot_contracts.py`
- Create: `tests/evaluation/test_stage_a_zero_shot_contracts.py`

**Interfaces:**
- Produces: `StageACandidate`, `StageAZeroShotEvaluationPlan`, `StageAEvaluationObservation`, `StageAEvaluationEvidence`, builders, strict JSON writers/loaders.

- [ ] Write failing tests for plan round-trip, candidate checkpoint/seed closure, complete Cartesian evidence, duplicate/missing observation rejection, checkpoint mismatch, and tamper rejection.
- [ ] Run `python -m pytest tests/evaluation/test_stage_a_zero_shot_contracts.py -q` and verify collection fails because the module is absent.
- [ ] Implement frozen records, digest payloads, exact field closure, canonical ordering, builders, and strict load/write functions.
- [ ] Re-run the focused tests and ensure all pass.
- [ ] Run Ruff format/check and MyPy for the new module.

### Task 2: Fold-level aggregation and validation selection

**Files:**
- Create: `trade_rl/evaluation/stage_a_zero_shot_gate.py`
- Create: `tests/evaluation/test_stage_a_zero_shot_gate.py`

**Interfaces:**
- Consumes: exact plan/evidence types from Task 1.
- Produces: `StageACandidateSummary`, `StageAValidationSelection`, `summarize_stage_a_candidate`, `select_stage_a_validation_candidate`.

- [ ] Write failing tests proving seeds/triplets are averaged within folds, bootstrap output is deterministic, the positive candidate is selected, and no candidate is selected below threshold.
- [ ] Run the gate test and verify import failure.
- [ ] Implement deterministic fold-resampling, content-addressed summaries, and deterministic validation selection.
- [ ] Re-run contract and gate tests.
- [ ] Run Ruff format/check and MyPy.

### Task 3: Selected-candidate-only sealed-test decision

**Files:**
- Modify: `trade_rl/evaluation/stage_a_zero_shot_gate.py`
- Modify: `tests/evaluation/test_stage_a_zero_shot_gate.py`

**Interfaces:**
- Produces: `StageASealedTestDecision`, `evaluate_stage_a_sealed_test`.

- [ ] Write failing tests that reject validation evidence at the test gate, reject test evidence containing unselected candidates, and pass/fail the selected candidate against the predeclared test threshold.
- [ ] Run the focused tests and verify the new interface is absent.
- [ ] Implement the fail-closed sealed-test decision and content-addressed result.
- [ ] Re-run all Stage A evaluation tests.

### Task 4: Documentation and full verification

**Files:**
- Create: `docs/operations/stage-a-zero-shot-evaluation-design.md`
- Create: `docs/operations/stage-a-zero-shot-evaluation-plan.md`

- [ ] Run focused evaluation tests.
- [ ] Run `python -m ruff format --check` and `python -m ruff check` on all changed Python files.
- [ ] Run `python -m mypy trade_rl/evaluation/stage_a_zero_shot_contracts.py trade_rl/evaluation/stage_a_zero_shot_gate.py`.
- [ ] Publish the branch and open a draft PR.
- [ ] Require complete repository CI, both OS compatibility jobs, Training image, and PostgreSQL Catalog on one exact head before merge.
