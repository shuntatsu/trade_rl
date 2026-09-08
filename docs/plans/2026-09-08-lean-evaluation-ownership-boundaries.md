# Lean Evaluation Ownership Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for the relocation and `superpowers:verification-before-completion` before any completion claim.

**Goal:** Reorganize `trade_rl.evaluation` into explicit comparison, robustness, and run ownership packages while preserving all evaluation semantics and the exact package-level public API.

**Architecture:** Keep `replay.py`, `metrics.py`, `evidence.py`, `series.py`, and the existing `gates/` package at evaluation root. Mechanically move comparison utilities to `comparison/`, capacity/trade/fold/perfect-information/walk-forward robustness logic to `robustness/`, and candidate orchestration/CLI to `runs/`. Old private flat paths are deleted with no compatibility shims.

**Tech Stack:** Python 3.12, NumPy/SciPy, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md` plus Amendment 1.

**Exact base:** `006278933f1ce499900d0f5019cbd5b92fb18fed` (verified Phase 3B final head).

## Quality Contract

### Objective
- Make the physical evaluation tree match the approved responsibilities without changing numerical, statistical, replay, gating, artifact, or candidate-selection behavior.

### Non-goals
- No new experiment-loop feature.
- No metric/statistical formula change.
- No candidate thresholds, strategy selection, fit/evaluation scope, replay, execution, accounting, risk, or profitability change.
- No broad docs cleanup; only executable current references required by moved CLI/module paths are updated here.

### Acceptance Criteria
1. Root KEEP files remain: `__init__.py`, `replay.py`, `metrics.py`, `evidence.py`, `series.py` and `gates/`.
2. `comparison/` owns `paired.py`, `strategies.py`, `bootstrap.py`, `seed_robustness.py`.
3. `robustness/` owns `capacity.py`, `closed_trades.py`, `fold_metrics.py`, `perfect_information/{bound.py,solver.py}`, and `walk_forward/{capabilities.py,folds.py,sealed_test.py,stitching.py}`.
4. `runs/` owns `candidate.py` and `candidate_suite.py`.
5. Every old moved flat/private path is absent; no forwarding shim survives.
6. Exact `trade_rl.evaluation.__all__` and package-level exported objects are preserved.
7. `python -m trade_rl.evaluation.runs.candidate` is the maintained CLI path and current README usage is updated.
8. Candidate artifact identity/determinism, fit/evaluation scope separation, per-symbol replay, walk-forward sealing/stitching, bootstrap/paired/seed statistics, perfect-information bound, capacity, closed-trade diagnostics, and fold metrics are behaviorally unchanged.
9. Normalized AST comparison against the exact base finds no unauthorized non-import semantic difference in moved/affected production modules.
10. Final helper-free exact HEAD passes Ruff, Format, Mypy, full `tests/`, package identity, and normal PR CI.

### Invariants
- `MarketExecutor + BookState` remain the economic/accounting authority used by replay.
- Dataset and candidate result identities remain deterministic.
- Evaluation remains independent per symbol where specified.
- Gate and sealed-test fail-closed behavior remains unchanged.
- Root package public export order/value remains unchanged.

### Failure Modes
- stale import points to deleted flat module;
- CLI path documented but no longer executable;
- import rewrite accidentally changes import membership/order-sensitive runtime behavior;
- walk-forward modules split into a cycle or lose sealed-test contracts;
- perfect-information solver/bound relationship changes;
- candidate-run identity or scope propagation changes;
- test passes while one old forwarding file remains.

### Test Oracle
- architecture layout tests;
- exact `evaluation.__all__` regression;
- candidate-run/suite artifact and scope tests;
- evaluation metrics/bootstrap/comparison/seed/capacity/closed-trade/fold/perfect-information/walk-forward/replay tests;
- normalized AST equivalence with authorized owner-path normalization only;
- exact-head full CI.

### Required Test Layers
- Static analysis: Ruff + Mypy.
- Unit/property/contract: existing evaluation tests.
- Integration/regression: candidate suite/run, replay + simulation/risk contracts, artifact identity tests.
- Architecture: final paths, retired path absence, public facade.
- Falsification: AST and stale-import scan against exact base.

### Quality Gate
- Do not call Phase 4A complete until all Acceptance Criteria are evidenced, helper files are absent, exact final HEAD has normal CI Green, and falsification finds no unresolved semantic difference.

---

## File Mapping

### KEEP at evaluation root

```text
KEEP trade_rl/evaluation/__init__.py
KEEP trade_rl/evaluation/replay.py
KEEP trade_rl/evaluation/metrics.py
KEEP trade_rl/evaluation/evidence.py
KEEP trade_rl/evaluation/series.py
KEEP trade_rl/evaluation/gates/**
```

### Comparison

```text
MOVE trade_rl/evaluation/bootstrap.py -> trade_rl/evaluation/comparison/bootstrap.py
MOVE trade_rl/evaluation/comparisons.py -> trade_rl/evaluation/comparison/paired.py
MOVE trade_rl/evaluation/seed_robustness.py -> trade_rl/evaluation/comparison/seed_robustness.py
MOVE trade_rl/evaluation/strategy_comparison.py -> trade_rl/evaluation/comparison/strategies.py
CREATE trade_rl/evaluation/comparison/__init__.py
```

### Robustness

```text
MOVE trade_rl/evaluation/capacity.py -> trade_rl/evaluation/robustness/capacity.py
MOVE trade_rl/evaluation/closed_trades.py -> trade_rl/evaluation/robustness/closed_trades.py
MOVE trade_rl/evaluation/fold_metrics.py -> trade_rl/evaluation/robustness/fold_metrics.py
MOVE trade_rl/evaluation/perfect_information_bound.py -> trade_rl/evaluation/robustness/perfect_information/bound.py
MOVE trade_rl/evaluation/_perfect_information_lp.py -> trade_rl/evaluation/robustness/perfect_information/solver.py
MOVE trade_rl/evaluation/walk_forward/__init__.py -> trade_rl/evaluation/robustness/walk_forward/__init__.py
MOVE trade_rl/evaluation/walk_forward/capabilities.py -> trade_rl/evaluation/robustness/walk_forward/capabilities.py
MOVE trade_rl/evaluation/walk_forward/folds.py -> trade_rl/evaluation/robustness/walk_forward/folds.py
MOVE trade_rl/evaluation/walk_forward/sealed_test.py -> trade_rl/evaluation/robustness/walk_forward/sealed_test.py
MOVE trade_rl/evaluation/walk_forward/stitching.py -> trade_rl/evaluation/robustness/walk_forward/stitching.py
CREATE trade_rl/evaluation/robustness/__init__.py
CREATE trade_rl/evaluation/robustness/perfect_information/__init__.py
```

### Runs

```text
MOVE trade_rl/evaluation/candidate_run.py -> trade_rl/evaluation/runs/candidate.py
MOVE trade_rl/evaluation/candidate_suite.py -> trade_rl/evaluation/runs/candidate_suite.py
CREATE trade_rl/evaluation/runs/__init__.py
```

## Task 1: Establish RED architecture contract

- Create `tests/architecture/test_lean_evaluation_layout.py`.
- Assert the final required files/directories exist.
- Assert all moved old private paths are absent.
- Assert root KEEP files remain.
- Hard-code the existing `trade_rl.evaluation.__all__` sequence and assert exact equality.
- Run full CI through a Draft PR; expected failure is structural only because old layout remains.

## Task 2: Mechanical relocation

- Rename every mapped production file without editing non-import code.
- Create minimal package `__init__.py` facades only where useful for explicit ownership; do not duplicate implementations.
- Rewrite imports repository-wide using exact AST/module names, not substring replacement.
- Update `evaluation/__init__.py` imports while preserving its exact `__all__`.
- Update current README candidate CLI path to `python -m trade_rl.evaluation.runs.candidate`.
- Do not create compatibility forwarding modules.

## Task 3: Targeted GREEN

Run at minimum:

```bash
uv run ruff check trade_rl/evaluation tests/evaluation tests/architecture
uv run ruff format --check trade_rl/evaluation tests/evaluation tests/architecture
uv run mypy trade_rl/evaluation
uv run pytest -q tests/evaluation tests/architecture
```

Then run the full repository suite. Any failure must be investigated; do not weaken assertions or retain an old path solely to satisfy a stale test.

## Task 4: Falsification review

Against exact base `006278933f1ce499900d0f5019cbd5b92fb18fed`:

1. Normalize only the authorized old->new evaluation import-owner map.
2. For every moved production file and every KEEP file whose imports changed, require exact import membership and exact ordered non-import top-level AST equality.
3. Require exact `evaluation.__all__` equality.
4. Scan production/tests/current README for retired module imports/CLI path.
5. Assert migration/falsification helper workflows/scripts are absent from the final tree.

## Task 5: Exact-head final gate

Normal PR CI on the final helper-free HEAD must show:

```text
Ruff PASS
Format PASS
Mypy PASS
pytest -q tests: 0 failures
Package identity PASS
```

Keep the PR Draft. Do not merge without explicit user authorization.
