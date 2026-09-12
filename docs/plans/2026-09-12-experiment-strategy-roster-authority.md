# Experiment Strategy Roster Authority Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `StudyPlan` the one Controlled Experiment semantic authority for the maintained eight-strategy roster and PPO-seed-invariant subset, while keeping Run Core as the independent execution authority.

**Architecture:** Add non-serialized `ClassVar` rosters to `StudyPlan`, derive them from the existing candidate/control Study contract, and replace independent Experiment-layer roster literals with those authorities. Keep `candidate_suite.py` production-independent from Experiments; verify execution/Study consistency in tests. Compress the durable architecture doc so it names the source authority instead of repeating the seven deterministic names.

**Tech Stack:** Python 3.12, dataclasses/ClassVar, pytest, Ruff, Mypy, GitHub Actions, uv build/package verification.

**Spec:** `docs/specs/2026-09-12-experiment-strategy-roster-authority.md`

## Global Constraints

- `CANDIDATE_STRATEGY_NAMES` and `CONTROL_STRATEGY_NAMES` values/order remain unchanged.
- `ppo` remains the only maintained strategy allowed to vary with `ppo_seed`.
- No Study/Evidence/analysis/artifact schema version or digest semantics change.
- No production dependency from `evaluation/runs` to `evaluation/experiments`.
- Factor-specific `FACTOR_RULES[*].unaffected_strategies` remain independently owned policy.
- No public facade expansion, generic StrategyRoster type, registry, or generated metadata.
- This is stacked on PR #469; do not treat it as merge-ready to `main` until #469 is integrated and this branch is revalidated against current `main`.

---

### Task 1: Define the strategy-roster authority contract with RED tests

**Files:**
- Create: `tests/architecture/test_experiment_strategy_roster_authority.py`
- Modify: `tests/evaluation/test_candidate_suite.py`
- Verify existing: `tests/evaluation/experiments/test_contracts.py`

**Interfaces:**
- Consumes: existing `CANDIDATE_STRATEGY_NAMES`, `CONTROL_STRATEGY_NAMES`, `StudyPlan`, `FACTOR_RULES`, and `run_lean_candidate_suite` behavior.
- Produces: failing contracts for `StudyPlan.STRATEGY_NAMES`, `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES`, removal of independent production roster literals, FactorRule subset validity, and Run Core/Study ordered-roster consistency.

- [ ] **Step 1: Add the failing architecture contract**

Create `tests/architecture/test_experiment_strategy_roster_authority.py` with the following contract shape:

```python
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from trade_rl.evaluation.experiments.contracts import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import FACTOR_RULES

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "trade_rl" / "evaluation" / "experiments"


def _module_level_literal_rosters(path: Path) -> tuple[tuple[str, ...], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rosters: list[tuple[str, ...]] = []
    for statement in tree.body:
        value = None
        if isinstance(statement, ast.Assign):
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        if not value.elts or any(
            not isinstance(element, ast.Constant) or not isinstance(element.value, str)
            for element in value.elts
        ):
            continue
        rosters.append(tuple(element.value for element in value.elts))
    return tuple(rosters)


def test_study_plan_owns_strategy_rosters_without_serializing_them() -> None:
    assert StudyPlan.STRATEGY_NAMES == (
        *CONTROL_STRATEGY_NAMES,
        *CANDIDATE_STRATEGY_NAMES,
    )
    assert StudyPlan.STRATEGY_NAMES == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
    )
    assert StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
    )
    dataclass_fields = {field.name for field in fields(StudyPlan)}
    assert "STRATEGY_NAMES" not in dataclass_fields
    assert "PPO_SEED_INVARIANT_STRATEGY_NAMES" not in dataclass_fields


def test_experiment_consumers_do_not_redeclare_study_rosters() -> None:
    forbidden = {
        StudyPlan.STRATEGY_NAMES,
        StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES,
    }
    for name in ("analysis.py", "evidence.py", "delta.py"):
        assert forbidden.isdisjoint(_module_level_literal_rosters(EXPERIMENTS / name))


def test_factor_rules_reference_only_study_strategies() -> None:
    allowed = frozenset(StudyPlan.STRATEGY_NAMES)
    for rule in FACTOR_RULES.values():
        assert rule.unaffected_strategies <= allowed
```

The exact-literal assertions intentionally remain in a test rather than being derived entirely from production, so accidental simultaneous source changes do not redefine the oracle.

- [ ] **Step 2: Change the Candidate Suite test into a cross-layer consistency oracle**

In `tests/evaluation/test_candidate_suite.py`, import `StudyPlan` and replace only the final hard-coded `calls["names"]` expected tuple with:

```python
assert calls["names"] == StudyPlan.STRATEGY_NAMES
```

Do not change the existing fit/execution assertions.

- [ ] **Step 3: Run the formal RED gate**

Run the repository's normal PR CI on this test-only head. The accepted RED requires:

```text
Ruff: pass
Format: pass
production Mypy: pass
architecture-tooling Mypy: pass
pytest: fail only the newly introduced strategy-roster authority conditions
```

Expected root causes before production change:

```text
StudyPlan has no STRATEGY_NAMES
StudyPlan has no PPO_SEED_INVARIANT_STRATEGY_NAMES
analysis/evidence/delta still contain independent literal rosters
```

If Ruff/Format/Mypy fails first, fix test hygiene only and rerun; do not count that as formal RED.

- [ ] **Step 4: Commit the RED test state**

Use a commit message equivalent to:

```text
test: define experiment strategy roster authority
```

---

### Task 2: Make StudyPlan the semantic roster owner and migrate Experiment consumers

**Files:**
- Modify: `trade_rl/evaluation/experiments/contracts/study.py`
- Modify: `trade_rl/evaluation/experiments/analysis.py`
- Modify: `trade_rl/evaluation/experiments/evidence.py`
- Modify: `trade_rl/evaluation/experiments/delta.py`
- Test: `tests/architecture/test_experiment_strategy_roster_authority.py`
- Test: existing Experiment analysis/evidence/delta/contracts suites

**Interfaces:**
- Consumes: module-level `CANDIDATE_STRATEGY_NAMES` and `CONTROL_STRATEGY_NAMES`.
- Produces: `StudyPlan.STRATEGY_NAMES: ClassVar[tuple[str, ...]]` and `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES: ClassVar[tuple[str, ...]]`.

- [ ] **Step 1: Add non-serialized roster ClassVars to StudyPlan**

Immediately after `FIXED_RESOLVED_FIELDS`, add:

```python
STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
    *CONTROL_STRATEGY_NAMES,
    *CANDIDATE_STRATEGY_NAMES,
)
PPO_SEED_INVARIANT_STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
    *CONTROL_STRATEGY_NAMES,
    *tuple(name for name in CANDIDATE_STRATEGY_NAMES if name != "ppo"),
)
```

Do not add either value to `StudyPlan.to_payload()`. Do not change the module-level candidate/control constants or `__all__`.

- [ ] **Step 2: Migrate analysis to StudyPlan authority**

In `analysis.py`:

- import `StudyPlan` from `trade_rl.evaluation.experiments.contracts`;
- delete module-local `_STRATEGIES` and `_DETERMINISTIC_STRATEGIES`;
- replace roster length/order/iteration uses with `StudyPlan.STRATEGY_NAMES`;
- replace deterministic/non-PPO iterations with `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES`.

Preserve all analysis schemas, payload structure, seed handling, and error text unless formatting requires line wrapping.

- [ ] **Step 3: Migrate EvidenceSet verification to StudyPlan authority**

In `evidence.py`:

- delete `_DETERMINISTIC_STRATEGIES` and `_EXPECTED_STRATEGIES`;
- compare observed name sets to `frozenset(StudyPlan.STRATEGY_NAMES)`;
- iterate `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES` for cross-PPO-seed raw-return equality.

Do not alter EvidenceSet fingerprint construction, semantic config, publication, loading, or exception categories.

- [ ] **Step 4: Migrate controlled-delta roster validation only**

In `delta.py`:

- delete module-local `_STRATEGIES`;
- compare observed ordered Run rosters to `StudyPlan.STRATEGY_NAMES`.

Do not rewrite `FACTOR_RULES[*].unaffected_strategies`; those sets are factor-specific policy, not duplicate full-roster authority.

- [ ] **Step 5: Run focused GREEN verification**

Run at minimum:

```text
pytest tests/architecture/test_experiment_strategy_roster_authority.py
pytest tests/evaluation/test_candidate_suite.py
pytest tests/evaluation/experiments/test_contracts.py
pytest tests/evaluation/experiments/test_analysis.py
pytest tests/evaluation/experiments/test_evidence.py
pytest tests/evaluation/experiments/test_delta.py
```

Also run Ruff/Format and Mypy for the changed production modules.

Expected: all pass with no schema/digest fixture changes.

- [ ] **Step 6: Review the production diff before committing**

Confirm:

```text
candidate_suite.py production unchanged
CANDIDATE_STRATEGY_NAMES unchanged
CONTROL_STRATEGY_NAMES unchanged
StudyPlan.to_payload() unchanged
FACTOR_RULES contents unchanged
no experiments __all__ / contracts __all__ expansion
```

- [ ] **Step 7: Commit the minimal GREEN implementation**

Use a commit message equivalent to:

```text
refactor: centralize experiment strategy roster authority
```

---

### Task 3: Compress the durable docs and complete verification

**Files:**
- Modify: `docs/architecture/controlled-experiment-loop.md`
- Modify: `docs/README.md`
- Delete after verified implementation: `docs/specs/2026-09-12-experiment-strategy-roster-authority.md`
- Delete after verified implementation: `docs/plans/2026-09-12-experiment-strategy-roster-authority.md`

**Interfaces:**
- Consumes: verified source/test contract from Tasks 1-2.
- Produces: current-only durable documentation with no duplicated seven-item strategy roster and no completed ephemeral docs.

- [ ] **Step 1: Replace the deterministic strategy list with an authority statement**

In `docs/architecture/controlled-experiment-loop.md`, replace the paragraph plus seven bullet names after the PPO-seed rule with wording equivalent to:

```text
EvidenceSet内で変えてよいのは `ppo_seed` だけである。EvidenceSetのsemantic configとそのdigestからは `ppo_seed` を除外し、seed policyはordered `ppo_seeds`として別にidentityへbindする。各Run artifactには実際の `ppo_seed` を保持する。

Studyで維持するordered strategy rosterは `StudyPlan.STRATEGY_NAMES`、そのうちPPO seedを変えてもraw returnsが完全一致しなければならないrosterは `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES` をsource contractの正本とする。現行契約では `ppo` だけがPPO seed依存を許される。rosterをarchitecture docs側で別途列挙・管理しない。
```

This is a replacement/compression, not additive history.

- [ ] **Step 2: Run the full exact-head quality gate before removing Active docs**

Run the full repository gate:

```text
Ruff
Format check
production Mypy
architecture-tooling Mypy
full pytest
uv build
tracked source closure: sdist
tracked source closure: direct wheel
sdist -> rebuilt wheel source closure
clean non-editable installed public import/package identity smoke
Candidate Run CLI --help
Canonical bootstrap CLI --help
package identity
```

Record the exact head and results. Because this is a stacked PR, also verify its parent is still the expected #469 head or its updated descendant.

- [ ] **Step 3: Perform falsification review**

Explicitly inspect whether the tests would catch these plausible wrong changes:

```text
add a strategy to StudyPlan but not Candidate Suite
add a strategy to Candidate Suite but not StudyPlan
put ppo into PPO_SEED_INVARIANT_STRATEGY_NAMES
reintroduce an 8-name tuple in analysis/evidence/delta
put a typo/non-Study name into any FactorRule.unaffected_strategies
turn a ClassVar into a dataclass field
make candidate_suite import evaluation.experiments
```

Where cheap, use mutation/probe execution; otherwise identify the exact test that rejects the error.

- [ ] **Step 4: Remove completed ephemeral spec/plan and restore docs index**

After GREEN + falsification evidence is complete:

- delete `docs/specs/2026-09-12-experiment-strategy-roster-authority.md`;
- delete this plan;
- restore `docs/README.md` to `現在Activeなspecはない。` / `現在Activeなplanはない。` unless another independent Active doc exists on the stacked parent at that time.

- [ ] **Step 5: Rerun final exact-head verification after docs cleanup**

The completion CI must run on the final tree after ephemeral docs are removed. Do not reuse the earlier full CI if the head changed.

- [ ] **Step 6: Retarget/revalidate after #469 integration before main merge readiness**

If #469 has merged to `main`:

```text
retarget this PR from refactor/experiment-semantic-authorities to main
bring current main into the feature branch without force-push/history rewrite
verify behind_by=0 and expected merge-base
rerun the entire final exact-head gate
```

If #469 is still open, keep this PR Draft and stacked; do not call it ready for main integration.

- [ ] **Step 7: Final review report**

Record:

- exact durable diff;
- Acceptance Criteria mapping;
- formal RED result;
- focused GREEN result;
- final full CI result;
- falsification findings;
- parent/main topology;
- unverified items and residual risk.
