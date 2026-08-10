# Default Target-Weight Full-Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the maintained full-research runner default to the three-profile target-weight growth walk-forward catalog while retaining `training-full.json` only as an explicit legacy template.

**Architecture:** Add one private module-level `Path` constant to the phase runner and route its no-override branch through the existing `walk-forward-target-weight-constrained-growth.json` template. Preserve the current explicit `--training-template` branch, configuration schemas, artifact identities, and all existing profile payloads. Lock the contract with a focused source-backed regression test and synchronize the operational documentation.

**Tech Stack:** Python 3.12, `pathlib`, `runpy`, JSON configuration, pytest, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Do not stop, rewrite, resume, or otherwise mutate any existing training generation or Docker container.
- Do not change reward coefficients, PPO/Lagrangian PPO behavior, BC, risk, execution, folds, seeds, timesteps, or model capacity.
- Do not rename or delete existing JSON profiles; historical paths and artifacts must remain resolvable.
- `training-full.json` must remain accepted through explicit `--training-template training-full.json` selection.
- The implicit default catalog must contain exactly the three target-weight growth profiles and must not contain `training-full.json`.
- Production status remains `NO-GO`.
- Every production behavior change must be preceded by a regression test that is observed failing for the expected reason.

---

### Task 1: Add the failing default-workflow contract test

**Files:**
- Create: `tests/examples/test_full_research_default_workflow.py`

**Interfaces:**
- Consumes: `examples/binance-multitimeframe/run_full_research_state.py`, its private `_example_template(value: str, *, field: str) -> Path` helper, and the existing target-weight workflow JSON.
- Produces: A regression contract requiring `_DEFAULT_WALK_FORWARD_TEMPLATE: Path` and the exact default candidate list.

- [ ] **Step 1: Create the focused regression test**

```python
from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"
RUNNER_PATH = EXAMPLE_ROOT / "run_full_research_state.py"
EXPECTED_DEFAULT_CANDIDATES = (
    (
        "target-weight-growth-gamma-one-ppo",
        "training-target-weight-growth-ppo.json",
    ),
    (
        "target-weight-constrained-growth-gamma-one",
        "training-target-weight-constrained-growth.json",
    ),
    (
        "target-weight-constrained-growth-discounted-168h",
        "training-target-weight-constrained-growth-discounted.json",
    ),
)


def _runner_namespace() -> dict[str, Any]:
    return runpy.run_path(str(RUNNER_PATH))


def _candidate_rows(path: Path) -> tuple[tuple[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (str(candidate["name"]), str(candidate["run_file"]))
        for candidate in payload["candidates"]
    )


def test_default_full_research_uses_target_weight_growth_catalog() -> None:
    namespace = _runner_namespace()

    assert "_DEFAULT_WALK_FORWARD_TEMPLATE" in namespace
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    assert isinstance(template, Path)
    assert template == (
        EXAMPLE_ROOT / "walk-forward-target-weight-constrained-growth.json"
    ).resolve()
    assert _candidate_rows(template) == EXPECTED_DEFAULT_CANDIDATES


def test_training_full_is_available_only_through_explicit_template_selection() -> None:
    namespace = _runner_namespace()
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    example_template = namespace["_example_template"]

    assert "training-full.json" not in {
        run_file for _, run_file in _candidate_rows(template)
    }
    assert example_template(
        "training-full.json",
        field="training template",
    ) == (EXAMPLE_ROOT / "training-full.json").resolve()
```

- [ ] **Step 2: Commit the test before production code**

```bash
git add tests/examples/test_full_research_default_workflow.py
git commit -m "test: require target-weight default workflow"
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/examples/test_full_research_default_workflow.py
```

Expected: failure at
`assert "_DEFAULT_WALK_FORWARD_TEMPLATE" in namespace`, proving that the current runner has no explicit target-weight default contract. Existing unrelated tests must not be changed to obtain RED.

---

### Task 2: Wire the runner to the target-weight workflow

**Files:**
- Modify: `examples/binance-multitimeframe/run_full_research_state.py`
- Test: `tests/examples/test_full_research_default_workflow.py`

**Interfaces:**
- Consumes: `_EXAMPLE_DIR: Path` and the existing
  `walk-forward-target-weight-constrained-growth.json` file.
- Produces: `_DEFAULT_WALK_FORWARD_TEMPLATE: Path`, used by
  `BinanceFullResearchStages._develop()` when no explicit template is supplied.

- [ ] **Step 1: Add the minimal default constant near the existing path constants**

```python
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WALK_FORWARD_TEMPLATE = (
    _EXAMPLE_DIR / "walk-forward-target-weight-constrained-growth.json"
)
_SUPERVISED_BOOTSTRAP_ARTIFACTS = frozenset(
    {"cuda-preflight.json", "entrypoint-provenance.json", "heartbeat.json"}
)
```

- [ ] **Step 2: Replace only the implicit workflow assignment**

Replace:

```python
workflow_template = _EXAMPLE_DIR / "walk-forward-full.json"
```

with:

```python
workflow_template = _DEFAULT_WALK_FORWARD_TEMPLATE
```

Do not alter the `requested_training` branch. That branch must continue to validate the explicitly named file, reject resume checkpoints, use the target-weight workflow as its evaluation envelope, and replace the candidate list with the explicit payload.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run:

```bash
uv run pytest -q tests/examples/test_full_research_default_workflow.py
```

Expected: `2 passed`.

- [ ] **Step 4: Run the related profile and runner tests**

Run:

```bash
uv run pytest -q \
  tests/examples/test_full_research_default_workflow.py \
  tests/examples/test_single_symbol_training_templates.py \
  tests/examples/test_binance_multitimeframe_full_assets.py \
  tests/workflows/test_sealed_ledger_profiles.py
```

Expected: all tests pass. The legacy `walk-forward-full.json` profile tests remain valid because the file is retained for reproducibility; only the runner default changes.

- [ ] **Step 5: Commit the minimal production change**

```bash
git add \
  examples/binance-multitimeframe/run_full_research_state.py \
  tests/examples/test_full_research_default_workflow.py
git commit -m "fix: default full research to target-weight growth"
```

---

### Task 3: Synchronize maintained workflow documentation

**Files:**
- Modify: `docs/operations/docker-gpu-full-training.md`
- Modify: `docs/SINGLE_SYMBOL.md`

**Interfaces:**
- Consumes: The runtime default defined in Task 2 and the three existing standalone profile files.
- Produces: Operator-facing documentation that distinguishes the implicit three-profile workflow from explicit legacy `training-full.json` runs.

- [ ] **Step 1: Replace the runbook's default-configuration section**

The section must state that the canonical default is
`walk-forward-target-weight-constrained-growth.json`, list the three candidate
profiles, and record the common maintained settings:

```text
symbol/action: BTCUSDT / one direct target-weight action
encoder/policy: hierarchical_sequence_v2 / MultiInputPolicy
BC: Oracle teacher / 45 epochs / 10% chronological validation
PPO learning rate: linear 0.00012 -> 0.000012
n_envs / n_steps / batch / epochs: 8 / 128 / 256 / 10
seeds: 0, 1, 2
device: cuda
timesteps per seed: 524288
```

It must identify the candidate objectives as gamma-one PPO control, gamma-one
Lagrangian PPO, and 168-hour discounted Lagrangian ablation. It must explicitly
state that `training-full.json` is a legacy mixed-shaping comparison selected
only with `--training-template training-full.json`, and that existing generations
remain pinned to their source/image/config identity.

- [ ] **Step 2: Clarify the single-symbol configuration catalog**

Update `docs/SINGLE_SYMBOL.md` so the configuration section separates:

```text
Default full-research workflow:
- walk-forward-target-weight-constrained-growth.json
- its three target-weight growth run_file profiles

Explicit legacy comparison:
- training-full.json
- walk-forward-full.json retained for historical/reproducibility use
```

Retain the existing schema, action-width, historical compatibility, and
production `NO-GO` statements.

- [ ] **Step 3: Review documentation against the actual JSON values**

Verify every numeric value against the standalone profile files. Do not copy the
legacy `training-full.json` values into the default section.

- [ ] **Step 4: Commit the documentation**

```bash
git add docs/operations/docker-gpu-full-training.md docs/SINGLE_SYMBOL.md
git commit -m "docs: describe target-weight full-research default"
```

---

### Task 4: Verify, review, and publish the pull request

**Files:**
- Review all changed files.
- No new production files beyond Tasks 1-3.

**Interfaces:**
- Consumes: Final branch head from Tasks 1-3.
- Produces: One reviewable PR based on `main`, with exact-head test evidence and no mutation of the active generation.

- [ ] **Step 1: Run formatting, lint, and targeted typing**

```bash
uv run ruff check \
  examples/binance-multitimeframe/run_full_research_state.py \
  tests/examples/test_full_research_default_workflow.py
uv run ruff format --check --diff \
  examples/binance-multitimeframe/run_full_research_state.py \
  tests/examples/test_full_research_default_workflow.py
uv run mypy \
  examples/binance-multitimeframe/run_full_research_state.py \
  tests/examples/test_full_research_default_workflow.py
```

Expected: all commands succeed with no issues.

- [ ] **Step 2: Re-run the focused and related tests**

```bash
uv run pytest -q \
  tests/examples/test_full_research_default_workflow.py \
  tests/examples/test_single_symbol_training_templates.py \
  tests/examples/test_binance_multitimeframe_full_assets.py \
  tests/workflows/test_sealed_ledger_profiles.py
```

Expected: all pass.

- [ ] **Step 3: Review the final diff**

Confirm:

- only the no-override default is changed;
- the explicit-template branch is byte-for-byte behaviorally unchanged;
- no JSON profile payload changed;
- no resume, checkpoint, reward, risk, execution, or selection-gate code changed;
- no generated files, secrets, temporary workflows, or local artifacts are present;
- the active generation is not referenced by a mutable path or operation.

- [ ] **Step 4: Open a Draft PR and run exact-head CI**

The PR body must include What, Why, Design, scope/non-goals, RED evidence, final
verification, active-run isolation, and remaining risk. Keep it Draft until all
required jobs on one final head succeed.

- [ ] **Step 5: Verify the exact final head**

Require success for Windows and Ubuntu compatibility, training image, frontend,
Ruff, formatting, MyPy, import architecture, dead-code, recovery/serving smoke,
full pytest/coverage, critical coverage, and package identity. Confirm zero
unresolved review threads.

- [ ] **Step 6: Mark Ready for review without merging**

Do not merge to `main` without a separate explicit user instruction.
