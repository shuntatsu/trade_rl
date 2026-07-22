# Sequence Projection Equivalence Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one backend-sensitive float32 elementwise gradient test with strict float64 historical equivalence plus bounded float32 semantic-gradient contracts.

**Architecture:** Production code remains unchanged. The existing sequence-policy test module will contain reusable legacy-path and gradient-comparison helpers, one strict double-precision equivalence test, and one single-precision semantic test. Temporary GitHub Actions workflows provide RED evidence and 100-run Linux/Windows stability evidence, then are removed before final exact-head verification.

**Tech Stack:** Python 3.12, PyTorch, pytest, GitHub Actions, Ubuntu, Windows.

## Global Constraints

- Do not modify `trade_rl/rl/sequence_policy.py` or any production module.
- Keep one direct historical selection-after-projection comparison.
- Use `float64` with `rtol=1e-9` and `atol=1e-10` for elementwise output/input/parameter-gradient equivalence.
- Use float32 cosine similarity `>= 0.999999` and relative L2 error `<= 2e-5` for nonzero gradient pairs.
- Require exact zero output and exact zero input gradient for the fully unavailable sample.
- Do not add global Torch tolerance settings or automatic final-CI reruns.
- Run at least 100 focused repetitions on both Ubuntu and Windows.
- Production remains `NO-GO`; no training, serving, execution, selection, or release behavior changes.
- Keep the pull request Draft and unmerged.

---

### Task 1: Record an expected RED contract for the test decomposition

**Files:**
- Create: `.github/scripts/check_sequence_projection_test_contract.py`
- Create: `.github/workflows/tmp-sequence-projection-red.yml`

**Interfaces:**
- Consumes: `tests/rl/test_sequence_policy_core.py`
- Requires functions named `test_projection_after_selection_matches_legacy_in_float64` and `test_projection_after_selection_preserves_float32_gradient_semantics`.
- Forbids the old function name `test_projection_after_selection_matches_legacy_outputs_and_gradients`.

- [ ] **Step 1: Add the failing source contract**

```python
from pathlib import Path

source = Path("tests/rl/test_sequence_policy_core.py").read_text(encoding="utf-8")
required = (
    "def test_projection_after_selection_matches_legacy_in_float64()",
    "def test_projection_after_selection_preserves_float32_gradient_semantics()",
)
missing = [name for name in required if name not in source]
if missing:
    raise SystemExit(f"missing stable projection contracts: {missing}")
if "def test_projection_after_selection_matches_legacy_outputs_and_gradients()" in source:
    raise SystemExit("backend-sensitive legacy test still exists")
```

- [ ] **Step 2: Run the RED workflow on the exact branch head**

The workflow installs `uv sync --extra dev --extra export --extra train-sb3 --extra studio`, runs the script, captures output, and uploads it even on failure.

Expected: FAIL because the two new tests do not exist and the old test still exists.

- [ ] **Step 3: Record the run ID, artifact ID, digest, and exact head**

Store them in the draft PR body and later in `docs/verification/2026-07-23-sequence-projection-equivalence.md`.

---

### Task 2: Replace the flaky test with two stable semantic contracts

**Files:**
- Modify: `tests/rl/test_sequence_policy_core.py:148-201`

**Interfaces:**
- Produces helper `_projection_equivalence_case(dtype: torch.dtype)` returning the encoder, availability mask, legacy and maintained inputs, outputs, and named gradients.
- Produces helper `_relative_l2(left: torch.Tensor, right: torch.Tensor) -> float`.
- Produces helper `_assert_gradient_semantics(left: torch.Tensor, right: torch.Tensor) -> None`.

- [ ] **Step 1: Replace the old test with reusable helpers**

Use a fixed seed, dropout zero, one CPU thread, and this legacy path:

```python
legacy_sequence = encoder.projection(encoder.forward_sequence(legacy_input))
positions = torch.arange(12).expand_as(available)
indices = positions.masked_fill(~available, -1).max(dim=1).values
safe = indices.clamp_min(0)
legacy_selected = legacy_sequence[torch.arange(3), safe]
legacy = torch.where(
    (indices >= 0).unsqueeze(1),
    legacy_selected,
    torch.zeros_like(legacy_selected),
)
```

Capture input and named parameter gradients after `square().sum().backward()`, clear gradients, then run the maintained `encoder(optimized_input, available)` path.

- [ ] **Step 2: Add strict float64 historical equivalence**

```python
def test_projection_after_selection_matches_legacy_in_float64() -> None:
    case = _projection_equivalence_case(torch.float64)
    torch.testing.assert_close(case.optimized, case.legacy, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        case.optimized_input_gradient,
        case.legacy_input_gradient,
        rtol=1e-9,
        atol=1e-10,
    )
    for name, gradient in case.optimized_parameter_gradients.items():
        torch.testing.assert_close(
            gradient,
            case.legacy_parameter_gradients[name],
            rtol=1e-9,
            atol=1e-10,
        )
    assert torch.count_nonzero(case.optimized[2]) == 0
    assert torch.count_nonzero(case.optimized_input_gradient[2]) == 0
```

- [ ] **Step 3: Add float32 semantic-gradient assertions**

```python
def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right)), 1e-12)
    return float(torch.linalg.vector_norm(left - right)) / denominator


def _assert_gradient_semantics(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left_flat))
    right_norm = float(torch.linalg.vector_norm(right_flat))
    if left_norm == 0.0 and right_norm == 0.0:
        return
    cosine = float(torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0))
    assert cosine >= 0.999999
    assert _relative_l2(left, right) <= 2e-5
```

The float32 test must also assert output closeness with `rtol=1e-5`, `atol=2e-6`, exact zero output/input gradient for sample 2, finite nonzero input gradients for samples 0 and 1, and matching named gradient sets.

- [ ] **Step 4: Run focused validation**

```bash
uv run ruff check tests/rl/test_sequence_policy_core.py
uv run ruff format --check tests/rl/test_sequence_policy_core.py
uv run pytest tests/rl/test_sequence_policy_core.py -k projection_after_selection -q
```

Expected: two tests pass.

- [ ] **Step 5: Run the source contract again**

```bash
uv run python .github/scripts/check_sequence_projection_test_contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit the verified test change**

```bash
git add tests/rl/test_sequence_policy_core.py
git commit -m "test: stabilize sequence projection equivalence"
```

---

### Task 3: Prove repeated Linux and Windows stability

**Files:**
- Create: `.github/workflows/tmp-sequence-projection-stability.yml`

**Interfaces:**
- Runs only the two focused projection tests.
- Matrix: `ubuntu-latest`, `windows-latest`.
- Repetitions: 100 per operating system.

- [ ] **Step 1: Add the repetition matrix**

Use Python to invoke pytest 100 times and stop on the first failure:

```python
import subprocess
import sys

command = [
    sys.executable,
    "-m",
    "pytest",
    "tests/rl/test_sequence_policy_core.py",
    "-k",
    "projection_after_selection",
    "-q",
]
for iteration in range(1, 101):
    print(f"iteration={iteration}", flush=True)
    subprocess.run(command, check=True)
```

- [ ] **Step 2: Run the workflow and retain artifacts**

Each OS writes its complete output to a separate artifact. Expected: 100/100 successful repetitions on both operating systems.

- [ ] **Step 3: Record run, job, artifact, and digest evidence**

Add exact IDs to the verification document and PR body.

---

### Task 4: Remove temporary infrastructure and perform exact-head verification

**Files:**
- Delete: `.github/scripts/check_sequence_projection_test_contract.py`
- Delete: `.github/workflows/tmp-sequence-projection-red.yml`
- Delete: `.github/workflows/tmp-sequence-projection-stability.yml`
- Create: `docs/verification/2026-07-23-sequence-projection-equivalence.md`

**Interfaces:**
- Final effective change contains the test module and documentation only.

- [ ] **Step 1: Delete temporary script and workflows**

Verify the final comparison contains no `tmp-` workflow and no patch script.

- [ ] **Step 2: Run normal exact-head CI**

Required successful checks:

```text
Studio Vitest, typecheck, production build, fixed viewport
workflow security
Ruff and format
Mypy
Import Linter
dead-code report
Serving smoke
full pytest and coverage
critical branch coverage
CLI smoke
Ubuntu compatibility
Windows compatibility
training-image and non-root probe
PostgreSQL Compose, readiness, migration, tests, shutdown
```

- [ ] **Step 3: Write verification evidence**

Document the original failure, unchanged-rerun success, RED contract, two-test design, 100x2 matrix, final test count, coverage, artifact digests, and production-code unchanged boundary.

- [ ] **Step 4: Update the Draft PR body**

Include final exact head and all evidence. Keep the PR Draft and unmerged.
