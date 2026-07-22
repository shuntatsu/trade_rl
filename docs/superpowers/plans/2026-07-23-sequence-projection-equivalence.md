# Sequence Projection Equivalence Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the backend-sensitive float32 elementwise gradient test with strict float64 historical equivalence and bounded float32 semantic-gradient contracts.

**Architecture:** Production code remains unchanged. One test helper captures outputs, input gradients, and named parameter gradients for historical and maintained graphs. A permanent architecture contract and narrowly triggered cross-platform workflow prevent the flaky assertion from returning.

**Tech Stack:** Python 3.12, PyTorch, pytest, AST, GitHub Actions, Ubuntu, Windows.

## Global Constraints

- Base the independent PR on current `main` commit `b39103aee26f9f80ab3b908fbf374c21ca1604a0`.
- Do not modify any file under `trade_rl/`.
- Use float64 `rtol=1e-9`, `atol=1e-10` for strict historical equivalence.
- Use float32 cosine similarity `>= 0.999999` and relative L2 error `<= 2e-5` for nonzero gradients.
- Require exact zero output and input gradient for the fully unavailable row.
- Preserve the prior RED, focused GREEN, and 100x2 stability evidence by immutable run and artifact IDs.
- Keep permanent CI narrowly path-filtered.
- Keep the PR Draft and unmerged.

---

### Task 1: Replace the unstable equivalence test

**Files:**
- Modify: `tests/rl/test_sequence_policy_core.py`

**Interfaces:**
- Produce `_ProjectionEquivalenceCase`.
- Produce `_projection_equivalence_case(dtype: torch.dtype)`.
- Produce `_relative_l2(left, right)` and `_assert_gradient_semantics(left, right)`.
- Produce `test_projection_after_selection_matches_legacy_in_float64`.
- Produce `test_projection_after_selection_preserves_float32_gradient_semantics`.

- [ ] Add `dataclass` import.
- [ ] Replace the old test block with the reviewed helper and two contracts.
- [ ] Run Ruff formatting and checking.
- [ ] Run repository-wide Mypy.
- [ ] Run `pytest tests/rl/test_sequence_policy_core.py -k projection_after_selection -q`.
- [ ] Commit only after every check passes.

Expected focused result: two passing tests.

---

### Task 2: Preserve the stable test structure

**Files:**
- Create: `tests/architecture/test_sequence_projection_stability.py`

**Interface:**

```python
def test_sequence_projection_equivalence_uses_stable_contracts() -> None:
    tree = ast.parse(Path("tests/rl/test_sequence_policy_core.py").read_text())
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "test_projection_after_selection_matches_legacy_in_float64",
        "test_projection_after_selection_preserves_float32_gradient_semantics",
    } <= names
    assert "test_projection_after_selection_matches_legacy_outputs_and_gradients" not in names
```

- [ ] Add the AST contract.
- [ ] Run the new architecture test with both focused projection tests.
- [ ] Commit after GREEN.

---

### Task 3: Add a narrowly targeted permanent matrix

**Files:**
- Create: `.github/workflows/sequence-projection-stability.yml`

**Workflow contract:**

- Trigger on `push` and `pull_request` to `main` only when one of four relevant paths changes.
- Use pinned `actions/checkout`, `astral-sh/setup-uv`, and `actions/upload-artifact` SHAs.
- Use read-only contents permission.
- Matrix: Ubuntu and Windows.
- Repeat the architecture contract and two projection tests 10 times.
- Upload one log artifact per OS.

- [ ] Add the workflow.
- [ ] Verify workflow-security checks.
- [ ] Confirm 10/10 repetitions on both operating systems.

---

### Task 4: Record and verify the independent PR

**Files:**
- Create: `docs/verification/2026-07-23-sequence-projection-equivalence.md`

- [ ] Record original failure and unchanged-rerun behavior.
- [ ] Record RED run `29953598828` and artifact `8543101802`.
- [ ] Record focused GREEN run `29953725692` and artifact `8543175208`.
- [ ] Record 100x2 run `29953836687` and Linux/Windows artifact digests.
- [ ] Confirm effective diff contains no production file and no temporary workflow/script.
- [ ] Run normal exact-head CI, PostgreSQL Catalog, and targeted stability workflow.
- [ ] Record final head, run IDs, test counts, coverage, and artifact digests in the PR body.
- [ ] Keep the PR Draft and unmerged.
