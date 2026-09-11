# Agent Repository Eval Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, versioned Agent-UX evaluation corpus and rubric so repository-structure changes can be compared using fresh-agent tasks without turning model outputs or run logs into a new current-tree authority.

**Architecture:** Keep evaluation definitions inside repository tooling under `tools/agent_repo/evals/`. Store only task definitions and rubric semantics; do not store model outputs, transcripts, or generated reports. Validation/scoring is deterministic local Python, while actual fresh-agent execution remains an external/manual Extended verification step.

**Tech Stack:** Python 3.12 stdlib (`dataclasses`, `json`, `pathlib`), pytest, existing `tools.agent_repo` CLI from the core plan.

**Spec:** `docs/specs/2026-09-11-agent-repository-control-plane-v1-design.md`

## Global Constraints

- Agent Eval is not a PR hard gate in v1.
- Do not encode exact implementation paths as the primary success criterion; evaluate semantic authority reuse, boundary compliance, scope discipline, and verification selection.
- Do not commit model transcripts, hidden chain-of-thought, generated patches from benchmark runs, or run-specific logs.
- Do not require a specific model/provider for the task corpus.
- Do not allow benchmark tasks to become production/domain authority.
- Baseline and post-change evaluations must use the same task IDs and rubric version when compared.

---

## File map

Create:

```text
tools/agent_repo/evals/__init__.py
tools/agent_repo/evals/v1.json
tools/agent_repo/eval_suite.py
tests/agent_repo/test_eval_suite.py
```

Modify:

```text
tools/agent_repo/__main__.py
docs/AGENTS.md
```

No production `trade_rl/**` file changes are allowed.

---

### Task 1: Define and validate the v1 evaluation corpus

**Files:**
- Create: `tools/agent_repo/evals/__init__.py`
- Create: `tools/agent_repo/evals/v1.json`
- Create: `tools/agent_repo/eval_suite.py`
- Create: `tests/agent_repo/test_eval_suite.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    prompt: str
    semantic_goal: str
    critical_failures: tuple[str, ...]
    review_questions: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class EvalRubric:
    dimensions: tuple[str, ...]
    max_score_per_dimension: int
    critical_dimensions: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class EvalSuite:
    schema_version: str
    rubric_version: str
    tasks: tuple[EvalTask, ...]
    rubric: EvalRubric


def load_eval_suite(path: Path) -> EvalSuite: ...
```

- [ ] **Step 1: Write RED schema-validation tests**

Test that the loader rejects:

```text
unknown top-level keys
duplicate task_id
empty prompt
empty semantic_goal
unknown rubric dimension in critical_dimensions
max_score_per_dimension < 1
no tasks
```

and accepts one minimal exact fixture.

Run:

```bash
uv run pytest -q tests/agent_repo/test_eval_suite.py
```

Expected: import failure because `eval_suite` does not exist.

- [ ] **Step 2: Implement strict JSON decoding**

Use exact-key validation rather than permissive `.get()` defaults. The initial schema is:

```json
{
  "schema_version": "agent_repo_eval_suite_v1",
  "rubric_version": "agent_repo_rubric_v1",
  "rubric": {
    "dimensions": [
      "authority_discovery",
      "authority_reuse",
      "boundary_compliance",
      "scope_discipline",
      "test_discovery",
      "verification_selection",
      "compatibility_awareness",
      "context_efficiency"
    ],
    "max_score_per_dimension": 2,
    "critical_dimensions": [
      "authority_reuse",
      "boundary_compliance",
      "scope_discipline"
    ]
  },
  "tasks": []
}
```

- [ ] **Step 3: Add five task families from the approved design**

`v1.json` must contain these stable task IDs and prompts:

```text
run-config-extension
sha256-validation-reuse
dataset-scope-change
run-artifact-compatible-extension
binance-fallback-fix
```

Prompts should describe user intent without naming the expected implementation file. Example:

```text
Add one new Candidate Run configuration field that affects resolved run semantics. Preserve existing public and persisted contracts unless a change is required, and determine the necessary verification.
```

Critical failures for that task include creating an independent parallel run-config authority, bypassing the maintained public/capability boundary, or claiming completion without checking persisted Run compatibility.

- [ ] **Step 4: Keep the rubric semantic rather than path-prescriptive**

For every task, use review questions such as:

```text
Did the agent identify and reuse the existing semantic authority?
Did it introduce an overlapping Config/validator/codec/resolver without justification?
Did it stay within the requested scope?
Did it identify the relevant contract/integration/architecture tests?
Did it distinguish persisted compatibility from Python import compatibility?
```

Do not encode `must edit foo.py` as a general rubric rule.

- [ ] **Step 5: Run loader tests**

```bash
uv run pytest -q tests/agent_repo/test_eval_suite.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tools/agent_repo/evals tools/agent_repo/eval_suite.py tests/agent_repo/test_eval_suite.py
 git commit -m "feat: add agent repository eval corpus"
```

---

### Task 2: Add deterministic rubric scoring for observed outcomes

**Files:**
- Modify: `tools/agent_repo/eval_suite.py`
- Modify: `tests/agent_repo/test_eval_suite.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class DimensionScore:
    dimension: str
    score: int
    evidence: str

@dataclass(frozen=True, slots=True)
class EvalScore:
    task_id: str
    rubric_version: str
    dimensions: tuple[DimensionScore, ...]
    total: int
    maximum: int
    critical_failure: bool


def score_eval(
    suite: EvalSuite,
    *,
    task_id: str,
    scores: Mapping[str, tuple[int, str]],
) -> EvalScore: ...
```

- [ ] **Step 1: Write RED tests for exact dimension closure**

Assert scoring rejects missing/extra dimensions and scores outside `0..max_score_per_dimension`.

- [ ] **Step 2: Define critical failure semantics**

A score of `0` in any critical dimension sets `critical_failure=True`; low scores in non-critical dimensions do not. The scorer does not infer evidence from an Agent transcript.

- [ ] **Step 3: Implement total/max calculation**

Use:

```python
maximum = len(rubric.dimensions) * rubric.max_score_per_dimension
total = sum(item.score for item in dimension_scores)
```

Keep evaluator-written evidence strings non-empty so every score has an observable reason.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest -q tests/agent_repo/test_eval_suite.py
 git add tools/agent_repo/eval_suite.py tests/agent_repo/test_eval_suite.py
 git commit -m "feat: score agent repository eval outcomes"
```

---

### Task 3: Expose eval definitions and scoring through the repository CLI

**Files:**
- Modify: `tools/agent_repo/__main__.py`
- Modify: `tests/agent_repo/test_cli.py`

**Interfaces:**

```bash
uv run python -m tools.agent_repo eval-list
uv run python -m tools.agent_repo eval-show TASK_ID
uv run python -m tools.agent_repo eval-score TASK_ID SCORE_JSON
```

`SCORE_JSON` is evaluator-authored input, not an Agent transcript.

- [ ] **Step 1: Write CLI RED tests**

Assert `eval-list` returns task IDs and rubric version; `eval-show` returns prompt/semantic goal/critical failures/review questions; unknown task ID exits non-zero.

- [ ] **Step 2: Implement CLI loading from the checked-in v1 corpus**

Resolve the corpus relative to `eval_suite.py`:

```python
DEFAULT_SUITE = Path(__file__).with_name("evals") / "v1.json"
```

Do not use network or environment-specific paths.

- [ ] **Step 3: Implement score-file input without persisting output**

`eval-score` reads the supplied JSON, prints the computed `EvalScore` as JSON, and writes no file.

- [ ] **Step 4: Run CLI tests and commit**

```bash
uv run pytest -q tests/agent_repo/test_cli.py tests/agent_repo/test_eval_suite.py
 git add tools/agent_repo/__main__.py tests/agent_repo
 git commit -m "feat: expose agent eval rubric CLI"
```

---

### Task 4: Establish the baseline execution protocol

**Files:**
- Modify: `docs/AGENTS.md`

**Interfaces:**
- Baseline/post-change fresh-agent runs are Extended verification, not pytest.
- Run output is summarized in the relevant PR/issue; raw transcripts/logs are not committed.

- [ ] **Step 1: Add an Agent Eval protocol section**

Document this exact procedure:

```text
1. Select the same eval suite/rubric version for baseline and comparison.
2. Start a fresh Agent session with repository access but without prior task-specific conversation context.
3. Give only the task prompt plus normal repository instructions.
4. Do not reveal evaluator-only critical failures/review questions before the Agent finishes.
5. Review the resulting diff, commands/tests chosen, public/private imports, and stated residual risks.
6. Score all rubric dimensions with short evidence strings.
7. Record aggregate score, critical failures, changed-path count, and notable unnecessary exploration in the PR/issue summary only.
8. Do not commit transcript/model reasoning/run logs.
```

- [ ] **Step 2: Run docs/CLI tests**

```bash
uv run pytest -q tests/agent_repo/test_eval_suite.py tests/agent_repo/test_cli.py tests/architecture/test_current_docs_layout.py
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add docs/AGENTS.md
 git commit -m "docs: define agent repository eval protocol"
```

---

### Task 5: Execute baseline and post-core comparison without creating repository artifacts

**Files:**
- No repository file is created for raw run data.
- PR/issue body/comment only for aggregate evidence.

**Interfaces:**
- Compare the same five task IDs with rubric `agent_repo_rubric_v1`.

- [ ] **Step 1: Run the five tasks against the pre-control-plane baseline**

Use the exact baseline commit recorded in the implementation PR. For each task record externally:

```text
total / maximum
critical_failure true|false
new overlapping authority created? yes|no
scope-expansion observed? yes|no
unnecessary verification observed? short note
```

- [ ] **Step 2: Run the same five tasks against the implemented control-plane HEAD**

Do not change prompts or rubric between runs.

- [ ] **Step 3: Evaluate the design criterion**

The control plane is acceptable for v1 when there is no new critical failure and no clear regression in aggregate authority reuse/boundary compliance/scope discipline. A score improvement is useful evidence but not a proof of general Agent correctness.

- [ ] **Step 4: If a task regresses, inspect the cause before changing the benchmark**

Do not weaken a rubric item to make the new design look better. Fix misleading tooling/routing when the rubric reflects the approved architecture; change the benchmark only when independent review shows the task/rubric itself is invalid.

- [ ] **Step 5: Final verification**

Run:

```bash
uv run pytest -q tests/agent_repo tests/architecture
```

and include Agent Eval aggregate evidence in the implementation PR's Verification/Falsification section.
