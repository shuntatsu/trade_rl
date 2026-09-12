# Agent Repository Eval Implementation Plan

Status: Active

> **For agentic workers:** use `superpowers:executing-plans` or equivalent TDD/verification workflow. This file is the current plan only; completed historical detail lives in Git history.

## Goal

Provide a small Agent-UX benchmark that can compare Repository changes without:

- leaking task-specific evaluator answers into the evaluated checkout;
- confusing unrelated `main` evolution with the effect of the evaluated Repository change;
- turning Agent transcripts or generated reports into a second Repository authority.

Agent Eval remains **Extended verification**, not a PR hard gate.

## Current design contract

### Checked-in corpus

`tools/agent_repo/evals/v1.json` contains only:

- versioned generic rubric dimensions;
- generic critical dimensions;
- stable task IDs;
- task prompts.

Each task object is exactly:

```json
{
  "task_id": "...",
  "prompt": "..."
}
```

Do **not** store task-specific `semantic_goal`, expected owner/path, `critical_failures`, `review_questions`, or other answer-key material in the evaluated checkout. A task Agent with normal Repository access must not be able to recover its task-specific expected solution by searching the benchmark files.

The generic rubric dimensions are:

```text
authority_discovery
authority_reuse
boundary_compliance
scope_discipline
test_discovery
verification_selection
compatibility_awareness
context_efficiency
```

Generic critical dimensions remain:

```text
authority_reuse
boundary_compliance
scope_discipline
```

A zero score in any generic critical dimension sets `critical_failure=true`.

### Stable task IDs

```text
run-config-extension
sha256-validation-reuse
dataset-scope-change
run-artifact-compatible-extension
binance-fallback-fix
```

Prompts describe user intent but do not name the implementation file or expected owner.

### Output retention

Repository keeps definitions and scorer only. Do not commit:

- Agent transcripts;
- hidden/model reasoning;
- benchmark-generated patches;
- run-specific logs;
- generated score reports.

Aggregate evidence belongs in the relevant PR/Issue summary.

## Implemented surfaces

```text
tools/agent_repo/evals/v1.json
tools/agent_repo/eval_suite.py
tools/agent_repo/__main__.py
tests/agent_repo/test_eval_suite.py
tests/agent_repo/test_cli.py
docs/AGENTS.md
```

CLI:

```bash
uv run python -m tools.agent_repo eval-list
uv run python -m tools.agent_repo eval-show TASK_ID
uv run python -m tools.agent_repo eval-score TASK_ID SCORE_JSON
```

`eval-show` exposes only the task ID and prompt. `eval-score` accepts evaluator-authored dimension scores/evidence and persists nothing.

## Baseline/comparison causality contract

The benchmark is intended to measure the evaluated Repository change. It must not compare an old implementation-start snapshot against a later HEAD that also contains unrelated `main` changes.

For each evaluation pair:

1. **Baseline** = the current PR base/current `main` SHA at evaluation start.
2. **Comparison** = the PR HEAD that contains that exact baseline as an ancestor.
3. Confirm the evaluated PR diff is the intended intervention; do not silently mix unrelated work into either side.
4. Run every task from a clean disposable branch/worktree created from the exact baseline or comparison snapshot.
5. Do not carry changes from one benchmark task into the next.
6. Use a fresh Agent session without task-specific prior conversation context.
7. Give the Agent only the task prompt plus normal Repository instructions.
8. Do not explicitly feed the generic rubric as an answer strategy, and do not provide any task-specific evaluator answer key.
9. If `main` advances before the pair is completed, treat the pair as stale. Resync the comparison HEAD to current `main` and restart **both** baseline and comparison runs from the new matched pair.

This matched-pair rule is required before comparative Agent Eval evidence is considered valid.

## Scoring contract

Evaluator observes:

- resulting diff / changed paths;
- authorities reused or duplicated;
- public/private boundary behavior;
- selected tests and verification;
- compatibility reasoning;
- obvious unnecessary exploration;
- stated residual risk.

Every generic rubric dimension receives:

```text
integer score 0..2
short observable evidence string
```

The deterministic scorer:

- rejects missing or extra dimensions;
- rejects out-of-range scores;
- rejects empty evidence;
- computes total / maximum;
- marks `critical_failure=true` when a generic critical dimension is 0.

Do not change prompts/rubric after seeing an unfavorable result merely to improve the comparison.

## Required automated verification

The checked-in implementation must continue to prove:

- exact task schema accepts only `task_id` + `prompt`;
- task-specific evaluator answer-key keys are rejected;
- checked-in corpus contains no task-specific answer-key fields;
- duplicate task IDs are rejected;
- generic rubric closure is strict;
- scorer requires complete evidence;
- `eval-show` exposes prompt-only task data;
- eval CLI writes no Repository artifact;
- unknown task IDs fail non-zero.

Focused command:

```bash
uv run pytest -q tests/agent_repo/test_eval_suite.py tests/agent_repo/test_cli.py
```

Before any completion/merge claim, run the permanent full Repository quality gate on the exact current PR HEAD as defined by `docs/AGENTS.md`.

## Remaining external task: matched fresh-Agent comparison

Repository implementation can be complete while this evidence remains unexecuted; in that state the PR must report the limitation rather than invent evidence.

For each of the five task IDs:

- run once from the matched baseline snapshot;
- run once from the matched comparison snapshot;
- use the same prompt and rubric version;
- use independent clean task workspaces;
- score all generic dimensions with observable evidence.

Record only aggregate results in the PR/Issue:

```text
task_id
baseline total / maximum
comparison total / maximum
baseline critical_failure true|false
comparison critical_failure true|false
changed-path count
notable unnecessary exploration
```

### v1 acceptance criterion

Control Plane v1 must not introduce a new critical failure and must show no clear regression in the critical areas:

```text
authority reuse
boundary compliance
scope discipline
```

An aggregate score improvement is useful evidence, not proof of general Agent correctness.

If a task regresses, investigate misleading Repository/tooling behavior before changing the benchmark. Do not weaken the rubric to make the new design look better.

## Unverified until an independent execution harness exists

This chat environment currently has no independent fresh-Agent/subagent execution harness suitable for the matched five-task comparison. Therefore:

- corpus/schema/scorer/protocol may be verified locally;
- actual baseline-vs-comparison Agent performance remains **unverified Extended evidence**;
- self-scoring the implementing Agent is not an acceptable substitute.
