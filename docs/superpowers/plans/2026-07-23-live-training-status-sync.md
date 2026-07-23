# Live Training Status Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the maintained research-status document so it describes the merged Live Training environment/current-episode isolation contract and permanently rejects the obsolete stream-isolation-gap wording.

**Architecture:** Keep the change documentation-only. Add one executable current-documentation regression contract, then minimally update `docs/RESEARCH_STATUS.md`; historical audit and verification files remain unchanged.

**Tech Stack:** Python 3.12, pytest, Markdown, GitHub Actions.

## Global Constraints

- Production status remains `NO-GO`.
- Direct exchange routing remains `NOT_IMPLEMENTED`.
- Profitability claim remains `NONE`.
- Do not modify product, Studio, training, evaluation, Serving, release, or execution code.
- Do not rewrite historical audit or verification records.

---

### Task 1: Add the current-status regression contract

**Files:**
- Modify: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: existing `_text(path: Path) -> str` helper and `ROOT` constant.
- Produces: `test_resolved_live_training_isolation_is_not_described_as_open()`.

- [ ] **Step 1: Write the failing test**

Append before `test_internal_markdown_links_resolve()`:

```python
def test_resolved_live_training_isolation_is_not_described_as_open() -> None:
    research_status = _text(ROOT / "docs" / "RESEARCH_STATUS.md").lower()
    for stale in (
        "available_for_diagnostic_replay_with_stream_isolation_gap",
        "one confirmed diagnostic limitation remains",
        "without an environment or episode selector",
    ):
        assert stale not in research_status
    for current in (
        "available_for_diagnostic_replay_with_episode_isolation",
        "producer-issued `episode_id`",
        "selected vector environment",
        "current episode",
        "historical records with `null` identity",
    ):
        assert current in research_status
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
uv run pytest tests/test_current_documentation_contract.py::test_resolved_live_training_isolation_is_not_described_as_open -q
```

Expected: FAIL because the obsolete capability token and limitation paragraph still exist and the current markers are absent.

- [ ] **Step 3: Commit the RED contract**

```bash
git add tests/test_current_documentation_contract.py
git commit -m "test: guard resolved live training status"
```

### Task 2: Synchronize the maintained research status

**Files:**
- Modify: `docs/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: the PR #85 selected-environment/current-episode contract and PR #103 producer-issued nullable `episode_id` contract.
- Produces: current capability wording accepted by Task 1.

- [ ] **Step 1: Update the current capability block**

Change the status date to `2026-07-23` and replace:

```text
TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY_WITH_STREAM_ISOLATION_GAP
```

with:

```text
TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY_WITH_EPISODE_ISOLATION
```

- [ ] **Step 2: Update the baseline summary**

Extend the baseline paragraph to state that the current baseline includes selected-environment/current-episode replay isolation and producer-issued telemetry episode identity. Add the current architecture closeout link without removing the dated historical links.

- [ ] **Step 3: Replace the obsolete telemetry limitation paragraph**

Use this maintained-state wording:

```markdown
The maintained browser path selects one vector environment and derives the chart, cursor, price, PnL, baseline, drawdown, events, playback, and jump controls from that environment's current episode. Producer-issued nullable `episode_id` values are preferred for episode selection. Historical records with `null` identity retain the terminal and counter-rollback fallback, so existing `training_telemetry_v1` streams remain readable.

Telemetry remains exploratory and is excluded from fitting, checkpoint selection, configuration selection, sealed evaluation, run identity, promotion, release approval, Serving activation, and order routing. The isolation contract prevents false cross-environment or cross-episode continuity; it does not turn telemetry into profitability or exchange-execution evidence.
```

- [ ] **Step 4: Run the focused test to verify GREEN**

Run:

```bash
uv run pytest tests/test_current_documentation_contract.py::test_resolved_live_training_isolation_is_not_described_as_open -q
```

Expected: PASS.

- [ ] **Step 5: Run the complete maintained-document contract**

Run:

```bash
uv run pytest tests/test_current_documentation_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the documentation correction**

```bash
git add docs/RESEARCH_STATUS.md
git commit -m "docs: sync live training isolation status"
```

### Task 3: Verify and publish the independent remediation PR

**Files:**
- Verify: `docs/RESEARCH_STATUS.md`
- Verify: `tests/test_current_documentation_contract.py`
- Verify: `docs/superpowers/specs/2026-07-23-live-training-status-sync-design.md`
- Verify: `docs/superpowers/plans/2026-07-23-live-training-status-sync.md`

**Interfaces:**
- Consumes: Task 1 RED contract and Task 2 documentation correction.
- Produces: one reviewable documentation-remediation pull request.

- [ ] **Step 1: Run static formatting checks for the changed Python test**

```bash
uv run ruff check tests/test_current_documentation_contract.py
uv run ruff format --check tests/test_current_documentation_contract.py
```

Expected: both commands pass.

- [ ] **Step 2: Inspect the branch diff**

```bash
git diff main...HEAD --check
git diff --stat main...HEAD
```

Expected: only the spec, plan, maintained status document, and documentation contract test are changed; no whitespace errors.

- [ ] **Step 3: Push and open a Draft PR**

Use title:

```text
docs: sync resolved live training status
```

The PR body must state that the old status contradicted PR #85, PR #103, and the architecture closeout; the change is documentation-only; production remains `NO-GO`; and the new test prevents recurrence.

- [ ] **Step 4: Require exact-head CI**

Do not merge until the normal GitHub Actions CI for the PR head succeeds. Record the exact head SHA and workflow run in the PR or a dated verification note if one is added.
