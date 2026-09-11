# Agent Repository Merge Safety Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR + current-main-inclusive tested-head CI the normal integration path for Agent-generated changes, add a concise PR quality contract, and establish verifiable main-branch protection without requiring unnecessary human approvals in a solo-maintainer repository.

**Architecture:** Keep code-level policy in `.github/pull_request_template.md`, `AGENTS.md`, and architecture tests. Treat branch protection/ruleset configuration as a separate repository-setting action with explicit read-back verification; do not claim protection when the available GitHub administration surface cannot apply or verify the setting.

**Tech Stack:** GitHub pull requests, GitHub Actions, repository branch/ruleset settings, Python architecture tests for checked-in policy files.

**Spec:** `docs/specs/2026-09-11-agent-repository-control-plane-v1-design.md`

## Global Constraints

- Do not enable automatic merge.
- Do not require a second human approval solely for ceremony in a solo-maintainer repository.
- Do not permit Agent automation to force-push, rewrite history, or delete `main`.
- Required CI evidence must correspond to the PR's current final HEAD.
- Integration invariant: tested PR head contains current `main`. If `main` advances, old PR-head Green is stale integration evidence until a new PR head containing that `main` commit is tested.
- If repository-setting mutation cannot be performed through the available administrative surface, stop that task and report protection as unverified; do not substitute prose for actual protection.
- Branch/ruleset settings must be read back after mutation before reporting success.

---

## File map

Create:

```text
.github/pull_request_template.md
tests/architecture/test_pr_quality_contract.py
```

Modify:

```text
AGENTS.md
docs/AGENTS.md
docs/architecture/package-boundaries.md
```

Repository setting target outside Git tree:

```text
main branch protection / ruleset
```

---

### Task 1: Add a concise PR quality contract

**Files:**
- Create: `.github/pull_request_template.md`
- Create: `tests/architecture/test_pr_quality_contract.py`

**Interfaces:**
- Template headings are stable routing prompts, not a requirement for long prose.

- [ ] **Step 1: Write a failing architecture test**

Create `tests/architecture/test_pr_quality_contract.py` with required headings:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"
REQUIRED = (
    "## Objective",
    "## Non-goals",
    "## Acceptance Criteria",
    "## Invariants",
    "## Failure Modes",
    "## Test Oracle",
    "## Changed authorities / public surfaces",
    "## Tests / verification",
    "## Falsification",
    "## Unverified items",
    "## Residual risk",
)


def test_pr_template_contains_quality_contract() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED if heading not in text]
    assert missing == []
```

Run:

```bash
uv run pytest -q tests/architecture/test_pr_quality_contract.py
```

Expected: FAIL because the template does not exist.

- [ ] **Step 2: Create the short template**

Use this content shape:

```markdown
## Objective
<!-- What observable outcome does this change deliver? -->

## Non-goals
<!-- What is deliberately not changed? Keep short. -->

## Acceptance Criteria
<!-- Concrete success conditions. -->

## Invariants
<!-- Existing behavior/contracts that must remain true. -->

## Failure Modes
<!-- Relevant ways this could break; omit unrelated generic risks. -->

## Test Oracle
<!-- What observations distinguish correct from incorrect behavior? -->

## Changed authorities / public surfaces
<!-- New/changed semantic owners, public API, schemas, effects. Write "None" when none. -->

## Tests / verification
<!-- Commands and observed results on the current final HEAD. -->

## Falsification
<!-- Wrong implementations/adversarial cases checked, when material. -->

## Unverified items
<!-- Explicitly state what was not checked. -->

## Residual risk
<!-- What remains possible after the evidence above? -->
```

Do not add mandatory essay-length instructions.

- [ ] **Step 3: Run the contract test**

```bash
uv run pytest -q tests/architecture/test_pr_quality_contract.py
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add .github/pull_request_template.md tests/architecture/test_pr_quality_contract.py
 git commit -m "docs: add agent pr quality contract"
```

---

### Task 2: Make the integration policy explicit in Agent routing

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/architecture/package-boundaries.md`

**Interfaces:**
- `main` is not an Agent work branch.
- Agent-created changes use a dedicated branch/PR.
- Merge remains an explicitly authorized action.
- tested PR head contains current `main` before merge authorization.

- [ ] **Step 1: Add root routing language**

Add a short Git/PR section to `AGENTS.md`:

```text
Agent implementation work uses a dedicated branch or worktree and a PR. Do not treat direct mutation of main as the normal path. Before merge authorization, verify the tested PR head contains current main and verify the current CI result for that same PR head. If main advanced after a Green run, create and test a new PR head containing current main rather than reusing the stale Green. Agent work must not force-push, rewrite history, or delete main. Merge remains an explicit user-authorized action.
```

- [ ] **Step 2: Add the verification detail to `docs/AGENTS.md`**

Document both stale-head and stale-base invalidation: an older successful workflow run is not evidence for a moved PR head, and a Green PR head that does not contain current `main` is not current integration evidence. Also document that branch protection/ruleset state must be read back before claiming the repository is protected.

- [ ] **Step 3: Add durable architecture ownership**

Add a concise `Repository integration boundary` section to `docs/architecture/package-boundaries.md` stating:

```text
GitHub PR/CI/ruleset configuration governs integration safety but is not a runtime package authority. Code architecture tests may verify checked-in PR/CI policy files; actual branch protection is verified from GitHub state. Integration evidence requires the tested PR head to contain current main and the permanent CI result to belong to that same tested head; if main advances, the PR must be synchronized and retested before merge authorization.
```

- [ ] **Step 4: Run docs/architecture tests and commit**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py tests/architecture/test_pr_quality_contract.py
 git add AGENTS.md docs
 git commit -m "docs: define agent integration safety"
```

---

### Task 3: Establish the target main-branch protection policy

**Files:**
- No Git-tree file change is required in this task.
- Repository settings: `main` branch protection/ruleset.

**Interfaces:**
- Target branch: `main`
- Required CI context: the permanent CI job corresponding to `.github/workflows/ci.yml` job name `Lean Core`.
- Target integration evidence: tested PR head contains current `main` and `Lean Core` succeeded on that same head.

- [ ] **Step 1: Read current protection state immediately before mutation**

Verify:

```text
main current SHA
protected true|false
required status check contexts
strict / branch-up-to-date requirement if exposed
force-push setting
deletion setting
PR requirement / approval count if exposed
admin enforcement / bypass behavior if exposed
```

If the administration surface cannot read these fields, do not guess.

- [ ] **Step 2: Apply the minimal solo-maintainer target policy using an administrative GitHub surface**

Target behavior:

```text
Require changes to main through pull requests.
Required approving reviews: 0.
Require the current permanent CI check `Lean Core` before merge/update of main.
Require the PR branch to be up to date with current `main` before merge.
Block force pushes.
Block branch deletion.
Apply protection to administrators/maintainers when the selected GitHub protection mechanism supports it; this is necessary to protect against Agent actions performed with maintainer credentials.
Do not enable auto-merge.
```

If GitHub's chosen ruleset/protection mechanism cannot represent this exact combination, stop and document the unsupported field instead of silently substituting a weaker policy. If strict branch-up-to-date enforcement cannot be configured, keep the checked-in process invariant and manually verify `merge-base(current main, PR head) == current main` immediately before merge authorization; do not claim GitHub enforces it.

- [ ] **Step 3: Read back the resulting settings**

Required observations before claiming success:

```text
main reports protected/enforced
Lean Core is in required checks
strict / branch-up-to-date enforcement matches the approved target, if supported
force-push is disabled
deletion is disabled
PR requirement is active
required approval count remains 0
admin/maintainer bypass behavior matches the approved target
```

- [ ] **Step 4: Record only the resulting state in the PR Verification section**

Do not commit a generated settings dump. Record the key read-back facts and the timestamp/HEAD in the PR body or comment. Separate GitHub-enforced settings from process-level invariants that still require explicit read-only verification.

---

### Task 4: Prove protection does not weaken or deadlock the normal workflow

**Files:**
- No new permanent file unless a real defect requires one.

**Interfaces:**
- Validation uses a harmless documentation-only PR after protection is active.

- [ ] **Step 1: Create a temporary docs-only branch from current main**

Make one harmless Markdown whitespace/comment-only change on a dedicated test branch and open a Draft PR. Do not touch runtime or CI behavior.

- [ ] **Step 2: Verify current-main-inclusive CI evidence**

Observe current `main` SHA, the exact PR head SHA, and the workflow run for that SHA. Verify `merge-base(current main, PR head) == current main`; equivalently, tested PR head contains current `main`. The required `Lean Core` check must report success for that same PR head before it is considered current integration evidence. If `main` advances, repeat synchronization and CI rather than reusing the old run.

- [ ] **Step 3: Verify direct-main mutation is rejected by the protection mechanism**

Do not attempt a direct-main write merely to prove protection. That probe is unsafe when the policy is absent or misconfigured because failure of the protection would mutate `main`.

Use GitHub read-back of branch protection/ruleset and, when available, effective rule evaluation. If no non-mutating API can prove direct-write rejection, record the item as **not safely probed** unless the user separately authorizes a controlled main-mutation test.

- [ ] **Step 4: Close the temporary PR without merging and delete only its disposable branch if verified safe**

Before branch deletion, confirm it is not `main`, not the head of any other open PR, and contains no unique intended work.

- [ ] **Step 5: Confirm main/integration evidence did not become stale during the probe**

Compare main SHA before/after the probe. The temporary probe itself must not move `main`. If `main` moved because of unrelated legitimate work, treat the previously observed PR-head CI as stale integration evidence and re-evaluate against the new main instead of reporting the old run as current.

---

### Task 5: Final falsification and quality review

**Files:**
- Existing files only.

- [ ] **Step 1: Falsification — remove one required PR heading**

Temporarily remove `## Failure Modes`; `tests/architecture/test_pr_quality_contract.py` must fail. Restore.

- [ ] **Step 2: Falsification — reject stale HEAD and stale base evidence**

During review, point verification evidence at either an old PR HEAD or a Green PR HEAD that predates a current-main advance. Verify the process rejects both as stale rather than treating a previous successful CI run as current integration evidence. No code mutation is needed; this is a review-procedure falsification.

- [ ] **Step 3: Run final repository gates on the implementation PR head**

Run the permanent CI and confirm the exact final HEAD passes. Immediately before merge authorization, also verify that this tested PR head contains the then-current `main`. Do not claim branch protection itself is validated by pytest; use GitHub read-back separately.

- [ ] **Step 4: Final report**

Report separately:

```text
checked-in PR quality contract: verified by architecture test
final-head CI: verified by GitHub run on exact SHA
current-main containment: verified by compare/merge-base immediately before merge authorization
branch protection/ruleset: verified by GitHub read-back, or explicitly unverified
main direct-update rejection probe: verified / not safely testable
remaining bypass/recovery risk
```
