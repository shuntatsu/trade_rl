# Agent Repository Merge Safety Implementation Plan

Status: Active

## Goal

Keep PR + current-main-inclusive tested-head CI as the normal integration path for Agent-generated changes, and finish the one remaining external control: verifiable GitHub protection/ruleset enforcement for `main`.

This file is the **current plan only**. Completed implementation detail and earlier design iterations live in Git history and in the superseded implementation/review PR #464.

## Current checked-in state

The Git-tree portion is implemented and contract-tested:

- `.github/pull_request_template.md` provides the concise PR quality contract.
- `AGENTS.md` and `docs/AGENTS.md` require dedicated branch/PR integration rather than normal direct work on `main`.
- Integration invariant: **tested PR head contains current `main`**.
- **Require the PR branch to be up to date with current `main` before merge.**
- Exact-head CI is required; old Green is invalid after the PR head changes.
- If `main` advances, old PR-head Green is also stale integration evidence until a new PR head containing that `main` commit is tested.
- force-push, history rewrite, and `main` deletion are outside normal Agent authority.
- merge still requires explicit user authorization.
- a destructive direct-main rejection probe is not part of the default verification path.

The checked-in architecture contract is covered by `tests/architecture/test_pr_quality_contract.py` and the normal repository quality gate.

## Remaining external task: protect `main`

Branch protection / ruleset is a GitHub repository setting, not a Git-tree property. It must not be inferred from prose, tests, or a successful PR workflow.

### Target policy

For the current solo/small-maintainer repository, prefer the simplest policy that preserves tested integration evidence:

- normal changes enter through PRs;
- `Lean Core` is required before merge;
- the PR must be up to date with `main` before merge, or a future merge queue must test the merge-group SHA against current `main`;
- force pushes to `main` are disabled;
- `main` deletion is disabled;
- no ceremonial extra human approval is required solely because the repository is Agent-operated.

Do not weaken this to a loose required-check policy where a PR can merge using CI from a head that no longer contains current `main`.

### Safe application protocol

Only perform the repository-setting mutation when an administrative GitHub surface is actually available and authorized.

1. Read current `main` SHA and current protection/ruleset state.
2. Apply the target policy through the administrative settings surface.
3. Immediately read the effective setting back from GitHub.
4. Verify at minimum:
   - PR requirement is effective;
   - required `Lean Core` check is effective;
   - current-main/up-to-date requirement (or merge-queue equivalent) is effective;
   - force-push is disabled;
   - branch deletion is disabled;
   - any admin/maintainer bypass is understood and explicitly reported.
5. Do **not** attempt a real direct write to `main` merely to prove rejection unless the user separately authorizes a controlled destructive test. If protection is missing, that probe would itself mutate `main`.
6. Record the read-back evidence in the active PR/Issue; do not store generated settings dumps in the repository.

### Current environment limitation

The GitHub connection available in this chat exposes protection/ruleset **read-back** but no branch-protection administration write action. Current read-back reports `main` unprotected with required status checks off.

Therefore this task remains **unverified/unapplied** here. Do not report branch protection as enabled until a write-capable administrative surface is used and the resulting policy is read back.

## Required pre-merge verification

Immediately before any merge authorization:

1. re-read current `main` SHA;
2. verify the active PR head contains that exact `main` commit;
3. verify the active PR's current exact HEAD is the HEAD that passed the permanent CI;
4. ensure no superseded PR is also eligible for merge;
5. read back branch-protection/ruleset state and report it accurately;
6. if `main` or the PR head moved, invalidate the old evidence and rerun the required checks.

The permanent repository quality gate remains defined in `docs/AGENTS.md`; this plan does not replace it.

## Completion criterion

This plan can leave the current tree only after either:

- the target `main` protection/ruleset is applied and read-back verified; or
- the repository owner explicitly accepts protection as an external remaining limitation and moves that durable limitation into the appropriate current architecture/governance documentation or issue.

Until then, keep this plan Active but do not treat the already-implemented Git-tree tasks as unfinished work.
