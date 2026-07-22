# Sequence Projection Equivalence — Clean Main Rebase Verification

## Scope

This note records the clean carry-forward of the already verified `AUD-CI-002` test remediation from the original stacked PR #86 onto the current `main` branch.

The original verification record remains:

- `docs/verification/2026-07-23-sequence-projection-equivalence.md`

That record contains the RED source contract, focused GREEN evidence, Ubuntu and Windows 100-repetition stability matrix, artifact digests, and cleanup-head verification from the original stacked branch. It is retained unchanged as historical evidence.

## Why a clean branch was required

PR #86 started from PR #84 head `703427cb162694a8b4990fe4e2ef17ea59a77f7a`. The narrower Live Training fix was later merged independently through PR #85, so PR #86 became non-mergeable and displayed unrelated stacked changes.

The remediation was therefore recreated from current `main` head:

```text
750ad2208fede137378dd4a0c0061ae91ef683d6
```

Initial clean commit:

```text
21aafb89c98c7f23e108405198c2d4867f50caeb
```

## Effective scope

The clean change contains only:

- the approved design;
- the approved implementation plan;
- the original evidence record;
- the sequence projection equivalence test update;
- this clean-rebase note.

No file under `trade_rl/` changes. Production sequence encoder behavior is unchanged.

## Evidence carried forward

The original branch proved:

- expected RED source-contract failure in run `29953598828`;
- focused GREEN in run `29953725692`;
- 100 successful repetitions on Ubuntu and 100 on Windows in run `29953836687`;
- cleanup-head CI success in run `29954450099`;
- PostgreSQL Catalog success in run `29954450146`;
- `1205 passed, 2 skipped`, total coverage `83.47%`.

The clean PR must also pass normal exact-head CI on its own head. Final run identifiers and artifact evidence are recorded in the PR after completion.

## Safety boundary

- Production remains `NO-GO`.
- No production module changes.
- No training, serving, execution, selection, promotion, release, artifact, or exchange-routing behavior changes.
- The change stabilizes test evidence without removing direct historical-equivalence coverage.
