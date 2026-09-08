# Agent and documentation contract

This file is the detailed routing and maintenance contract for agents working in this repository. Root `AGENTS.md` is only the bootstrap; keep detailed documentation policy here so the two files cannot drift into competing authorities.

## Before working

1. Read `docs/README.md`.
2. Read the current authority for the subsystem you will change.
3. Inspect the relevant source, tests, CI, and package exports before changing documentation claims.
4. Confirm whether the task changes a maintained contract or only implementation details.

Do not infer current behavior from completed plans or Git chronology alone.

## Evidence and authority

Use this order when deciding what the current system actually does:

1. executable source, schemas, and tests;
2. current architecture/research documentation;
3. executable run instructions and examples;
4. empirical result artifacts;
5. Git history and completed design chronology.

A lower item does not silently override a higher one. A mismatch is a defect to investigate. Distinguish implementation, CI verification, empirical evaluation, profitability, and Production authorization; they are not interchangeable states.

## Documentation routing

- `docs/architecture/`: current architecture, responsibility boundaries, dependency direction, public contracts, and invariants.
- `docs/research/`: current research stage, evidence protocol, candidate comparison state, and explicit unknowns.
- `docs/specs/`: active or in-review design specifications only. Create it only when active work needs a normative design artifact.
- `docs/plans/`: active implementation plans only. Create it only for work that still needs execution tracking.
- `LICENSES/`: permanent licensing, provenance, SPDX license texts, and third-party notices.

Do not create `docs/history/` or `docs/archive/`. Git history is the archive for completed design/implementation chronology.

Use `lowercase-kebab-case.md` for new Markdown filenames unless an established permanent filename such as `README.md` or `AGENTS.md` applies.

## Update triggers

Update the smallest current authority that owns the changed contract:

- package/file ownership, dependency direction, public/private boundary → `docs/architecture/package-boundaries.md`;
- causality, strategy intent, risk/execution separation, accounting, replay flow → `docs/architecture/lean-core.md`;
- candidate set, fit/evaluation scope, evidence stage, winner/no-winner decision, profitability or Production state → `docs/research/current-status.md`;
- documentation layout, routing, retention, or agent procedure → `docs/README.md` and/or this file;
- user-facing run command or top-level project status → root `README.md` plus the owning current document.

When a code change makes a current statement false, update the owning document in the same scoped change. Link to one authority instead of copying normative text into several places.

## Retention

Current docs describe the current system. Completed specs and plans should be deleted after their durable requirements have been reflected into current architecture/research authorities. Historical existence is not a reason to keep compatibility documentation in the working tree.

Never delete or relocate required `LICENSES/` compliance material as documentation cleanup. Do not convert completed material into `docs/history/` or `docs/archive/`.

## Before finishing

Verify the observable result, not only that Markdown was edited:

- current documentation paths and links resolve;
- documented package/module paths exist;
- retired private paths and obsolete commands are absent from current authority docs;
- current research status does not overstate empirical evidence, profitability, or Production authorization;
- completed specs/plans are removed when no longer active;
- `LICENSES/` material is intact;
- documentation contract tests, relevant targeted tests, static checks, and required full CI run on the exact final HEAD.

Report any unverified item or residual risk explicitly.
