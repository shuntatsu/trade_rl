# Documentation Refresh Design

## Status

Approved design for the selective documentation refresh on `integration/cost-aware-causal-teacher-final2`.

This document is a historical design artifact. After implementation, current behavior is authoritative in the maintained top-level documentation under `docs/`, repository `README.md`, `START.md`, source code, configuration schemas, and executable examples.

## Objective

Rebuild the documentation set around the current repository architecture and contracts so that a reader can distinguish:

- what the software currently implements;
- what CI has verified;
- what still requires empirical research evidence;
- what is operationally maintained;
- what is historical design or implementation material;
- what is explicitly not production-ready.

The refresh must reduce stale paths, duplicated explanations, historical implementation narrative in maintained documents, and plan/runbook mixing without deleting security, licensing, provenance, migration, serving, evaluation, or research-boundary information that remains contractually relevant.

## Non-goals

- No production logic changes solely to make documentation easier to write.
- No claim of profitability, production readiness, or empirical admission unsupported by evidence.
- No removal of historical specifications that are still useful for design provenance; historical material is relocated or clearly marked instead.
- No mass renaming of public schemas, CLI commands, artifact formats, or runtime paths.
- No cosmetic folder churn unrelated to documentation responsibility.
- No merging to `main` as part of the refresh.

## Current problems

### Maintained documentation and history are mixed

`docs/README.md` correctly states that top-level documentation is the maintained source of truth and that implementation plans are historical. However, `docs/operations/` currently contains a mixture of runbooks and design/implementation-plan documents. A reader cannot reliably infer whether an `operations/` file is executable current guidance or implementation history from its location alone.

### Maintained documents still contain historical narrative

Current architecture/reference documents include implementation-era detail such as specific pull-request references and transition history. This conflicts with the repository's own documentation rule that maintained documents describe the current system rather than development history.

### Stale path and location references remain possible

The repository has recently moved Docker assets, licensing notices, test support, and configuration ownership. Documentation needs to be revalidated against current repository paths rather than repaired piecemeal.

### Responsibility overlap is too high

README, ARCHITECTURE, RESEARCH_STATUS, UNIVERSAL_TRAINING, CONFIGURATION, and some operations documents repeat overlapping explanations of architecture, policy identity, production status, causal boundaries, and training stages. Duplication increases drift risk.

## Target information architecture

```text
README.md
  Repository entry point only:
  status / capability boundary / quickstart / repository map / key links

START.md
  Shortest maintained setup and execution path

docs/
├── README.md
│   Documentation index, source-of-truth rules, reading order
│
├── ARCHITECTURE.md
│   Responsibilities, dependency direction, data flow, identity,
│   execution, training/evaluation boundaries, serving boundaries
│
├── CONFIGURATION.md
│   Maintained schemas, configuration fields, versioned contracts
│
├── RESEARCH_STATUS.md
│   Implemented vs CI-verified vs empirically unverified vs NO-GO
│
├── SINGLE_SYMBOL.md
│   Maintained one-run/one-instrument contract
│
├── UNIVERSAL_TRAINING.md
│   Current U3-U6 shared-policy research workflow and causal teacher
│
├── REWARD_OBJECTIVE.md
├── EXECUTION_ROBUSTNESS.md
├── MULTITIMEFRAME_RESEARCH.md
├── BINANCE.md
├── NAUTILUS_MIGRATION.md
├── LICENSING.md
├── LICENSING_PROVENANCE.md
│   Maintained reference documents with one primary responsibility each
│
├── operations/
│   Only currently executable operational runbooks and verification guides
│
├── performance/
│   Hardware/performance-specific measurements and candidate settings
│
└── implementation-plans/
    ├── README.md
    ├── specs/
    └── plans/
        Historical design and implementation material
```

## Document responsibilities

### Repository README

Keep it short enough to answer:

1. What is Trade RL?
2. What is maintained today?
3. What is explicitly not production-ready?
4. What is the shortest way to run it?
5. Where should a reader go next?

Detailed model architecture, policy identity, evaluation semantics, serving internals, and historical compatibility details should link to their owning documents instead of being duplicated in README.

### docs/README.md

Act as the documentation router and source-of-truth policy. It should enumerate maintained documents by user goal, define the maintained/historical split, and state update rules.

### ARCHITECTURE.md

Describe only the current architecture contract:

- responsibility map and import direction;
- canonical data flow;
- causal data boundary;
- action/observation/reward ownership;
- policy/checkpoint identity;
- execution state flow;
- training/evaluation separation;
- artifact/catalog boundaries;
- serving/release boundaries;
- privileged runner boundary.

Historical PR numbers, implementation chronology, and superseded design discussion do not belong here.

### RESEARCH_STATUS.md

Use explicit status categories and avoid conflating them:

- implemented;
- CI verified;
- empirically evaluated;
- admitted/promoted;
- production authorized.

The current `NO-GO` status and absence of profitability claims remain explicit.

### UNIVERSAL_TRAINING.md

Remain the canonical reference for U3-U6, including:

- train-symbol data boundaries;
- causal-alpha teacher fitting and label cutoffs;
- holdout/admission separation;
- candidate selection;
- generator/checkpoint identity and resume fail-closed behavior;
- BC and critic warm-start sequence;
- U5/U6 shared teacher package;
- full-research execution;
- monitor semantics;
- software-success vs research-success distinction.

It must reflect the newly enforced causal-alpha checkpoint generator-code identity.

### operations/

Keep only documents that instruct a user how to execute or verify a currently supported operation. Design documents and implementation plans move into `docs/implementation-plans/` while preserving their historical content.

## Content migration rules

A document is moved to historical material if its primary purpose is one of:

- describing a proposed design before implementation;
- recording implementation sequencing;
- describing a completed hardening phase rather than a current operational procedure;
- naming transient branch/PR/commit state as the main subject.

A document remains maintained if it provides current:

- runtime behavior;
- operational procedure;
- configuration/reference semantics;
- migration/compatibility requirement still enforced by code;
- security/provenance/licensing requirement;
- research interpretation boundary.

Do not delete a historical document merely because the implementation is complete. Relocate or clearly classify it unless it is exact duplicate content with no unique provenance value.

## Cross-document ownership rules

Each concept has one primary owner:

- repository status and entry point: `README.md`
- documentation structure: `docs/README.md`
- system structure: `docs/ARCHITECTURE.md`
- configuration fields: `docs/CONFIGURATION.md`
- empirical/production state: `docs/RESEARCH_STATUS.md`
- maintained single-symbol contract: `docs/SINGLE_SYMBOL.md`
- Universal U3-U6: `docs/UNIVERSAL_TRAINING.md`
- reward semantics: `docs/REWARD_OBJECTIVE.md`
- execution-model limitations: `docs/EXECUTION_ROBUSTNESS.md`
- market-data acquisition: `docs/BINANCE.md`
- licensing: `docs/LICENSING*.md` and `LICENSES/`

Secondary documents link to the owner rather than reproducing long normative sections.

## Accuracy requirements

Documentation must be checked against current code and repository structure for at least:

- Python version and dependency commands;
- CLI commands;
- Docker paths and compose paths;
- maintained example config paths;
- configuration schema names and versions;
- action/observation/reward schema versions;
- policy/checkpoint/export/serving identity versions;
- licensing and third-party notice paths;
- Universal causal teacher behavior;
- generator-code checkpoint identity;
- current production/research status;
- Import Linter responsibility order;
- direct-exchange-routing non-goal.

If a statement cannot be supported by code, maintained configuration, executable examples, CI contract, or an explicitly identified empirical artifact, it must be weakened or removed rather than guessed.

## Documentation contract tests

The refresh should strengthen documentation tests so a green test means something observable.

### Required oracles

- Every maintained relative Markdown link resolves.
- Repository file references in maintained docs resolve when they are presented as current paths.
- Current CLI examples reference existing scripts/configs where statically checkable.
- Docker paths reference current `docker/` locations.
- Licensing links point to `LICENSE`, `LICENSES/`, and the actual notice file.
- Maintained schema/version claims agree with code constants or maintained configuration fixtures where practical.
- Maintained docs do not treat `implementation-plans/` as runtime authority.
- Historical plan files are not indexed as current runbooks.

Tests must fail closed when a required target is absent. They must not use `if path.exists()` / `if path.is_file()` to silently skip a required maintained target.

## Acceptance criteria

1. `README.md` becomes an entry point rather than a second architecture document.
2. `docs/README.md` accurately routes all maintained documentation and defines current-vs-history ownership.
3. Top-level maintained documents contain no stale Docker/license paths discovered during the refresh.
4. Current architecture/reference documents do not depend on transient PR numbers or completed implementation chronology for normative behavior.
5. `docs/operations/` contains only current runbooks/verification procedures; design/plan material is relocated to history.
6. `UNIVERSAL_TRAINING.md` documents generator-code identity in causal-alpha checkpoint resume behavior.
7. `RESEARCH_STATUS.md` preserves the distinction between software completion, CI verification, empirical evidence, profitability, and production authorization.
8. Documentation link/path contract tests are fail-closed for required current targets.
9. Security, provenance, licensing, migration, sealed-evaluation, causal, and production-NO-GO boundaries are preserved.
10. Documentation changes do not require unrelated production behavior changes.

## Invariants

- Production status remains `NO-GO` unless independently changed by actual authorization/evidence.
- No profitability claim is introduced.
- Direct exchange order routing remains documented as unimplemented.
- Historical artifacts are never rewritten to imply new identity semantics.
- Maintained one-symbol and Universal shared-policy contracts remain distinct.
- Causal and sealed-evaluation boundaries are not weakened.
- Licensing provenance remains traceable.
- Documentation history is preserved where it contains unique design/provenance information.

## Failure modes

- Broken internal links after moving historical documents.
- Current runbooks accidentally moved into history.
- Historical design treated as current runtime authority.
- Current docs referencing deleted/moved files.
- Schema/version text drifting from code.
- README becoming too detailed again and duplicating architecture.
- Removing a security or migration explanation as "old" when it still protects a current invariant.
- Documentation test passing despite not examining a required file.
- Universal teacher documentation omitting resume identity constraints.

## Test oracle

Correctness is not "Markdown renders". The observable oracle is:

- current paths/links resolve;
- maintained docs agree with current source/configuration contracts;
- historical docs are clearly separated;
- docs tests fail when a required maintained target is removed or renamed;
- full repository checks remain green after the documentation-only changes.

## Required test layers

- targeted documentation-contract tests;
- architecture/repository-layout tests affected by moved docs;
- license contract tests if licensing references change;
- Ruff/format/type/import checks to ensure no incidental repository regression;
- full Python test suite because documentation contract tests are part of the suite;
- frontend checks only if frontend documentation or referenced scripts/configuration are changed;
- same-head GitHub CI before final completion claim.

## Quality gate

Do not mark the refresh complete unless:

- all acceptance criteria are mapped to concrete changes;
- no maintained link/path failures remain;
- required docs tests are fail-closed;
- targeted tests pass;
- full required static checks pass;
- full pytest passes;
- relevant CI workflows pass on the final HEAD;
- the final diff is reviewed for accidental deletion of security/provenance/migration material;
- an independent/falsification review searches for stale paths, transient PR/commit references, silent documentation-test skips, and duplicated normative sections;
- remaining uncertainty and empirical limitations are explicitly reported.

## Implementation sequence

1. Inventory maintained and historical documents.
2. Classify each `docs/operations/` file as current runbook, verification guide, design, plan, or obsolete duplicate.
3. Move historical design/plan material without changing its meaning.
4. Rewrite documentation index and repository README around primary ownership.
5. Refresh current reference documents against source/configuration/examples.
6. Update Universal checkpoint identity documentation.
7. Strengthen documentation contract tests before removing stale assertions.
8. Run targeted documentation/layout/license tests.
9. Run static checks and full pytest.
10. Perform independent stale-reference/duplication/fail-closed review.
11. Run same-head CI and update PR metadata with verified results.

## Remaining design constraint

The refresh is intentionally selective rather than a blank-slate rewrite. Existing maintained documents that already express current contracts correctly should be edited in place, not rewritten merely for stylistic uniformity. This minimizes the risk of dropping subtle safety, provenance, or research-boundary semantics while still producing a coherent documentation system.
