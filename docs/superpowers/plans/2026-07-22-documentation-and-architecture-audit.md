# Documentation and Architecture Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile maintained explanatory Markdown with the current `main` implementation, then record a code-backed architecture audit without overstating research or production readiness.

**Architecture:** Documentation is treated as a maintained contract over the repository's enforced dependency layers, artifact identities, execution semantics, evidence boundaries, and capability status. The audit compares those claims against the current import-linter contracts and concrete training, execution, telemetry, catalog, Studio, and serving paths.

**Tech Stack:** Markdown, Python 3.12, Import Linter, pytest, Ruff, Mypy, GitHub Actions.

## Global Constraints

- Direct exchange routing remains outside the repository capability boundary.
- Production status remains `NO-GO` and no profitability claim may be introduced.
- Current implementation truth takes precedence over historical design documents.
- Historical verification evidence must retain exact commit and workflow identities.
- Architecture findings must distinguish confirmed defects, performance risks, maintenance risks, and deliberate compatibility paths.

---

### Task 1: Reconcile maintained entry-point documentation

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `START.md`
- Modify: `studio/README.md`

**Interfaces:**
- Consumes: current CLI extras, Studio telemetry contracts, PostgreSQL catalog contracts, serving bundle schema, and stateful execution defaults.
- Produces: copy-paste setup and capability descriptions that match current code.

- [ ] Correct dependency-install commands and serving schema references.
- [ ] Add concise descriptions of the PostgreSQL metadata catalog, append-only training telemetry, and conservative stateful order simulator.
- [ ] Keep exploratory visualization, deterministic evaluation, paper serving, and live exchange execution explicitly separated.
- [ ] Link the architecture and research-status documents from the entry points.

### Task 2: Reconcile architecture and research-status contracts

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: `.importlinter`, current package structure, PR #73 through PR #75 verification records, and serving bundle v5.
- Produces: one current responsibility map, dependency-order description, execution boundary, catalog boundary, telemetry boundary, and evidence-status summary.

- [ ] Make the documented layer order match `.importlinter` exactly.
- [ ] Document packages that are not currently covered by the layer stack instead of implying enforcement.
- [ ] Describe stateful order persistence, current-bar capacity, path assumptions, promotion evidence, and compatibility execution paths.
- [ ] Update the current evidence status while preserving `NO-GO` and historical archived results.

### Task 3: Record the architecture audit

**Files:**
- Create: `docs/verification/2026-07-22-documentation-and-architecture-audit.md`

**Interfaces:**
- Consumes: current source and maintained documentation at the audited commit.
- Produces: prioritized findings with concrete paths, impact, and remediation direction.

- [ ] Record the audited head and inspection scope.
- [ ] Record strengths that are enforced by tests or import contracts.
- [ ] Record confirmed gaps, including execution-path divergence, telemetry boundary enforcement, telemetry read scaling, strict parsing, ambiguous stream discovery, and duplicated canonical JSON logic.
- [ ] Separate architecture integrity from empirical profitability and operational release authorization.

### Task 4: Verify and publish

**Files:**
- Verify all modified Markdown and repository checks through the existing CI workflow.

**Interfaces:**
- Consumes: the complete documentation diff.
- Produces: a reviewable pull request with exact-head verification status.

- [ ] Review the final diff for stale schema versions, contradictory capability claims, and broken relative links.
- [ ] Run or observe `ruff check .`, `ruff format --check .`, `mypy .`, `lint-imports`, pytest/coverage, Studio checks, platform checks, PostgreSQL checks, and training-image checks through the repository workflow.
- [ ] Open a draft pull request and report any checks that remain pending or unavailable rather than claiming success without evidence.
